import os
import glob
import pickle
import re
import time
from dataclasses import dataclass

import jax
import jax.numpy as jp

from tron1_mjx_env import Tron1MjxEnv


os.chdir(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class TrainConfig:
    n_envs: int = 64
    total_timesteps: int = 1_000_000
    rollout_len: int = 128
    update_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_coef: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_size: int = 64
    init_log_std: float = -0.3
    checkpoint_every: int = 100_000
    seed: int = 0


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_bool(name, default):
    return os.environ.get(name, str(int(default))).lower() in {"1", "true", "yes", "on"}


CONFIG = TrainConfig(
    n_envs=_env_int("MJX_N_ENVS", 64),
    total_timesteps=_env_int("MJX_TOTAL_TIMESTEPS", 1_000_000),
    rollout_len=_env_int("MJX_ROLLOUT_LEN", 128),
    update_epochs=_env_int("MJX_UPDATE_EPOCHS", 4),
    num_minibatches=_env_int("MJX_NUM_MINIBATCHES", 4),
    learning_rate=_env_float("MJX_LEARNING_RATE", 3e-4),
    clip_coef=_env_float("MJX_CLIP_COEF", 0.2),
    ent_coef=_env_float("MJX_ENT_COEF", 0.001),
    vf_coef=_env_float("MJX_VF_COEF", 0.5),
    max_grad_norm=_env_float("MJX_MAX_GRAD_NORM", 0.5),
    init_log_std=_env_float("MJX_INIT_LOG_STD", -0.3),
    checkpoint_every=_env_int("MJX_CHECKPOINT_EVERY", 100_000),
)
SAVE_ENABLED = _env_bool("MJX_SAVE", True)
RUN_NAME = os.environ.get("MJX_RUN_NAME", "v12")
CHECKPOINT_DIR = f"./mjx_checkpoints_{RUN_NAME}/"
FINAL_PATH = f"tron1_balance_{RUN_NAME}_final.pkl"
BEST_PATH = f"tron1_balance_{RUN_NAME}_best.pkl"


def init_layer(key, in_dim, out_dim, scale=1.0):
    w_key, _ = jax.random.split(key)
    w = jax.random.normal(w_key, (in_dim, out_dim)) * (scale / jp.sqrt(in_dim))
    b = jp.zeros((out_dim,))
    return {"w": w, "b": b}


def init_params(key, obs_dim, action_dim, hidden_size, init_log_std):
    keys = jax.random.split(key, 7)
    return {
        "policy": [
            init_layer(keys[0], obs_dim, hidden_size, jp.sqrt(2.0)),
            init_layer(keys[1], hidden_size, hidden_size, jp.sqrt(2.0)),
            init_layer(keys[2], hidden_size, action_dim, 0.01),
        ],
        "value": [
            init_layer(keys[3], obs_dim, hidden_size, jp.sqrt(2.0)),
            init_layer(keys[4], hidden_size, hidden_size, jp.sqrt(2.0)),
            init_layer(keys[5], hidden_size, 1, 1.0),
        ],
        "log_std": jp.ones((action_dim,)) * init_log_std,
    }


def mlp(layers, x):
    for layer in layers[:-1]:
        x = jp.tanh(x @ layer["w"] + layer["b"])
    return x @ layers[-1]["w"] + layers[-1]["b"]


def forward(params, obs):
    mean = mlp(params["policy"], obs)
    value = mlp(params["value"], obs).squeeze(-1)
    log_std = jp.broadcast_to(params["log_std"], mean.shape)
    return mean, log_std, value


def gaussian_log_prob(action, mean, log_std):
    var = jp.exp(2.0 * log_std)
    log_prob = -0.5 * (((action - mean) ** 2) / var + 2.0 * log_std + jp.log(2.0 * jp.pi))
    return log_prob.sum(axis=-1)


def gaussian_entropy(log_std):
    entropy = 0.5 + 0.5 * jp.log(2.0 * jp.pi) + log_std
    return entropy.sum(axis=-1)


def sample_action(params, obs, key):
    mean, log_std, value = forward(params, obs)
    noise = jax.random.normal(key, mean.shape)
    action = mean + jp.exp(log_std) * noise
    action = jp.clip(action, -1.0, 1.0)
    log_prob = gaussian_log_prob(action, mean, log_std)
    return action, log_prob, value


def tree_zeros_like(tree):
    return jax.tree.map(jp.zeros_like, tree)


def adam_init(params):
    return {
        "m": tree_zeros_like(params),
        "v": tree_zeros_like(params),
        "t": jp.array(0, dtype=jp.int32),
    }


def tree_l2_norm(tree):
    leaves = jax.tree.leaves(tree)
    return jp.sqrt(sum([jp.sum(x * x) for x in leaves]))


def clip_grads(grads, max_norm):
    norm = tree_l2_norm(grads)
    scale = jp.minimum(1.0, max_norm / (norm + 1e-8))
    return jax.tree.map(lambda g: g * scale, grads)


def adam_update(params, opt_state, grads, learning_rate):
    grads = clip_grads(grads, CONFIG.max_grad_norm)
    t = opt_state["t"] + 1
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = jax.tree.map(lambda m, g: beta1 * m + (1.0 - beta1) * g, opt_state["m"], grads)
    v = jax.tree.map(lambda v, g: beta2 * v + (1.0 - beta2) * (g * g), opt_state["v"], grads)
    m_hat = jax.tree.map(lambda x: x / (1.0 - beta1**t), m)
    v_hat = jax.tree.map(lambda x: x / (1.0 - beta2**t), v)
    params = jax.tree.map(
        lambda p, mh, vh: p - learning_rate * mh / (jp.sqrt(vh) + eps),
        params,
        m_hat,
        v_hat,
    )
    return params, {"m": m, "v": v, "t": t}


def compute_gae(rewards, dones, values, last_value, gamma, gae_lambda):
    def scan_fn(carry, xs):
        next_value, next_advantage = carry
        reward, done, value = xs
        next_non_terminal = 1.0 - done
        delta = reward + gamma * next_value * next_non_terminal - value
        advantage = delta + gamma * gae_lambda * next_non_terminal * next_advantage
        return (value, advantage), advantage

    _, advantages = jax.lax.scan(
        scan_fn,
        (last_value, jp.zeros_like(last_value)),
        (rewards[::-1], dones[::-1], values[::-1]),
    )
    advantages = advantages[::-1]
    returns = advantages + values
    return advantages, returns


def flatten_time_batch(x):
    return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])


def save_checkpoint(path, params, opt_state, total_steps):
    payload = {
        "params": jax.device_get(params),
        "opt_state": jax.device_get(opt_state),
        "total_steps": total_steps,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["params"], payload["opt_state"], int(payload["total_steps"])


def find_latest_checkpoint(directory):
    files = glob.glob(os.path.join(directory, f"tron1_balance_{RUN_NAME}_*_steps.pkl"))
    files += glob.glob(os.path.join(directory, "tron1_balance_mjx_*_steps.pkl"))
    if os.path.exists(FINAL_PATH):
        files.append(FINAL_PATH)
    if not files:
        return None, 0

    def extract_steps(path):
        match = re.search(r"_(\d+)_steps\.pkl$", path)
        if match:
            return int(match.group(1))
        try:
            _, _, steps = load_checkpoint(path)
            return steps
        except Exception:
            return 0

    latest = max(files, key=extract_steps)
    return latest, extract_steps(latest)


def main():
    env = Tron1MjxEnv.create()
    config = CONFIG
    if SAVE_ENABLED:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    _, sample_obs = env.reset(jax.random.PRNGKey(config.seed + 12345))
    obs_dim = int(sample_obs.shape[0])
    action_dim = env.model.nu
    key = jax.random.PRNGKey(config.seed)
    key, params_key, reset_key = jax.random.split(key, 3)
    params = init_params(
        params_key,
        obs_dim,
        action_dim,
        config.hidden_size,
        config.init_log_std,
    )
    opt_state = adam_init(params)
    total_steps = 0
    best_ep_len = -1.0

    if SAVE_ENABLED:
        checkpoint_path, resumed_at = find_latest_checkpoint(CHECKPOINT_DIR)
        if checkpoint_path:
            params, opt_state, total_steps = load_checkpoint(checkpoint_path)
            best_ep_len = 0.0
            print(f"체크포인트 발견: {checkpoint_path} ({resumed_at:,} 스텝)")
        else:
            print("새로 학습 시작")

    reset_keys = jax.random.split(reset_key, config.n_envs)
    states, obs = jax.vmap(env.reset)(reset_keys)
    ep_returns = jp.zeros((config.n_envs,), dtype=jp.float32)
    ep_lengths = jp.zeros((config.n_envs,), dtype=jp.int32)

    batched_step = jax.vmap(env.step, in_axes=(0, 0))
    batched_reset_if_done = jax.vmap(env.reset_if_done, in_axes=(0, 0, 0, 0, 0))

    def rollout_step(carry, _):
        params, key, states, obs, ep_returns, ep_lengths = carry
        key, action_key, reset_key = jax.random.split(key, 3)
        actions, log_probs, values = sample_action(params, obs, action_key)
        next_states, next_obs, rewards, dones, truncated = batched_step(states, actions)
        ep_returns = ep_returns + rewards
        ep_lengths = ep_lengths + 1
        reset_keys = jax.random.split(reset_key, config.n_envs)
        next_states, reset_obs, finished = batched_reset_if_done(
            next_states, next_obs, dones, truncated, reset_keys
        )
        finished_f = finished.astype(jp.float32)
        transition = {
            "obs": obs,
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "rewards": rewards,
            "dones": finished_f,
            "episode_returns": jp.where(finished, ep_returns, 0.0),
            "episode_lengths": jp.where(finished, ep_lengths, 0),
        }
        ep_returns = jp.where(finished, 0.0, ep_returns)
        ep_lengths = jp.where(finished, 0, ep_lengths)
        return (params, key, next_states, reset_obs, ep_returns, ep_lengths), transition

    def collect_rollout(params, key, states, obs, ep_returns, ep_lengths):
        carry, batch = jax.lax.scan(
            rollout_step,
            (params, key, states, obs, ep_returns, ep_lengths),
            None,
            length=config.rollout_len,
        )
        _, key, states, last_obs, ep_returns, ep_lengths = carry
        _, _, last_value = forward(params, last_obs)
        advantages, returns = compute_gae(
            batch["rewards"],
            batch["dones"],
            batch["values"],
            last_value,
            config.gamma,
            config.gae_lambda,
        )
        batch["advantages"] = advantages
        batch["returns"] = returns
        return (key, states, last_obs, ep_returns, ep_lengths), batch

    def ppo_loss(params, batch):
        mean, log_std, values = forward(params, batch["obs"])
        new_log_probs = gaussian_log_prob(batch["actions"], mean, log_std)
        entropy = gaussian_entropy(log_std).mean()

        ratio = jp.exp(new_log_probs - batch["log_probs"])
        policy_loss_1 = batch["advantages"] * ratio
        policy_loss_2 = batch["advantages"] * jp.clip(
            ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef
        )
        policy_loss = -jp.minimum(policy_loss_1, policy_loss_2).mean()

        value_loss = 0.5 * ((batch["returns"] - values) ** 2).mean()
        loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy
        approx_kl = (batch["log_probs"] - new_log_probs).mean()
        return loss, {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
        }

    def update_minibatch(carry, minibatch):
        params, opt_state = carry
        (loss, metrics), grads = jax.value_and_grad(ppo_loss, has_aux=True)(params, minibatch)
        params, opt_state = adam_update(params, opt_state, grads, config.learning_rate)
        metrics["loss"] = loss
        return (params, opt_state), metrics

    def train_update(params, opt_state, key, states, obs, ep_returns, ep_lengths):
        key, rollout_key, shuffle_key = jax.random.split(key, 3)
        (rollout_key, states, obs, ep_returns, ep_lengths), batch = collect_rollout(
            params, rollout_key, states, obs, ep_returns, ep_lengths
        )

        train_batch = {
            key: value
            for key, value in batch.items()
            if key not in {"episode_returns", "episode_lengths"}
        }
        train_batch = jax.tree.map(flatten_time_batch, train_batch)
        advantages = train_batch["advantages"]
        train_batch["advantages"] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = config.n_envs * config.rollout_len
        minibatch_size = batch_size // config.num_minibatches

        def epoch_update(carry, epoch_key):
            params, opt_state = carry
            permutation = jax.random.permutation(epoch_key, batch_size)
            shuffled = jax.tree.map(lambda x: x[permutation], train_batch)
            minibatches = jax.tree.map(
                lambda x: x.reshape((config.num_minibatches, minibatch_size) + x.shape[1:]),
                shuffled,
            )
            (params, opt_state), metrics = jax.lax.scan(
                update_minibatch, (params, opt_state), minibatches
            )
            return (params, opt_state), metrics

        epoch_keys = jax.random.split(shuffle_key, config.update_epochs)
        (params, opt_state), metrics = jax.lax.scan(
            epoch_update, (params, opt_state), epoch_keys
        )
        done_count = batch["dones"].sum()
        ep_rew_mean = batch["episode_returns"].sum() / jp.maximum(done_count, 1.0)
        ep_len_mean = batch["episode_lengths"].sum() / jp.maximum(done_count, 1.0)
        rollout_metrics = {
            "reward_mean": train_batch["rewards"].mean(),
            "done_count": done_count,
            "return_mean": train_batch["returns"].mean(),
            "ep_rew_mean": ep_rew_mean,
            "ep_len_mean": ep_len_mean,
        }
        metrics = jax.tree.map(lambda x: x[-1].mean(), metrics)
        metrics.update(rollout_metrics)
        return params, opt_state, key, states, obs, ep_returns, ep_lengths, metrics

    train_update = jax.jit(train_update)

    steps_per_update = config.n_envs * config.rollout_len
    if steps_per_update % config.num_minibatches != 0:
        raise ValueError(
            "n_envs * rollout_len must be divisible by num_minibatches: "
            f"{steps_per_update} % {config.num_minibatches} != 0"
        )

    remaining_steps = max(0, config.total_timesteps - total_steps)
    num_updates = remaining_steps // steps_per_update
    if remaining_steps > 0 and num_updates == 0:
        num_updates = 1

    if num_updates == 0:
        print(f"이미 목표 스텝에 도달했습니다: {total_steps:,}/{config.total_timesteps:,}")
        return

    print("MJX PPO 학습 시작")
    print(f"환경 수: {config.n_envs}, rollout: {config.rollout_len}, update 수: {num_updates}")
    print(f"한 update당 transition: {steps_per_update:,}\n")
    print(
        "하이퍼파라미터: "
        f"lr={config.learning_rate:g}, clip={config.clip_coef:g}, "
        f"ent={config.ent_coef:g}, grad_norm={config.max_grad_norm:g}, "
        f"init_log_std={config.init_log_std:g}\n"
    )

    started = time.perf_counter()
    for update in range(1, num_updates + 1):
        params, opt_state, key, states, obs, ep_returns, ep_lengths, metrics = train_update(
            params, opt_state, key, states, obs, ep_returns, ep_lengths
        )
        jax.block_until_ready(metrics["loss"])
        total_steps += steps_per_update

        elapsed = time.perf_counter() - started
        sps = int(total_steps / max(elapsed, 1e-6))
        print(
            f"[{update:04d}/{num_updates}] "
            f"steps={total_steps:,} "
            f"reward={float(metrics['reward_mean']):+.3f} "
            f"ep_rew={float(metrics['ep_rew_mean']):+.3f} "
            f"ep_len={float(metrics['ep_len_mean']):.1f} "
            f"done={int(metrics['done_count'])} "
            f"loss={float(metrics['loss']):+.3f} "
            f"v_loss={float(metrics['value_loss']):+.3f} "
            f"entropy={float(metrics['entropy']):+.3f} "
            f"sps={sps:,}"
        )

        ep_len = float(metrics["ep_len_mean"])
        if SAVE_ENABLED and ep_len > best_ep_len:
            best_ep_len = ep_len
            save_checkpoint(BEST_PATH, params, opt_state, total_steps)
            print(f"  best 저장: {BEST_PATH} (ep_len={best_ep_len:.1f})")

        if SAVE_ENABLED and total_steps % config.checkpoint_every < steps_per_update:
            path = os.path.join(CHECKPOINT_DIR, f"tron1_balance_{RUN_NAME}_{total_steps}_steps.pkl")
            save_checkpoint(path, params, opt_state, total_steps)

    if SAVE_ENABLED:
        save_checkpoint(FINAL_PATH, params, opt_state, total_steps)
        print(f"\n학습 완료! 모델 저장: {FINAL_PATH}")
    else:
        print("\n학습 완료! MJX_SAVE=0이라 모델 저장은 건너뜀")


if __name__ == "__main__":
    main()
