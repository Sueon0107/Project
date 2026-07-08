import time

import jax
import jax.numpy as jp

from tron1_mjx_env import Tron1MjxEnv


BATCH_SIZE = 128
ROLLOUT_LEN = 256


def main():
    env = Tron1MjxEnv.create()
    key = jax.random.PRNGKey(1)
    reset_keys = jax.random.split(key, BATCH_SIZE)
    states, obs = jax.vmap(env.reset)(reset_keys)

    batched_step = jax.vmap(env.step, in_axes=(0, 0))
    batched_reset_if_done = jax.vmap(env.reset_if_done, in_axes=(0, 0, 0, 0, 0))

    def rollout_step(carry, _):
        key, states, obs, ep_returns, ep_lengths, completed_returns, completed_lengths, completed_count = carry
        key, action_key, reset_key = jax.random.split(key, 3)
        actions = jax.random.uniform(
            action_key,
            (BATCH_SIZE, env.model.nu),
            minval=-1.0,
            maxval=1.0,
        )

        states, next_obs, reward, done, truncated = batched_step(states, actions)
        finished = done | truncated
        ep_returns = ep_returns + reward
        ep_lengths = ep_lengths + 1

        completed_returns = completed_returns + jp.where(finished, ep_returns, 0.0)
        completed_lengths = completed_lengths + jp.where(finished, ep_lengths, 0)
        completed_count = completed_count + finished.astype(jp.int32)

        reset_keys = jax.random.split(reset_key, BATCH_SIZE)
        states, next_obs, finished = batched_reset_if_done(
            states, next_obs, done, truncated, reset_keys
        )
        ep_returns = jp.where(finished, 0.0, ep_returns)
        ep_lengths = jp.where(finished, 0, ep_lengths)

        carry = (
            key,
            states,
            next_obs,
            ep_returns,
            ep_lengths,
            completed_returns,
            completed_lengths,
            completed_count,
        )
        metrics = {
            "reward": reward,
            "done": done,
            "truncated": truncated,
        }
        return carry, metrics

    @jax.jit
    def run_rollout(key, states, obs):
        zeros_f = jp.zeros((BATCH_SIZE,), dtype=jp.float32)
        zeros_i = jp.zeros((BATCH_SIZE,), dtype=jp.int32)
        carry = (
            key,
            states,
            obs,
            zeros_f,
            zeros_i,
            zeros_f,
            zeros_i,
            zeros_i,
        )
        carry, metrics = jax.lax.scan(rollout_step, carry, None, length=ROLLOUT_LEN)
        return carry, metrics

    started = time.perf_counter()
    carry, metrics = run_rollout(key, states, obs)
    jax.block_until_ready(metrics["reward"])
    elapsed = time.perf_counter() - started

    completed_returns = carry[5]
    completed_lengths = carry[6]
    completed_count = carry[7]
    total_completed = completed_count.sum()
    mean_return = completed_returns.sum() / jp.maximum(total_completed, 1)
    mean_length = completed_lengths.sum() / jp.maximum(total_completed, 1)

    print("batch_size", BATCH_SIZE)
    print("rollout_len", ROLLOUT_LEN)
    print("transitions", BATCH_SIZE * ROLLOUT_LEN)
    print("elapsed_sec", round(elapsed, 3))
    print("completed_episodes", int(total_completed))
    print("mean_completed_return", float(mean_return))
    print("mean_completed_length", float(mean_length))
    print("last_reward_mean", float(metrics["reward"][-1].mean()))


if __name__ == "__main__":
    main()
