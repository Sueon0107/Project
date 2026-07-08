import jax
import jax.numpy as jp

from tron1_mjx_env import Tron1MjxEnv


def main():
    env = Tron1MjxEnv.create()
    key = jax.random.PRNGKey(0)
    state, obs = env.reset(key)

    step_fn = jax.jit(env.step)
    action = jp.zeros(env.model.nu)

    steps = 0
    for _ in range(100):
        state, obs, reward, done, truncated = step_fn(state, action)
        steps += 1
        if bool(done) or bool(truncated):
            break

    print("obs_shape", obs.shape)
    print("steps", steps)
    print("reward", float(reward))
    print("done", bool(done))
    print("truncated", bool(truncated))
    print("height", float(state["data"].qpos[2]))

    batch_size = 32
    keys = jax.random.split(key, batch_size)
    states, obs = jax.vmap(env.reset)(keys)
    batched_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0)))
    actions = jp.zeros((batch_size, env.model.nu))
    states, obs, reward, done, truncated = batched_step(states, actions)

    print("batch_obs_shape", obs.shape)
    print("batch_reward_shape", reward.shape)
    print("batch_done_shape", done.shape)


if __name__ == "__main__":
    main()
