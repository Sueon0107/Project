import os

import jax
import jax.numpy as jp
import numpy as np

from tron1_env import Tron1BalanceEnv
from tron1_mjx_env import Tron1MjxEnv
from tron1_mjx_env import quat_to_rp as mjx_quat_to_rp


os.chdir(os.path.dirname(os.path.abspath(__file__)))

N_STEPS = int(os.environ.get("COMPARE_STEPS", 60))
ACTION_MODE = os.environ.get("COMPARE_ACTION", "zero")
SEED = int(os.environ.get("COMPARE_SEED", 0))


def mujoco_roll_pitch(env):
    return env._quat_to_rp(env.data.qpos[3:7])


def mjx_roll_pitch(data):
    roll, pitch = mjx_quat_to_rp(data.qpos[3:7])
    return float(roll), float(pitch)


def make_action(step, n_joints):
    if ACTION_MODE == "zero":
        return np.zeros(n_joints, dtype=np.float32)
    if ACTION_MODE == "sin":
        x = 0.2 * np.sin(step * 0.2)
        return np.array([0.0, x, -x, 0.0, -x, x], dtype=np.float32)
    raise ValueError(f"Unknown COMPARE_ACTION={ACTION_MODE!r}")


def main():
    np.random.seed(SEED)
    mujoco_env = Tron1BalanceEnv()
    mjx_env = Tron1MjxEnv.create()

    obs_mj, _ = mujoco_env.reset(seed=SEED)
    mjx_state, obs_mjx = mjx_env.reset(jax.random.PRNGKey(SEED))
    mjx_step = jax.jit(mjx_env.step)

    print(f"action_mode={ACTION_MODE}, steps={N_STEPS}, seed={SEED}")
    print(
        "step | "
        "mj_h mj_roll mj_pitch mj_reward mj_done | "
        "mx_h mx_roll mx_pitch mx_reward mx_done | "
        "dh droll dpitch"
    )

    for step in range(1, N_STEPS + 1):
        action = make_action(step, mujoco_env.n_joints)

        obs_mj, reward_mj, done_mj, truncated_mj, _ = mujoco_env.step(action)
        mjx_state, obs_mjx, reward_mjx, done_mjx, truncated_mjx = mjx_step(
            mjx_state, jp.asarray(action)
        )

        mj_h = float(mujoco_env.data.qpos[2])
        mj_roll, mj_pitch = mujoco_roll_pitch(mujoco_env)

        mx_data = mjx_state["data"]
        mx_h = float(mx_data.qpos[2])
        mx_roll, mx_pitch = mjx_roll_pitch(mx_data)

        print(
            f"{step:04d} | "
            f"{mj_h:+.3f} {mj_roll:+.3f} {mj_pitch:+.3f} "
            f"{reward_mj:+.3f} {int(done_mj or truncated_mj)} | "
            f"{mx_h:+.3f} {mx_roll:+.3f} {mx_pitch:+.3f} "
            f"{float(reward_mjx):+.3f} {int(bool(done_mjx) or bool(truncated_mjx))} | "
            f"{(mx_h - mj_h):+.3f} {(mx_roll - mj_roll):+.3f} {(mx_pitch - mj_pitch):+.3f}"
        )

        if done_mj or truncated_mj or bool(done_mjx) or bool(truncated_mjx):
            print("terminated")
            break

    mujoco_env.close()


if __name__ == "__main__":
    main()
