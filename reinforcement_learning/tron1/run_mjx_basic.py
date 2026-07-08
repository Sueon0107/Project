import os
import pickle
import sys
import time

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np


os.chdir(os.path.dirname(os.path.abspath(__file__)))

XML_PATH = "tron1-mujoco-sim/robot-description/pointfoot/PF_TRON1A/xml/robot.xml"
DEFAULT_MODEL_PATH = "tron1_balance_basic_v2_final.pkl"


class ViewerEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.n_joints = self.model.nu
        self.steps_per_ctrl = max(1, int(round(1.0 / 60.0 / self.model.opt.timestep)))
        self.max_steps = 1000
        self.action_scale = 0.8
        self._step_count = 0
        self._target_q = np.zeros(self.n_joints)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[7:] += np.random.uniform(-0.05, 0.05, self.n_joints)
        self.data.qvel[:] = 0.0
        self._step_count = 0
        self._target_q = np.zeros(self.n_joints)
        mujoco.mj_forward(self.model, self.data)
        return self.obs()

    def obs(self):
        q = self.data.qpos[7:].copy()
        dq = self.data.qvel[6:].copy()
        roll, pitch = self._quat_to_rp(self.data.qpos[3:7])
        rp = np.array([roll, pitch])
        ang_vel = self.data.qvel[3:6].copy()
        return np.concatenate([q, dq, rp, ang_vel]).astype(np.float32)

    def step(self, action):
        target = action * self.action_scale
        self._target_q = 0.2 * target + 0.8 * self._target_q
        self._target_q = np.clip(self._target_q, -1.5, 1.5)

        kp, kd = 60.0, 3.0
        for _ in range(self.steps_per_ctrl):
            q = self.data.qpos[7:]
            dq = self.data.qvel[6:]
            self.data.ctrl[:] = kp * (self._target_q - q) + kd * (0.0 - dq)
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self.obs()
        done = self._is_fallen()
        truncated = self._step_count >= self.max_steps
        reward = self._reward(done)
        return obs, reward, done, truncated

    def _quat_to_rp(self, quat):
        w, x, y, z = quat
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        return roll, pitch

    def _is_fallen(self):
        h = self.data.qpos[2]
        roll, pitch = self._quat_to_rp(self.data.qpos[3:7])
        return h < 0.40 or abs(pitch) > 0.8 or abs(roll) > 0.8

    def _reward(self, fallen):
        h = self.data.qpos[2]
        roll, pitch = self._quat_to_rp(self.data.qpos[3:7])
        r_height = np.clip((h - 0.40) / (0.78 - 0.40), -1.0, 1.0)
        r_upright = 1.0 - abs(pitch) / 0.8 - abs(roll) / 0.8
        reward = r_height + r_upright
        if fallen:
            reward -= 30.0
        return reward

    def close(self):
        pass


def load_params(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["params"], payload.get("total_steps", 0)


def mlp(layers, x):
    for layer in layers[:-1]:
        x = jp.tanh(x @ layer["w"] + layer["b"])
    return x @ layers[-1]["w"] + layers[-1]["b"]


def forward(params, obs):
    mean = mlp(params["policy"], obs)
    value = mlp(params["value"], obs).squeeze(-1)
    log_std = jp.broadcast_to(params["log_std"], mean.shape)
    return mean, log_std, value


def predict_action(params, obs):
    obs = jp.asarray(obs[None, :], dtype=jp.float32)
    mean, _, _ = forward(params, obs)
    return jp.clip(mean[0], -1.0, 1.0)


def run():
    model_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "MJX_MODEL_PATH", DEFAULT_MODEL_PATH
    )
    params, total_steps = load_params(model_path)
    policy = jax.jit(lambda obs: predict_action(params, obs))

    env = ViewerEnv()
    print(f"모델 로드 완료: {model_path} ({total_steps:,} 스텝)")
    print("basic 뷰어 실행 중... (창 닫으면 종료)\n")

    control_dt = 1.0 / 60.0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20

        episode = 0
        while viewer.is_running():
            obs = env.reset()
            episode += 1
            total_reward = 0.0
            steps = 0

            print(f"[에피소드 {episode}] 시작")

            while viewer.is_running():
                action = policy(obs)
                action = jax.device_get(action)
                obs, reward, done, truncated = env.step(action)
                total_reward += reward
                steps += 1

                t_start = time.perf_counter()
                viewer.sync()
                remaining = control_dt - (time.perf_counter() - t_start)
                if remaining > 0:
                    time.sleep(remaining)

                if truncated:
                    print(
                        f"  -> 시간 종료 | {steps}스텝 ({steps / 60:.1f}초) "
                        f"| 보상합계: {total_reward:.1f}"
                    )
                    time.sleep(1.0)
                    break

                if done:
                    print(
                        f"  -> 넘어짐 | {steps}스텝 ({steps / 60:.1f}초) "
                        f"| 보상합계: {total_reward:.1f}"
                    )
                    time.sleep(1.0)
                    break

    env.close()


if __name__ == "__main__":
    run()
