import os
import time
import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from tron1_env import Tron1BalanceEnv

os.chdir(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = "tron1_balance_v4_final"

def run():
    env   = Tron1BalanceEnv()
    model = PPO.load(MODEL_PATH, env=env)

    print(f"모델 로드 완료: {MODEL_PATH}.zip")
    print("뷰어 실행 중... (창 닫으면 종료)\n")

    steps_per_frame = max(1, int(round(1.0 / 60.0 / env.model.opt.timestep)))

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.distance  = 3.0
        v.cam.elevation = -20

        episode = 0
        while v.is_running():
            obs, _ = env.reset()
            episode += 1
            total_reward = 0
            steps = 0

            print(f"[에피소드 {episode}] 시작")

            while v.is_running():
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1

                # 뷰어 sync (물리는 env.step 안에서 이미 실행됨)
                v.sync()

                if truncated:
                    env._step_count = 0  # 카운터만 초기화, 물리 상태 유지
                if done:
                    print(f"  → 넘어짐 | {steps}스텝 ({steps/60:.1f}초) | 보상합계: {total_reward:.1f}")
                    time.sleep(1.0)
                    break

    env.close()

if __name__ == "__main__":
    run()
