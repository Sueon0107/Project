import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from tron1_env import Tron1BalanceEnv

os.chdir(os.path.dirname(os.path.abspath(__file__)))

N_ENVS = 16

def make_env():
    def _init():
        return Monitor(Tron1BalanceEnv())
    return _init

def main():
    print("환경 검증 중...")
    check_env(Tron1BalanceEnv(), warn=True)
    print("환경 검증 완료\n")

    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path="./checkpoints_v4/",
        name_prefix="tron1_balance"
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        device="cpu",
        tensorboard_log="./logs_v4/",
    )

    print(f"학습 시작! ({N_ENVS}개 환경 병렬)")
    model.learn(
        total_timesteps=5_000_000,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save("tron1_balance_v4_final")
    print("\n학습 완료! 모델 저장: tron1_balance_v4_final.zip")
    env.close()

if __name__ == "__main__":
    main()
