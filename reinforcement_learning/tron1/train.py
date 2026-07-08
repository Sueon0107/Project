import os
import re
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from tron1_env import Tron1BalanceEnv

os.chdir(os.path.dirname(os.path.abspath(__file__)))

N_ENVS          = 16
TOTAL_TIMESTEPS = 3_000_000
CHECKPOINT_DIR  = "./checkpoints_v6/"

def make_env():
    def _init():
        return Monitor(Tron1BalanceEnv())
    return _init

def find_latest_checkpoint(directory):
    files = glob.glob(os.path.join(directory, "*_steps.zip"))
    if not files:
        return None, 0
    def extract_steps(f):
        m = re.search(r'_(\d+)_steps\.zip$', f)
        return int(m.group(1)) if m else 0
    latest = max(files, key=extract_steps)
    return latest, extract_steps(latest)

def main():
    print("환경 검증 중...")
    check_env(Tron1BalanceEnv(), warn=True)
    print("환경 검증 완료\n")

    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="tron1_balance"
    )

    checkpoint_path, resumed_at = find_latest_checkpoint(CHECKPOINT_DIR)

    if checkpoint_path:
        print(f"체크포인트 발견: {checkpoint_path} ({resumed_at:,} 스텝)")
        model = PPO.load(checkpoint_path, env=env, device="cpu",
                         verbose=1, tensorboard_log="./logs_v6/")
        remaining = TOTAL_TIMESTEPS - resumed_at
    else:
        print("새로 학습 시작")
        model = PPO(
            "MlpPolicy", env, verbose=1,
            n_steps=2048, batch_size=64, n_epochs=10,
            learning_rate=3e-4, device="cpu",
            tensorboard_log="./logs_v6/",
        )
        remaining = TOTAL_TIMESTEPS

    print(f"학습 시작! ({N_ENVS}개 환경 병렬, 남은 스텝: {remaining:,})")
    model.learn(
        total_timesteps=remaining,
        callback=checkpoint_cb,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    model.save("tron1_balance_v6_final")
    print("\n학습 완료! 모델 저장: tron1_balance_v6_final.zip")
    env.close()

if __name__ == "__main__":
    main()
