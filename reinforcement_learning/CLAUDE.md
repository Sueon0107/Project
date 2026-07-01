# TRON1 보행 강화학습 프로젝트

## 프로젝트 개요

**목표**: TRON1 2족 로봇의 균형잡기 → 보행 강화학습 구현

**담당**: 학부연구생

**현재 단계**: MuJoCo 시뮬레이션 기반 PPO 균형잡기 학습 실험 중

---

## 기술 스택 (실제 사용 중)

| 항목 | 버전/도구 |
|------|---------|
| 운영체제 | Ubuntu 24.04 LTS |
| ROS | ROS 2 Jazzy (강화학습엔 미사용) |
| 시뮬레이터 | MuJoCo (tron1-mujoco-sim, 순수 Python) |
| 강화학습 프레임워크 | Stable-Baselines3 (PPO) |
| 프로그래밍 언어 | Python 3 |

> **왜 Gazebo 안 쓰나?**
> tron1-gazebo-ros2는 Gazebo Classic 기반 → ROS 2 Jazzy(Gazebo Harmonic)와 호환 안 됨
> → tron1-mujoco-sim으로 전환 (ROS 의존성 없음, 순수 Python)

---

## 현재 프로젝트 구조

```
reinforcement_learning/
├── CLAUDE.md
└── tron1/
    ├── tron1-mujoco-sim/                          # MuJoCo 시뮬레이터
    │   └── robot-description/pointfoot/PF_TRON1A/
    │       └── xml/robot.xml                      # MuJoCo 로봇 모델
    ├── tron1_env.py        # Gymnasium 환경 래퍼
    ├── train.py            # PPO 학습 스크립트
    ├── run.py              # 학습된 모델 시각화
    ├── standing.py         # 수동 자세 탐색 도구
    ├── tron1_balance_final.zip      # v1 학습 결과
    ├── tron1_balance_v2_final.zip   # v2 학습 결과
    ├── tron1_balance_v3_final.zip   # v3 학습 결과
    ├── tron1_balance_v4_final.zip   # v4 학습 결과 (최신)
    ├── checkpoints/        # v1 체크포인트
    ├── checkpoints_v2/     # v2 체크포인트
    ├── checkpoints_v3/     # v3 체크포인트
    └── checkpoints_v4/     # v4 체크포인트 (최신)
```

---

## TRON1 로봇 구조

- **타입**: 2족 보행 로봇 (Point Foot = 발목 없는 점 접촉)
- **관절**: 6 DOF (좌/우 각 3개: abad, hip, knee)
  - abad_L: axis +X / abad_R: axis +X
  - hip_L: axis +Y / hip_R: axis -Y (반대!)
  - knee_L: axis -Y / knee_R: axis +Y (반대!)
  - **L/R 축이 반대이므로 대칭 동작 시 R쪽 값 부호 반전 필요**
- **물리 파라미터** (q=0 기준):
  - base 초기 높이: 0.82 m
  - 발 높이: 0.04 m (지면보다 4cm 위 → 자유낙하 상태)
  - 실제 서있을 때 base 높이: ≈ 0.78 m

---

## 강화학습 구현

### Observation (18차원)
```
관절 위치 q   (6) : abad_L, hip_L, knee_L, abad_R, hip_R, knee_R
관절 속도 dq  (6) : 위와 동일 순서
base 높이     (1) : qpos[2]
roll, pitch   (2) : 쿼터니언 → 오일러 변환
각속도        (3) : qvel[3:6]
```

### Action (6차원)
- **직접 타겟 각도 방식**: `target_q = action * 0.8` (스무딩 적용)
- 스무딩: `target_q = 0.2 * new_target + 0.8 * prev_target` (급격한 토크 방지)
- PD 제어: `torque = Kp*(target_q - q) + Kd*(0 - dq)`, Kp=60, Kd=3

### Reward
```python
r_alive   = 1.0
r_height  = (h - 0.40) / (0.78 - 0.40)   # 0(쓰러짐) ~ 1(완전히 섬)
r_upright = 1.0 - |pitch|/0.8 - |roll|/0.8
reward = r_alive + r_height + r_upright
if fallen: reward -= 10.0
```

### Done 조건
```python
h < 0.40 or |pitch| > 0.8 or |roll| > 0.8
```

### 학습 설정 (현재 v4)
```
알고리즘 : PPO (Stable-Baselines3)
병렬 환경 : 16개 (SubprocVecEnv)
네트워크 : MLP [64, 64] / Tanh / Actor+Critic 분리
총 스텝  : 5,000,000
n_steps  : 2048 (환경당)
batch    : 64
lr       : 3e-4
device   : cpu
```

---

## 학습 실험 기록

| 버전 | 변경사항 | ep_len_mean | 비고 |
|------|---------|-------------|------|
| v1 | 기본 구현, 환경 1개, 500k 스텝 | ~83 | delta 액션, pitch만 체크 |
| v2 | 직접 타겟 액션, h<0.55, roll 추가, 8환경 | ~46 | done 조건 너무 엄격 |
| v3 | 액션 스무딩, h<0.40, 8환경, 500k 스텝 | v2보다 개선 | |
| v4 | 16환경, 5M 스텝 | 진행 중 | |

### 설계 교훈

- **직접 타겟 각도는 스무딩 필수**: Kp=60에서 0.8 rad 즉시 명령 → 48 Nm 충격
- **done 높이 임계값**: 너무 높으면(0.55) 에피소드 너무 짧아 학습 안 됨
- **roll 무시하면 안 됨**: pitch만 체크하면 옆으로 넘어지는 패턴 학습 못 막음
- **5M+ 스텝 필요**: Point Foot 균형은 어려운 문제, 500k는 부족

---

## 주요 파일 사용법

```bash
cd ~/Project/reinforcement_learning/tron1

# 수동 자세 탐색 (i/k=hip, j/l=knee, u/o=abad)
python3 standing.py

# 학습
python3 train.py

# 결과 시각화
python3 run.py
```

---

## 병렬 학습 (SubprocVecEnv)

- SB3 내장 기능, 설치 없이 사용
- 환경당 OS 프로세스 1개 생성
- 28코어 기준 16개 권장 (이상은 통신 오버헤드 > 속도 이득)
- MuJoCo GPU 병렬화(MJX/JAX)는 별도 구현 필요, 난이도 높음
- ep_len/rew_mean 보려면 `Monitor` 래퍼 필수

---

## 다음 과제 후보

- [ ] 5M 스텝 결과 확인 및 분석
- [ ] 초기 자세 개선 (hip/knee 약간 구부린 상태로 reset)
- [ ] 네트워크 크기 확대 [256, 256]
- [ ] 보행(locomotion)으로 확장

---

**마지막 수정**: 2026-06-30
