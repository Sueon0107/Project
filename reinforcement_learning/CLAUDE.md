# TRON1 보행 강화학습 프로젝트

## 프로젝트 개요

**목표**: TRON1 2족 로봇의 보행 강화학습 구현 및 시뮬레이션

**담당**: 학부연구생 (강화학습 기본 개념 학습 완료)

**지시**: 교수님으로부터 TRON1 보행 강화학습 수행 지시 받음

---

## 배경 지식

### 강화학습(Reinforcement Learning) 기본 개념
- **Agent**: 로봇(TRON1)이 환경과 상호작용하며 학습
- **Environment**: Gazebo 시뮬레이터
- **State**: 로봇의 센서값 (각 관절의 위치, 속도, IMU 등)
- **Action**: 관절에 가하는 토크 또는 속도 명령
- **Reward**: 보행 속도, 에너지 효율성, 안정성 등을 고려한 보상 신호
- **Policy**: 상태에서 최적의 행동을 선택하는 전략

### TRON1 로봇
- **타입**: 2족 보행 로봇 (Bipedal, 휴머노이드)
- **다리 구조**: 좌/우 2개 다리, 각 다리마다 3개 모터화 관절 (abad, hip, knee)
  - **abad**: 좌우 이동 관절
  - **hip**: 엉덩이 관절
  - **knee**: 무릎 관절
- **발 형태**: PF(Point Foot), SF(Soft Foot), WF(Wide Foot) 등 여러 버전
- **용도**: 시뮬레이션 기반 강화학습 연구
- **시뮬레이터**: Gazebo (ROS 2 Iron 환경)
- **개발사**: LIMX Dynamics

---

## 기술 스택

| 항목 | 버전/도구 |
|------|---------|
| 운영체제 | Ubuntu 22.04 LTS |
| ROS | ROS 2 Iron |
| 시뮬레이터 | Gazebo |
| 강화학습 프레임워크 | (미정 - Stable-Baselines3, Ray RLlib, PyTorch 등 고려) |
| 프로그래밍 언어 | Python 3, C++ |

---

## 현재 프로젝트 구조

```
/home/sueon/project/
├── reinforcement_learning/
│   ├── tron1/                              # 메인 시뮬레이터 패키지
│   │   ├── pointfoot_gazebo/              # Gazebo 환경 설정 및 제어 인터페이스
│   │   ├── limxsdk-sim/                   # SDK 시뮬레이션 모듈
│   │   ├── robot-description/             # 로봇 모델 파일 (URDF/Xacro)
│   │   │   ├── pointfoot/                 # 2족 로봇 모델
│   │   │   │   ├── PF_TRON1A/B           # Point Foot TRON1
│   │   │   │   ├── SF_TRON1A/B           # Soft Foot TRON1
│   │   │   │   └── WF_TRON1A/B           # Wide Foot TRON1
│   │   │   └── wheellegged/              # 다른 로봇 플랫폼
│   │   ├── README.md                      # 환경 설정 가이드
│   │   └── LICENSE
│   └── CLAUDE.md                          # 이 파일
```

### 아직 클론 필요한 저장소
- `limxsdk-lowlevel`: 모션 제어 인터페이스 (선택사항)
- `robot-visualization`: 시각화 도구 (선택사항)

---

## 학습 로드맵

### Phase 1: 환경 구축 (1-2주)
- [ ] ROS 2 Iron 설치
- [ ] 필수 패키지 설치
- [ ] 워크스페이스 생성 및 컴파일
- [ ] Gazebo 시뮬레이터 실행 확인
- [ ] ROBOT_TYPE을 PF_TRON1A로 설정
- [ ] 기본 제어 예제 실행

### Phase 2: 강화학습 이론 학습 (2-3주)
- [ ] Policy Gradient 방법 이해 (REINFORCE, PPO)
- [ ] Actor-Critic 아키텍처
- [ ] 보행 로봇의 보상 함수 설계
- [ ] Stable-Baselines3 튜토리얼 따라하기

### Phase 3: 기본 실험 (2-3주)
- [ ] Simple locomotion task (직진)
- [ ] 기본 보상 함수 구현
- [ ] 학습 커브 관찰 및 분석

### Phase 4: 고급 과제 (진행 중)
- [ ] 다양한 이동 패턴 학습 (좌우회전, 복잡한 지형)
- [ ] 에너지 효율성 최적화
- [ ] 현실(Real Robot)로의 전이(Transfer Learning)

---

## 주요 개념 및 용어

| 용어 | 설명 |
|------|------|
| **Policy** | 상태 → 행동 매핑, π(a\|s) |
| **Reward** | 각 스텝에서 받는 점수 |
| **Episode** | 초기 상태에서 끝나는 상태까지의 한 사이클 |
| **Trajectory** | 상태-행동-보상의 시퀀스 |
| **Convergence** | 정책이 최적해에 수렴하는 것 |
| **Exploration** | 미지의 행동 탐색 |
| **Exploitation** | 알려진 좋은 행동 선택 |

---

## 강화학습 알고리즘 선택 가이드

**현재 권장**: PPO (Proximal Policy Optimization)
- 안정적이고 구현이 간단함
- 보행 로봇 학습에 많이 사용됨
- Stable-Baselines3에 구현되어 있음

**대안**:
- **TRPO**: PPO보다 더 안정적이나 복잡
- **A3C**: 병렬 학습 가능, 계산 효율적
- **SAC**: 샘플 효율성 높음
- **TD3**: 연속 제어에 최적화

---

## 중요 파일 및 모듈

### Gazebo 시뮬레이션
- **`pointfoot_gazebo/`**: Gazebo 환경 및 제어 인터페이스
  - `launch/empty_world.launch.py`: 빈 세계 시뮬레이션 시작
  - `config/pointfoot_gazebo.yaml`: ROS 2 Control 설정
  - `worlds/empty_world.world`: Gazebo 월드 파일

### 로봇 모델 (URDF/Xacro)
- **`robot-description/pointfoot/PF_TRON1A/`** (권장)
  - `xacro/robot.xacro`: 로봇 구조 정의
  - `xacro/leg.xacro`: 다리(leg) 정의 (3 DOF: abad, hip, knee)
  - `xacro/const.xacro`: 로봇 물리 파라미터
  - `meshes/`: 로봇 3D 모델 파일

### 로봇 제어
- **`limxsdk-sim/`**: 로봇 제어 SDK
  - 관절 명령 전송 (토크/속도)
  - 센서 데이터 수신 (위치, 속도, IMU)

### 로봇 상태
- 2개 다리: 좌(L), 우(R)
- 각 다리 3개 관절 × 2 = 총 6개 ActuatedJoint
- IMU 센서: 방향(quaternion), 각속도, 선가속도

---

## 다음 단계

1. **환경 구축 완료 후**: README.md의 "Set up the Development Environment" 섹션 따라하기
2. **필수 저장소 클론**:
   ```bash
   mkdir -p ~/limx_ws/src
   cd ~/limx_ws/src
   git clone https://github.com/limxdynamics/robot-description.git
   git clone https://github.com/limxdynamics/limxsdk-lowlevel.git
   git clone https://github.com/limxdynamics/robot-visualization.git
   ```
3. **강화학습 라이브러리 설치**:
   ```bash
   pip install stable-baselines3[extra]
   pip install gymnasium  # 강화학습 환경 표준
   ```
4. **Custom Gym Environment 만들기**: TRON1을 Gym 환경으로 래핑

---

## 문제 해결 및 주의사항

### 일반 주의사항
- ROS 2 Iron 버전 명확히 설치 (다른 버전과 호환 안 될 수 있음)
- `source ~/.bashrc` 또는 `source /opt/ros/iron/setup.bash`로 환경 초기화
- 시뮬레이션은 CPU/GPU 성능에 따라 시간이 오래 걸릴 수 있음

### 강화학습 학습 팁
- 작은 네트워크부터 시작해서 점진적으로 키우기
- Reward shaping을 신중히 - 과도한 보상이 이상한 행동 유발 가능
- 충분한 탐색 시간 주기 (exploration)
- 학습 곡선과 에피소드 수를 모니터링

---

## 참고 자료

### 강화학습 이론
- [Spinning Up in Deep RL](https://spinningup.openai.com/) - OpenAI
- [Sutton & Barto: Reinforcement Learning](http://incompleteideas.net/book/the-book.html)
- [Deep RL Course](https://huggingface.co/deep-rl-course/unit0/introduction) - HuggingFace

### 실제 구현
- [Stable-Baselines3 문서](https://stable-baselines3.readthedocs.io/)
- [Gymnasium (구 OpenAI Gym)](https://gymnasium.farama.org/)
- [ROS 2 공식 문서](https://docs.ros.org/en/iron/)

### TRON1 로봇
- GitHub: https://github.com/limxdynamics/tron1-gazebo-ros2
- 로봇 설명: https://github.com/limxdynamics/robot-description

---

## 중요 정정사항

### TRON1은 2족(Bipedal) 로봇입니다!
- ❌ 이전: 4족(Quadruped)이라고 잘못 이해
- ✅ 정정: **2족 로봇 (좌/우 2개 다리)**
- 각 다리마다 3개의 모터화 관절 (abad, hip, knee)
- **"Pointfoot"은 다리의 개수가 아니라 발의 형태를 나타냅니다**
  - PF = Point Foot (가는 발)
  - SF = Soft Foot (부드러운 발)  
  - WF = Wide Foot (넓은 발)

### 강화학습 State/Action 정의
- **State (12차원)**: 
  - 좌다리 3관절의 위치/속도 (6D)
  - 우다리 3관절의 위치/속도 (6D)
  - IMU 센서값 (방향, 각속도, 가속도 등)
- **Action (6차원)**:
  - 6개 관절에 대한 토크/속도 명령

---

## 업데이트 기록

- **2026-06-26**: 초기 CLAUDE.md 작성 (4족으로 잘못 이해)
- **2026-06-26**: TRON1이 2족 로봇임을 확인하고 전체 내용 정정
  - robot-description 저장소의 URDF 파일 분석으로 확인
  - Pointfoot의 정확한 의미 파악 (발의 형태, 다리 개수 아님)
  - 로봇 구조 명확화 (2 다리 × 3 관절 = 6 DOF)

---

**마지막 수정**: 2026-06-26
