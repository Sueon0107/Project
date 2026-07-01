import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

XML_PATH = "tron1-mujoco-sim/robot-description/pointfoot/PF_TRON1A/xml/robot.xml"

class Tron1BalanceEnv(gym.Env):
    """
    TRON1 중심잡기 환경
    목표: 넘어지지 않고 최대한 오래 서있기
    """

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data  = mujoco.MjData(self.model)

        self.n_joints       = self.model.nu          # 6
        self.steps_per_ctrl = max(1, int(round(1.0 / 60.0 / self.model.opt.timestep)))
        self.max_steps      = 1000                   # 에피소드 최대 스텝

        # Action: 각 관절의 목표 각도 델타 (-1 ~ 1 정규화, 실제로 ±0.5 rad 범위)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )
        self.action_scale = 0.8  # rad (직접 타겟 각도)

        # Observation: 관절 위치(6) + 관절 속도(6) + base 높이(1) + 기울기 roll/pitch(2) + 각속도(3)
        obs_dim = 6 + 6 + 1 + 2 + 3  # = 18
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._step_count = 0
        self._target_q   = np.zeros(self.n_joints)

        # 렌더링
        self._viewer = None

    def _get_obs(self):
        q   = self.data.qpos[7:].copy()    # 관절 위치 (6)
        dq  = self.data.qvel[6:].copy()    # 관절 속도 (6)
        h   = np.array([self.data.qpos[2]])  # base 높이 (1)

        # base 기울기: 쿼터니언 → roll, pitch
        quat     = self.data.qpos[3:7]     # [w, x, y, z]
        roll, pitch = self._quat_to_rp(quat)
        rp       = np.array([roll, pitch])  # (2)

        # base 각속도 (3)
        ang_vel  = self.data.qvel[3:6].copy()

        return np.concatenate([q, dq, h, rp, ang_vel]).astype(np.float32)

    def _quat_to_rp(self, quat):
        w, x, y, z = quat
        roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
        return roll, pitch

    def _is_fallen(self):
        h          = self.data.qpos[2]
        roll, pitch = self._quat_to_rp(self.data.qpos[3:7])
        return h < 0.40 or abs(pitch) > 0.8 or abs(roll) > 0.8

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 관절에 약간의 노이즈 추가 (학습 다양성)
        self.data.qpos[7:] += self.np_random.uniform(-0.05, 0.05, self.n_joints)
        self.data.qvel[:]   = 0.0

        self._step_count = 0

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        # 직접 타겟에 스무딩 적용 (급격한 토크 변화 방지)
        target = action * self.action_scale
        self._target_q = 0.2 * target + 0.8 * self._target_q
        self._target_q = np.clip(self._target_q, -1.5, 1.5)

        # PD 제어로 물리 스텝 실행
        Kp, Kd = 60.0, 3.0
        for _ in range(self.steps_per_ctrl):
            q  = self.data.qpos[7:]
            dq = self.data.qvel[6:]
            self.data.ctrl[:] = Kp * (self._target_q - q) + Kd * (0.0 - dq)
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs       = self._get_obs()
        fallen    = bool(self._is_fallen())
        truncated = bool(self._step_count >= self.max_steps)

        # 보상 설계
        h               = self.data.qpos[2]
        roll, pitch     = self._quat_to_rp(self.data.qpos[3:7])
        r_height        = (h - 0.40) / (0.78 - 0.40)       # 0(쓰러짐) ~ 1(완전히 섬)
        r_upright       = 1.0 - abs(pitch) / 0.8 - abs(roll) / 0.8  # roll/pitch 모두 반영
        r_alive         = 1.0
        reward          = r_alive + r_height + r_upright

        if fallen:
            reward -= 10.0  # 넘어지면 패널티

        if self.render_mode == "human" and self._viewer is not None:
            self._viewer.sync()

        return obs, reward, fallen, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self._viewer.cam.distance  = 3.0
                self._viewer.cam.elevation = -20

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
