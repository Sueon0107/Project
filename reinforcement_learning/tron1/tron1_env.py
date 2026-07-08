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

        self.n_joints       = self.model.nu
        self.steps_per_ctrl = max(1, int(round(1.0 / 60.0 / self.model.opt.timestep)))
        self.max_steps      = 1000

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )
        self.action_scale = 0.8

        obs_dim = 6 + 6 + 2 + 3  # = 17 (높이 제거)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._step_count    = 0
        self._target_q      = np.zeros(self.n_joints)
        self._prev_lin_vel  = np.zeros(3)

        # 발/바닥 geom ID 캐싱
        self._foot_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'foot_L_collision'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'foot_R_collision'),
        }
        self._floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')

        self._viewer = None

    def _get_obs(self):
        q       = self.data.qpos[7:].copy()
        dq      = self.data.qvel[6:].copy()
        quat    = self.data.qpos[3:7]
        roll, pitch = self._quat_to_rp(quat)
        rp      = np.array([roll, pitch])
        ang_vel = self.data.qvel[3:6].copy()
        return np.concatenate([q, dq, rp, ang_vel]).astype(np.float32)

    def _quat_to_rp(self, quat):
        w, x, y, z = quat
        roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
        return roll, pitch

    def _is_fallen(self):
        h           = self.data.qpos[2]
        roll, pitch = self._quat_to_rp(self.data.qpos[3:7])
        return h < 0.40 or abs(pitch) > 0.8 or abs(roll) > 0.8

    def _foot_contacts(self):
        """바닥에 닿아있는 발 수 반환 (0, 1, 2)"""
        contacts = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self._floor_geom_id in pair:
                contacts |= pair & self._foot_geom_ids
        return len(contacts)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[7:] += self.np_random.uniform(-0.05, 0.05, self.n_joints)
        self.data.qvel[:]   = 0.0

        self._step_count   = 0
        self._target_q     = np.zeros(self.n_joints)
        self._prev_lin_vel = np.zeros(3)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        target = action * self.action_scale
        self._target_q = 0.2 * target + 0.8 * self._target_q
        self._target_q = np.clip(self._target_q, -1.5, 1.5)

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

        h               = self.data.qpos[2]
        roll, pitch     = self._quat_to_rp(self.data.qpos[3:7])

        r_alive   = 1.0
        r_height  = (h - 0.40) / (0.78 - 0.40)
        r_upright = 1.0 - abs(pitch) / 0.8 - abs(roll) / 0.8

        # 발 접지 패널티
        n_contacts = self._foot_contacts()
        r_contact  = (n_contacts / 2.0) - 1.0   # 두 발 접지=0, 한 발=−0.5, 공중=−1.0

        # 선가속도 패널티 (IMU로 측정 가능)
        dt_ctrl       = self.steps_per_ctrl * self.model.opt.timestep
        curr_lin_vel  = self.data.qvel[0:3].copy()
        lin_acc       = (curr_lin_vel - self._prev_lin_vel) / dt_ctrl
        self._prev_lin_vel = curr_lin_vel
        r_acc = -min(np.linalg.norm(lin_acc[:2]), 10.0) * 0.3  # 수평 가속도만, 최대 -3.0

        reward = r_alive + r_height + r_upright + r_contact + r_acc

        if fallen:
            reward -= 10.0

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
