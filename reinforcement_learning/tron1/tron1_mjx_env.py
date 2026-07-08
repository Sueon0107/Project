import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx


XML_PATH = "tron1-mujoco-sim/robot-description/pointfoot/PF_TRON1A/xml/robot.xml"


@dataclass(frozen=True)
class Tron1MjxConfig:
    action_scale: float = 0.8
    ctrl_hz: float = 60.0
    kp: float = 60.0
    kd: float = 3.0
    max_steps: int = 1000
    gait_period: int = 40


def _mjx_compatible_xml(path):
    root = ET.parse(path).getroot()
    keep_collision = {
        "floor",
        "base_collision",
        "abad_L_collision",
        "hip_L_collision",
        "knee_L_collision",
        "foot_L_collision",
        "abad_R_collision",
        "hip_R_collision",
        "knee_R_collision",
        "foot_R_collision",
    }

    for parent in root.iter():
        for child in list(parent):
            if child.tag == "mesh":
                parent.remove(child)
            elif child.tag == "geom" and (
                child.get("class") == "visual" or child.get("mesh") is not None
            ):
                parent.remove(child)

    for geom in root.iter("geom"):
        if geom.get("type") == "cylinder":
            geom.set("type", "capsule")
        if geom.get("name") in keep_collision:
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
        else:
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

    return ET.tostring(root, encoding="unicode")


def quat_to_rp(quat):
    w, x, y, z = quat
    roll = jp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = jp.arcsin(jp.clip(2 * (w * y - z * x), -1, 1))
    return roll, pitch


@dataclass
class Tron1MjxEnv:
    model: mujoco.MjModel
    mjx_model: mjx.Model
    config: Tron1MjxConfig
    steps_per_ctrl: int
    foot_geom_ids: jp.ndarray
    floor_geom_id: int

    @classmethod
    def create(cls, xml_path=XML_PATH, config=Tron1MjxConfig()):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = xml_path if os.path.isabs(xml_path) else os.path.join(base_dir, xml_path)
        model = mujoco.MjModel.from_xml_string(_mjx_compatible_xml(full_path))
        mjx_model = mjx.put_model(model)
        steps_per_ctrl = max(1, int(round((1.0 / config.ctrl_hz) / model.opt.timestep)))
        foot_geom_ids = jp.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "foot_L_collision"),
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "foot_R_collision"),
            ],
            dtype=jp.int32,
        )
        floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        return cls(model, mjx_model, config, steps_per_ctrl, foot_geom_ids, floor_geom_id)

    def reset(self, key):
        data = mjx.make_data(self.model)
        noise = jax.random.uniform(key, (self.model.nu,), minval=-0.05, maxval=0.05)
        data = data.replace(qpos=data.qpos.at[7:].add(noise), qvel=jp.zeros_like(data.qvel))
        data = mjx.forward(self.mjx_model, data)
        phase = jax.random.uniform(key, (), minval=0.0, maxval=1.0)
        state = {
            "data": data,
            "target_q": jp.zeros(self.model.nu),
            "prev_lin_vel": jp.zeros(3),
            "step_count": jp.array(0, dtype=jp.int32),
            "phase": phase,
        }
        return state, self.obs(data, phase)

    def obs(self, data, phase):
        q = data.qpos[7:]
        dq = data.qvel[6:]
        roll, pitch = quat_to_rp(data.qpos[3:7])
        rp = jp.array([roll, pitch])
        ang_vel = data.qvel[3:6]
        clock = jp.array([jp.sin(2.0 * jp.pi * phase), jp.cos(2.0 * jp.pi * phase)])
        return jp.concatenate([q, dq, rp, ang_vel, clock]).astype(jp.float32)

    def step(self, state, action):
        data = state["data"]
        target = jp.asarray(action) * self.config.action_scale
        target_q = 0.2 * target + 0.8 * state["target_q"]
        target_q = jp.clip(target_q, -1.5, 1.5)

        def physics_step(data, _):
            q = data.qpos[7:]
            dq = data.qvel[6:]
            ctrl = self.config.kp * (target_q - q) + self.config.kd * (0.0 - dq)
            data = data.replace(ctrl=ctrl)
            return mjx.step(self.mjx_model, data), None

        data, _ = jax.lax.scan(physics_step, data, None, length=self.steps_per_ctrl)
        step_count = state["step_count"] + 1
        phase = (state["phase"] + 1.0 / self.config.gait_period) % 1.0
        obs = self.obs(data, phase)

        h = data.qpos[2]
        roll, pitch = quat_to_rp(data.qpos[3:7])
        fallen = (h < 0.40) | (jp.abs(pitch) > 0.8) | (jp.abs(roll) > 0.8)
        truncated = step_count >= self.config.max_steps

        r_alive = 0.5
        r_height = (h - 0.40) / (0.78 - 0.40)
        r_upright = 1.0 - jp.abs(pitch) / 0.8 - jp.abs(roll) / 0.8
        r_air = jp.where(self._foot_contacts(data) == 0, -0.5, 0.0)
        r_phase = self._phase_reward(data, phase)

        dt_ctrl = self.steps_per_ctrl * self.model.opt.timestep
        curr_lin_vel = data.qvel[0:3]
        lin_acc = (curr_lin_vel - state["prev_lin_vel"]) / dt_ctrl
        r_acc = -jp.minimum(jp.linalg.norm(lin_acc[:2]), 10.0) * 0.15

        reward = r_alive + r_height + r_upright + r_air + r_acc + r_phase
        reward = jp.where(fallen, reward - 30.0, reward)

        next_state = {
            "data": data,
            "target_q": target_q,
            "prev_lin_vel": curr_lin_vel,
            "step_count": step_count,
            "phase": phase,
        }
        return next_state, obs, reward, fallen, truncated

    def reset_if_done(self, state, obs, done, truncated, key):
        finished = done | truncated
        reset_state, reset_obs = self.reset(key)

        def choose(reset_value, current_value):
            return jp.where(finished, reset_value, current_value)

        state = jax.tree.map(choose, reset_state, state)
        obs = jp.where(finished, reset_obs, obs)
        return state, obs, finished

    def _foot_contacts(self, data):
        geoms = data.contact.geom
        floor_touch = jp.any(geoms == self.floor_geom_id, axis=1)
        left_touch = jp.any(geoms == self.foot_geom_ids[0], axis=1) & floor_touch
        right_touch = jp.any(geoms == self.foot_geom_ids[1], axis=1) & floor_touch
        n_contacts = left_touch.any().astype(jp.float32) + right_touch.any().astype(jp.float32)
        return n_contacts

    def _contact_reward(self, data):
        n_contacts = self._foot_contacts(data)
        return (n_contacts / 2.0) - 1.0

    def _phase_reward(self, data, phase):
        geoms = data.contact.geom
        floor_touch = jp.any(geoms == self.floor_geom_id, axis=1)
        left_contact = (jp.any(geoms == self.foot_geom_ids[0], axis=1) & floor_touch).any()
        right_contact = (jp.any(geoms == self.foot_geom_ids[1], axis=1) & floor_touch).any()
        foot_z = data.geom_xpos[self.foot_geom_ids, 2]
        left_lift = foot_z[0] > 0.055
        right_lift = foot_z[1] > 0.055

        left_stance = phase < 0.5
        stance_contact = jp.where(left_stance, left_contact, right_contact)
        swing_contact = jp.where(left_stance, right_contact, left_contact)
        swing_clear = jp.where(left_stance, right_lift, left_lift)
        r_stance = jp.where(stance_contact, 0.8, -1.5)
        r_swing_contact = jp.where(swing_contact, -2.0, 0.4)
        r_swing_clear = jp.where(swing_clear, 1.0, -0.8)
        return r_stance + r_swing_contact + r_swing_clear
