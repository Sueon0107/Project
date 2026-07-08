import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx


XML_PATH = "tron1-mujoco-sim/robot-description/pointfoot/PF_TRON1A/xml/robot.xml"


@dataclass(frozen=True)
class Tron1MjxBasicConfig:
    action_scale: float = 0.8
    ctrl_hz: float = 60.0
    kp: float = 60.0
    kd: float = 3.0
    max_steps: int = 1000
    min_height: float = 0.40
    target_height: float = 0.78
    max_tilt: float = 0.8
    fall_penalty: float = 30.0


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
class Tron1MjxBasicEnv:
    model: mujoco.MjModel
    mjx_model: mjx.Model
    config: Tron1MjxBasicConfig
    steps_per_ctrl: int

    @classmethod
    def create(cls, xml_path=XML_PATH, config=Tron1MjxBasicConfig()):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = xml_path if os.path.isabs(xml_path) else os.path.join(base_dir, xml_path)
        model = mujoco.MjModel.from_xml_string(_mjx_compatible_xml(full_path))
        mjx_model = mjx.put_model(model)
        steps_per_ctrl = max(1, int(round((1.0 / config.ctrl_hz) / model.opt.timestep)))
        return cls(model, mjx_model, config, steps_per_ctrl)

    def reset(self, key):
        data = mjx.make_data(self.model)
        noise = jax.random.uniform(key, (self.model.nu,), minval=-0.05, maxval=0.05)
        data = data.replace(qpos=data.qpos.at[7:].add(noise), qvel=jp.zeros_like(data.qvel))
        data = mjx.forward(self.mjx_model, data)
        state = {
            "data": data,
            "target_q": jp.zeros(self.model.nu),
            "step_count": jp.array(0, dtype=jp.int32),
        }
        return state, self.obs(data)

    def obs(self, data):
        q = data.qpos[7:]
        dq = data.qvel[6:]
        roll, pitch = quat_to_rp(data.qpos[3:7])
        rp = jp.array([roll, pitch])
        ang_vel = data.qvel[3:6]
        return jp.concatenate([q, dq, rp, ang_vel]).astype(jp.float32)

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
        obs = self.obs(data)

        h = data.qpos[2]
        roll, pitch = quat_to_rp(data.qpos[3:7])
        fallen = (
            (h < self.config.min_height)
            | (jp.abs(pitch) > self.config.max_tilt)
            | (jp.abs(roll) > self.config.max_tilt)
        )
        truncated = step_count >= self.config.max_steps

        height_span = self.config.target_height - self.config.min_height
        r_height = jp.clip((h - self.config.min_height) / height_span, -1.0, 1.0)
        r_upright = 1.0 - jp.abs(pitch) / self.config.max_tilt - jp.abs(roll) / self.config.max_tilt
        reward = r_height + r_upright
        reward = jp.where(fallen, reward - self.config.fall_penalty, reward)

        next_state = {
            "data": data,
            "target_q": target_q,
            "step_count": step_count,
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
