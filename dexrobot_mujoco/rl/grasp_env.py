import os
import tempfile
from dataclasses import dataclass, field

import mujoco
import numpy as np

from dexrobot_mujoco.wrapper import MJSimWrapper

from .math_utils import (
    axis_angle_to_matrix,
    clamp_norm,
    matrix_to_rot6d,
    quat_wxyz_to_xyzw,
    rotation_error,
)
from .scene_builder import build_grasp_scene_xml, sample_object_spec


@dataclass
class GraspEnvConfig:
    seed: int = 0
    episode_horizon: int = 250
    arm_action_scale_pos: float = 0.015
    arm_action_scale_rot: float = 0.12
    hand_action_scale: float = 0.08
    success_height_margin: float = 0.08
    contact_threshold: float = 1e-4
    object_types: list = field(default_factory=lambda: ["box", "thin_box", "cylinder", "capsule", "ellipsoid"])
    frame_skip: int = 10


class DexRobotGraspEnv:
    ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
    HAND_JOINTS = [
        "r_f_joint1_1", "r_f_joint1_2", "r_f_joint1_3", "r_f_joint1_4",
        "r_f_joint2_1", "r_f_joint2_2", "r_f_joint2_3", "r_f_joint2_4",
        "r_f_joint3_1", "r_f_joint3_2", "r_f_joint3_3", "r_f_joint3_4",
        "r_f_joint4_1", "r_f_joint4_2", "r_f_joint4_3", "r_f_joint4_4",
        "r_f_joint5_1", "r_f_joint5_2", "r_f_joint5_3", "r_f_joint5_4",
    ]
    PALM_BODY = "right_hand_base"
    OBJECT_BODY = "grasp_object"
    OBJECT_JOINT = "grasp_object_joint"
    TOUCH_SENSORS = [
        "touch_r_f_link1_pad",
        "touch_r_f_link2_pad",
        "touch_r_f_link3_pad",
        "touch_r_f_link4_pad",
        "touch_r_f_link5_pad",
    ]
    PAD_SITES = [
        "site_r_f_link1_pad",
        "site_r_f_link2_pad",
        "site_r_f_link3_pad",
        "site_r_f_link4_pad",
        "site_r_f_link5_pad",
    ]

    def __init__(self, config=None):
        self.config = config or GraspEnvConfig()
        self.rng = np.random.default_rng(self.config.seed)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.temp_dir = tempfile.mkdtemp(prefix="dexrobot_grasp_", dir=repo_root)
        self.scene_path = os.path.join(self.temp_dir, "generated_grasp_scene.xml")
        self.repo_root = repo_root
        self.mj = None
        self.current_object_spec = None
        self.arm_action_dim = 6
        self.hand_action_dim = len(self.HAND_JOINTS)
        self.action_dim = self.arm_action_dim + self.hand_action_dim
        self.step_count = 0
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.success_counter = 0
        self.table_height = 0.82
        self._build_sim()

    def _build_sim(self):
        self.current_object_spec = sample_object_spec(self.rng, self.config.object_types)
        build_grasp_scene_xml(self.scene_path, self.current_object_spec, self.repo_root)
        self.mj = MJSimWrapper(self.scene_path, seed=self.config.seed)
        self._cache_model_handles()
        self._set_initial_pose()

    def _cache_model_handles(self):
        self.arm_qpos_idx = [self.mj.get_qpos_addr(name) for name in self.ARM_JOINTS]
        self.arm_qvel_idx = [self.mj.get_qvel_addr(name) for name in self.ARM_JOINTS]
        self.hand_qpos_idx = [self.mj.get_qpos_addr(name) for name in self.HAND_JOINTS]
        self.hand_qvel_idx = [self.mj.get_qvel_addr(name) for name in self.HAND_JOINTS]
        self.palm_body_id = self.mj.get_link_id(self.PALM_BODY)
        self.object_body_id = self.mj.get_link_id(self.OBJECT_BODY)
        self.object_joint_qpos_idx = self.mj.get_qpos_addr(self.OBJECT_JOINT)
        self.object_joint_qvel_idx = self.mj.get_qvel_addr(self.OBJECT_JOINT)
        self.arm_dof_idx = np.array(self.arm_qvel_idx, dtype=np.int32)
        self.site_ids = [
            mujoco.mj_name2id(self.mj.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in self.PAD_SITES
        ]
        self.sensor_slices = {}
        adr = 0
        for sensor_id in range(self.mj.model.nsensor):
            name = mujoco.mj_id2name(self.mj.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
            dim = int(self.mj.model.sensor_dim[sensor_id])
            self.sensor_slices[name] = slice(adr, adr + dim)
            adr += dim
        self.actuator_ranges = {
            name: tuple(self.mj.model.actuator_ctrlrange[i])
            for i, name in enumerate(self.mj.actuator_names)
        }
        self.hand_actuator_names = [f"act_{name}" for name in self.HAND_JOINTS]

    def _set_initial_pose(self):
        arm_home = np.array([0.0, -0.75, 1.0, 0.0, 1.1, 0.0], dtype=np.float64)
        hand_home = np.array(
            [0.2, 0.15, 0.1, 0.1, 0.12, 0.18, 0.18, 0.18, 0.0, 0.18, 0.18, 0.18, 0.12, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18],
            dtype=np.float64,
        )
        for name, value in zip(self.ARM_JOINTS, arm_home):
            self.mj.set_qpos(name, float(value))
            self.mj.set_control(name, float(value))
        for joint_name, act_name, value in zip(self.HAND_JOINTS, self.hand_actuator_names, hand_home):
            self.mj.set_qpos(joint_name, float(value))
            self.mj.set_control(act_name, float(value))
        mujoco.mj_forward(self.mj.model, self.mj.data)
        for _ in range(20):
            self.mj.step()

    def reset(self):
        self.step_count = 0
        self.prev_action[:] = 0.0
        self.success_counter = 0
        self._build_sim()
        return self._get_obs(), self._get_info()

    def _get_sensor(self, name):
        return self.mj.data.sensordata[self.sensor_slices[name]].copy()

    def _body_pos(self, body_id):
        return self.mj.data.xpos[body_id].copy()

    def _body_rot(self, body_id):
        return self.mj.data.xmat[body_id].reshape(3, 3).copy()

    def _object_pose(self):
        qpos = self.mj.data.qpos[self.object_joint_qpos_idx:self.object_joint_qpos_idx + 7].copy()
        return qpos[:3], qpos[3:7]

    def _object_velocity(self):
        return self.mj.data.qvel[self.object_joint_qvel_idx:self.object_joint_qvel_idx + 6].copy()

    def _get_obs(self):
        arm_qpos = self.mj.data.qpos[self.arm_qpos_idx].copy()
        arm_qvel = self.mj.data.qvel[self.arm_qvel_idx].copy()
        hand_qpos = self.mj.data.qpos[self.hand_qpos_idx].copy()
        hand_qvel = self.mj.data.qvel[self.hand_qvel_idx].copy()
        palm_pos = self._body_pos(self.palm_body_id)
        palm_rot = self._body_rot(self.palm_body_id)
        object_pos, object_quat = self._object_pose()
        object_vel = self._object_velocity()
        touch = np.concatenate([self._get_sensor(name) for name in self.TOUCH_SENSORS], axis=0)
        pad_positions = np.concatenate([self.mj.data.site_xpos[sid].copy() for sid in self.site_ids], axis=0)
        obs = np.concatenate(
            [
                arm_qpos,
                arm_qvel,
                hand_qpos,
                hand_qvel,
                palm_pos,
                matrix_to_rot6d(palm_rot),
                object_pos,
                quat_wxyz_to_xyzw(object_quat),
                object_vel,
                object_pos - palm_pos,
                touch,
                pad_positions,
                self.prev_action,
            ],
            axis=0,
        )
        return obs.astype(np.float32)

    def _solve_arm_ik(self, target_pos, target_rot):
        current_q = self.mj.data.qpos[self.arm_qpos_idx].copy()
        palm_pos = self._body_pos(self.palm_body_id)
        palm_rot = self._body_rot(self.palm_body_id)
        pos_err = target_pos - palm_pos
        rot_err = rotation_error(target_rot, palm_rot)
        task_err = np.concatenate([pos_err, rot_err], axis=0)

        jacp = np.zeros((3, self.mj.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.mj.model.nv), dtype=np.float64)
        mujoco.mj_jacBody(self.mj.model, self.mj.data, jacp, jacr, self.palm_body_id)
        jac = np.vstack([jacp[:, self.arm_dof_idx], jacr[:, self.arm_dof_idx]])
        damping = 1e-3 * np.eye(6, dtype=np.float64)
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping, task_err)
        dq = clamp_norm(dq, 0.2)
        q_target = current_q + dq
        for i, name in enumerate(self.ARM_JOINTS):
            lower, upper = self.actuator_ranges[name]
            q_target[i] = np.clip(q_target[i], lower, upper)
        return q_target

    def _apply_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        arm_action = action[:6]
        hand_action = action[6:]

        palm_pos = self._body_pos(self.palm_body_id)
        palm_rot = self._body_rot(self.palm_body_id)
        target_pos = palm_pos + arm_action[:3] * self.config.arm_action_scale_pos
        target_rot = axis_angle_to_matrix(arm_action[3:] * self.config.arm_action_scale_rot) @ palm_rot
        arm_targets = self._solve_arm_ik(target_pos, target_rot)
        for name, value in zip(self.ARM_JOINTS, arm_targets):
            self.mj.send_control(name, float(value))

        hand_qpos = self.mj.data.qpos[self.hand_qpos_idx].copy()
        desired_hand = hand_qpos + hand_action * self.config.hand_action_scale
        for actuator_name, value in zip(self.hand_actuator_names, desired_hand):
            lower, upper = self.actuator_ranges[actuator_name]
            self.mj.send_control(actuator_name, float(np.clip(value, lower, upper)))

    def _count_bad_collisions(self):
        bad = 0
        for idx in range(self.mj.data.ncon):
            contact = self.mj.data.contact[idx]
            body1 = self.mj.model.geom_bodyid[contact.geom1]
            body2 = self.mj.model.geom_bodyid[contact.geom2]
            if self.object_body_id in (body1, body2):
                continue
            names = [
                mujoco.mj_id2name(self.mj.model, mujoco.mjtObj.mjOBJ_BODY, body1),
                mujoco.mj_id2name(self.mj.model, mujoco.mjtObj.mjOBJ_BODY, body2),
            ]
            if any(name is None for name in names):
                continue
            robot_hit = any(name.startswith("r_f_") or name.startswith("link") or name == "right_hand_base" for name in names)
            scene_hit = any("table" in name or "scene" in name for name in names)
            if robot_hit and scene_hit:
                bad += 1
        return bad

    def _compute_reward(self, action):
        object_pos, _ = self._object_pose()
        object_vel = self._object_velocity()
        palm_pos = self._body_pos(self.palm_body_id)
        palm_rot = self._body_rot(self.palm_body_id)
        touch_values = np.array([self._get_sensor(name)[0] for name in self.TOUCH_SENSORS], dtype=np.float64)
        active_contacts = int(np.sum(touch_values > self.config.contact_threshold))
        palm_to_object = object_pos - palm_pos
        reach_dist = np.linalg.norm(palm_to_object)

        palm_forward = palm_rot[:, 2]
        approach_dir = palm_to_object / max(reach_dist, 1e-6)
        alignment = max(0.0, float(np.dot(palm_forward, approach_dir)))

        object_height = float(object_pos[2])
        lifted = object_height > (self.table_height + self.config.success_height_margin)
        stable_grasp = active_contacts >= 2 and np.linalg.norm(object_vel[:3]) < 0.25

        reward = 0.0
        reward += 1.5 * np.exp(-6.0 * reach_dist)
        reward += 0.6 * alignment
        reward += 0.35 * active_contacts
        reward += 0.6 if stable_grasp else 0.0
        reward += 2.5 if (lifted and stable_grasp) else 0.0
        reward -= 0.04 * float(np.linalg.norm(action - self.prev_action))
        reward -= 0.03 * float(np.linalg.norm(action))
        reward -= 0.1 * self._count_bad_collisions()

        if lifted and stable_grasp:
            self.success_counter += 1
        else:
            self.success_counter = 0
        success = self.success_counter >= 10
        if success:
            reward += 10.0

        info = {
            "reach_distance": reach_dist,
            "alignment": alignment,
            "active_contacts": active_contacts,
            "object_height": object_height,
            "lifted": lifted,
            "stable_grasp": stable_grasp,
            "success": success,
        }
        return reward, info

    def _get_info(self):
        _, info = self._compute_reward(self.prev_action)
        return info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self._apply_action(action)
        for _ in range(self.config.frame_skip):
            self.mj.step()
        self.step_count += 1
        reward, info = self._compute_reward(action)
        self.prev_action = np.clip(action, -1.0, 1.0)

        object_pos, _ = self._object_pose()
        terminated = bool(info["success"] or object_pos[2] < 0.72)
        truncated = self.step_count >= self.config.episode_horizon
        return self._get_obs(), float(reward), terminated, truncated, info

    def close(self):
        self.mj.viewer = None
