from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Dict, Tuple, List
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import importlib
from isaaclab.utils.math import (
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    sample_uniform,
    saturate,
)

import cv2
import numpy as np

BOLD = "\033[1m"
YELLOW = "\033[33m"
RESET = "\033[0m"


class MLP1024_512_256(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = 1 * 5 * 17   # (1,5,17) flatten → 85

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # (B,85)
        out = self.mlp(x)          # (B,5)
        return out
    

class Embedder(nn.Module):
    def __init__(self, in_dim=85, hidden=(128, 64), out_dim=2, dropout=0.0):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.LayerNorm(h1),

            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.LayerNorm(h2),
            nn.Dropout(dropout),

            nn.Linear(h2, out_dim)
        )
        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


if TYPE_CHECKING:
    from .sr_bottle_v1_env_cfg import ShadowHandBottleEnvCfg


class ShadowHandBottleEnv(DirectRLEnv):
    """Shadow Robot Hand bottle manipulation environment using IsaacLab."""

    cfg: ShadowHandBottleEnvCfg
    
    def __init__(
        self,
        cfg: ShadowHandBottleEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        # Store method before super().__init__
        self.method = cfg.method
        print(f"Environment method: {self.method}")
        print(f"DEBUG: num_envs passed to __init__: {cfg.num_envs}")
        print(f"DEBUG: kwargs: {kwargs}")

        # Call parent constructor - this sets up scene, device, etc.
        super().__init__(cfg, render_mode, **kwargs)
        print(f"DEBUG: self.num_envs after super().__init__: {self.num_envs}")
        # ================== Dimensions and Counters ==================
        self.num_hand_dofs = self.hand.num_joints
        self.total_hand_dof = self.hand.num_joints  # Shadow Hand specific
        self.total_arm_dof = 0  # Not using arm in this task

        self.dof_vel_scale = cfg.dof_vel_scale
        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        # Frame stacking
        self.n_stack_frame = self.cfg.n_stack_frame
        self.single_frame_obs_dim = self._compute_obs_dim()

        # Counters
        self.count = 0.0
        self.vel_params = torch.ones(1, device=self.device)

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        print('self.actuated_dof_indices', self.actuated_dof_indices)
        self.actuated_dof_indices.sort()
        print('self.actuated_dof_indices.sort()', self.actuated_dof_indices)
        print('self.hand.joint_names', self.hand.joint_names)


        # ================== DOF Targets and Tracking ==================
        self.hand_dof_targets = torch.zeros(
            (self.num_envs, self.total_hand_dof), dtype=torch.float, device=self.device
        )
        self.prev_targets = torch.zeros(
            (self.num_envs, self.total_hand_dof), dtype=torch.float, device=self.device
        )
        self.cur_targets = torch.zeros(
            (self.num_envs, self.total_hand_dof), dtype=torch.float, device=self.device
        )
        self.last_actions = torch.zeros(
            (self.num_envs, 22), dtype=torch.float, device=self.device
        )

        # ================== PID Control ==================
        self.pid_integral = torch.zeros(
            (self.num_envs, self.total_hand_dof), dtype=torch.float, device=self.device
        )
        self.pre_error = torch.zeros(
            (self.num_envs, self.total_hand_dof), dtype=torch.float, device=self.device
        )

        # PID gains (will be randomized per env)
        self.p_gain = torch.ones(
            (self.num_envs, self.total_hand_dof), device=self.device, dtype=torch.float
        ) * self.cfg.p_gain

        self.d_gain = torch.ones(
            (self.num_envs, self.total_hand_dof), device=self.device, dtype=torch.float
        ) * self.cfg.d_gain

        self.i_gain = torch.ones(
            (self.num_envs, self.total_hand_dof), device=self.device, dtype=torch.float
        ) * self.cfg.i_gain

        # ================== Joint Limits ==================
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]
        # print('**************************************', self.total_hand_dof)

        # ================== Body Indices ==================
        # Fingertip bodies
        self.fingertip_bodies = []
        for body_name in self.cfg.fingertip_body_names:
            self.fingertip_bodies.append(self.hand.body_names.index(body_name))
        self.fingertip_bodies.sort()

        self.thumbtip_bodies = self.hand.body_names.index('robot0_thdistal')

        self.num_fingertips = len(self.fingertip_bodies)

        # # Tactile sensor bodies (17 per finger)
        # self.tactile_sensor_bodies_ff = [
        #     self.hand.body_names.index(name) for name in self.cfg.tactile_sensor_body_names["ff"]
        # ]
        # self.tactile_sensor_bodies_mf = [
        #     self.hand.body_names.index(name) for name in self.cfg.tactile_sensor_body_names["mf"]
        # ]
        # self.tactile_sensor_bodies_rf = [
        #     self.hand.body_names.index(name) for name in self.cfg.tactile_sensor_body_names["rf"]
        # ]
        # self.tactile_sensor_bodies_lf = [
        #     self.hand.body_names.index(name) for name in self.cfg.tactile_sensor_body_names["lf"]
        # ]
        # self.tactile_sensor_bodies_th = [
        #     self.hand.body_names.index(name) for name in self.cfg.tactile_sensor_body_names["th"]
        # ]

        # Bottle body indices
        self.bottle_cap_body_idx = self.bottle.body_names.index("link1")
        self.bottle_base_body_idx = self.bottle.body_names.index("link2")

        self.bottle_joint_id = self.bottle.joint_names.index('b_joint')

        # list of actuated joints
        self.bottle_cap_marker_handles = list()
        for cap_marker_name in cfg.cap_marker_handle_names:
            self.bottle_cap_marker_handles.append(self.bottle.body_names.index(cap_marker_name))
        self.bottle_cap_marker_handles.sort()

        # print("CAP_MARKER_HANDLES", self.bottle_cap_marker_handles)

        # ================== Bottle DOF Tracking ==================
        self.last_bottle_dof_pos = torch.zeros(
            (self.num_envs, 1), device=self.device, dtype=torch.float
        )

        # ================== Tactile and Slip Detection ==================
        # Z-component signs for slip direction
        self.z_sign_nonthumb = torch.zeros(
            (self.num_envs, 4, 1), dtype=torch.float, device=self.device
        )
        self.z_sign_thumb = torch.zeros(
            (self.num_envs, 1, 1), dtype=torch.float, device=self.device
        )

        # Slip detection outputs
        self.slip_detection = torch.zeros(
            (self.num_envs, 5), dtype=torch.float, device=self.device
        )
        self.contact_detection = torch.zeros(
            (self.num_envs, 5), dtype=torch.float, device=self.device
        )

        # Previous fingertip positions for velocity calculation
        self.pre_tip_nonthumb = torch.zeros(
            (self.num_envs, 4, 1, 3), dtype=torch.float, device=self.device
        )
        self.pre_tip_thumb = torch.zeros(
            (self.num_envs, 1, 1, 3), dtype=torch.float, device=self.device
        )

        # Velocity tracking (EMA filtered)
        self.pre_vel_nonthumb = torch.zeros((self.num_envs, 4), device=self.device)
        self.pre_vel_thumb = torch.zeros((self.num_envs, 1), device=self.device)

        # Normal force magnitudes (EMA filtered)
        self.pre_normal_mag_ff = torch.zeros((self.num_envs,1), device=self.device)
        self.pre_normal_mag_mf = torch.zeros((self.num_envs,1), device=self.device)
        self.pre_normal_mag_rf = torch.zeros((self.num_envs,1), device=self.device)
        self.pre_normal_mag_lf = torch.zeros((self.num_envs,1), device=self.device)
        self.pre_normal_mag_th = torch.zeros((self.num_envs,1), device=self.device)

        # Virtual torques (for slip-based control)
        self.pre_tau_ff = torch.zeros((self.num_envs,), device=self.device)
        self.pre_tau_mf = torch.zeros((self.num_envs,), device=self.device)
        self.pre_tau_rf = torch.zeros((self.num_envs,), device=self.device)
        self.pre_tau_lf = torch.zeros((self.num_envs,), device=self.device)
        self.pre_tau_th = torch.zeros((self.num_envs,), device=self.device)

        self.last_cube_dof_pos = torch.zeros(
            (self.num_envs, 1), device=self.device, dtype=torch.float
        )

        # ================== Contact Forces ==================
        # Cap force
        self.cap_force = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.pre_cap_force = torch.zeros((self.num_envs, 3), device=self.device)

        # Finger forces (raw)
        self.finger_force_ff = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.finger_force_mf = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.finger_force_rf = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.finger_force_lf = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.finger_force_th = torch.zeros((self.num_envs, 1, 3), device=self.device)

        # Finger forces (EMA filtered)
        self.pre_finger_force_ff = torch.zeros((self.num_envs, 3), device=self.device)
        self.pre_finger_force_mf = torch.zeros((self.num_envs, 3), device=self.device)
        self.pre_finger_force_rf = torch.zeros((self.num_envs, 3), device=self.device)
        self.pre_finger_force_lf = torch.zeros((self.num_envs, 3), device=self.device)
        self.pre_finger_force_th = torch.zeros((self.num_envs, 3), device=self.device)

        # Force norms (for observation)
        self.force_norm = torch.zeros((self.num_envs, 5), dtype=torch.float, device=self.device)

        # ================== Tactile Sensor Data ==================
        self.data_sensor_ff = torch.zeros((self.num_envs, 17), device=self.device)
        self.data_sensor_mf = torch.zeros((self.num_envs, 17), device=self.device)
        self.data_sensor_rf = torch.zeros((self.num_envs, 17), device=self.device)
        self.data_sensor_lf = torch.zeros((self.num_envs, 17), device=self.device)
        self.data_sensor_th = torch.zeros((self.num_envs, 17), device=self.device)

        # Data collection buffers (if needed)
        self.data_slip_vel = torch.zeros((self.num_envs, 5), device=self.device)
        self.data_joint_pos = torch.zeros((self.num_envs, 22), device=self.device)
        self.data_list = []
        self.label_list = []

        # ================== Force Tracking and Rewards ==================
        # Sum of forces over episode
        self.sum_force = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)
        print('self.sum_force init',self.sum_force.shape)

        # Slip tracking
        self.slip_sum = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        self.slip_avr = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)

        # Desired force targets (randomized per episode)
        self.desired_force_norm = torch.zeros((self.num_envs,1), dtype=torch.float, device=self.device)

        # Force trajectory (if using time-varying targets)
        steps = 600  # 10 seconds at 60Hz
        self.desired_force_traj = torch.zeros(
            (self.num_envs, steps), device=self.device, dtype=torch.float
        )

        # Work tracking
        self.right_control_work = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float
        )
        self.control_work = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float
        )

        # ================== Random Forces (for domain randomization) ==================
        self.num_bodies = self.hand.num_bodies
        self.rb_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device
        )
        self.force_progress_buf = torch.zeros_like(self.episode_length_buf)

        # ================== Random  ==================
        self.episode_additive_noise_names = []
        self.episode_affine_noise_names = []

        # --------------------------------------------------
        # reward modufunctione
        # --------------------------------------------------
        self.reward_settings = {
            "failure_penalty": self.cfg.failure_penalty,
            "rotation_reward_scale": self.cfg.rotation_reward_scale,
            "cube_rotation_reward_scale": self.cfg.cube_rotation_reward_scale,
            "action_penalty_scale": self.cfg.action_penalty_scale,
            "left_action_penalty_scale": self.cfg.left_action_penalty_scale,
            "right_action_penalty_scale": self.cfg.right_action_penalty_scale,
            "distance_penalty_scale": self.cfg.distance_penalty_scale,
            "force_reward_scale": self.cfg.force_reward_scale,
            "distance_reward_scale": self.cfg.distance_reward_scale,
            "reorient_reward_scale": self.cfg.reorient_reward_scale,
            "work_penalty_scale": self.cfg.work_penalty_scale,
            "left_work_penalty_scale": self.cfg.left_work_penalty_scale,
            "right_work_penalty_scale": self.cfg.right_work_penalty_scale,
            "thumb_mult": self.cfg.thumb_mult,
            "cap_mult": self.cfg.cap_mult,

            "cap_tactile_mult": self.cfg.cap_tactile_mult,
            "gaiting_mult": self.cfg.gaiting_mult,
            "slip_mult": self.cfg.slip_mult,
            "scaling_factor": self.cfg.scaling_factor,
            "slip_diff_ratio": self.cfg.slip_diff_ratio,
            "release_bonus": self.cfg.release_bonus,
            "grasping_qual_onoff": self.cfg.grasping_qual_onoff,

            "pose_diff_penalty_scale": self.cfg.pose_diff_penalty_scale,
            "drop_threshold": self.cfg.drop_threshold,

            "angle_penalty_scale": self.cfg.angle_penalty_scale,
            "hand_init_pose_penalty": self.cfg.hand_init_pose_penalty,
            "finger_distance_reward_scale": self.cfg.finger_distance_reward_scale,
            "reach_goal_bonus": self.cfg.reach_goal_bonus,
            "rotate_axis": self.cfg.rotate_axis,
            "reset_by_z_angle": self.cfg.reset_by_z_angle,

            "use_curriculum": self.cfg.use_curriculum

        }

        rewarder_name = self.cfg.rewarder
        reward_module_path = f"isaaclab_tasks.direct.shadow_hand_lit.rewarder.{rewarder_name}"
        reward_module = importlib.import_module(reward_module_path)
        reward_function_builder = getattr(reward_module, "build")

        # Setup reward function
        self.reward_function = reward_function_builder(
            device=self.device,
            num_envs=self.num_envs,
            reward_settings=self.reward_settings,
        )

        # randomizer_name = self.cfg.randomizer
        # randomizer_module_path = f"isaaclab_tasks.direct.shadow_hand_lit.randomizer.{randomizer_name}"
        # randomizer_module = importlib.import_module(randomizer_module_path)
        # randomizer_function_builder = getattr(randomizer_module, "build")

        # # Setup reward function
        # self.randomizer_function = randomizer_function_builder(
        #     device=self.device,
        #     num_envs=self.num_envs,
        #     reward_settings=self.reward_settings,
        # )

        # # ================== Load ML Models ==================
        # if self.cfg.method in ["slip_force_nn_model", "tactile_model"]:
        #     self._load_ml_models()

        # ================== Unit Tensors ==================
        self.x_unit_tensor = torch.tensor(
            [1, 0, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor(
            [0, 1, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor(
            [0, 0, 1], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))

        # ================== Initial Observation Buffer ==================
        # Initialize observation buffer with correct stacking
        total_obs_dim = self.single_frame_obs_dim * self.n_stack_frame
        self.obs_buf = torch.zeros(
            (self.num_envs, total_obs_dim), dtype=torch.float, device=self.device
        )

        # ================== State Buffer (for asymmetric training) ==================
        if self.cfg.asymmetric_obs:
            self.states_buf = torch.zeros(
                (self.num_envs, self.cfg.state_space), dtype=torch.float, device=self.device
            )

        # ================== Initialize Environments ==================
        # Set default hand pose
        self.hand_default_dof_pos = self.hand.data.default_joint_pos[0].clone()

        # Generate cap dynamics randomization (damping values)
        # These will be different per environment
        # self.env_cap_dynamics = self._generate_cap_dynamics_randomization()

        print(f"✓ Shadow Hand Bottle Environment initialized")
        print(f"  - Method: {self.method}")
        print(f"  - Num envs: {self.num_envs}")
        print(f"  - Observation dim: {total_obs_dim} ({self.single_frame_obs_dim} × {self.n_stack_frame})")
        # print(f"  - Action dim: {self.num_actions}")
        if self.cfg.asymmetric_obs:
            print(f"  - State dim: {self.cfg.state_space}")


    def _compute_obs_dim(self) -> int:
        """Compute observation dimension based on method."""
        base_dim = 22 + 22 + 3 + 22  # dof_pos + dof_vel + bottle_pos + prev_targets

        if self.cfg.method == "joint_model":
            return base_dim
        elif self.cfg.method == "slip_model":
            return base_dim + 5  # + slip detection
        elif self.cfg.method == "slip_force_model":
            return base_dim + 5 + 5  # + slip + force
        elif self.cfg.method == "slip_force_nn_model":
            return base_dim + 5 + 5  # + nn_slip + force
        else:  # tactile_model
            return base_dim + 85  # + full tactile input (5 fingers × 17 sensors)


    @staticmethod
    def _fill(buf, x, start_pos):
        width = x.size(1)
        buf[:, start_pos : start_pos + width] = x
        return buf, start_pos + width


    def _setup_scene(self):
    # """Setup the scene with hand, bottle, and ground plane."""
        # Add Shadow Hand
        self.hand = Articulation(self.cfg.robot_cfg)

        # Add bottle object
        self.bottle = Articulation(self.cfg.bottle_cfg)  # Bottle is articulated (has joints)

        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.contact_sensor2 = ContactSensor(self.cfg.contact_sensor2)

        # print(f"DEBUG: Scene num_envs: {self.scene.num_envs}")
        # self.contact_forces = ContactSensorCfg(
        # prim_path="/World/envs/env_.*/robot", update_period=0.0, history_length=6, debug_vis=True
        # )
        # # we need to explicitly filter collisions for CPU simulation
        # if self.device == "cpu":
        #     self.scene.filter_collisions(global_prim_paths=[self.cfg.prim_path])
        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Clone and replicate environments
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        
        # Register to scene
        self.scene.articulations["robot"] = self.hand
        self.scene.articulations["bottle"] = self.bottle
        self.scene.sensors["contact_forces"] = self.contact_sensor
        self.scene.sensors["contact_forces_bottle"] = self.contact_sensor2
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        

    def _get_rewards(self) -> torch.Tensor:
        self.states = {
            "dof_pos": self.hand_dof_pos,
            "dof_vel": self.hand_dof_vel,
            "cube_pos": self.bottle_pos,
            "cube_quat": self.bottle_rot,
            "cube_vel": self.bottle_velocities,
            "prev_cube_quat": self.prev_cube_quat,
            "cube_angvel": self.bottle_angvel,
            "right_thumb_tips_pos": self.right_thumb_tips_pos,
            "right_nonthumb_tips_pos": self.right_nonthumb_tips_pos,
            "pre_right_thumb_tips_pos": self.pre_tip_thumb.clone(),
            "pre_right_nonthumb_tips_pos": self.pre_tip_nonthumb.clone(),
            "cube_dof_pos": self.bottle_dof_pos,
            "cube_dof_vel": self.bottle_dof_vel,
            "last_cube_dof_pos": self.last_bottle_dof_pos,
            "cube_base_pos": self.bottle_base_pos,
            "cube_cap_pos": self.bottle_cap_pos,
            "cube_cap_marker_pos": self.bottle_pos_orig[:,self.bottle_cap_marker_handles],
            "right_work": self.right_control_work,
            "work": self.control_work,

            #tactile sensing
            "right_link_ff_tip_sensor_pos": self.right_link_ff_tip_sensor_pos,
            "right_link_ff_tip_sensor_quat": self.right_link_ff_tip_sensor_rot,

            "right_link_mf_tip_sensor_pos": self.right_link_mf_tip_sensor_pos,
            "right_link_mf_tip_sensor_quat": self.right_link_mf_tip_sensor_rot,

            "right_link_rf_tip_sensor_pos": self.right_link_rf_tip_sensor_pos,
            "right_link_rf_tip_sensor_quat": self.right_link_rf_tip_sensor_rot,

            "right_link_lf_tip_sensor_pos": self.right_link_lf_tip_sensor_pos,
            "right_link_lf_tip_sensor_quat": self.right_link_lf_tip_sensor_rot,

            "right_link_th_tip_sensor_pos": self.right_link_th_tip_sensor_pos,
            "right_link_th_tip_sensor_quat": self.right_link_th_tip_sensor_rot,
            "z_sign_nonthumb": self.z_sign_nonthumb,
            "z_sign_thumb": self.z_sign_thumb,
            "slip_detection": self.slip_detection,
            "slip_avr" : self.slip_avr,
            "finger_force_ff" : self.finger_force_ff,
            "finger_force_mf" : self.finger_force_mf,
            "finger_force_rf" : self.finger_force_rf,
            "finger_force_lf" : self.finger_force_lf,
            "finger_force_th" : self.finger_force_th,
            "cap_force": self.cap_force,
            "desired_force_norm": self.desired_force_norm
        }


        # print('Debug ---------self.bottle_dof_pos', self.bottle_dof_pos)
        # print('Debug ---------self.last_bottle_dof_pos', self.last_bottle_dof_pos)
        
        total_reward, _, info = self.reward_function.forward(
            self.reset_buf,
            self.episode_length_buf,
            self.actions,
            self.states,
            self.reward_settings,
            self.max_episode_length,
        )
        # print('Debug ---------total_reward', total_reward)
               # log reward components
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["reward"] = info["rot_score"]
        # self.extras["log"]["dist_goal"] = goal_dist.mean()
        return total_reward

    def _get_observations(self) -> dict:
        # follow original compute_observations
        obs = self.compute_observations()

        if self.cfg.asymmetric_obs:
            states = self.compute_full_state()

        observations = {"policy": obs}
        if self.cfg.asymmetric_obs:
            observations = {"policy": obs, "critic": states}
        return observations
    
    def compute_full_state(self):
        states = torch.cat(
            (
                # hand
                self.hand_dof_pos,
                self.hand_dof_vel,
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                # object
                self.bottle_pos.reshape(self.num_envs, -1),
                self.bottle_rot.reshape(self.num_envs, -1),
                self.bottle_base_pos.reshape(self.num_envs, -1),
                self.bottle_cap_pos.reshape(self.num_envs, -1),
                self.bottle_pos_orig[:,self.bottle_cap_marker_handles].reshape(self.num_envs, -1),
                # # goal
                # self.in_hand_pos,
                # self.goal_rot,
                # quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                # # fingertips
                # self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                # self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                # self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # self.cfg.force_torque_obs_scale
                # * self.fingertip_force_sensors.view(self.num_envs, self.num_fingertips * 6),
                # # actions
                # self.actions,
            ),
            dim=-1,
        )

        # print('Debug--------------------------------', states.shape)
        return states
        
    def compute_observations(self):


        previous_dof_pos = self.hand_dof_pos.clone()
        
        pre_right_thumb_tips_pos =  self.pre_tip_thumb.clone()
        pre_right_nonthumb_tips_pos = self.pre_tip_nonthumb.clone()

        dof_vel_scaled = self.hand_dof_vel * self.dof_vel_scale

        if self.cfg.no_dof_vel:
            dof_vel_scaled = torch.zeros_like(dof_vel_scaled)

        self.cap_base_pos =self.bottle_pos_orig.reshape(self.num_envs, -1)
        # print("self.cap_base_pos", self.cap_base_pos.shape)

        cube_pos = self.bottle_pos
        # print("self.cube_pos", cube_pos)


        if self.cfg.no_obj_pos:
            cube_pos = torch.zeros_like(cube_pos)

        if self.cfg.no_cap_base:
            self.cap_base_pos = torch.zeros_like(self.cap_base_pos)

        self.cube_state = self.bottle.data.body_state_w
        cube_quat = self.bottle.data.body_quat_w
        if self.cfg.no_obj_quat:
            cube_quat = torch.zeros_like(cube_quat)

        obs_prev_target = self.prev_targets.clone()
        randomized_prev_target = obs_prev_target
        
        # # # tactile observations: 
        # # 1. magnitude 
        # # example marker positions in cap frame
        # marker_offsets = torch.tensor([
        #     [ 0.0400,  0.0000, -0.0105],
        #     [ 0.0370,  0.0153, -0.0105],
        #     ...
        # ], device=self.device)

        # cap_pose = self.bottle.data.body_pos_w[:, cap_id]
        # marker_pos_w = cap_pose.unsqueeze(1) + marker_offsets
        cap_marker_pos =  self.bottle_pos_orig[:,self.bottle_cap_marker_handles]

        right_thumb_tips_pos =  self.fingertip_pos[:, 4:5,:]
        right_nonthumb_tips_pos =  self.fingertip_pos[:, 0:4,:]

        right_thumb_tips_pos = right_thumb_tips_pos.unsqueeze(2)  # [N, F2, 1, 3]
        right_nonthumb_tips_pos = right_nonthumb_tips_pos.unsqueeze(2)  # [N, F2, 1, 3]

        # pre_right_thumb_tips_pos = pre_right_thumb_tips_pos.unsqueeze(2)  # [N, F2, 1, 3]
        # pre_right_nonthumb_tips_pos = pre_right_nonthumb_tips_pos.unsqueeze(2)  # [N, F2, 1, 3]

        # print("right_nonthumb_tips_pos", right_nonthumb_tips_pos.shape)

        cube_center_pose = self.bottle_pos
        cube_angle = self.bottle_dof_pos
        last_cube_angle = self.last_bottle_dof_pos

        # #print(cube_center_quat)

        cube_center_pose = cube_center_pose.unsqueeze(1).unsqueeze(1)# [N, 1, 1, 3]
        vec_center_tip_nonthumb = right_nonthumb_tips_pos - cube_center_pose
        vec_center_tip_thumb = right_thumb_tips_pos-cube_center_pose

        temp_tip_non_thumb_vel = (right_nonthumb_tips_pos - pre_right_nonthumb_tips_pos) / self.cfg.sim.dt
        temp_tip_thumb_vel = (right_thumb_tips_pos - pre_right_thumb_tips_pos) / self.cfg.sim.dt

        # xy plane projection.
        temp_tip_non_thumb_vel[..., 2] = 0.0
        temp_tip_thumb_vel[..., 2] = 0.0
        tip_non_thumb_vel = temp_tip_non_thumb_vel.clone()
        tip_thumb_vel = temp_tip_thumb_vel.clone()

        cap_marker_pos = cap_marker_pos.unsqueeze(1)  # [N, 1, K2, 3] 
        
        right_link_ff_tip_sensor_pos = self.right_link_ff_tip_sensor_pos.unsqueeze(2)  # [N, F2, 1, 3]
        right_link_mf_tip_sensor_pos = self.right_link_mf_tip_sensor_pos.unsqueeze(2)  # [N, F2, 1, 3]

        right_link_rf_tip_sensor_pos = self.right_link_rf_tip_sensor_pos.unsqueeze(2)  # [N, F2, 1, 3]
        right_link_lf_tip_sensor_pos = self.right_link_lf_tip_sensor_pos.unsqueeze(2)  # [N, F2, 1, 3]
        right_link_th_tip_sensor_pos = self.right_link_th_tip_sensor_pos.unsqueeze(2)  # [N, F2, 1, 3]

        #################################################### cap to finger 
        dist_cap_to_sensor_ff = torch.norm(
            right_link_ff_tip_sensor_pos - cap_marker_pos, dim=-1, p=2
        )  # (N, F2, K2)

        dist_cap_to_sensor_mf = torch.norm(
            right_link_mf_tip_sensor_pos - cap_marker_pos, dim=-1, p=2
        )  # (N, F2, K2)
        dist_cap_to_sensor_rf = torch.norm(
            right_link_rf_tip_sensor_pos - cap_marker_pos, dim=-1, p=2
        )  # (N, F2, K2)

        dist_cap_to_sensor_lf = torch.norm(
            right_link_lf_tip_sensor_pos - cap_marker_pos, dim=-1, p=2
        )  # (N, F2, K2)
        dist_cap_to_sensor_th = torch.norm(
            right_link_th_tip_sensor_pos - cap_marker_pos, dim=-1, p=2
        )  # (N, F2, K2)

        dist_cap_to_sensor_ff = torch.min(dist_cap_to_sensor_ff, -1)[0]
        dist_cap_to_sensor_mf = torch.min(dist_cap_to_sensor_mf, -1)[0]
        dist_cap_to_sensor_rf = torch.min(dist_cap_to_sensor_rf, -1)[0]
        dist_cap_to_sensor_lf = torch.min(dist_cap_to_sensor_lf, -1)[0]
        dist_cap_to_sensor_th = torch.min(dist_cap_to_sensor_th, -1)[0]

        #################################################### center to finger 

        dist_cap_to_right_nonthumb = torch.norm(
            right_nonthumb_tips_pos - cap_marker_pos, dim=-1, p=2
        )  # (N, F2, K2)
        dist_cap_to_right_thumb = torch.norm(
            right_thumb_tips_pos - cap_marker_pos, dim=-1, p=2
        )  # (N, F2, K2)


        dist_cap_to_right_nonthumb = torch.min(dist_cap_to_right_nonthumb, -1)[0]
        dist_cap_to_right_thumb = torch.min(dist_cap_to_right_thumb, -1)[0]

        min_dis_val_ff = dist_cap_to_right_nonthumb[:, 0] # 
        min_dis_val_mf = dist_cap_to_right_nonthumb[:, 1] # 
        min_dis_val_rf = dist_cap_to_right_nonthumb[:, 2] #
        min_dis_val_lf = dist_cap_to_right_nonthumb[:, 3] #
        min_dis_val_th = dist_cap_to_right_thumb[:, 0] # 

        self.plot_min_dis_val_ff = min_dis_val_ff
        self.plot_min_dis_val_mf = min_dis_val_mf
        self.plot_min_dis_val_rf = min_dis_val_rf
        self.plot_min_dis_val_lf = min_dis_val_lf
        self.plot_min_dis_val_th = min_dis_val_th

        #################################################### center to finger 

        dist_center_to_sensor_ff = torch.norm(
            right_link_ff_tip_sensor_pos - cube_center_pose, dim=-1, p=2
        )  # (N, F2, K2)

        dist_center_to_sensor_mf = torch.norm(
            right_link_mf_tip_sensor_pos - cube_center_pose, dim=-1, p=2
        )  # (N, F2, K2)
        dist_center_to_sensor_rf = torch.norm(
            right_link_rf_tip_sensor_pos - cube_center_pose, dim=-1, p=2
        )  # (N, F2, K2)

        dist_center_to_sensor_lf = torch.norm(
            right_link_lf_tip_sensor_pos - cube_center_pose, dim=-1, p=2
        )  # (N, F2, K2)
        dist_center_to_sensor_th = torch.norm(
            right_link_th_tip_sensor_pos - cube_center_pose, dim=-1, p=2
        )  # (N, F2, K2)

        dist_center_to_sensor_ff = torch.min(dist_center_to_sensor_ff, -1)[0]
        dist_center_to_sensor_mf = torch.min(dist_center_to_sensor_mf, -1)[0]
        dist_center_to_sensor_rf = torch.min(dist_center_to_sensor_rf, -1)[0]
        dist_center_to_sensor_lf = torch.min(dist_center_to_sensor_lf, -1)[0]
        dist_center_to_sensor_th = torch.min(dist_center_to_sensor_th, -1)[0]


        ################################################## center to finger
        temp_sensor_input_ff = dist_cap_to_sensor_ff
        temp_sensor_input_mf = dist_cap_to_sensor_mf
        temp_sensor_input_rf = dist_cap_to_sensor_rf
        temp_sensor_input_lf = dist_cap_to_sensor_lf
        temp_sensor_input_th = dist_cap_to_sensor_th

        #contact_threshold = 0.005 or 0.01
        radius = 0.04
        contact_offset = 0.005

        contact_threshold = contact_offset
        force_k = 0.001

        sensor_input_ff = torch.where(temp_sensor_input_ff > contact_threshold, 0.0, 1/(temp_sensor_input_ff + 0.01)*0.002) #10.0
        sensor_input_mf = torch.where(temp_sensor_input_mf > contact_threshold, 0.0, 1/(temp_sensor_input_mf + 0.01)*0.002)
        sensor_input_rf = torch.where(temp_sensor_input_rf > contact_threshold, 0.0, 1/(temp_sensor_input_rf + 0.01)*0.002)
        sensor_input_lf = torch.where(temp_sensor_input_lf > contact_threshold, 0.0, 1/(temp_sensor_input_lf + 0.01)*0.002)
        sensor_input_th = torch.where(temp_sensor_input_th > contact_threshold, 0.0, 1/(temp_sensor_input_th + 0.01)*0.002)

        mlp_sensor_input_ff = torch.where(sensor_input_ff > 0.0, 1.0, 0.0)
        mlp_sensor_input_mf = torch.where(sensor_input_mf > 0.0, 1.0, 0.0)
        mlp_sensor_input_rf = torch.where(sensor_input_rf > 0.0, 1.0, 0.0)
        mlp_sensor_input_lf = torch.where(sensor_input_lf > 0.0, 1.0, 0.0)
        mlp_sensor_input_th = torch.where(sensor_input_th > 0.0, 1.0, 0.0)

        mlp_input = torch.stack([mlp_sensor_input_ff, mlp_sensor_input_mf, mlp_sensor_input_rf, mlp_sensor_input_lf, mlp_sensor_input_th], dim=1)


        #print(outputs.shape)


        temp_sensor_input_ff = torch.where(temp_sensor_input_ff > contact_threshold, 0.0, 0.1)
        temp_sensor_input_mf = torch.where(temp_sensor_input_mf > contact_threshold, 0.0, 0.1)
        temp_sensor_input_rf = torch.where(temp_sensor_input_rf > contact_threshold, 0.0, 0.1)
        temp_sensor_input_lf = torch.where(temp_sensor_input_lf > contact_threshold, 0.0, 0.1)
        temp_sensor_input_th = torch.where(temp_sensor_input_th > contact_threshold, 0.0, 0.1)

        sum_dist_cap_to_sensor_ff_padded = temp_sensor_input_ff.sum(dim=1)
        sum_dist_cap_to_sensor_mf_padded = temp_sensor_input_mf.sum(dim=1)
        sum_dist_cap_to_sensor_rf_padded = temp_sensor_input_rf.sum(dim=1)
        sum_dist_cap_to_sensor_lf_padded = temp_sensor_input_lf.sum(dim=1)
        sum_dist_cap_to_sensor_th_padded = temp_sensor_input_th.sum(dim=1)

        #print("temp_sensor_input_ff", temp_sensor_input_ff.shape)


        self.data_sensor_ff  = sensor_input_ff
        self.data_sensor_mf  = sensor_input_mf
        self.data_sensor_rf  = sensor_input_rf
        self.data_sensor_lf  = sensor_input_lf
        self.data_sensor_th  = sensor_input_th
        

        ####################################################################################################

        vec_norm_1 = torch.norm(vec_center_tip_nonthumb, dim=-1, keepdim=True)
        vec_norm_2 = torch.norm(vec_center_tip_thumb, dim=-1, keepdim=True)
        vel_norm_3 = torch.norm(tip_non_thumb_vel, dim=-1, keepdim=True)
        vel_norm_4 = torch.norm(tip_thumb_vel, dim=-1, keepdim=True)

        # print("vel_norm_3.squeeze()", vel_norm_3.shape)
        # print("el_norm_4.reshape(self.num_envs, 1)", vel_norm_4.shape)



        eps = 1e-8
        vec_unit_center_tip_nonthumb = vec_center_tip_nonthumb / (vec_norm_1 + eps)
        vec_unit_center_tip_thumb = vec_center_tip_thumb / (vec_norm_2 + eps)
        vec_unit_tip_non_thumb_vel = tip_non_thumb_vel / (vel_norm_3 + eps)
        vec_unit_tip_thumb_vel = tip_thumb_vel / (vel_norm_4 + eps)

        # print('vec_unit_center_tip_nonthumb', vec_unit_center_tip_nonthumb.shape)
        # print('vec_unit_tip_non_thumb_vel', vec_unit_tip_non_thumb_vel.shape)

        non_thum_cross = torch.cross(vec_unit_center_tip_nonthumb, vec_unit_tip_non_thumb_vel, dim=-1)
        thum_cross = torch.cross(vec_unit_center_tip_thumb, vec_unit_tip_thumb_vel, dim=-1)

        z_component_nonthumb = non_thum_cross[..., 2]
        z_component_thumb = thum_cross[..., 2]

        # z (-1, 0, 1)
        self.z_sign_nonthumb = torch.sign(z_component_nonthumb)
        self.z_sign_thumb = torch.sign(z_component_thumb)

        # # slip 
        # diff_angle = (
        #     cube_angle - last_cube_angle
        # )
        # angular_vel = diff_angle/120
        # #should be modified here : 

        # tan_vel = angular_vel*0.04
        # print("tan_vel", tan_vel.shape)
        # # print("vel_norm_3.squeeze()", vel_norm_3.shape)
        # # print("el_norm_4.reshape(self.num_envs, 1)", vel_norm_4.shape)
        # diff_rel_vel = tan_vel.view(self.num_envs, 2, 1) - vel_norm_3.view(self.num_envs, 1, 4)
        # abs_diff_rel_vel = diff_rel_vel.abs()
        # diff_thumb_rel_vel = tan_vel.view(self.num_envs, 2, 1) - vel_norm_4.view(self.num_envs, 1, 1)   # size(2,1) - size(2,1)
        # abs_diff_thumb_rel_vel = diff_thumb_rel_vel.abs()

        # abs_diff_rel_vel = abs_diff_rel_vel.squeeze(0)
        # abs_diff_thumb_rel_vel = abs_diff_thumb_rel_vel.squeeze(0)

        # print('abs_diff_rel_vel',abs_diff_rel_vel.shape)
        # print('abs_diff_thumb_rel_vel',abs_diff_thumb_rel_vel.shape)

        # vel_alpha = 0.75

        # final_abs_vel_nonthumb = vel_alpha*abs_diff_rel_vel + (1-vel_alpha)*self.pre_vel_nonthumb
        # self.pre_vel_nonthumb = final_abs_vel_nonthumb.clone()


        # final_abs_vel_thumb = vel_alpha*abs_diff_thumb_rel_vel + (1-vel_alpha)*self.pre_vel_thumb
        # self.pre_vel_thumb = final_abs_vel_thumb.clone()


        # # tuned
        # slip_tau = 0.01 # Can you trust?

        # finger_force_alpha = 0.4 # 0.3

        # #print(self.finger_force_lf)

        # final_finger_force_ff = finger_force_alpha * self.finger_force_ff + (1-finger_force_alpha)*self.pre_finger_force_ff
        # final_finger_force_mf = finger_force_alpha * self.finger_force_mf + (1-finger_force_alpha)*self.pre_finger_force_mf
        # final_finger_force_rf = finger_force_alpha * self.finger_force_rf + (1-finger_force_alpha)*self.pre_finger_force_rf
        # final_finger_force_lf = finger_force_alpha * self.finger_force_lf + (1-finger_force_alpha)*self.pre_finger_force_lf
        # final_finger_force_th = finger_force_alpha * self.finger_force_th + (1-finger_force_alpha)*self.pre_finger_force_th

        # #print(final_finger_force_lf)

        # self.pre_finger_force_ff = final_finger_force_ff
        # self.pre_finger_force_mf = final_finger_force_mf
        # self.pre_finger_force_rf = final_finger_force_rf
        # self.pre_finger_force_lf = final_finger_force_lf
        # self.pre_finger_force_th = final_finger_force_th

        # final_cap_force = finger_force_alpha * self.cap_force + (1-finger_force_alpha)*self.pre_cap_force

        # cap_force_norm = torch.norm(final_cap_force,dim=-1,p=2)

        # contact_ff = torch.where(sum_dist_cap_to_sensor_ff_padded > 0.0, 1.0, 0.0)
        # contact_mf = torch.where(sum_dist_cap_to_sensor_mf_padded > 0.0, 1.0, 0.0)
        # contact_rf = torch.where(sum_dist_cap_to_sensor_rf_padded > 0.0, 1.0, 0.0)
        # contact_lf = torch.where(sum_dist_cap_to_sensor_lf_padded > 0.0, 1.0, 0.0)
        # contact_th = torch.where(sum_dist_cap_to_sensor_th_padded > 0.0, 1.0, 0.0)

        # non_contact_penalty = contact_ff * contact_mf * contact_rf * contact_lf * contact_th



        # finger_force_ff_norm = torch.norm(final_finger_force_ff,dim=-1,p=2)
        # finger_force_mf_norm = torch.norm(final_finger_force_mf,dim=-1,p=2)
        # finger_force_rf_norm = torch.norm(final_finger_force_rf,dim=-1,p=2)
        # finger_force_lf_norm = torch.norm(final_finger_force_lf,dim=-1,p=2)
        # finger_force_th_norm = torch.norm(final_finger_force_th,dim=-1,p=2)

        # force_contact_ff = torch.where(finger_force_ff_norm > 0, 1.0, 0.0) #
        # force_contact_mf = torch.where(finger_force_mf_norm > 0, 1.0, 0.0) #
        # force_contact_rf = torch.where(finger_force_rf_norm > 0, 1.0, 0.0) #
        # force_contact_lf = torch.where(finger_force_lf_norm > 0, 1.0, 0.0) #
        # force_contact_th = torch.where(finger_force_th_norm > 0, 1.0, 0.0) #

        # non_force_contact_penalty = force_contact_ff * force_contact_mf * force_contact_rf * force_contact_lf * force_contact_th


        # self.plot_sensor_sum_ff  = finger_force_ff_norm
        # self.plot_sensor_sum_mf  = finger_force_mf_norm
        # self.plot_sensor_sum_rf  = finger_force_rf_norm
        # self.plot_sensor_sum_lf  = finger_force_lf_norm
        # self.plot_sensor_sum_th  = finger_force_th_norm

        # #print("self.plot_sensor_sum_th  ",self.plot_sensor_sum_th )

        # #temp_param = 10.0

        # print("final_abs_vel_nonthumb", final_abs_vel_nonthumb.shape)
        # print("final_abs_vel_thumb", final_abs_vel_thumb.squeeze(-1).squeeze(-1).shape)

        # # label 
        # slip_ff = torch.where((sum_dist_cap_to_sensor_ff_padded > 0.0) & (force_contact_ff > 0), final_abs_vel_nonthumb[:,0], 0.0)
        # slip_mf = torch.where((sum_dist_cap_to_sensor_mf_padded > 0.0) & (force_contact_mf > 0), final_abs_vel_nonthumb[:,1], 0.0)
        # slip_rf = torch.where((sum_dist_cap_to_sensor_rf_padded > 0.0) & (force_contact_rf > 0), final_abs_vel_nonthumb[:,2], 0.0)
        # slip_lf = torch.where((sum_dist_cap_to_sensor_lf_padded > 0.0) & (force_contact_lf > 0), final_abs_vel_nonthumb[:,3], 0.0)
        # slip_th = torch.where((sum_dist_cap_to_sensor_th_padded > 0.0) & (force_contact_th > 0), final_abs_vel_thumb[:,0], 0.0)

        # print('slip_ff', slip_ff.shape)
        # print('slip_lf', slip_lf.shape)
        # print('slip_th', slip_th.shape)

        # self.slip_detection = torch.stack([slip_ff, slip_mf, slip_rf, slip_lf, slip_th], dim=1)  # (2, 5)

        # finger_force_ff_norm = torch.where((sum_dist_cap_to_sensor_ff_padded > 0.0) & (finger_force_ff_norm > 0), finger_force_ff_norm, 0.0)
        # finger_force_mf_norm = torch.where((sum_dist_cap_to_sensor_mf_padded > 0.0) & (finger_force_mf_norm > 0), finger_force_mf_norm, 0.0)
        # finger_force_rf_norm = torch.where((sum_dist_cap_to_sensor_rf_padded > 0.0) & (finger_force_rf_norm > 0), finger_force_rf_norm, 0.0)
        # finger_force_lf_norm = torch.where((sum_dist_cap_to_sensor_lf_padded > 0.0) & (finger_force_lf_norm > 0), finger_force_lf_norm, 0.0)
        # finger_force_th_norm = torch.where((sum_dist_cap_to_sensor_th_padded > 0.0) & (finger_force_th_norm > 0), finger_force_th_norm, 0.0)

        # self.force_norm = torch.stack([finger_force_ff_norm, 
        # finger_force_mf_norm, 
        # finger_force_rf_norm, 
        # finger_force_lf_norm, 
        # finger_force_th_norm], dim=1)  # (2, 5)



        # self.data_slip_vel = self.slip_detection 

        # self.slip_detection_sum = slip_ff + slip_mf + slip_rf + slip_lf + slip_th


        # f_norm_all = finger_force_ff_norm + finger_force_mf_norm + finger_force_rf_norm + finger_force_lf_norm + finger_force_th_norm


        # print('f_norm_all', f_norm_all.shape)
        # print('sum_force', self.sum_force.shape)

        # self.sum_force += f_norm_all
        # temp_progress_buf = torch.where(self.episode_length_buf > 0,self.episode_length_buf , 1)
        # current_force_norm_avr = self.sum_force / temp_progress_buf.float()

        # self.slip_sum += torch.abs(self.slip_detection_sum)

        # self.slip_avr = self.slip_sum / temp_progress_buf.float()

        # self.plot_slip_avr = self.slip_avr

        # #print("self.plot_slip_avr  ",self.plot_slip_avr )

        # #print(self.slip_avr)
 
        # slip_avr_input = self.slip_avr.unsqueeze(1)

        # self.data_joint_pos = dof_pos_scaled
  
        # # self.sum_force =  self.sum_force +  finger_force_ff_norm + finger_force_mf_norm + finger_force_rf_norm + finger_force_lf_norm + finger_force_th_norm
        # # self.sum_force_avr = self.sum_force /self.episode_length_buf.float()

        # # print("self.sum_force :", current_force_norm_avr)
        # # print("self.desired_norm :", f_d_norm)
        # # print("self.env_cap_dynamics : ", torch.norm(self.env_cap_dynamics,dim=-1))
        # # print(torch.abs(f_d_norm - current_force_norm_avr))

        # obs_tactile_input =  mlp_input.reshape(self.num_envs, -1) 

        # print('******************', randomized_prev_target.shape)
        # print('******************', cube_pos)
        # print('******************', dof_pos_scaled.shape)
        
        # 69
        if self.method == "joint_model":
            frame_obs_buf = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos[:, self.actuated_dof_indices], self.hand_dof_lower_limits[:, self.actuated_dof_indices], self.hand_dof_upper_limits[:, self.actuated_dof_indices]),
                self.cfg.vel_obs_scale * self.hand_dof_vel[:, self.actuated_dof_indices],
                cube_pos, # 3
                randomized_prev_target[:, self.actuated_dof_indices], #22
            ),
            dim=-1,
            )
        
        # print("Debug self.hand_dof_lower_limits", self.hand_dof_lower_limits)
        # print("Debug self.hand_dof_upper_limits", self.hand_dof_upper_limits)

        # if self.use_reward_obs:
        #     frame_obs_buf = torch.cat(
        #         (frame_obs_buf, self.reward_function.get_observation()), dim=-1
        #     )
        self.cube_shape_id = []


        # if self.cfg.no_obj_id:
        #     cube_id_obs = torch.zeros_like(self.cube_shape_id)
        # else:
        #     cube_id_obs = self.cube_shape_id


        # frame_obs_buf = torch.cat((frame_obs_buf, cube_id_obs), dim=-1)

        #print("check 3333",frame_obs_buf.shape)
        if torch.isnan(frame_obs_buf).int().sum() > 0:
            print("Nan Detected in IsaacGym simulation.")

        frame_obs_buf = torch.nan_to_num(
            frame_obs_buf, nan=0.0, posinf=1.0, neginf=-1.0
        )
        frame_obs_buf = torch.clamp(frame_obs_buf, -100.0, 100.0)

        frame_obs_buf = frame_obs_buf

        # if type(self.obs_buf) is not torch.Tensor:
        #     self.obs_buf = self.obs_buf['policy']

        if self.n_stack_frame == 1:
           obs_buf = frame_obs_buf.clone()
        else:
            obs_buf = torch.cat(
                (
                    frame_obs_buf[:, : self.single_frame_obs_dim],
                    frame_obs_buf[:, : self.single_frame_obs_dim],
                ),
                dim=-1,
            )
        # print('Debug obs_buf ************', obs_buf)

        return obs_buf
    

    def _compute_intermediate_values(self):
        # data for hand
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.fingertip_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.fingertip_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )

        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.fingertip_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        # data for bottle
        self.bottle_pos = self.bottle.data.root_pos_w - self.scene.env_origins
        self.bottle_rot = self.bottle.data.root_quat_w
        self.bottle_velocities = self.bottle.data.root_vel_w
        self.bottle_linvel = self.bottle.data.root_lin_vel_w
        self.bottle_angvel = self.bottle.data.root_ang_vel_w
        self.bottle_dof_pos = self.bottle.data.joint_pos[:, self.bottle_joint_id]
        self.bottle_dof_vel = self.bottle.data.joint_vel[:, self.bottle_joint_id]

        self.bottle_pos_orig = self.bottle.data.body_pos_w - self.scene.env_origins.unsqueeze(1)
        self.bottle_base_pos = self.bottle_pos_orig[:, self.bottle_base_body_idx]
        self.bottle_cap_pos = self.bottle_pos_orig[:, self.bottle_cap_body_idx]

        self.right_thumb_tips_pos = self.fingertip_pos[:, 4:5,:]
        self.right_nonthumb_tips_pos = self.fingertip_pos[:, 0:4,:]

        # print('Debug-----------self.bottle.data.root_pos_w', self.bottle.data.root_pos_w.shape)
        # print('Debug-----------self.bottle.data.body_pos_w',self.bottle.data.body_pos_w-self.scene.env_origins.unsqueeze(1))
        
        # pre_right_thumb_tips_pos = pre_right_thumb_tips_pos.unsqueeze(2)  # [N, F2, 1, 3]
        # pre_right_nonthumb_tips_pos = pre_right_nonthumb_tips_pos.unsqueeze(2)  # [N, F2, 1, 3]

        cap_marker_pos = self.bottle_pos_orig[:, self.bottle_cap_marker_handles]

        cap_marker_pos = cap_marker_pos.unsqueeze(1)  # [N, 1, K2, 3]

        self.right_thumb_tips_pos = self.right_thumb_tips_pos.unsqueeze(2)  # [N, F2, 1, 3]
        self.right_nonthumb_tips_pos = self.right_nonthumb_tips_pos.unsqueeze(2)  # [N, F2, 1, 3]

        cube_center_pose = self.bottle.data.root_pos_w
        cube_center_pose = cube_center_pose.unsqueeze(1).unsqueeze(1)# [N, 1, 1, 3]
        

        # slip_detection = states["slip_detection"]

        dist_cap_to_right_nonthumb = torch.norm(
            self.right_nonthumb_tips_pos - cap_marker_pos, dim=-1, p=2
        )  # (N, F2, K2)
        dist_cap_to_right_thumb = torch.norm(
            self.right_thumb_tips_pos - cap_marker_pos, dim=-1, p=2
        )  # (N, F2, K2)
        # print('Debug-----------right_thumb_tips_pos',self.right_thumb_tips_pos)
        # print('Debug-----------cap_marker_pos',cap_marker_pos)

        dist_cap_to_right_nonthumb = torch.min(dist_cap_to_right_nonthumb, -1)[0]
        dist_cap_to_right_thumb = torch.min(dist_cap_to_right_thumb, -1)[0]

        self.min_dis_val_ff = dist_cap_to_right_nonthumb[:, 0] # 
        self.min_dis_val_mf = dist_cap_to_right_nonthumb[:, 1] # 
        self.min_dis_val_rf = dist_cap_to_right_nonthumb[:, 2] #
        self.min_dis_val_lf = dist_cap_to_right_nonthumb[:, 3] #
        self.min_dis_val_th = dist_cap_to_right_thumb[:, 0] # 

        ####################################################################################33
        #################################################### sensor

        self.right_link_ff_tip_sensor_pos =  self.fingertip_pos[:, 0:1,:]
        self.right_link_mf_tip_sensor_pos =  self.fingertip_pos[:, 1:2,:]
        self.right_link_rf_tip_sensor_pos =  self.fingertip_pos[:, 2:3,:]
        self.right_link_lf_tip_sensor_pos =  self.fingertip_pos[:, 3:4,:]
        self.right_link_th_tip_sensor_pos =  self.fingertip_pos[:, 4:5,:]

        self.right_link_ff_tip_sensor_rot =  self.fingertip_rot[:, 0:1,:]
        self.right_link_mf_tip_sensor_rot =  self.fingertip_rot[:, 1:2,:]
        self.right_link_rf_tip_sensor_rot =  self.fingertip_rot[:, 2:3,:]
        self.right_link_lf_tip_sensor_rot =  self.fingertip_rot[:, 3:4,:]
        self.right_link_th_tip_sensor_rot =  self.fingertip_rot[:, 4:5,:]


        # right_link_ff_tip_sensor_pos = right_link_ff_tip_sensor_pos.unsqueeze(2)  # [N, F2, 1, 3]
        # right_link_mf_tip_sensor_pos = right_link_mf_tip_sensor_pos.unsqueeze(2)  # [N, F2, 1, 3]

        # right_link_rf_tip_sensor_pos = right_link_rf_tip_sensor_pos.unsqueeze(2)  # [N, F2, 1, 3]
        # right_link_lf_tip_sensor_pos = right_link_lf_tip_sensor_pos.unsqueeze(2)  # [N, F2, 1, 3]
        # right_link_th_tip_sensor_pos = right_link_th_tip_sensor_pos.unsqueeze(2)  # [N, F2, 1, 3]
        ####################################################################################33
        ####################### contact
        contact_forces = self.contact_sensor.data.net_forces_w
        self.cap_force = self.contact_sensor2.data.net_forces_w
        # print('Debug-------------------contact_forces', self.contact_sensor2.data.net_forces_w[:, self.bottle_cap_body_idx,:])

        self.finger_force_ff = contact_forces[:, 0, :]
        self.finger_force_mf = contact_forces[:, 1, :]
        self.finger_force_rf = contact_forces[:, 2, :]
        self.finger_force_lf = contact_forces[:, 3, :]
        self.finger_force_th = contact_forces[:, 4, :]

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        self._compute_intermediate_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        reset_buf = torch.zeros_like(time_out)
        # print('Debug -------self.episode_length_buf',self.episode_length_buf)

        # Early termination.
        reset_buf = torch.where(
            self.bottle.data.root_pos_w[:,2] < self.cfg.drop_threshold,
            torch.ones_like(time_out),
            reset_buf,
        )

        distance_termination = 0.07

        # print('Debug----------- bottle',self.bottle.data.root_pos_w[:,0:3])

        # print('Debug----------- self.min_dis_val_ff', self.min_dis_val_ff)

        reset_buf = torch.where(
            self.min_dis_val_ff > distance_termination,
            torch.ones_like(reset_buf),
            reset_buf,
        )

        reset_buf = torch.where(
            self.min_dis_val_mf > distance_termination,
            torch.ones_like(reset_buf),
            reset_buf,
        )

        reset_buf = torch.where(
            self.min_dis_val_rf > distance_termination,
            torch.ones_like(reset_buf),
            reset_buf,
        )

        reset_buf = torch.where(
            self.min_dis_val_lf > distance_termination,
            torch.ones_like(reset_buf),
            reset_buf,
        )

        reset_buf = torch.where(
            self.min_dis_val_th > distance_termination,
            torch.ones_like(reset_buf),
            reset_buf,
        )
        # print('Debug _reset_buf ************',reset_buf)
        # print('Debug -------self.episode_length_buf',time_out)

        return reset_buf, time_out


    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        # reset articulation and rigid body attributes via IsaacLab helpers
        super()._reset_idx(env_ids)

        # print('Debug _reset_idx ************')
        object_default_state = self.bottle.data.default_root_state.clone()[env_ids]
        # print('Debug object_default_state ************', self.bottle.data.body_pose_w[:,0:3])


        # print('Debug object_default_state ************', self.hand.data.default_joint_pos)

        # reset controller setup
        self.virtual_torque_rate = self.cfg.virtual_torque_rate
        
        self.randomize_p_gain_lower = self.cfg.p_gain_val * self.cfg.p_gain_lower  # 0.30
        self.randomize_p_gain_upper = self.cfg.p_gain_val * self.cfg.p_gain_upper  # 0.60
        self.randomize_d_gain_lower = self.cfg.d_gain_val * self.cfg.d_gain_lower  # 0.75
        self.randomize_d_gain_upper = self.cfg.d_gain_val * self.cfg.d_gain_upper  # 1.05

        self.p_gain[env_ids] = sample_uniform(
            self.randomize_p_gain_lower,
            self.randomize_p_gain_upper,
            (len(env_ids), self.num_hand_dofs),
            device=self.device,
        ).squeeze(1)
        self.d_gain[env_ids] = sample_uniform(
            self.randomize_d_gain_lower,
            self.randomize_d_gain_upper,
            (len(env_ids), self.num_hand_dofs),
            device=self.device,
        ).squeeze(1)

        # reset hand joints
        dof_pos = self.hand.data.default_joint_pos[env_ids]
        dof_pos = dof_pos + torch.randn_like(dof_pos)*self.cfg.reset_dof_pos_noise
        # print('Debug dof_pos ************', dof_pos)

        dof_vel =  torch.zeros_like(self.hand_dof_vel[env_ids])

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        self.desired_force_norm = 10 + 130 * torch.rand(
           (self.num_envs,), dtype=torch.float, device=self.device
           ) 
        # # reset cube dof
        # for i, axis_noise_scale in enumerate(
        #     [
        #         self.cfg.object_init_pos_noise_x_scale,
        #         self.object_init_pos_noise_y_scale,
        #         self.object_init_pos_noise_z_scale,
        #     ]
        # ):
        #     pos[:, i] += torch.randn_like(pos[:, i]) * axis_noise_scale
        # # print("POS AFTER", pos)

        object_default_state[:, 0:3] = object_default_state[:, 0:3] + self.cfg.reset_dof_pos_noise *torch.randn_like(object_default_state[:, 0:3])*0 + self.scene.env_origins[env_ids]
        # object_default_state[:, 0:3] = object_default_state[:, 0:3]

        # randomize rotation around X and Y
        axis_noise_scale = [self.cfg.object_init_quat_noise_x, self.cfg.object_init_quat_noise_y, self.cfg.object_init_quat_noise_z]
        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        object_default_state[:, 3:7] = randomize_rotation(rot_noise[:, 0]*self.cfg.object_init_quat_noise_x, rot_noise[:, 1]*self.cfg.object_init_quat_noise_y, self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        object_default_state[:, 7:] = torch.zeros_like(self.bottle.data.default_root_state[env_ids, 7:])
        self.bottle.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.bottle.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

        self.last_actions[env_ids,:] = 0.0
        self.last_cube_dof_pos[env_ids] = 0.0
        self.pid_integral[env_ids,:] = 0.0

        self.sum_force[env_ids] = 0.0
        self.slip_sum[env_ids] = 0.0

        #grid = torch.arange(10, 131, 20, device=self.device, dtype=torch.float)

        self.desired_force_norm = 10 + 130 * torch.rand(
           (self.num_envs,), dtype=torch.float, device=self.device
           ) 

        # cap_limits = torch.tensor([3.0, 30.0]) 
        # temp_camp = (self.env_cap_dynamics_norm - cap_limits[0])/(cap_limits[1] - cap_limits[0])

        # force_limits =  torch.tensor([10.0, 130.0]) 
        # self.desired_force_norm =  force_limits[0] + temp_camp*(force_limits[1]-force_limits[0])
        


        self.pre_finger_force_ff = torch.zeros((self.num_envs,3)).to(self.device)
        self.pre_finger_force_mf = torch.zeros((self.num_envs,3)).to(self.device)
        self.pre_finger_force_rf = torch.zeros((self.num_envs,3)).to(self.device)
        self.pre_finger_force_lf = torch.zeros((self.num_envs,3)).to(self.device)
        self.pre_finger_force_th = torch.zeros((self.num_envs,3)).to(self.device)

        self.pre_cap_force = torch.zeros((self.num_envs,3)).to(self.device)

        self.force_progress_buf[env_ids] = -1000

        # # Reset torque setup
        # (
        #     torque_lower,
        #     torque_upper,
        # ) = self.randomizer.get_object_dof_friction_scaling_setup()
        # torque_lower, torque_upper = (
        #     self.brake_torque * torque_lower,
        #     self.brake_torque * torque_upper,
        # )
        # self.object_brake_torque[env_ids] = (
        #     torch.rand(len(env_ids)).to(self.device) * (torque_upper - torque_lower)
        #     + torque_lower
        # )


        #self.left_control_work = torch.zeros_like(self.left_control_work)
        self.right_control_work = torch.zeros_like(self.right_control_work)
        self.control_work = torch.zeros_like(self.control_work)

        self.reward_function.reset(env_ids)

        # self.successes[env_ids] = 0
        self._compute_intermediate_values()


    def _pre_physics_step(self, actions: torch.Tensor):
        """Pre-process actions before applying to simulation."""
        self.actions = actions.clone().clamp(-2.0, 2.0)

        # print("actions", self.actions)
        assert torch.isnan(self.actions).int().sum() == 0, "nan detected"

        self.action_moving_average = 0.75
        #smooth our action.
        self.actions = self.actions * self.action_moving_average + self.last_actions * (
            1.0 - self.action_moving_average
        )

        self.cur_targets[:, self.actuated_dof_indices] = (
           self.cur_targets[:, self.actuated_dof_indices] +  self.cfg.controller_action_scale *self.actions
        )

        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )


        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices].clone()

        self.last_actions = self.actions

        self.last_bottle_dof_pos = self.bottle.data.joint_pos[:, self.bottle_joint_id].clone()

        self.pre_tip_nonthumb = self.right_nonthumb_tips_pos.clone()
        self.pre_tip_thumb = self.right_thumb_tips_pos.clone()

        # #  Unknown usage
        # if self.force_scale > 0.0:
        #     self.rb_forces *= torch.pow(
        #         self.force_decay, self.cfg.sim.dt / self.force_decay_interval
        #     )

        #     obj_mass = to_torch(
        #         [
        #             self.gym.get_actor_rigid_body_properties(
        #                 env, self.gym.find_actor_handle(env, "cube")
        #             )[0].mass
        #             for env in self.envs
        #         ],
        #         device=self.device,
        #     )

        #     prob = self.force_prob
        #     force_indices_candidate = (
        #         torch.less(torch.rand(self.num_envs, device=self.device), prob)
        #     ).nonzero()
        #     last_force_progress = self.force_progress_buf[force_indices_candidate]
        #     current_progress =self.episode_length_buf[force_indices_candidate]

        #     valid_indices = torch.where(
        #         current_progress
        #         > last_force_progress
        #         + torch.randint(
        #             20, 50, (len(force_indices_candidate),), device=self.device
        #         ).unsqueeze(-1)
        #     )[0]


        #     force_indices = force_indices_candidate[valid_indices]

        #     self.force_progress_buf[force_indices] =self.episode_length_buf[force_indices]

        #     step =self.episode_length_buf[force_indices]  # [N, 1]
        #     horizon_decay = torch.pow(self.force_horizon_decay, step)

        #     for i, axis_scale in enumerate(
        #         [self.force_scale_x, self.force_scale_y, self.force_scale_z]
        #     ):
        #         self.rb_forces[force_indices, self.bottle_base_handle, i] = (
        #             torch.randn(
        #                 self.rb_forces[force_indices, self.bottle_base_handle, i].shape,
        #                 device=self.device,
        #             )
        #             * horizon_decay
        #             * obj_mass[force_indices]
        #             * axis_scale
        #             * self.force_scale
        #         )
        #     self.gym.apply_rigid_body_force_tensors(
        #         self.sim,
        #         gymtorch.unwrap_tensor(self.rb_forces),
        #         None,
        #         gymapi.ENV_SPACE,
        #     )

    def _apply_action(self) -> None:
        """Apply PD control to reach target positions."""
        # Get current state
        dof_pos = self.hand.data.joint_pos
        dof_vel = self.hand.data.joint_vel
        cube_quat = self.bottle.data.body_quat_w

        # Compute position error
        error = self.cur_targets - dof_pos

        # error = self.hand.data.default_joint_pos - dof_pos


        # print('Debug********* cur_targets', self.cur_targets)
        # print('Debug********* error', error)

        # self.hand.set_joint_position_target(
        #     self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        # )
        
        # PID control with integral term
        self.pid_integral += error * self.cfg.sim.dt
        self.pid_integral = torch.clamp(self.pid_integral, -10.0, 10.0)
        hand_torques = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        # Compute control torques
        hand_torques = (
            self.p_gain * error
            - self.d_gain * dof_vel
            + self.i_gain * self.pid_integral
        )
        
        # Clip torques to safe limits
        hand_torques = torch.clamp(hand_torques, -6.0, 6.0)
        # print('Debug********* hand_torques', hand_torques[:, self.actuated_dof_indices])
        # Apply torques to actuated joints
        self.hand.set_joint_effort_target(
            hand_torques[:, self.actuated_dof_indices],
            joint_ids=self.actuated_dof_indices
        )
        
        # Track control work for reward computation
        dof_vel_finite_diff = dof_vel
        all_work = hand_torques * dof_vel_finite_diff

        self.right_control_work += all_work[:, self.actuated_dof_indices].abs().sum(-1) * self.cfg.sim.dt
        self.control_work = self.right_control_work.clone()
        
        # Store for next iteration
        self.prev_dof_pos = dof_pos.clone()
        self.prev_cube_quat = cube_quat.clone()


    def slip_data_collector(self):

        self.data_list = []
        self.label_list= []
        

        return self.data_all, self.label_all


    def smooth_clamp(self, x, min_val=0.0, max_val=17.0):
        scale = max_val - min_val
        #print(torch.tanh(x))
        return (( torch.tanh(x)) * scale)
    
    def sensor2matrix_tensor(self, raw_sensor_data):

        temp_values = torch.zeros((self.num_envs,4,5))

        for i in range(self.num_envs):
            temp_values[i,0,0] = -1
            temp_values[i,0,1] = raw_sensor_data[i,4]
            temp_values[i,0,2] = -1
            temp_values[i,0,3] = raw_sensor_data[i,7]
            temp_values[i,0,4] = -1

            temp_values[i,1,0] = raw_sensor_data[i,9]
            temp_values[i,1,1] = raw_sensor_data[i,3]
            temp_values[i,1,2] = raw_sensor_data[i,1]
            temp_values[i,1,3] = raw_sensor_data[i,6]
            temp_values[i,1,4] = raw_sensor_data[i,11]

            temp_values[i,2,0] = raw_sensor_data[i,8]
            temp_values[i,2,1] = raw_sensor_data[i,2]
            temp_values[i,2,2] = raw_sensor_data[i,0]
            temp_values[i,2,3] = raw_sensor_data[i,5]
            temp_values[i,2,4] = raw_sensor_data[i,10]

            temp_values[i,3,0] = raw_sensor_data[i,16]
            temp_values[i,3,1] = raw_sensor_data[i,13]
            temp_values[i,3,2] = raw_sensor_data[i,12]
            temp_values[i,3,3] = raw_sensor_data[i,14]
            temp_values[i,3,4] = raw_sensor_data[i,15]

        return temp_values

    def sensor2matrix_np(self, raw_sensor_data):

        temp_values = np.zeros((4,5))
        temp_values[0,0] = -1
        temp_values[0,1] = raw_sensor_data[13]
        temp_values[0,2] = -1
        temp_values[0,3] = raw_sensor_data[11]
        temp_values[0,4] = -1

        temp_values[1,0] = raw_sensor_data[4]
        temp_values[1,1] = raw_sensor_data[9]
        temp_values[1,2] = raw_sensor_data[12]
        temp_values[1,3] = raw_sensor_data[5]
        temp_values[1,4] = raw_sensor_data[6]

        temp_values[2,0] = raw_sensor_data[0]
        temp_values[2,1] = raw_sensor_data[1]
        temp_values[2,2] = raw_sensor_data[16]
        temp_values[2,3] = raw_sensor_data[7]
        temp_values[2,4] = raw_sensor_data[15]

        temp_values[3,0] = raw_sensor_data[3]
        temp_values[3,1] = raw_sensor_data[2]
        temp_values[3,2] = raw_sensor_data[8]
        temp_values[3,3] = raw_sensor_data[14]
        temp_values[3,4] = raw_sensor_data[10]

        return temp_values
    
        # Additive Noise
    def set_episode_additive_noise_scale(self, scale, name):
        setattr(self, f"{name}_episode_additive_noise_scale", scale)

    def get_episode_additive_noise_scale(self, name):
        return getattr(self, f"{name}_episode_additive_noise_scale")

    def set_episode_additive_noise(self, x, name):
        setattr(self, f"episode_{name}_additive_noise", x)
        if name not in self.episode_additive_noise_names:
            self.episode_additive_noise_names.append(name)

    def get_episode_additive_noise(self, name):
        try:
            return getattr(self, f"episode_{name}_additive_noise")
        except Exception:
            return None
        
            # Affine Noise
    def set_episode_affine_noise_scale(self, scale, name):
        setattr(self, f"{name}_episode_affine_noise_scale", scale)

    def get_episode_affine_noise_scale(self, name):
        return getattr(self, f"{name}_episode_affine_noise_scale")
    
    def get_episode_affine_noise(self, name):
        try:
            return getattr(self, f"episode_{name}_affine_noise")
        except Exception:
            return None
        
    def set_episode_affine_noise(self, x, name):
        setattr(self, f"episode_{name}_affine_noise", x)
        if name not in self.episode_affine_noise_names:
            self.episode_affine_noise_names.append(name)

    def _get_white_noise(self, x, scale):
        return torch.randn_like(x) * scale
    


    def randomize_with_noise(
        self,
        x,
        perstep_noise_scale,
        episode_additive_noise_scale=None,
        episode_affine_noise_scale=None,
        apply_episode_additive_noise=False,
        apply_episode_affine_noise=False,
        apply_latency=False,
        name="",
        return_raw=False,
    ):
        # Initialize affine noise and additive noise, if necessary.
        if apply_episode_additive_noise and name != "":
            if self.get_episode_additive_noise(name) is None:
                if episode_additive_noise_scale is None:
                    episode_additive_noise_scale = (
                        self.get_episode_additive_noise_scale(name)
                    )
                episode_additive_noise = self._get_white_noise(
                    x, episode_additive_noise_scale
                )
                self.set_episode_additive_noise(episode_additive_noise, name)

        if apply_episode_affine_noise and name != "":
            if self.get_episode_affine_noise(name) is None:
                if episode_affine_noise_scale is None:
                    episode_affine_noise_scale = self.get_episode_affine_noise_scale(
                        name
                    )
                episode_affine_noise = self._get_white_noise(
                    x, episode_affine_noise_scale
                )
                self.set_episode_affine_noise(episode_affine_noise, name)

        if apply_episode_affine_noise:
            affine_coeff = 1.0 + self.get_episode_affine_noise(name)
        else:
            affine_coeff = 1.0

        # Get episode noise
        if apply_episode_additive_noise:
            episode_additive_noise = self.get_episode_additive_noise(name)
        else:
            episode_additive_noise = 0.0

        # print("Affine", affine_coeff)
        # print("Additive", episode_additive_noise)

        # Get perstep noise
        perstep_noise = self._get_white_noise(x, perstep_noise_scale)

        # Create randomization
        randomized_x = affine_coeff * x + episode_additive_noise + perstep_noise

        if return_raw:
            return {
                "affine": affine_coeff,
                "additive": episode_additive_noise,
                "perstep": perstep_noise,
            }, randomized_x

        else:
            return randomized_x

    
    def friction_coeffs(self, start: float, end: float, num_envs: int, *,
                        inclusive: bool = True,
                        dtype=np.float32) -> np.ndarray:
        """
        Return an array of friction coefficients evenly spaced between `start` and `end`.
        
        Args:
            start (float): Starting friction coefficient.
            end (float): Ending friction coefficient.
            num_envs (int): Number of environments (number of samples).
            inclusive (bool): If True, include the end value. If False, exclude it.
            dtype: Data type of the resulting array. Default is np.float32.
        
        Returns:
            np.ndarray: Array of shape (num_envs,) containing friction coefficients.
        """
        # Validate the number of environments
        if num_envs <= 0:
            raise ValueError("num_envs must be >= 1")
        # Generate evenly spaced values using np.linspace
        arr = np.linspace(start, end, num=num_envs, endpoint=inclusive, dtype=dtype)
        return arr
    
    def friction_coeffs_array(self,
                    start: Union[float, Sequence[float]],
                    end: Optional[float] = None,
                    num_envs: int = 1, *,
                    inclusive: bool = True,
                    dtype=np.float32) -> np.ndarray:
        """
        Return an array of friction coefficients.

        두 가지 모드:
        1) start가 시퀀스(리스트/배열)면: 그 값들을 균등 반복해 길이 num_envs로 맞춰서 반환
        - 예: values=[3,6,9,12], num_envs=8 -> [3,3,6,6,9,9,12,12]
        - num_envs가 값 개수로 나누어떨어지지 않으면, 앞에서부터 1개씩 더 배분
            (예: num_envs=10 -> [3,3,3,6,6,6,9,9,12,12])

        2) start가 float이고 end도 주어지면: np.linspace(start, end, num_envs, endpoint=inclusive)

        Args:
            start: float 또는 시퀀스(예: [3,6,9,12])
            end: float (linspace 모드에서만 사용)
            num_envs: 결과 배열 길이 (>=1)
            inclusive: linspace 모드에서 끝값 포함 여부
            dtype: 반환 배열 dtype

        Returns:
            np.ndarray: shape (num_envs,)
        """
        # Validate
        if num_envs <= 0:
            raise ValueError("num_envs must be >= 1")

        # 시퀀스 입력 모드
        if isinstance(start, (list, tuple, np.ndarray)) and end is None:
            values = np.asarray(start, dtype=dtype)
            if values.size == 0:
                raise ValueError("values must contain at least one element")
            k = values.size
            base = num_envs // k
            rem = num_envs % k
            # 앞에서부터 rem개에 +1씩 더 배분
            counts = np.array([base + (i < rem) for i in range(k)], dtype=int)
            arr = np.repeat(values, counts).astype(dtype, copy=False)
            # 방어적으로 길이 보정(이론상 필요 없지만 혹시 모를 경우)
            if arr.size != num_envs:
                arr = arr[:num_envs] if arr.size > num_envs else np.pad(arr, (0, num_envs - arr.size), mode='edge')
            return arr

        # linspace 모드
        if not isinstance(start, (list, tuple, np.ndarray)) and end is not None:
            return np.linspace(float(start), float(end),
                            num=num_envs, endpoint=inclusive, dtype=dtype)

        # 잘못된 사용
        raise ValueError(
            "Provide either: (values sequence, end=None) OR (start float, end float)."
        )

    def make_force(self, progress_buf):
        hz = 60
        desired_min = 0.0
        desired_max = 80.0

        # 구간 step 수
        t1 = 1 * hz      # 1초
        t3 = 3 * hz      # 3초

        cycle_steps = 12 * hz
        # cycle 반복
        phase = progress_buf % cycle_steps

        force = torch.zeros_like(phase, dtype=torch.float32)

        # 0~1s: 0 유지
        mask = phase < t1
        force[mask] = desired_min

        # 1~4s: 0→80 증가 (linear)
        mask = (phase >= t1) & (phase < t1 + t3)
        force[mask] = (phase[mask] - t1) / t3 * (desired_max - desired_min)

        # 4~5s: 80 유지
        mask = (phase >= t1 + t3) & (phase < t1 + t3 + t1)
        force[mask] = desired_max

        # 5~8s: 80→0 감소 (linear)
        mask = (phase >= t1 + t3 + t1) & (phase < t1 + t3 + t1 + t3)
        force[mask] = desired_max - (phase[mask] - (t1 + t3 + t1)) / t3 * (desired_max - desired_min)

        # 8~9s: 0 유지
        mask = (phase >= t1 + t3 + t1 + t3) & (phase < t1 + t3 + t1 + t3 + t1)
        force[mask] = desired_min

        # 9~12s: 0→80 증가 (linear)
        mask = (phase >= t1 + t3 + t1 + t3 + t1)
        force[mask] = (phase[mask] - (t1 + t3 + t1 + t3 + t1)) / t3 * (desired_max - desired_min)

        return force
#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )

@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat(
        [
            vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
            torch.cos(angle[idx, :] / 2.0),
        ],
        dim=-1,
    )

    # Reshape and return output
    quat = quat.reshape(
        list(input_shape)
        + [
            4,
        ]
    )
    return quat
