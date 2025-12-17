# Copyright (c) 2025, Shadow Robot Company.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG
from isaaclab.actuators import ImplicitActuatorCfg

import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class EventCfg:
    """Configuration for randomization events."""

    # # -- robot enable if use internal ctr
    # robot_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="reset",
    #     min_step_count_between_reset=720,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "static_friction_range": (0.7, 1.3),
    #         "dynamic_friction_range": (1.0, 1.0),
    #         "restitution_range": (1.0, 1.0),
    #         "num_buckets": 250,
    #     },
    # )
    # robot_joint_stiffness_and_damping = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     min_step_count_between_reset=720,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "stiffness_distribution_params": (0.75, 1.5),
    #         "damping_distribution_params": (0.3, 3.0),
    #         "operation": "scale",
    #         "distribution": "log_uniform",
    #     },
    # )
    # robot_joint_pos_limits = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     min_step_count_between_reset=720,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "lower_limit_distribution_params": (0.00, 0.01),
    #         "upper_limit_distribution_params": (0.00, 0.01),
    #         "operation": "add",
    #         "distribution": "gaussian",
    #     },
    # )
    # robot_tendon_properties = EventTerm(
    #     func=mdp.randomize_fixed_tendon_parameters,
    #     min_step_count_between_reset=720,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", fixed_tendon_names=".*"),
    #         "stiffness_distribution_params": (0.75, 1.5),
    #         "damping_distribution_params": (0.3, 3.0),
    #         "operation": "scale",
    #         "distribution": "log_uniform",
    #     },
    # )

    # -- Object (Bottle) Physics Material Randomization
    bottle_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("bottle"),
            "static_friction_range": (0.9, 1.1),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )


    # -- Object Mass Randomization
    bottle_mass_randomization = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("bottle"),
            "mass_distribution_params": (0.9, 1.1),  # from mass_randomization_lower/upper
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # # -- Object Scale Randomization
    # bottle_scale_randomization = EventTerm(
    #     func=mdp.randomize_rigid_body_scale,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("bottle"),
    #         "scale_range": (0.9, 1.5),  # from scale_randomization_lower/upper
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )

    # -- Gravity Randomization (if not disabled)
    # Commented out since disable_gravity: True in original config
    # reset_gravity = EventTerm(
    #     func=mdp.randomize_physics_scene_gravity,
    #     mode="interval",
    #     is_global_time=True,
    #     interval_range_s=(10.0, 10.0),
    #     params={
    #         "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
    #         "operation": "add",
    #         "distribution": "gaussian",
    #     },
    # )


@configclass
class ShadowHandBottleEnvCfg(DirectRLEnvCfg):
    """Configuration for Shadow Hand Bottle Manipulation task."""

    # ================== Environment Settings ==================
    decimation = 2  # controlFrequencyInv
    # episodeLength = 1000
    episode_length_s = 16.6666  # 1000 timesteps
    num_envs = 4096
    env_spacing = 0.5

    # Action and observation spaces
    action_space = 22  # All Shadow Hand DOFs
    observation_space = 138  # Will be computed dynamically based on method
    state_space = 124 if True else 0  # computeState = True, fullState = False
    asymmetric_obs = True  # computeState = True

    # Observation type and settings
    obs_type = "full"
    method = "joint_model"  # Options: joint_model, slip_model, slip_force_model, slip_force_nn_model, tactile_model
    n_stack_frame = 2

    # Clipping
    clip_observations = 10.0
    clip_actions = 1.0

    # ================== Simulation Settings ==================
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,  # 0.0166
        render_interval=decimation,
        # disable_contact_processing=False,
        physics_material=RigidBodyMaterialCfg(
            static_friction=5.0,
            dynamic_friction=5.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            solver_type=1,  # TGS solver
            max_position_iteration_count=8,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            enable_stabilization=True,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**21,
            gpu_found_lost_pairs_capacity=2**21,
            gpu_found_lost_aggregate_pairs_capacity=2**24,
            gpu_total_aggregate_pairs_capacity=2**21,
            gpu_collision_stack_size=2**26,
            gpu_heap_capacity=2**26,
            gpu_temp_buffer_capacity=2**25,
            gpu_max_num_partitions=8,
        ),
    )

    # ================== Scene Configuration ==================
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        replicate_physics=True,
        # clone_in_fabric=True,
    )
    # ================== Robot Configuration ==================
        # robot
    robot_cfg: ArticulationCfg = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.05, -0.045, 1.35),
            rot=(0.0, 1.0, 1.0, 0.0),
            joint_pos={
            "robot0_FFJ3": 0.0,
            "robot0_FFJ2": 0.8,
            "robot0_FFJ1": 0.4,
            "robot0_FFJ0": 0.0,
            "robot0_MFJ3": 0.0,
            "robot0_MFJ2": 0.8,
            "robot0_MFJ1": 0.4,
            "robot0_MFJ0": 0.0,
            "robot0_RFJ3": 0.0,
            "robot0_RFJ2": 0.8,
            "robot0_RFJ1": 0.4,
            "robot0_RFJ0": 0.0,
            "robot0_LFJ4": 0.0,
            "robot0_LFJ3": 0.0,
            "robot0_LFJ2": 0.8,
            "robot0_LFJ1": 0.4,
            "robot0_LFJ0": 0.0,
            "robot0_THJ4": -0.2,
            "robot0_THJ3": 0.8,
            "robot0_THJ2": 0.0,
            "robot0_THJ1": 0.5,
            "robot0_THJ0": -0.1,
        },
        ),
        actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["robot0_WR.*", "robot0_(FF|MF|RF|LF|TH)J(3|2|1)", "robot0_(LF|TH)J4", "robot0_THJ0"],
            effort_limit_sim={
                "robot0_WRJ1": 0.5,
                "robot0_WRJ0": 0.5,
                "robot0_(FF|MF|RF|LF)J1": 0.5,
                "robot0_FFJ(3|2)": 0.5,
                "robot0_MFJ(3|2)": 0.5,
                "robot0_RFJ(3|2)": 0.5,
                "robot0_LFJ(4|3|2)": 0.5,
                "robot0_THJ4": 0.5,
                "robot0_THJ3": 0.5,
                "robot0_THJ(2|1)": 0.5,
                "robot0_THJ0": 0.5,
            },
            stiffness={
                "robot0_WRJ.*": 5.0,
                "robot0_(FF|MF|RF|LF|TH)J(3|2|1)": 0.1,
                "robot0_(LF|TH)J4": 0.1,
                "robot0_THJ0": 0.1,
            },
            damping={
                "robot0_WRJ.*": 0.01,
                "robot0_(FF|MF|RF|LF|TH)J(3|2|1)": 0.01,
                "robot0_(LF|TH)J4": 0.01,
                "robot0_THJ0": 0.01,
            },
            friction=0.01,
            armature=0.001
        ),
        },
    )

    # Actuated joints (all 22 DOFs)
    actuated_joint_names =  [
            "robot0_FFJ3",
            "robot0_FFJ2",
            "robot0_FFJ1",
            "robot0_FFJ0",
            "robot0_MFJ3",
            "robot0_MFJ2",
            "robot0_MFJ1",
            "robot0_MFJ0",
            "robot0_RFJ3",
            "robot0_RFJ2",
            "robot0_RFJ1",
            "robot0_RFJ0",
            "robot0_LFJ4",
            "robot0_LFJ3",
            "robot0_LFJ2",
            "robot0_LFJ1",
            "robot0_LFJ0",
            "robot0_THJ4",
            "robot0_THJ3",
            "robot0_THJ2",
            "robot0_THJ1",
            "robot0_THJ0",
        ]
    
    

    # [ 'robot0_WRJ1', 'robot0_WRJ0',

    # Fingertip bodies for contact sensing
    fingertip_body_names = [
        "robot0_ffdistal",
        "robot0_mfdistal",
        "robot0_rfdistal",
        "robot0_lfdistal",
        "robot0_thdistal",
    ]
    
    # # Tactile sensor bodies (17 per finger)
    # tactile_sensor_body_names = {
    #     "ff": [f"rh_ff_taxel_{i}" for i in range(17)],
    #     "mf": [f"rh_mf_taxel_{i}" for i in range(17)],
    #     "rf": [f"rh_rf_taxel_{i}" for i in range(17)],
    #     "lf": [f"rh_lf_taxel_{i}" for i in range(17)],
    #     "th": [f"rh_th_taxel_{i}" for i in range(17)],
    # }
    
    # # # ================== Bottle Configuration ==================
    # bottle_cfg = ArticulationCfg(
    # prim_path="/World/envs/env_.*/bottle",
    # spawn=sim_utils.UsdFileCfg(
    #     usd_path="/home/cshen/drlr/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand_open_bottle/assets/single_bottle/sr_model/sr_model.usd",
    #     activate_contact_sensors=False,
    # ),
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         pos=(0.68, -0.052, 1.265), #1.05, -0.045, 1.35
    #         rot=(0, 0, 0, 1),
    #         joint_pos={ "b_joint": 0.0,
    #                     "brake_joint": 0.00},
    #         joint_vel={".*": 0.0},
    # ),
    #     actuators={
    #         "bottle_joints": ImplicitActuatorCfg(
    #             joint_names_expr=['b_joint', 'brake_joint'],
    #             effort_limit=1.0,
    #             velocity_limit=3.0,
    #             stiffness=0.0,
    #             damping=3.0,
    #             friction=0.1,
    #             armature=0.0001
    #         ),
    # },
    # )

    bottle_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Bottle",
        spawn=sim_utils.UrdfFileCfg(
            asset_path="/home/cshen/drlr/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand_open_bottle/assets/single_bottle/sr_model.urdf",
            usd_dir= "/home/cshen/drlr/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand_open_bottle/assets/single_bottle/usd",
            usd_file_name="sr_model.usd",
            fix_base=True,  # 
            merge_fixed_joints=False,
            force_usd_conversion=True,
            activate_contact_sensors=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,  # ✓ Fixed
                    damping=3.0,
                ),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1000.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.68, -0.052, 1.265), #1.05, -0.045, 1.35
            rot=(0, 0, 0, 1),
            joint_pos={ "b_joint": 0.0,
                        "brake_joint": 0.00},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "bottle_joints": ImplicitActuatorCfg(
                joint_names_expr=['b_joint', 'brake_joint'],
                effort_limit=1.0,
                velocity_limit=3.0,
                stiffness=0.0,
                damping=3.0,
                friction=0.1,
                armature=0.0001
            ),
        },
    )

    cap_marker_handle_names =  [
            "l0",
            "l1",
            "l2",
            "l3",
            "l4",
            "l5",
            "l6",
            "l7",
            "l8",
            "l9",
            "l10",
            "l11",
            "l12",
            "l13",
            "l14",
            "l15",
        ]

    base_marker_handle_names = [
        "l20",
        "l21",
        "l22",
        "l23",
        "l24",
        "l25",
        "l26",
        "l27",
    ]

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/(robot0_ffdistal|robot0_mfdistal|robot0_rfdistal|robot0_lfdistal|robot0_thdistal)", history_length=3, update_period=0.005, track_air_time=True
    )

    contact_sensor2 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Bottle/link1", update_period=0.005, history_length=3, debug_vis=True
        )
    

    # ================== Controller Settings ==================
    action_moving_average = 0.75  # actionEMA
    controller_action_scale = 0.1  # controllerActionScale
    p_gain = 10.0  # kp 3
    d_gain = 0.1  # kd
    i_gain = 0.0  # Not in original config

    # ================== Reset/Initialization Settings ==================
    reset_position_noise = 0.25  # startPositionNoise
    reset_rotation_noise = 0.785  # startRotationNoise (in radians)
    reset_dof_pos_noise = 0.005  # From randomization_setup: hand_init_qpos_noise_scale
    reset_dof_vel_noise = 0.0

    # Object initialization noise
    object_init_pos_noise_x = 0.0  # object_init_pos_noise_x_scale
    object_init_pos_noise_y = 0.0  # object_init_pos_noise_y_scale
    object_init_pos_noise_z = 0.02  # object_init_pos_noise_z_scale
    object_init_quat_noise_x = 0.001  # object_init_quat_noise_x_scale
    object_init_quat_noise_y = 0.0  # object_init_quat_noise_y_scale
    object_init_quat_noise_z = 0.001  # object_init_quat_noise_z_scale

  # Randomization Setup
    randomizer = "sr_twist_v1"
    observation_noise_scale= 0.005
    hand_init_qpos_noise_scale= 0.05

    # Initialization noises
    object_init_pos_noise_x_scale= 0.0 #0.02
    object_init_pos_noise_y_scale= 0.0 #0.02
    object_init_pos_noise_z_scale= 0.00 #0.005

    object_init_quat_noise_x_scale= 0.0  # 0.0 # This is the rotation angle scale around x axis.
    object_init_quat_noise_y_scale= 0.0 # 0.04 # This is the rotation angle scale around x axis.
    object_init_quat_noise_z_scale= 0.0 # 0.75 # This is the rotation angle scale around z axis.

    cube_pos_noise_scale= 0.02 # TODO()= Legacy?
    cap_pos_noise_scale= 0.02 # TODO()= Legacy?

    bottle_obs_shift_scale= 0.01
    bottle_obs_shift_reset_prob= 0.05


    scale_randomization_lower= 0.98
    scale_randomization_upper= 1.02

    mass_randomization_lower= 0.8
    mass_randomization_upper= 1.2

    mass_value_lower= 0.030
    mass_value_upper= 0.150

    randomize_mass_by_value= False

    friction_randomization_lower= 0.75 # 0.75
    friction_randomization_upper= 1.25 # 1.25

    cab_friction_lower= 1.0 # 0.9
    cab_friction_upper= 5.0 # 1.5

    # virtual_torque_rate= 0.30 # 0.30
    # virtual_torque_lower= 0.5
    # virtual_torque_upper= 1.5

    virtual_torque_rate= 0.1 # 0.3
    virtual_torque_lower= 1.0
    virtual_torque_upper= 1.0 


    p_gain_lower= 0.8
    p_gain_upper= 1.1
    d_gain_lower= 0.7
    d_gain_upper= 1.2

    # p_gain_lower= 1.0
    # p_gain_upper= 1.0
    # d_gain_lower= 1.0
    # d_gain_upper= 1.0


    object_dof_friction_lower= 0.8 # 0.8 
    object_dof_friction_upper= 1.2 # 1.2

    # latency modeling
    frame_latency_prob= 0.1
    action_drop_prob= 0.0
    action_latency_prob= 0.1

    # perstep noises
    prev_target_noise_scale= 0.0
    bottle_obs_noise_scale= 0.02
    dofpos_noise_scale= 0.4
    action_noise_scale= 0.2
    separate_bottle_perstep_noise= False

    # episode noises (active since jan19)
    prev_target_episode_additive_noise_scale= 0.0
    prev_target_episode_affine_noise_scale= 0.0

    action_episode_additive_noise_scale= 0.0
    action_episode_affine_noise_scale= 0.0

    dofpos_episode_additive_noise_scale= 0.0
    dofpos_episode_affine_noise_scale= 0.0

    bottle_episode_additive_noise_scale= 0.0
    bottle_episode_affine_noise_scale= 0.0


    # ================== Reward Settings ==================
    rewarder = "sr_twist_v1"
    # Reward scales
    failure_penalty = -100.0
    rotation_reward_scale = 1000.0
    cube_rotation_reward_scale = 1.0
    action_penalty_scale = 0.0001
    left_action_penalty_scale = 0.0
    right_action_penalty_scale = 1.0
    distance_penalty_scale = 0.0
    force_reward_scale = 10.0
    distance_reward_scale = 0.0
    reorient_reward_scale = 0.0
    work_penalty_scale = 1.0
    left_work_penalty_scale = 3.0
    right_work_penalty_scale = 0.01

    # Multipliers
    thumb_mult = 8.0
    cap_mult = 8.0
    cap_tactile_mult = 2.0
    gaiting_mult = 10.0
    slip_mult = 15.0
    scaling_factor = 15.0
    slip_diff_ratio = 0.01
    release_bonus = 4.0
    grasping_qual_onoff = 1.0

    # Penalties/bonuses
    pose_diff_penalty_scale = 0.0
    drop_threshold = 0.4
    angle_penalty_scale = 20.0
    hand_init_pose_penalty = 0.0
    finger_distance_reward_scale = 2.5
    reach_goal_bonus = 200.0
    rotate_axis = "z"
    reset_by_z_angle = True

    # Curriculum
    use_curriculum = False
    screw_curriculum = [50_000_000, 120_000_000]
    screw_curriculum_reward_scale = [0.0, 500.0]

    # Cap center point
    cap_center_point = [0.70, 0.00]

    # ================== Observation Settings ==================
    dof_vel_scale = 0.1  # dofVelocityScale
    vel_obs_scale = 0.2
    force_torque_obs_scale = 10.0

    # Observation toggles
    no_dof_vel = True
    no_obj_quat = True
    no_obj_id = False
    no_cap_base = False
    no_obj_pos = True

    # ================== Randomization Settings ==================
    # Observation noise
    observation_noise_scale = 0.005
    bottle_obs_noise_scale = 0.02
    dofpos_noise_scale = 0.4
    action_noise_scale = 0.2
    prev_target_noise_scale = 0.0

    # Bottle observation shift
    bottle_obs_shift_scale = 0.01
    bottle_obs_shift_reset_prob = 0.05
    separate_bottle_perstep_noise = False

    # Episode noises
    prev_target_episode_additive_noise_scale = 0.0
    prev_target_episode_affine_noise_scale = 0.0
    action_episode_additive_noise_scale = 0.0
    action_episode_affine_noise_scale = 0.0
    dofpos_episode_additive_noise_scale = 0.0
    dofpos_episode_affine_noise_scale = 0.0
    bottle_episode_additive_noise_scale = 0.0
    bottle_episode_affine_noise_scale = 0.0

    # Controller
    p_gain_lower = 0.8
    p_gain_upper =  1.1
    d_gain_lower =  0.7
    d_gain_upper =  1.2

    p_gain_val = 3
    d_gain_val = 0.1
    actionEMA = 0.75 # 0.75
    controllerActionScale = 0.1

    # p_gain_lower= 1.0
    # p_gain_upper= 1.0
    # d_gain_lower= 1.0
    # d_gain_upper= 1.0


    object_dof_friction_lower= 0.8 # 0.8 
    object_dof_friction_upper= 1.2 # 1.2


    # Latency modeling
    frame_latency_prob = 0.1
    action_drop_prob = 0.0
    action_latency_prob = 0.1

    # Virtual torque
    virtual_torque_rate = 0.1
    virtual_torque_lower = 1.0
    virtual_torque_upper = 1.0

    # Bottle cap dynamics randomization
    cab_friction_lower = 1.0
    cab_friction_upper = 5.0

    # Object DOF friction randomization
    object_dof_friction_lower = 0.8
    object_dof_friction_upper = 1.2

    # ================== Force Randomization ==================
    force_prob = 0.2
    force_decay = 0.98
    force_scale = 0.0  # Set to 0.0 to disable
    force_scale_x = 10.0
    force_scale_y = 5.0
    force_scale_z = 0.5
    force_decay_interval = 0.1
    force_horizon_decay = 0.97

    # ================== ML Model Paths ==================
    slip_model_path = "checkpoints/mlp_1024_512_256_slip.pth"
    tactile_embedder_path = "embedder_v1.pt"

    # ================== Success Criteria ==================
    success_tolerance = 0.1
    max_consecutive_success = 0
    av_factor = 0.1

    # ================== Events ==================
    events: EventCfg = EventCfg()


# ================== Pre-configured Variants ==================

@configclass
class ShadowHandBottleEnvCfg_JointModel(ShadowHandBottleEnvCfg):
    """Configuration for joint model (no tactile/slip)."""
    method = "joint_model"
    observation_space = 69  # 22 + 22 + 3 + 22


@configclass
class ShadowHandBottleEnvCfg_SlipModel(ShadowHandBottleEnvCfg):
    """Configuration with slip detection."""
    method = "slip_model"
    observation_space = 74  # 69 + 5


@configclass
class ShadowHandBottleEnvCfg_SlipForceModel(ShadowHandBottleEnvCfg):
    """Configuration with slip detection and force sensing."""
    method = "slip_force_model"
    observation_space = 79  # 69 + 5 + 5


@configclass
class ShadowHandBottleEnvCfg_SlipForceNNModel(ShadowHandBottleEnvCfg):
    """Configuration with neural network slip prediction and force sensing."""
    method = "slip_force_nn_model"
    observation_space = 79  # 69 + 5 (NN predicted) + 5


@configclass
class ShadowHandBottleEnvCfg_TactileModel(ShadowHandBottleEnvCfg):
    """Configuration with full tactile input (5 fingers × 17 sensors)."""
    method = "tactile_model"
    observation_space = 154  # 69 + 85


@configclass
class ShadowHandBottleEnvCfg_Play(ShadowHandBottleEnvCfg):
    """Configuration for play/evaluation (no randomization)."""
    num_envs = 64
    
    # Disable all randomization
    events: EventCfg = EventCfg()
    events.robot_physics_material = None
    # events.robot_joint_stiffness_and_damping = None
    events.bottle_physics_material = None
    events.bottle_mass_randomization = None
    # events.bottle_scale_randomization = None
    
    # No noise
    observation_noise_scale = 0.0
    bottle_obs_noise_scale = 0.0
    dofpos_noise_scale = 0.0
    action_noise_scale = 0.0