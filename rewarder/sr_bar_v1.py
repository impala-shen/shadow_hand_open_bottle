import torch
import torch.nn
import torch.nn.functional as F
from isaacgymenvs.tasks.rewarder.base import BaseRewardFunction
import cv2
import numpy as np

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def xyzw_to_wxyz(quat):
    new_quat = quat.clone()
    new_quat[:, :1] = quat[:, -1:]
    new_quat[:, 1:] = quat[:, :-1]
    return new_quat


class RewardFunction(BaseRewardFunction):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rot_score = torch.zeros((self.num_envs,)).to(self.device)
        self.translation_score = torch.zeros((self.num_envs,)).to(self.device)
        self.z_score = torch.zeros((self.num_envs,)).to(self.device)
        self.slip_sum = torch.zeros((self.num_envs,)).to(self.device)

        self.pre_normal_mag_ff = torch.zeros((self.num_envs,)).to(self.device)
        self.pre_normal_mag_mf = torch.zeros((self.num_envs,)).to(self.device)
        self.pre_normal_mag_rf = torch.zeros((self.num_envs,)).to(self.device)
        self.pre_normal_mag_lf = torch.zeros((self.num_envs,)).to(self.device)
        self.pre_normal_mag_th = torch.zeros((self.num_envs,)).to(self.device)

        self.pre_diff_ff = torch.zeros((self.num_envs,)).to(self.device)
        self.pre_diff_mf = torch.zeros((self.num_envs,)).to(self.device)
        self.pre_diff_rf = torch.zeros((self.num_envs,)).to(self.device)
        self.pre_diff_lf = torch.zeros((self.num_envs,)).to(self.device)
        self.pre_diff_th = torch.zeros((self.num_envs,)).to(self.device)

        self.pre_vel_nonthumb = torch.zeros((self.num_envs,4)).to(self.device)
        self.pre_vel_thumb = torch.zeros((self.num_envs,1)).to(self.device)

        self.pre_finger_force_ff = torch.zeros((self.num_envs,3)).to(self.device)
        self.pre_finger_force_mf = torch.zeros((self.num_envs,3)).to(self.device)
        self.pre_finger_force_rf = torch.zeros((self.num_envs,3)).to(self.device)
        self.pre_finger_force_lf = torch.zeros((self.num_envs,3)).to(self.device)
        self.pre_finger_force_th = torch.zeros((self.num_envs,3)).to(self.device)

        self.pre_cap_force = torch.zeros((self.num_envs,3)).to(self.device)

        self.force_norm_sum = torch.zeros((self.num_envs,)).to(self.device)

        self.force_norm_sum = torch.zeros((self.num_envs,)).to(self.device)
        self.force_ff_norm_sum = torch.zeros((self.num_envs,)).to(self.device)
        self.force_mf_norm_sum = torch.zeros((self.num_envs,)).to(self.device)
        self.force_rf_norm_sum = torch.zeros((self.num_envs,)).to(self.device)
        self.force_lf_norm_sum = torch.zeros((self.num_envs,)).to(self.device)
        self.force_th_norm_sum = torch.zeros((self.num_envs,)).to(self.device)


        self.sum_force_norm = torch.zeros((self.num_envs,)).to(self.device)
        self.count = 0.0


        return

    def forward(
        self,
        reset_buf,
        progress_buf,
        actions,
        states,
        reward_settings,
        max_episode_length,
    ):
        info_dict = {}

        # information 
        slip_avr = states["slip_avr"]
        #print(desired_force_norm)

        cap_force = states["cap_force"]

        finger_force_ff = states["finger_force_ff"]
        finger_force_mf = states["finger_force_mf"]
        finger_force_rf = states["finger_force_rf"]
        finger_force_lf = states["finger_force_lf"]
        finger_force_th = states["finger_force_th"]

        z_sign_nonthumb = states["z_sign_nonthumb"]
        z_sign_thumb = states["z_sign_thumb"]

        right_thumb_tips_pos = states["right_thumb_tips_pos"]
        right_nonthumb_tips_pos = states["right_nonthumb_tips_pos"]

        right_thumb_tips_vel_unit = states["right_thumb_tips_vel_unit"]
        right_non_thumb_tips_vel_unit = states["right_nonthumb_tips_vel_unit"]

        #print(right_thumb_tips_vel_unit.shape)
        #print(right_non_thumb_tips_vel_unit.shape)

        pre_right_thumb_tips_pos = states["pre_right_thumb_tips_pos"]
        pre_right_nonthumb_tips_pos = states["pre_right_nonthumb_tips_pos"]

        pre_right_thumb_tips_pos = pre_right_thumb_tips_pos.unsqueeze(2)  # [N, F2, 1, 3]
        pre_right_nonthumb_tips_pos = pre_right_nonthumb_tips_pos.unsqueeze(2)  # [N, F2, 1, 3]

        cap_marker_pos = states["cube_cap_marker_pos"]  # [N, K2, 3]

        right_link_ff_tip_sensor_pos =  states["right_link_ff_tip_sensor_pos"]
        right_link_mf_tip_sensor_pos =  states["right_link_mf_tip_sensor_pos"]
        right_link_rf_tip_sensor_pos =  states["right_link_rf_tip_sensor_pos"]
        right_link_lf_tip_sensor_pos =  states["right_link_lf_tip_sensor_pos"]
        right_link_th_tip_sensor_pos =  states["right_link_th_tip_sensor_pos"]

        cap_marker_pos = cap_marker_pos.unsqueeze(1)  # [N, 1, K2, 3]

        right_thumb_tips_pos = right_thumb_tips_pos.unsqueeze(2)  # [N, F2, 1, 3]
        right_nonthumb_tips_pos = right_nonthumb_tips_pos.unsqueeze(2)  # [N, F2, 1, 3]

        right_link_ff_tip_sensor_pos = right_link_ff_tip_sensor_pos.unsqueeze(2)  # [N, F2, 1, 3]
        right_link_mf_tip_sensor_pos = right_link_mf_tip_sensor_pos.unsqueeze(2)  # [N, F2, 1, 3]

        right_link_rf_tip_sensor_pos = right_link_rf_tip_sensor_pos.unsqueeze(2)  # [N, F2, 1, 3]
        right_link_lf_tip_sensor_pos = right_link_lf_tip_sensor_pos.unsqueeze(2)  # [N, F2, 1, 3]
        right_link_th_tip_sensor_pos = right_link_th_tip_sensor_pos.unsqueeze(2)  # [N, F2, 1, 3]

        cube_center_pose = states["cube_pos"]
        cube_center_pose = cube_center_pose.unsqueeze(1).unsqueeze(1)# [N, 1, 1, 3]

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

        ####################################################################################33
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
        idx_center_to_sensor_ff = torch.min(dist_center_to_sensor_ff, -1)[1]
        closest_sensor_pose_ff =  right_link_ff_tip_sensor_pos[torch.arange(right_link_ff_tip_sensor_pos.size(0)), idx_center_to_sensor_ff] 
    
        dist_center_to_sensor_mf = torch.min(dist_center_to_sensor_mf, -1)[0]
        idx_center_to_sensor_mf = torch.min(dist_center_to_sensor_mf, -1)[1]
        closest_sensor_pose_mf =  right_link_mf_tip_sensor_pos[torch.arange(right_link_mf_tip_sensor_pos.size(0)), idx_center_to_sensor_mf] 

        dist_center_to_sensor_rf = torch.min(dist_center_to_sensor_rf, -1)[0]
        idx_center_to_sensor_rf = torch.min(dist_center_to_sensor_rf, -1)[1]
        closest_sensor_pose_rf =  right_link_rf_tip_sensor_pos[torch.arange(right_link_rf_tip_sensor_pos.size(0)), idx_center_to_sensor_rf] 

        dist_center_to_sensor_lf = torch.min(dist_center_to_sensor_lf, -1)[0]
        idx_center_to_sensor_lf = torch.min(dist_center_to_sensor_lf, -1)[1]
        closest_sensor_pose_lf =  right_link_lf_tip_sensor_pos[torch.arange(right_link_lf_tip_sensor_pos.size(0)), idx_center_to_sensor_lf] 

        dist_center_to_sensor_th = torch.min(dist_center_to_sensor_th, -1)[0]
        idx_center_to_sensor_th = torch.min(dist_center_to_sensor_th, -1)[1]
        closest_sensor_pose_th =  right_link_th_tip_sensor_pos[torch.arange(right_link_th_tip_sensor_pos.size(0)), idx_center_to_sensor_th] 

        # print("closest_sensor_pose_ff : ",closest_sensor_pose_ff)
        # print("closest_sensor_pose_mf : ",closest_sensor_pose_mf)
        # print("closest_sensor_pose_rf : ",closest_sensor_pose_rf)
        # print("closest_sensor_pose_lf : ",closest_sensor_pose_lf)
        # print("closest_sensor_pose_th : ",closest_sensor_pose_th)


        # ##################################################3tensor calculation 

        temp_sensor_input_ff = dist_cap_to_sensor_ff
        temp_sensor_input_mf = dist_cap_to_sensor_mf
        temp_sensor_input_rf = dist_cap_to_sensor_rf
        temp_sensor_input_lf = dist_cap_to_sensor_lf
        temp_sensor_input_th = dist_cap_to_sensor_th

        #print("temp_sensor_input_ff :: ", temp_sensor_input_ff)

        ################################################## center to finger
        # temp_sensor_input_ff = dist_center_to_sensor_ff
        # temp_sensor_input_mf = dist_center_to_sensor_mf
        # temp_sensor_input_rf = dist_center_to_sensor_rf
        # temp_sensor_input_lf = dist_center_to_sensor_lf
        # temp_sensor_input_th = dist_center_to_sensor_th

        #contact_threshold = 0.005 or 0.01
        radius = 0.04
        contact_offset = 0.005
        
        contact_threshold = contact_offset
        force_k = 0.001

        finger_force_alpha = 0.4# 0.3

        final_cap_force = finger_force_alpha * cap_force + (1-finger_force_alpha)*self.pre_cap_force
        self.pre_cap_force = final_cap_force
        cap_force_norm = torch.norm(final_cap_force, dim=-1, p=2)

        #print(cap_force_norm.shape)

        final_finger_force_ff = finger_force_alpha * finger_force_ff + (1-finger_force_alpha)*self.pre_finger_force_ff
        final_finger_force_mf = finger_force_alpha * finger_force_mf + (1-finger_force_alpha)*self.pre_finger_force_mf
        final_finger_force_rf = finger_force_alpha * finger_force_rf + (1-finger_force_alpha)*self.pre_finger_force_rf
        final_finger_force_lf = finger_force_alpha * finger_force_lf + (1-finger_force_alpha)*self.pre_finger_force_lf
        final_finger_force_th = finger_force_alpha * finger_force_th + (1-finger_force_alpha)*self.pre_finger_force_th

        self.pre_finger_force_ff = final_finger_force_ff
        self.pre_finger_force_mf = final_finger_force_mf
        self.pre_finger_force_rf = final_finger_force_rf
        self.pre_finger_force_lf = final_finger_force_lf
        self.pre_finger_force_th = final_finger_force_th

        force_ff_norm = torch.norm(final_finger_force_ff, dim=-1, p=2)
        force_mf_norm = torch.norm(final_finger_force_mf, dim=-1, p=2)
        force_rf_norm = torch.norm(final_finger_force_rf, dim=-1, p=2)
        force_lf_norm = torch.norm(final_finger_force_lf, dim=-1, p=2)
        force_th_norm = torch.norm(final_finger_force_th, dim=-1, p=2)

        force_contact_ff = torch.where(force_ff_norm > 0, 1.0, 0.0) #
        force_contact_mf = torch.where(force_mf_norm > 0, 1.0, 0.0) #
        force_contact_rf = torch.where(force_rf_norm > 0, 1.0, 0.0) #
        force_contact_lf = torch.where(force_lf_norm > 0, 1.0, 0.0) #
        force_contact_th = torch.where(force_th_norm > 0, 1.0, 0.0) #

        non_force_contact_penalty = force_contact_ff * force_contact_mf * force_contact_rf * force_contact_lf * force_contact_th

        #print("force_lf_norm",force_lf_norm)


        temp_sensor_input_ff = torch.where(temp_sensor_input_ff > contact_threshold, 0.0, 1/(temp_sensor_input_ff + 0.01)*0.002) #10.0
        temp_sensor_input_mf = torch.where(temp_sensor_input_mf > contact_threshold, 0.0, 1/(temp_sensor_input_mf + 0.01)*0.002)
        temp_sensor_input_rf = torch.where(temp_sensor_input_rf > contact_threshold, 0.0, 1/(temp_sensor_input_rf + 0.01)*0.002)
        temp_sensor_input_lf = torch.where(temp_sensor_input_lf > contact_threshold, 0.0, 1/(temp_sensor_input_lf + 0.01)*0.002)
        temp_sensor_input_th = torch.where(temp_sensor_input_th > contact_threshold, 0.0, 1/(temp_sensor_input_th + 0.01)*0.002)

        #print("temp_sensor_input_ff",temp_sensor_input_ff.shape)

        temp_ff = temp_sensor_input_ff.sum(dim=1)
        temp_mf = temp_sensor_input_mf.sum(dim=1)
        temp_rf = temp_sensor_input_rf.sum(dim=1)
        temp_lf = temp_sensor_input_lf.sum(dim=1)
        temp_th = temp_sensor_input_th.sum(dim=1)

        non_contact_ff = torch.where(temp_ff > 0, 1.0, 0.0) #10.0
        non_contact_mf = torch.where(temp_mf > 0, 1.0, 0.0) #10.0
        non_contact_rf = torch.where(temp_rf > 0, 1.0, 0.0) #10.0
        non_contact_lf = torch.where(temp_lf > 0, 1.0, 0.0) #10.0
        non_contact_th = torch.where(temp_th > 0, 1.0, 0.0) #10.0

        non_force_contact_ff = torch.where(force_ff_norm > 0, 1.0, 0.0) #10.0
        non_force_contact_mf = torch.where(force_mf_norm > 0, 1.0, 0.0) #10.0
        non_force_contact_rf = torch.where(force_rf_norm > 0, 1.0, 0.0) #10.0
        non_force_contact_lf = torch.where(force_lf_norm > 0, 1.0, 0.0) #10.0
        non_force_contact_th = torch.where(force_th_norm > 0, 1.0, 0.0) #10.0

        non_force_contact_penalty = non_force_contact_ff * non_force_contact_mf * non_force_contact_rf * non_force_contact_lf
        
        non_contact_penalty = non_contact_ff * non_contact_mf * non_contact_rf 
      
        temp_sum_dist_cap_to_sensor_ff_padded =  torch.clamp(temp_ff, 0.0, 0.1)
        temp_sum_dist_cap_to_sensor_mf_padded =  torch.clamp(temp_mf, 0.0, 0.1)
        temp_sum_dist_cap_to_sensor_rf_padded =  torch.clamp(temp_rf, 0.0, 0.1)
        temp_sum_dist_cap_to_sensor_lf_padded =  torch.clamp(temp_lf, 0.0, 0.1)
        temp_sum_dist_cap_to_sensor_th_padded =  torch.clamp(temp_th, 0.0, 0.1)
        

        sum_dist_cap_to_sensor_ff_padded = temp_sum_dist_cap_to_sensor_ff_padded
        sum_dist_cap_to_sensor_mf_padded = temp_sum_dist_cap_to_sensor_mf_padded
        sum_dist_cap_to_sensor_rf_padded = temp_sum_dist_cap_to_sensor_rf_padded
        sum_dist_cap_to_sensor_lf_padded = temp_sum_dist_cap_to_sensor_lf_padded
        sum_dist_cap_to_sensor_th_padded = temp_sum_dist_cap_to_sensor_th_padded

        #print("Converted",sum_dist_cap_to_sensor_3_padded)
        reward_sensor_ff_dist_thumb = sum_dist_cap_to_sensor_ff_padded
        reward_sensor_mf_dist_thumb = sum_dist_cap_to_sensor_mf_padded
        reward_sensor_rf_dist_thumb = sum_dist_cap_to_sensor_rf_padded
        reward_sensor_lf_dist_thumb = sum_dist_cap_to_sensor_lf_padded
        reward_sensor_th_dist_thumb = sum_dist_cap_to_sensor_th_padded

        # Early termination.
        reset_buf = torch.where(
            states["cube_pos"][:, 2] < reward_settings["drop_threshold"],
            torch.ones_like(reset_buf),
            reset_buf,
        )
        reset_buf = torch.where(
            progress_buf >= max_episode_length - 1,
            torch.ones_like(reset_buf),
            reset_buf,
        )

        distance_termination = 0.07

        reset_buf = torch.where(
            min_dis_val_ff > distance_termination,
            torch.ones_like(reset_buf),
            reset_buf,
        )

        reset_buf = torch.where(
            min_dis_val_mf > distance_termination,
            torch.ones_like(reset_buf),
            reset_buf,
        )

        reset_buf = torch.where(
            min_dis_val_rf > distance_termination,
            torch.ones_like(reset_buf),
            reset_buf,
        )

        reset_buf = torch.where(
            min_dis_val_lf > distance_termination,
            torch.ones_like(reset_buf),
            reset_buf,
        )

        reset_buf = torch.where(
            min_dis_val_th > distance_termination,
            torch.ones_like(reset_buf),
            reset_buf,
        )

        reward = torch.zeros_like(reset_buf).reshape(-1)
        reward = reward.float()

        assert reward_settings["failure_penalty"] <= 0, "penalty must <= 0"
        reward = torch.where(
            min_dis_val_ff > distance_termination,
            reward + reward_settings["failure_penalty"],
            reward,
        )
        reward = torch.where(
            min_dis_val_mf > distance_termination,
            reward + reward_settings["failure_penalty"],
            reward,
        )
        reward = torch.where(
            min_dis_val_rf > distance_termination,
            reward + reward_settings["failure_penalty"],
            reward,
        )
        reward = torch.where(
            min_dis_val_lf > distance_termination,
            reward + reward_settings["failure_penalty"],
            reward,
        )
        # reward = torch.where(
        #     min_dis_val_th > distance_termination,
        #     reward + reward_settings["failure_penalty"],
        #     reward,
        # )

        # +---------------------------------------------------------------------------------+
        # |      Reward      |
        # +---------------------------------------------------------------------------------+

        # Contact pressure condition
        reward_tactile_1 = (
            reward_sensor_ff_dist_thumb
            + reward_sensor_mf_dist_thumb
            + reward_sensor_rf_dist_thumb
            + reward_sensor_lf_dist_thumb
            + reward_sensor_th_dist_thumb # *8.0
        )

        #print(reward_tactile_1)


        # w/o slip detections
        grasp_quality_ff = sum_dist_cap_to_sensor_ff_padded
        grasp_quality_mf = sum_dist_cap_to_sensor_mf_padded
        grasp_quality_rf = sum_dist_cap_to_sensor_rf_padded
        grasp_quality_lf = sum_dist_cap_to_sensor_lf_padded
        grasp_quality_th = sum_dist_cap_to_sensor_th_padded

        # grasp_quality_ff = torch.where(sum_dist_cap_to_sensor_ff_padded > 0.0, sum_dist_cap_to_sensor_ff_padded, *-300.0)
        # grasp_quality_mf = torch.where(sum_dist_cap_to_sensor_mf_padded > 0.0, sum_dist_cap_to_sensor_mf_padded, *-300.0)
        # grasp_quality_rf = torch.where(sum_dist_cap_to_sensor_rf_padded > 0.0, sum_dist_cap_to_sensor_rf_padded, *-300.0)
        # grasp_quality_lf = torch.where(sum_dist_cap_to_sensor_lf_padded > 0.0, sum_dist_cap_to_sensor_lf_padded, *-300.0)
        # grasp_quality_th = torch.where(sum_dist_cap_to_sensor_th_padded > 0.0, sum_dist_cap_to_sensor_th_padded, *-300.0)

        just_released_ff = (self.pre_normal_mag_ff > 0) & (sum_dist_cap_to_sensor_ff_padded == 0)
        just_released_mf = (self.pre_normal_mag_mf > 0) & (sum_dist_cap_to_sensor_mf_padded == 0)
        just_released_rf = (self.pre_normal_mag_rf > 0) & (sum_dist_cap_to_sensor_rf_padded == 0)
        just_released_lf = (self.pre_normal_mag_lf > 0) & (sum_dist_cap_to_sensor_lf_padded == 0)
        just_released_th = (self.pre_normal_mag_th > 0) & (sum_dist_cap_to_sensor_th_padded == 0)

        #print(just_released_ff)

        # 보너스 리워드 값 설정
        release_bonus_ff = torch.where(
            just_released_ff,
            torch.ones_like(grasp_quality_ff) * reward_settings["release_bonus"],
            torch.zeros_like(grasp_quality_ff)
            )
        release_bonus_mf = torch.where(
            just_released_mf,
            torch.ones_like(grasp_quality_mf) * reward_settings["release_bonus"],
            torch.zeros_like(grasp_quality_mf)
            )
        release_bonus_rf = torch.where(
            just_released_rf,
            torch.ones_like(grasp_quality_rf) * reward_settings["release_bonus"],
            torch.zeros_like(grasp_quality_rf)
            )
        release_bonus_lf = torch.where(
            just_released_lf,
            torch.ones_like(grasp_quality_lf) * reward_settings["release_bonus"],
            torch.zeros_like(grasp_quality_lf)
            )
        
        release_bonus_th = torch.where(
            just_released_th,
            torch.ones_like(grasp_quality_th) * reward_settings["release_bonus"],
            torch.zeros_like(grasp_quality_th)
            )

        release_bonus = release_bonus_ff + release_bonus_mf + release_bonus_rf + release_bonus_lf + release_bonus_th
        

        #print(release_bonus)

        reward = reward +  release_bonus

        #print(release_bonus)

        #print(grasp_quality)
        # s_ff = sum_dist_cap_to_sensor_ff_padded
        # s_mf = sum_dist_cap_to_sensor_mf_padded
        # s_rf = sum_dist_cap_to_sensor_rf_padded
        # s_lf = sum_dist_cap_to_sensor_lf_padded
        # s_th = sum_dist_cap_to_sensor_th_padded
        # stacked_sum_dist_cap = torch.stack([s_ff, s_mf, s_rf, s_lf,s_th], dim=0) 
        # max_sum, _ = torch.max(stacked_sum_dist_cap, dim=0) # tactile sensor 의 최대 값 at 각 환경당 
        # std_sum = torch.std(stacked_sum_dist_cap, dim=0)

        # sig_result = torch.where(max_sum != 0.0, torch.tensor(1.0), torch.tensor(0.0))
        # temp = torch.where(stacked_sum_dist_cap != 0.0, torch.tensor(1.0), torch.tensor(0.0))
        # sig_all_result = temp.sum(dim=0)

        # reward_tactile_2 = (sig_all_result)*(1/ (std_sum + 0.03))*2.0 #하나만 닿아도 이득

        reward_tactile = reward_tactile_1 #for display

        reward = reward + (
                    grasp_quality_ff + grasp_quality_mf + grasp_quality_rf + grasp_quality_lf + grasp_quality_th
                ) * reward_settings["cap_tactile_mult"]


        translation_reward =  (
            states["cube_dof_pos"] - states["last_cube_dof_pos"]
        )

        #print(translation_reward*100.0)

        translation_reward = torch.nan_to_num(
            translation_reward, nan=0.0, posinf=1.0, neginf=-1.0
        )
        translation_reward = torch.clamp(translation_reward, -0.02, 0.01)
        #rotation_reward = torch.clamp(rotation_reward, -0.02, 0.005)

        info_dict["cube_dof_pos_delta"] = translation_reward
        
        self.translation_score += translation_reward #

        #self.translation_score = states["cube_dof_pos"]

        reward = reward + translation_reward * reward_settings["translation_reward_scale"] * non_contact_penalty

        #print("rotation_reward : ",rotation_reward * reward_settings["rotation_reward_scale"] * non_contact_penalty)

        #final_translation_reward = translation_reward * reward_settings["translation_reward_scale"] * non_contact_penalty

        firm_translation_ff = grasp_quality_ff * translation_reward * reward_settings["translation_reward_scale"]* reward_settings["grasping_qual_onoff"]
        firm_translation_mf = grasp_quality_mf * translation_reward * reward_settings["translation_reward_scale"]* reward_settings["grasping_qual_onoff"]
        firm_translation_rf = grasp_quality_rf * translation_reward * reward_settings["translation_reward_scale"]* reward_settings["grasping_qual_onoff"]
        firm_translation_lf = grasp_quality_lf * translation_reward * reward_settings["translation_reward_scale"]* reward_settings["grasping_qual_onoff"]
        firm_translation_th = grasp_quality_th * translation_reward * reward_settings["translation_reward_scale"]* reward_settings["grasping_qual_onoff"]

        firm_translation_ff =  torch.clamp(firm_translation_ff, -2.0, 2.0)
        firm_translation_mf =  torch.clamp(firm_translation_mf, -2.0, 2.0)
        firm_translation_rf =  torch.clamp(firm_translation_rf, -2.0, 2.0)
        firm_translation_lf =  torch.clamp(firm_translation_lf, -2.0, 2.0)
        firm_translation_th =  torch.clamp(firm_translation_th, -2.0, 2.0)

        reward = (reward + (firm_translation_ff + firm_translation_mf + firm_translation_rf + firm_translation_lf + firm_translation_th)*non_contact_penalty)

        # success_termination = 0.18

        # reward = torch.where(
        #     states["cube_dof_pos"] > success_termination,
        #     reward + reward_settings["success_reward"],
        #     reward,
        # )

        # reset_buf = torch.where(
        #     states["cube_dof_pos"] > success_termination,
        #     torch.ones_like(reset_buf),
        #     reset_buf,
        # )



        # +---------------------------------------------------------------------------------+
        # |      penalty      |
        # +---------------------------------------------------------------------------------+

        #print(right_non_thumb_tips_vel_unit.shape)
        #print(z_sign_thumb[:,0,0])

        right_non_thumb_tips_vel_unit = right_non_thumb_tips_vel_unit.squeeze(2)
        # encourage y_slide 
        ff_temp = torch.where(right_non_thumb_tips_vel_unit[:,0,1] > 0, torch.ones_like(right_non_thumb_tips_vel_unit[:,0,1])* 2.0, right_non_thumb_tips_vel_unit[:,0,1]* 2.0)
        mf_temp = torch.where(right_non_thumb_tips_vel_unit[:,1,1] > 0, torch.ones_like(right_non_thumb_tips_vel_unit[:,1,1])* 2.0, right_non_thumb_tips_vel_unit[:,1,1]* 2.0)
        rf_temp = torch.where(right_non_thumb_tips_vel_unit[:,2,1] > 0, torch.ones_like(right_non_thumb_tips_vel_unit[:,2,1])* 2.0, right_non_thumb_tips_vel_unit[:,2,1]* 2.0)
        lf_temp = torch.where(right_non_thumb_tips_vel_unit[:,3,1] > 0, torch.ones_like(right_non_thumb_tips_vel_unit[:,3,1])* 2.0, right_non_thumb_tips_vel_unit[:,3,1]* 2.0)
        #th_temp = torch.where(z_sign_thumb[:,0,0] > 0, torch.ones_like(z_sign_thumb[:,0,0])* 2.0, z_sign_thumb[:,0,0]* 2.0)


        temp_gaiting_penalty_ff = sum_dist_cap_to_sensor_ff_padded * ff_temp * reward_settings["gaiting_mult"]
        temp_gaiting_penalty_mf = sum_dist_cap_to_sensor_mf_padded * mf_temp * reward_settings["gaiting_mult"]
        temp_gaiting_penalty_rf = sum_dist_cap_to_sensor_rf_padded * rf_temp * reward_settings["gaiting_mult"]
        temp_gaiting_penalty_lf = sum_dist_cap_to_sensor_lf_padded * lf_temp * reward_settings["gaiting_mult"]
        #temp_gaiting_penalty_th = sum_dist_cap_to_sensor_th_padded * th_temp * reward_settings["gaiting_mult"]

        #print(temp_gaiting_penalty_ff)

        gaiting_penalty_ff =  torch.clamp(temp_gaiting_penalty_ff, -5.0, 5.0)
        gaiting_penalty_mf =  torch.clamp(temp_gaiting_penalty_mf, -5.0, 5.0)
        gaiting_penalty_rf =  torch.clamp(temp_gaiting_penalty_rf, -5.0, 5.0)
        gaiting_penalty_lf =  torch.clamp(temp_gaiting_penalty_lf, -5.0, 5.0)
        #gaiting_penalty_th =  torch.clamp(temp_gaiting_penalty_th, -5.0, 5.0)


        gaiting_penalty = (gaiting_penalty_ff + gaiting_penalty_mf + gaiting_penalty_rf + gaiting_penalty_lf)

        reward = reward + gaiting_penalty

        # translation penalty 
        # cube_quat = states["cube_quat"]
        # cube_rotation_matrix = quaternion_to_matrix(
        #     xyzw_to_wxyz(cube_quat)
        # )  # [B, 3, 3]
        # z_axis = cube_rotation_matrix[:, :, 2]  # [B, 3]

        # #print("z_axis :: ",z_axis)

        # # Point to the right.
        # target_vector = (
        #     torch.FloatTensor([0.0, 0.0, 1.0])
        #     .to(self.device)
        #     .reshape(-1, 3)
        #     .repeat(z_axis.size(0), 1)
        # )
        # angle_difference = torch.arccos(torch.sum(z_axis * target_vector, dim=-1)) # 똑바로 하기 위한것 
        # angle_penalty = -torch.clamp(angle_difference, 0.0, 1.0)

        # reward = reward + angle_penalty * reward_settings["angle_penalty_scale"]

        # action penalty 
        right_action_penalty = -torch.sum(actions[:, :22] ** 2, dim=-1)

        action_penalty = (
            + right_action_penalty# off
        )
        reward = reward + action_penalty * reward_settings["action_penalty_scale"]

        #print(action_penalty * reward_settings["action_penalty_scale"])

        # work penalty 
        work_penalty = (
            -states["right_work"] * reward_settings["right_work_penalty_scale"]
        )
        reward = reward + work_penalty * reward_settings["work_penalty_scale"]

        # +---------------------------------------------------------------------------------+
        # |      Final_reward      |
        # +---------------------------------------------------------------------------------+
        
        
        reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-1.0)

        #print(reward)

        # performance metrics   
        info_dict["rot_score"] = (
            self.rot_score / progress_buf.float()
        )  # torch.clip(self.progress_buf.float(), min=1, max=200) # + 1e-4)
        info_dict["z_score"] = self.z_score / torch.clip(
            progress_buf.float() - 100, min=1
        )  # + 1e-4)

        #print(info_dict["rot_score"])
        #print("progress_buf ",progress_buf)

        info_dict["reward_tactile"] = reward_tactile

        info_dict["grasp_quality_ff"] = grasp_quality_ff
        info_dict["grasp_quality_mf"] = grasp_quality_mf
        info_dict["grasp_quality_rf"] = grasp_quality_rf
        info_dict["grasp_quality_lf"] = grasp_quality_lf
        info_dict["grasp_quality_th"] = grasp_quality_th

        info_dict["f_error"] = self.z_score
        info_dict["slip_error"] = self.z_score
        info_dict["translation_score"] = (
            self.translation_score / progress_buf.float()
        ) *100.0

        self.pre_normal_mag_ff = sum_dist_cap_to_sensor_ff_padded.clone()
        self.pre_normal_mag_mf = sum_dist_cap_to_sensor_mf_padded.clone()
        self.pre_normal_mag_rf = sum_dist_cap_to_sensor_rf_padded.clone()
        self.pre_normal_mag_lf = sum_dist_cap_to_sensor_lf_padded.clone()
        self.pre_normal_mag_th = sum_dist_cap_to_sensor_th_padded.clone()

        return reward, reset_buf, info_dict

    def smooth_clamp(self, x, min_val=0.0, max_val=17.0):
        scale = max_val - min_val
        #print(torch.tanh(x))
        return (( torch.tanh(x)) * scale)

    def reset(self, env_ids):
        self.z_score[env_ids] = 0.0
        self.rot_score[env_ids] = 0.0
        self.translation_score[env_ids] = 0.0
        self.slip_sum[env_ids] = 0.0
        self.force_norm_sum[env_ids] = 0.0

        self.pre_normal_mag_ff[env_ids] = 0.0
        self.pre_normal_mag_mf[env_ids] = 0.0
        self.pre_normal_mag_rf[env_ids] = 0.0
        self.pre_normal_mag_lf[env_ids] = 0.0
        self.pre_normal_mag_th[env_ids] = 0.0

        self.pre_diff_ff[env_ids] = 0.0
        self.pre_diff_mf[env_ids] = 0.0
        self.pre_diff_rf[env_ids] = 0.0
        self.pre_diff_lf[env_ids] = 0.0
        self.pre_diff_th[env_ids] = 0.0

        self.pre_vel_nonthumb[env_ids] = 0.0
        self.pre_vel_thumb[env_ids] = 0.0

        self.pre_finger_force_ff[env_ids] = 0.0
        self.pre_finger_force_mf[env_ids] = 0.0
        self.pre_finger_force_rf[env_ids] = 0.0
        self.pre_finger_force_lf[env_ids] = 0.0
        self.pre_finger_force_th[env_ids] = 0.0

        self.pre_cap_force[env_ids] = 0.0

        self.count = 0.0
        self.force_norm_sum[env_ids] = 0.0
        self.force_ff_norm_sum[env_ids] = 0.0
        self.force_mf_norm_sum[env_ids] = 0.0
        self.force_rf_norm_sum[env_ids] = 0.0
        self.force_lf_norm_sum[env_ids] = 0.0
        self.force_th_norm_sum[env_ids] = 0.0

        return

def build(**kwargs):
    return RewardFunction(**kwargs)
