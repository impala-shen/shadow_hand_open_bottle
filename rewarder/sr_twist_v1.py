import torch
import torch.nn
import torch.nn.functional as F
from isaaclab.utils.math import matrix_from_quat
from isaaclab_tasks.direct.shadow_hand_open_bottle.rewarder.base import BaseRewardFunction
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
        desired_force_norm = states["desired_force_norm"]
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

        pre_right_thumb_tips_pos = states["pre_right_thumb_tips_pos"]
        pre_right_nonthumb_tips_pos = states["pre_right_nonthumb_tips_pos"]

        # pre_right_thumb_tips_pos = pre_right_thumb_tips_pos.unsqueeze(2)  # [N, F2, 1, 3]
        # pre_right_nonthumb_tips_pos = pre_right_nonthumb_tips_pos.unsqueeze(2)  # [N, F2, 1, 3]

        cap_marker_pos = states["cube_cap_marker_pos"]  # [N, K2, 3]

        right_link_ff_tip_sensor_pos =  states["right_link_ff_tip_sensor_pos"]
        right_link_mf_tip_sensor_pos =  states["right_link_mf_tip_sensor_pos"]
        right_link_rf_tip_sensor_pos =  states["right_link_rf_tip_sensor_pos"]
        right_link_lf_tip_sensor_pos =  states["right_link_lf_tip_sensor_pos"]
        right_link_th_tip_sensor_pos =  states["right_link_th_tip_sensor_pos"]

        cap_marker_pos = cap_marker_pos.unsqueeze(1)  # [N, 1, K2, 3]

        right_link_ff_tip_sensor_pos = right_link_ff_tip_sensor_pos.unsqueeze(2)  # [N, F2, 1, 3]
        right_link_mf_tip_sensor_pos = right_link_mf_tip_sensor_pos.unsqueeze(2)  # [N, F2, 1, 3]

        right_link_rf_tip_sensor_pos = right_link_rf_tip_sensor_pos.unsqueeze(2)  # [N, F2, 1, 3]
        right_link_lf_tip_sensor_pos = right_link_lf_tip_sensor_pos.unsqueeze(2)  # [N, F2, 1, 3]
        right_link_th_tip_sensor_pos = right_link_th_tip_sensor_pos.unsqueeze(2)  # [N, F2, 1, 3]

        cube_center_pose = states["cube_pos"]
        cube_center_pose = cube_center_pose.unsqueeze(1).unsqueeze(1)# [N, 1, 1, 3]

        slip_detection = states["slip_detection"]

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

        min_temp_sensor_input_ff = torch.min(dist_cap_to_sensor_ff,-1)[0]
        min_temp_sensor_input_mf = torch.min(dist_cap_to_sensor_mf,-1)[0]
        min_temp_sensor_input_rf = torch.min(dist_cap_to_sensor_rf,-1)[0]
        min_temp_sensor_input_lf = torch.min(dist_cap_to_sensor_lf,-1)[0]
        min_temp_sensor_input_th = torch.min(dist_cap_to_sensor_th,-1)[0]

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

        # temp_sensor_input_ff = torch.where(temp_sensor_input_ff > contact_threshold, 0.0, 0.1) #10.0
        # temp_sensor_input_mf = torch.where(temp_sensor_input_mf > contact_threshold, 0.0, 0.1)
        # temp_sensor_input_rf = torch.where(temp_sensor_input_rf > contact_threshold, 0.0, 0.1)
        # temp_sensor_input_lf = torch.where(temp_sensor_input_lf > contact_threshold, 0.0, 0.1)
        # temp_sensor_input_th = torch.where(temp_sensor_input_th > contact_threshold, 0.0, 0.1)

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

        non_force_contact_penalty = non_force_contact_ff * non_force_contact_mf * non_force_contact_rf * non_force_contact_lf * non_force_contact_th
        
        non_contact_penalty = non_contact_ff * non_contact_mf * non_contact_rf * non_contact_lf * non_contact_th
        #print(non_contact_ff)
        #print(non_contact_mf)
        #print(non_contact_rf)
        #print(non_contact_lf)
        #print(non_contact_th)

        #print(temp_lf)

        # temp_sum_dist_cap_to_sensor_ff_padded =  torch.clamp(temp_ff, 0.0, 2.0)
        # temp_sum_dist_cap_to_sensor_mf_padded =  torch.clamp(temp_mf, 0.0, 2.0)
        # temp_sum_dist_cap_to_sensor_rf_padded =  torch.clamp(temp_rf, 0.0, 2.0)
        # temp_sum_dist_cap_to_sensor_lf_padded =  torch.clamp(temp_lf, 0.0, 2.0)
        # temp_sum_dist_cap_to_sensor_th_padded =  torch.clamp(temp_th, 0.0, 2.0)

        temp_sum_dist_cap_to_sensor_ff_padded =  torch.clamp(temp_ff, 0.0, 0.1)
        temp_sum_dist_cap_to_sensor_mf_padded =  torch.clamp(temp_mf, 0.0, 0.1)
        temp_sum_dist_cap_to_sensor_rf_padded =  torch.clamp(temp_rf, 0.0, 0.1)
        temp_sum_dist_cap_to_sensor_lf_padded =  torch.clamp(temp_lf, 0.0, 0.1)
        temp_sum_dist_cap_to_sensor_th_padded =  torch.clamp(temp_th, 0.0, 0.1)
        

        # EMA for sensor 

        # ema_ratio_sum = 0.7

        # sum_dist_cap_to_sensor_ff_padded = ema_ratio_sum*temp_sum_dist_cap_to_sensor_ff_padded + (1-ema_ratio_sum)*self.pre_normal_mag_ff
        # sum_dist_cap_to_sensor_mf_padded = ema_ratio_sum*temp_sum_dist_cap_to_sensor_mf_padded + (1-ema_ratio_sum)*self.pre_normal_mag_mf
        # sum_dist_cap_to_sensor_rf_padded = ema_ratio_sum*temp_sum_dist_cap_to_sensor_rf_padded + (1-ema_ratio_sum)*self.pre_normal_mag_rf
        # sum_dist_cap_to_sensor_lf_padded = ema_ratio_sum*temp_sum_dist_cap_to_sensor_lf_padded + (1-ema_ratio_sum)*self.pre_normal_mag_lf
        # sum_dist_cap_to_sensor_th_padded = ema_ratio_sum*temp_sum_dist_cap_to_sensor_th_padded + (1-ema_ratio_sum)*self.pre_normal_mag_th

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
        reward = torch.where(
            min_dis_val_th > distance_termination,
            reward + reward_settings["failure_penalty"],
            reward,
        )
        # print("min_dis_val_ff",min_dis_val_ff)
        # print("min_dis_val_mf",min_dis_val_mf)
        # print("min_dis_val_rf",min_dis_val_rf)
        # print("min_dis_val_lf",min_dis_val_lf)
        # print("min_dis_val_th",min_dis_val_th)
        #print(reward)
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

        reward_tactile = reward_tactile_1 #for display

        reward = reward + (
                    grasp_quality_ff + grasp_quality_mf + grasp_quality_rf + grasp_quality_lf + grasp_quality_th
                ) * reward_settings["cap_tactile_mult"]

        # print("grasp_quality_ff ",(
        #            grasp_quality_ff + grasp_quality_mf + grasp_quality_rf + grasp_quality_lf + grasp_quality_th
        #        ) * reward_settings["cap_tactile_mult"])

        # rotation reward
        # set_point_value = 0.006
        # rotation 
        rotation_reward =  (
            states["cube_dof_pos"] - states["last_cube_dof_pos"]
        ) 

        rotation_reward = torch.nan_to_num(
            rotation_reward, nan=0.0, posinf=1.0, neginf=-1.0
        )
        rotation_reward = torch.clamp(rotation_reward, -0.02, 0.01)
        #rotation_reward = torch.clamp(rotation_reward, -0.02, 0.005)

        info_dict["cube_dof_pos_delta"] = rotation_reward
        
        self.rot_score += rotation_reward #

        # print('reward', reward)
        # print('DEbug -----rotation_reward', rotation_reward)

        # reward = reward + rotation_reward * reward_settings["rotation_reward_scale"] * non_contact_penalty
        reward = reward + rotation_reward * reward_settings["rotation_reward_scale"]* non_contact_penalty
        # print('DEbug -----reward', reward)

        # print("rotation_reward : ",rotation_reward * reward_settings["rotation_reward_scale"] * non_contact_penalty)

        contact_ff = torch.where(sum_dist_cap_to_sensor_ff_padded > 0.0, 20.0, 0.0)
        contact_mf = torch.where(sum_dist_cap_to_sensor_mf_padded > 0.0, 20.0, 0.0)
        contact_rf = torch.where(sum_dist_cap_to_sensor_rf_padded > 0.0, 20.0, 0.0)
        contact_lf = torch.where(sum_dist_cap_to_sensor_lf_padded > 0.0, 20.0, 0.0)
        contact_th = torch.where(sum_dist_cap_to_sensor_th_padded > 0.0, 20.0, 0.0)

        #print(contact_ff)

        # reward = (reward 
        # + 2.0*(1/(grasp_quality_ff+1.0)) * rotation_reward * reward_settings["rotation_reward_scale"]* reward_settings["grasping_qual_onoff"]
        # + 2.0*(1/(grasp_quality_mf+1.0)) * rotation_reward * reward_settings["rotation_reward_scale"]* reward_settings["grasping_qual_onoff"]
        # + 2.0*(1/(grasp_quality_rf+1.0)) * rotation_reward * reward_settings["rotation_reward_scale"]* reward_settings["grasping_qual_onoff"]
        # + 2.0*(1/(grasp_quality_lf+1.0)) * rotation_reward * reward_settings["rotation_reward_scale"]* reward_settings["grasping_qual_onoff"]
        # + 2.0*(1/(grasp_quality_th+1.0)) * rotation_reward * reward_settings["rotation_reward_scale"]* reward_settings["grasping_qual_onoff"])

        firm_rotation_ff = grasp_quality_ff * rotation_reward * reward_settings["rotation_reward_scale"]* reward_settings["grasping_qual_onoff"]
        firm_rotation_mf = grasp_quality_mf * rotation_reward * reward_settings["rotation_reward_scale"]* reward_settings["grasping_qual_onoff"]
        firm_rotation_rf = grasp_quality_rf * rotation_reward * reward_settings["rotation_reward_scale"]* reward_settings["grasping_qual_onoff"]
        firm_rotation_lf = grasp_quality_lf * rotation_reward * reward_settings["rotation_reward_scale"]* reward_settings["grasping_qual_onoff"]
        firm_rotation_th = grasp_quality_th * rotation_reward * reward_settings["rotation_reward_scale"]* reward_settings["grasping_qual_onoff"]

        firm_rotation_ff =  torch.clamp(firm_rotation_ff, -2.0, 2.0)
        firm_rotation_mf =  torch.clamp(firm_rotation_mf, -2.0, 2.0)
        firm_rotation_rf =  torch.clamp(firm_rotation_rf, -2.0, 2.0)
        firm_rotation_lf =  torch.clamp(firm_rotation_lf, -2.0, 2.0)
        firm_rotation_th =  torch.clamp(firm_rotation_th, -2.0, 2.0)

        reward = (reward + 1.0*(firm_rotation_ff + firm_rotation_mf + firm_rotation_rf + firm_rotation_lf + firm_rotation_th)*non_contact_penalty)

        # print("firm_rotation_ff : ",(firm_rotation_ff + firm_rotation_mf + firm_rotation_rf + firm_rotation_lf + firm_rotation_th)*non_contact_penalty)

        #print((firm_rotation_ff + firm_rotation_mf + firm_rotation_rf + firm_rotation_lf + firm_rotation_th)*non_contact_penalty)

        reward_cap_dist_nonthumb = 0.1 / (dist_cap_to_right_nonthumb * 2 + 0.03)
        reward_cap_dist_nonthumb = torch.clamp(
            reward_cap_dist_nonthumb, 0, 0.1 / (0.01 * 2 + 0.03)
        ).sum(dim=-1)
        # print(reward_cap_dist_nonthumb)

        reward_cap_dist_thumb = 0.1 / (dist_cap_to_right_thumb * 2 + 0.03)
        reward_cap_dist_thumb = torch.clamp(
            reward_cap_dist_thumb, 0, 0.1 / (0.01 * 2 + 0.03)
        ).mean(dim=-1)

        #reward = (reward + reward_cap_dist_nonthumb * reward_settings["cap_mult"] + reward_cap_dist_thumb * reward_settings["thumb_mult"] )

        #distance_penalty = -(min_dis_val_ff + min_dis_val_mf + min_dis_val_rf + min_dis_val_lf + min_dis_val_th) * reward_settings["distance_penalty_scale"]
        dt = 1 / 120.0
        temp_tip_non_thumb_vel = (right_nonthumb_tips_pos - pre_right_nonthumb_tips_pos) / dt
        temp_tip_thumb_vel = (right_thumb_tips_pos - pre_right_thumb_tips_pos) / dt

        # xy plane projection.
        temp_tip_non_thumb_vel[..., 2] = 0.0
        temp_tip_thumb_vel[..., 2] = 0.0
        tip_non_thumb_vel = temp_tip_non_thumb_vel.clone()
        tip_thumb_vel = temp_tip_thumb_vel.clone()

        #print(tip_thumb_vel.shape)

        #print(temp_tip_non_thumb_vel)

        vel_norm_3 = torch.norm(tip_non_thumb_vel, dim=-1, keepdim=True)
        vel_norm_4 = torch.norm(tip_thumb_vel, dim=-1, keepdim=True)

        #print(vel_norm_4.shape)

        diff_angle = (
           states["cube_dof_pos"] - states["last_cube_dof_pos"]
        )
        angular_vel = states["cube_dof_vel"]


        #should be modified here : 

        tan_vel = angular_vel*0.04

        
        # print("tan_vel", tan_vel.shape)
        # print("vel_norm_3.squeeze()", vel_norm_3.shape)
        # print("el_norm_4.reshape(self.num_envs, 1)", vel_norm_4.shape)
        # diff_rel_vel = tan_vel.view(1, 2, 1) - vel_norm_3.view(1, 1, 4)
        # abs_diff_rel_vel = diff_rel_vel.abs()
        # diff_thumb_rel_vel = tan_vel.view(1, 2, 1) - vel_norm_4.view(1, 1, 1)   # size(2,1) - size(2,1)
        # abs_diff_thumb_rel_vel = diff_thumb_rel_vel.abs()

        # abs_diff_rel_vel = abs_diff_rel_vel.squeeze(0)
        # abs_diff_thumb_rel_vel = abs_diff_thumb_rel_vel.squeeze(0)

        # #print(vel_norm_3.squeeze().shape)
        # #print(tan_vel.shape)

        # vel_alpha = 0.75

        # final_abs_vel_nonthumb = vel_alpha*abs_diff_rel_vel + (1-vel_alpha)*self.pre_vel_nonthumb
        # self.pre_vel_nonthumb = final_abs_vel_nonthumb.clone()


        # final_abs_vel_thumb = vel_alpha*abs_diff_thumb_rel_vel + (1-vel_alpha)*self.pre_vel_thumb
        # self.pre_vel_thumb = final_abs_vel_thumb.clone()



        #print(abs_diff_thumb_rel_vel.shape)
        
        upper_bound = 0.005

        #print(vel_norm_3)

        # figner distance penalty
        # w/ slip detections

        temp_contact_status_ff = (reward_sensor_ff_dist_thumb > 0.0)
        temp_contact_status_mf = (reward_sensor_mf_dist_thumb > 0.0)
        temp_contact_status_rf = (reward_sensor_rf_dist_thumb > 0.0)
        temp_contact_status_lf = (reward_sensor_lf_dist_thumb > 0.0)
        temp_contact_status_th = (reward_sensor_th_dist_thumb > 0.0)

        # 
        contact_status_ff = torch.where(
            temp_contact_status_ff,
            torch.ones_like(grasp_quality_ff),
            torch.zeros_like(grasp_quality_ff)
            )
        
        contact_status_mf = torch.where(
            temp_contact_status_mf,
            torch.ones_like(grasp_quality_mf),
            torch.zeros_like(grasp_quality_mf)
            )
        
        contact_status_rf = torch.where(
            temp_contact_status_rf,
            torch.ones_like(grasp_quality_rf),
            torch.zeros_like(grasp_quality_rf)
            )
        contact_status_lf = torch.where(
            temp_contact_status_lf,
            torch.ones_like(grasp_quality_lf),
            torch.zeros_like(grasp_quality_lf)
            )
        contact_status_th = torch.where(
            temp_contact_status_th,
            torch.ones_like(grasp_quality_th),
            torch.zeros_like(grasp_quality_th)
            )

        diff_normal_mag_ff = contact_status_ff*(sum_dist_cap_to_sensor_ff_padded - self.pre_normal_mag_ff)
        diff_normal_mag_mf = contact_status_mf*(sum_dist_cap_to_sensor_mf_padded - self.pre_normal_mag_mf)
        diff_normal_mag_rf = contact_status_rf*(sum_dist_cap_to_sensor_rf_padded - self.pre_normal_mag_rf)
        diff_normal_mag_lf = contact_status_lf*(sum_dist_cap_to_sensor_lf_padded - self.pre_normal_mag_lf)
        diff_normal_mag_th = contact_status_th*(sum_dist_cap_to_sensor_th_padded - self.pre_normal_mag_th)

        ema_alpha = 0.7
        final_diff_normal_mag_ff = ema_alpha * diff_normal_mag_ff + (1 - ema_alpha)*self.pre_diff_ff
        final_diff_normal_mag_mf = ema_alpha * diff_normal_mag_mf + (1 - ema_alpha)*self.pre_diff_mf
        final_diff_normal_mag_rf = ema_alpha * diff_normal_mag_rf + (1 - ema_alpha)*self.pre_diff_rf
        final_diff_normal_mag_lf = ema_alpha * diff_normal_mag_lf + (1 - ema_alpha)*self.pre_diff_lf
        final_diff_normal_mag_th = ema_alpha * diff_normal_mag_th + (1 - ema_alpha)*self.pre_diff_th

        #print(final_diff_normal_mag_mf)

        self.pre_diff_ff = final_diff_normal_mag_ff.clone()
        self.pre_diff_mf = final_diff_normal_mag_mf.clone()
        self.pre_diff_rf = final_diff_normal_mag_rf.clone()
        self.pre_diff_lf = final_diff_normal_mag_lf.clone()
        self.pre_diff_th = final_diff_normal_mag_th.clone()
        #print(reward)

        # error_epslion = 10.0
        #f_error =  torch.abs(f_d_norm - force_norm_avr)
        # condition_error = torch.where(f_error < error_epslion, f_error, 200.0)
        # temp_f_error = torch.abs(f_d_norm - force_norm_avr)/(f_d_norm + 0.01)

        slip_error =  torch.abs(slip_avr)
        f_error =  torch.abs(slip_avr)

        temp_slip_error = torch.abs(slip_avr)/(0.01)

        #print(slip_avr)
       
        #slip_reward = torch.exp(-temp_slip_error/10.0)*reward_settings["slip_mult"]*non_force_contact_penalty*non_contact_penalty
        #adaptive_slip_reward = slip_reward

        slip_reward = torch.exp(-temp_slip_error/10.0)*reward_settings["slip_mult"]*non_force_contact_penalty*non_contact_penalty
        #temp_slip_rotation_reward = torch.clamp(rotation_reward, 0.02, 0.02)
        adaptive_slip_reward = slip_reward * rotation_reward * reward_settings["rotation_reward_scale"]

        #print("adaptive_slip_reward : ",adaptive_slip_reward)
        reward = reward  + 0.0*adaptive_slip_reward



        # +---------------------------------------------------------------------------------+
        # |      penalty      |
        # +---------------------------------------------------------------------------------+

        #non-contact penalty 
        #reward = reward  + non_contact_penalty


        # diff_normal_mag_ff = torch.where(diff_normal_mag_ff < 0.0, diff_normal_mag_ff, torch.zeros_like(diff_normal_mag_ff))
        # diff_normal_mag_mf = torch.where(diff_normal_mag_mf < 0.0, diff_normal_mag_mf, torch.zeros_like(diff_normal_mag_mf))
        # diff_normal_mag_rf = torch.where(diff_normal_mag_rf < 0.0, diff_normal_mag_rf, torch.zeros_like(diff_normal_mag_rf))
        # diff_normal_mag_lf = torch.where(diff_normal_mag_lf < 0.0, diff_normal_mag_lf, torch.zeros_like(diff_normal_mag_lf))
        # diff_normal_mag_th = torch.where(diff_normal_mag_th < 0.0, diff_normal_mag_th, torch.zeros_like(diff_normal_mag_th))

        # penalty_slip_ff =  slip_detection[:,0]*torch.clamp(diff_normal_mag_ff, -2.0, 0.0)*reward_settings["slip_mult"]
        # penalty_slip_mf =  slip_detection[:,1]*torch.clamp(diff_normal_mag_mf, -2.0, 0.0)*reward_settings["slip_mult"]
        # penalty_slip_rf =  slip_detection[:,2]*torch.clamp(diff_normal_mag_rf, -2.0, 0.0)*reward_settings["slip_mult"]
        # penalty_slip_lf =  slip_detection[:,3]*torch.clamp(diff_normal_mag_lf, -2.0, 0.0)*reward_settings["slip_mult"]
        # penalty_slip_th =  slip_detection[:,4]*torch.clamp(diff_normal_mag_th, -2.0, 0.0)*reward_settings["slip_mult"]

        # give penalties to an agent when the slip happens and the applied normal force is small. 

        #slip_penalty = (penalty_slip_ff + penalty_slip_mf + penalty_slip_rf + penalty_slip_lf + penalty_slip_th)

        #reward = reward + slip_penalty 

        #print(diff_normal_mag_ff)

        # finger gaiting penalty

        ff_temp = torch.where(z_sign_nonthumb[:,0,0] > 0, torch.ones_like(z_sign_nonthumb[:,0,0])* 2.0, z_sign_nonthumb[:,0,0]* 2.0)
        mf_temp = torch.where(z_sign_nonthumb[:,1,0] > 0, torch.ones_like(z_sign_nonthumb[:,1,0])* 2.0, z_sign_nonthumb[:,1,0]* 2.0)
        rf_temp = torch.where(z_sign_nonthumb[:,2,0] > 0, torch.ones_like(z_sign_nonthumb[:,2,0])* 2.0, z_sign_nonthumb[:,2,0]* 2.0)
        lf_temp = torch.where(z_sign_nonthumb[:,3,0] > 0, torch.ones_like(z_sign_nonthumb[:,3,0])* 2.0, z_sign_nonthumb[:,3,0]* 2.0)
        th_temp = torch.where(z_sign_thumb[:,0,0] > 0, torch.ones_like(z_sign_thumb[:,0,0])* 2.0, z_sign_thumb[:,0,0]* 2.0)


        temp_gaiting_penalty_ff = sum_dist_cap_to_sensor_ff_padded * ff_temp * reward_settings["gaiting_mult"]
        temp_gaiting_penalty_mf = sum_dist_cap_to_sensor_mf_padded * mf_temp * reward_settings["gaiting_mult"]
        temp_gaiting_penalty_rf = sum_dist_cap_to_sensor_rf_padded * rf_temp * reward_settings["gaiting_mult"]
        temp_gaiting_penalty_lf = sum_dist_cap_to_sensor_lf_padded * lf_temp * reward_settings["gaiting_mult"]
        temp_gaiting_penalty_th = sum_dist_cap_to_sensor_th_padded * th_temp * reward_settings["gaiting_mult"]

        #print(temp_gaiting_penalty_ff)

        gaiting_penalty_ff =  torch.clamp(temp_gaiting_penalty_ff, -5.0, 5.0)
        gaiting_penalty_mf =  torch.clamp(temp_gaiting_penalty_mf, -5.0, 5.0)
        gaiting_penalty_rf =  torch.clamp(temp_gaiting_penalty_rf, -5.0, 5.0)
        gaiting_penalty_lf =  torch.clamp(temp_gaiting_penalty_lf, -5.0, 5.0)
        gaiting_penalty_th =  torch.clamp(temp_gaiting_penalty_th, -5.0, 5.0)


        # gaiting_penalty_ff =  ff_temp * reward_settings["gaiting_mult"]
        # gaiting_penalty_mf =  mf_temp * reward_settings["gaiting_mult"]
        # gaiting_penalty_rf =  rf_temp * reward_settings["gaiting_mult"]
        # gaiting_penalty_lf =  lf_temp * reward_settings["gaiting_mult"]
        # gaiting_penalty_th =  th_temp * reward_settings["gaiting_mult"]


        gaiting_penalty = (gaiting_penalty_ff + gaiting_penalty_mf + gaiting_penalty_rf + gaiting_penalty_lf + gaiting_penalty_th)

        # print("gaiting_penalty", gaiting_penalty_mf)

        #print("gaiting_penalty_ff ", gaiting_penalty_ff)
        #print("gaiting_penalty_mf :", gaiting_penalty_mf)
        #print("gaiting_penalty_rf :", gaiting_penalty_rf)

        #print(gaiting_penalty)
        reward = reward + 1.0*gaiting_penalty

        # anlgue penalty 
        cube_quat = states["cube_quat"]
        cube_rotation_matrix = matrix_from_quat(cube_quat
        )  # [B, 3, 3]
        z_axis = cube_rotation_matrix[:, :, 2]  # [B, 3]

        # print("z_axis :: ",cube_quat)

        # Point to the right.
        target_vector = (
            torch.FloatTensor([0.0, 0.0, 1.0])
            .to(self.device)
            .reshape(-1, 3)
            .repeat(z_axis.size(0), 1)
        )
        angle_difference = torch.arccos(torch.sum(z_axis * target_vector, dim=-1)) # 똑바로 하기 위한것 
        angle_penalty = -torch.clamp(angle_difference, 0.0, 1.0)
        #print("angle_difference :: ",angle_difference)
        info_dict["cube_z_angle_difference"] = angle_difference

        z_score_update_idx = torch.where(progress_buf > 100)
        self.z_score[z_score_update_idx] += angle_difference[z_score_update_idx]

        # if episode length is over 100, reset if the angle is too large
        # if reward_settings["reset_by_z_angle"]:
        #     reset_buf = torch.where(
        #         torch.logical_and(progress_buf >= 100, angle_difference > 0.2),
        #         torch.ones_like(reset_buf),
        #         reset_buf,
        #     )


        reward = reward + 1.0*angle_penalty * reward_settings["angle_penalty_scale"]

        # print('angle_penalty_scale', angle_penalty * reward_settings["angle_penalty_scale"])

        # action penalty 
        right_action_penalty = -torch.sum(actions[:, :22] ** 2, dim=-1)

        action_penalty = (
            + right_action_penalty# off
        )
        reward = reward + 1.0*action_penalty * reward_settings["action_penalty_scale"]

        # print('action_penalty * reward_settings["action_penalty_scale"]',action_penalty * reward_settings["action_penalty_scale"])

        # work penalty 
        work_penalty = (
            -states["right_work"] * reward_settings["right_work_penalty_scale"]
        )
        reward = reward + 0.0*work_penalty * reward_settings["work_penalty_scale"]

        # print('work_penalty * reward_settings["work_penalty_scale"]',work_penalty * reward_settings["work_penalty_scale"])


        # +---------------------------------------------------------------------------------+
        # |      Final_reward      |
        # +---------------------------------------------------------------------------------+
        
        
        reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-1.0)

        # print('final**********',reward)

        reward = reward[:1, :]
        # print('final**********',reward)

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

        info_dict["f_error"] = f_error
        info_dict["slip_error"] = slip_error

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

    def make_force(self, progress_buf):
        hz = 60
        desired_min = 10.0
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

def build(**kwargs):
    return RewardFunction(**kwargs)
