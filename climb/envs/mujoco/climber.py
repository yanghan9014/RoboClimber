import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
from gymnasium.spaces import Box
import pdb

def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass)).copy()

class Climber(HumanoidEnv):
    def __init__(self, xml_file=None, keyframe=None, rand_start_keyframe=False, **kwargs):
        self.cur_mode = "train"
        render_mode = kwargs.get('render_mode', 'rgb_array')
        self._exclude_current_positions_from_observation = False

        if xml_file is not None:
            super().__init__(xml_file=xml_file, render_mode=render_mode)
        else:
            super().__init__(render_mode=render_mode)


        #################################
        # Selecting starting frame
        self.keyframe = keyframe
        self.rand_start_keyframe = rand_start_keyframe
        if rand_start_keyframe:
            self.height_trigger = [0.0, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]
            self.start_keyframe_max = 1
            self.start_qpos = None
            self.end_qpos =None
        self.qpos_t_2 = None

        #################################
        # include ladder rungs in observation space
        ladder_id = self.model.body('ladder').id
        # The geom id of the ladder rungs
        rung_low = self.model.body_geomadr[ladder_id]
        rung_high = rung_low + self.model.body_geomnum[ladder_id]
        self.rung_range = np.arange(rung_low, rung_high)

        self.include_rung_xz_pos = False
        obs_size = self.observation_space.shape[0]
        if self.include_rung_xz_pos:
            obs_size += (rung_high - rung_low) * 2
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )
        
        #################################
        # define legal contacts
        self.rf_body_id = self.model.body('right_foot').id
        self.lf_body_id = self.model.body('left_foot').id
        rh_id = self.model.body('finger_mid_r').id
        lh_id = self.model.body('finger_mid_l').id

        self.rf_geom_id = self.model.body_geomadr[self.rf_body_id]
        self.lf_geom_id = self.model.body_geomadr[self.lf_body_id]
        self.f_geom_id = np.array([self.rf_geom_id, self.lf_geom_id])
        self.rh_geom_ids = np.arange(self.model.geom('finger_mid_r1').id, self.model.geom('finger_mid_r1').id + self.model.body_geomnum[rh_id])
        self.lh_geom_ids = np.arange(self.model.geom('finger_mid_l1').id, self.model.geom('finger_mid_l1').id + self.model.body_geomnum[lh_id])
        self.uwaist_geom_id = self.model.geom('uwaist').id

        self.legal_contact_bodyid = np.array([self.model.body('ladder').id,
                                              self.model.body('right_shin').id,
                                              self.model.body('right_foot').id,
                                              self.model.body('left_shin').id,
                                              self.model.body('left_foot').id,
                                              self.model.body('right_hand').id, 
                                              self.model.body('finger_base_r').id,
                                              self.model.body('finger_mid_r').id,
                                              self.model.body('finger_tip_r').id,
                                              self.model.body('left_hand').id, 
                                              self.model.body('finger_base_l').id,
                                              self.model.body('finger_mid_l').id,
                                              self.model.body('finger_tip_l').id,
                                              ])


        #################################
        # reset values
        self._highest_reached_rung_reset_value = rung_low + 7 # assume the first 6 rungs can be reached easily
        self.highest_reached_rung = self._highest_reached_rung_reset_value

        self._foot_highest_reached_rung_reset_value = rung_low + 2 # assume the first 2 rungs can be reached easily
        self.foot_highest_reached_rung = self._foot_highest_reached_rung_reset_value

        self.max_height = 0
        self.max_height_rew_thresh = 0.1
        self.max_height_reward_weight = 1.0

        self.milestone_reward_weight_reset_val = 30
        self.milestone_reward_weight = self.milestone_reward_weight_reset_val
        self.foot_milestone_reward_weight_reset_val = 30
        self.foot_milestone_reward_weight = self.foot_milestone_reward_weight_reset_val

        #################################
        # reward weights
        self.upward_reward_weight = 50.0
        self.downward_reward_weight = 0.0

        self.smoothness_cost_weight = 0.1
        self.motion_cost_weight = 0.1

        self._healthy_reward = 1
        self.bad_contact_cost_weight = 3.0

        self.goal_reward_weight = 100.0

        self.good_contact_reward = 1.0

        self.hand_up_reward_weight = 0.1
        self.let_go_reward_weight = 0.1

        self.hip_qpos_reward_weight = 1.0
        self.ab_qpos_reward_weight = 1.0

    
    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2]
        return is_healthy

    def step(self, action):
        # print(self.data.cfrc_ext)
        # print(f"ladder: {self.data.cfrc_ext[1]}")
        # print(f"right_foot: {self.data.cfrc_ext[7]}")
        # print(f"left_foot: {self.data.cfrc_ext[10]}")
        # print(f"finger_base_r: {self.data.cfrc_ext[14]}")
        # print(f"finger_mid_r: {self.data.cfrc_ext[15]}")
        # print(f"finger_tip_r: {self.data.cfrc_ext[16]}")
        # print(f"finger_base_l: {self.data.cfrc_ext[20]}")
        # print(f"finger_mid_l: {self.data.cfrc_ext[21]}")
        # print(f"finger_tip_l: {self.data.cfrc_ext[22]}")
        # print(f"total x force: {sum([f[0] for f in self.data.cfrc_ext])}")
        # print(f"total y force: {sum([f[1] for f in self.data.cfrc_ext])}")
        # print(f"total z force: {sum([f[2] for f in self.data.cfrc_ext])}")
        xyz_position_before = mass_center(self.model, self.data)
        qpos_t_1 = self.data.qpos.copy()
        self.do_simulation(action, self.frame_skip)
        qpos_t = self.data.qpos.copy()
        xyz_position_after = mass_center(self.model, self.data)

        xyz_velocity = (xyz_position_after - xyz_position_before) / self.dt
        x_velocity, y_velocity, z_velocity = xyz_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(z_velocity, action, qpos_t_1, qpos_t)
        self.qpos_t_2 = qpos_t_1.copy()

        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "z_position": self.data.qpos[2],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "z_velocity": z_velocity,
            "highest_reached_rung": self.highest_reached_rung,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info
    
    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._include_cinert_in_observation is True:
            com_inertia = self.data.cinert[1:].flatten()
        else:
            com_inertia = np.array([])
        if self._include_cvel_in_observation is True:
            com_velocity = self.data.cvel[1:].flatten()
        else:
            com_velocity = np.array([])

        if self._include_qfrc_actuator_in_observation is True:
            actuator_forces = self.data.qfrc_actuator[6:].flatten()
        else:
            actuator_forces = np.array([])
        if self._include_cfrc_ext_in_observation is True:
            external_contact_forces = self.data.cfrc_ext[1:].flatten()
        else:
            external_contact_forces = np.array([])

        if self.include_rung_xz_pos:
            rung_rel_position_xz = (position[0:3:2] - self.rung_xpos).flatten()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        ret = np.concatenate((
            position,
            velocity,
            com_inertia,
            com_velocity,
            actuator_forces,
            external_contact_forces,))
        if self.include_rung_xz_pos:
            ret = np.concatenate((ret, rung_rel_position_xz,))
        return ret

    def _get_rew(self, z_velocity: float, action, qpos_t_1, qpos_t):
        # clip_z = 0.5
        if z_velocity > 0.1:
            upward_reward = self.upward_reward_weight * z_velocity
            # upward_reward = self.upward_reward_weight * min(clip_z, z_velocity)
        else:
            upward_reward = self.downward_reward_weight * z_velocity
        healthy_reward = self.healthy_reward

        # height = self.data.qpos[2]
        # max_height_reward = 0
        # self.max_height = max(height, self.max_height)
        # if height > self.max_height - self.max_height_rew_thresh:
        #     max_height_reward = self.max_height_reward_weight

        # contact_geoms = self.data.contact.geom.ravel()
        # contact_geom1 = self.data.contact.geom1
        # contact_geom2 = self.data.contact.geom2
        # contact_rungs = contact_geom1[np.isin(contact_geom1, self.rung_range)] 

        # reached_milestone = 0
        # if contact_rungs.size > 0:
        #     if contact_rungs.max() > self.highest_reached_rung:
        #         self.milestone_reward_weight = self.milestone_reward_weight_reset_val
        #         reached_milestone = 1
        #         self.highest_reached_rung = contact_rungs.max()
        #     elif contact_rungs.max() == self.highest_reached_rung:
        #         self.milestone_reward_weight *= 0.96
        #         reached_milestone = 1

        # foot_reached_milestone = 0
        # if np.any(np.isin(self.f_geom_id, contact_geom2)) and self.data.cfrc_ext[self.rf_body_id, -1] > 100 and self.data.cfrc_ext[self.lf_body_id, -1] > 100:
        #     f_idx = np.where(np.isin(contact_geom2, self.f_geom_id))[0]
        #     foot_contact = contact_geom1[f_idx].max()
        #     if foot_contact > self.foot_highest_reached_rung:
        #         self.foot_milestone_reward_weight = self.foot_milestone_reward_weight_reset_val
        #         foot_reached_milestone = 1
        #         self.foot_highest_reached_rung = foot_contact
        #     elif foot_contact == self.foot_highest_reached_rung:
        #         self.foot_milestone_reward_weight *= 0.96
        #         foot_reached_milestone = 1
            
        # milestone_reward = self.milestone_reward_weight * reached_milestone
        # foot_milestone_reward = self.foot_milestone_reward_weight * foot_reached_milestone

        # goal_reward = self.goal_reward_weight * (np.power(self.end_qpos - qpos_t_1, 2).sum() - np.power(self.end_qpos - qpos_t, 2).sum())

        # rh_xpos = self.data.geom_xpos[self.rh_geom_ids[1]]
        # lh_xpos = self.data.geom_xpos[self.lh_geom_ids[1]]
        # uwaist_xpos = self.data.geom_xpos[self.uwaist_geom_id]

        # hand_up = 0.0
        # let_go_reward = 0.0
        # if rh_xpos[-1] > uwaist_xpos[-1]:
        #     # if np.any(np.isin(self.rh_geom_ids, contact_geom2)):
        #     rh_idx = np.where(np.isin(contact_geom2, self.rh_geom_ids))[0]
        #     hand_up += np.any(np.isin(contact_geom1[rh_idx], self.rung_range))
        # else:
        #     rh_mean = -0.4
        #     rh_std = 0.2
        #     rh_variance = rh_std ** 2
        #     rh_coefficient = 1 / np.sqrt(2 * np.pi * rh_variance)
        #     rh_exponent = -((action[-2]  - rh_mean) ** 2) / (2 * rh_variance)
        #     let_go_reward -= rh_coefficient * np.exp(rh_exponent)

        # if lh_xpos[-1] > uwaist_xpos[-1]:
        #     # if np.any(np.isin(self.lh_geom_ids, contact_geom2)):
        #     lh_idx = np.where(np.isin(contact_geom2, self.lh_geom_ids))[0]
        #     hand_up += np.any(np.isin(contact_geom1[lh_idx], self.rung_range))
        # else:
        #     lh_mean = -0.4
        #     lh_std = 0.2
        #     lh_variance = lh_std ** 2
        #     lh_coefficient = 1 / np.sqrt(2 * np.pi * lh_variance)
        #     lh_exponent = -((action[-1] - lh_mean) ** 2) / (2 * lh_variance)
        #     let_go_reward -= lh_coefficient * np.exp(lh_exponent)
        
        # hand_up_reward = self.hand_up_reward_weight * hand_up
        # let_go_reward = self.let_go_reward_weight * let_go_reward

        # good_contact = 0
        # if np.isin(self.rf_geom_id, contact_geom2):
        #     rf_idx = np.where(self.rf_geom_id == contact_geom2)[0]
        #     good_contact += np.all(np.isin(contact_geom1[rf_idx], self.rung_range)) \
        #                     and self.data.cfrc_ext[self.rf_body_id, -1] > 100 \
        #                     # and abs(self.data.cfrc_ext[self.rf_body_id, -2]) < 100 \
        #                     # and abs(self.data.cfrc_ext[self.rf_body_id, -3]) < 100
        # if np.isin(self.lf_geom_id, contact_geom2):
        #     lf_idx = np.where(self.lf_geom_id == contact_geom2)[0]
        #     good_contact += np.all(np.isin(contact_geom1[lf_idx], self.rung_range)) \
        #                     and self.data.cfrc_ext[self.lf_body_id, -1] > 100 \
        #                     # and abs(self.data.cfrc_ext[self.lf_body_id, -2]) < 100 \
        #                     # and abs(self.data.cfrc_ext[self.lf_body_id, -3]) < 100
        # if good_contact == 2:
        #     good_contact = 5
        # good_contact_reward = self.good_contact_reward * good_contact

        # hip_mean = -0.65
        # hip_std = 0.4
        # hip_variance = hip_std ** 2
        # hip_coefficient = 1 / np.sqrt(2 * np.pi * hip_variance)
        # hip_exponent_r = -((self.data.qpos[self.data.joint('right_hip_y').id + 6] - hip_mean) ** 2) / (2 * hip_variance)
        # hip_exponent_l = -((self.data.qpos[self.data.joint('left_hip_y').id + 6] - hip_mean) ** 2) / (2 * hip_variance)
        # hip_qpos_reward = self.hip_qpos_reward_weight * hip_coefficient * (np.exp(hip_exponent_r) + np.exp(hip_exponent_l))

        # ab_mean = -0.4
        # ab_std = 0.15
        # ab_variance = ab_std ** 2
        # ab_coefficient = 1 / np.sqrt(2 * np.pi * ab_variance)
        # ab_exponent = -((self.data.qpos[self.data.joint('abdomen_y').id + 6] - ab_mean) ** 2) / (2 * ab_variance)
        # ab_qpos_reward = self.ab_qpos_reward_weight * ab_coefficient * np.exp(ab_exponent)
        
        rewards = healthy_reward + upward_reward

        smoothness_cost = 0
        motion_cost = 0
        # if self.qpos_t_2 is not None:
        #     smoothness_cost = self.smoothness_cost_weight * np.power(qpos_t[7:] + self.qpos_t_2[7:] - 2 * qpos_t_1[7:], 2).sum()
        #     motion_cost = self.motion_cost_weight * np.power(qpos_t[7:] - qpos_t_1[7:], 2).sum()

        ctrl_cost = self.control_cost(action)

        # bad_contact = np.any(~np.isin(self.model.geom_bodyid[contact_geoms], self.legal_contact_bodyid))
        # bad_contact_cost = self.bad_contact_cost_weight * bad_contact
        costs = ctrl_cost
        # costs = ctrl_cost + smoothness_cost + motion_cost

        reward = rewards - costs

        reward_info = {
            "reward_survive": healthy_reward,
            "reward_upward": upward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_smooth": -smoothness_cost,
            "reward_motion": -motion_cost,
            "goal_reward": 0,
            # "goal_reward": goal_reward,
            # "reward_contact": -contact_cost,
            # "sensor_data": self.data.sensordata.copy()
        }

        return reward, reward_info

    def reset_model(self):
        if not self.rand_start_keyframe and self.keyframe is None:
            return super().reset_model()
        # print(f"Resetting model to keyframe '{self.keyframe}'")
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        if self.rand_start_keyframe and self.cur_mode == "train":
            start_id = np.random.randint(0, self.start_keyframe_max)
            # start_id = 1
            # end_id = start_id + 1
            # self.end_qpos = self.model.key_qpos[end_id]
            key_qpos = self.model.key_qpos[start_id]
            key_qvel = self.model.key_qvel[start_id]
        elif self.keyframe is not None:
            key_qpos = self.model.key(self.keyframe).qpos
            key_qvel = self.model.key(self.keyframe).qvel
        else:
            key_qpos = self.model.key_qpos[0]
            key_qvel = self.model.key_qvel[0]
        qpos = key_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = key_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        
        self.milestone_reward_weight = self.milestone_reward_weight_reset_val
        self.foot_milestone_reward_weight = self.foot_milestone_reward_weight_reset_val
        self.highest_reached_rung = self._highest_reached_rung_reset_value
        self.foot_highest_reached_rung = self._foot_highest_reached_rung_reset_value
        # self.max_height = 0

        # self.rung_xpos = self.data.geom_xpos[self.rung_range][:,::2] # the rung_positions (excluding the y coordinate)
        observation = self._get_obs()
        return observation


    def render(self):
        return super().render()
