import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
import pdb

def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass)).copy()

class Climber(HumanoidEnv):
    def __init__(self, xml_file=None, keyframe=None, **kwargs):
        render_mode = kwargs.get('render_mode', 'rgb_array')
        if xml_file is not None:
            super().__init__(xml_file=xml_file, render_mode=render_mode)
        else:
            super().__init__(render_mode=render_mode)
        self._healthy_reward = 1
        self._upward_reward_weight = 50.0
        self.keyframe = keyframe
        self.xyz_position_after = self.data.xipos[17]
    
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
        self.xyz_position_before = self.xyz_position_after.copy() #self.data.xipos[17] #mass_center(self.model, self.data)
        # print(f"xyz_position_before: {xyz_position_before}")
        self.do_simulation(action, self.frame_skip)
        self.xyz_position_after = self.data.xipos[17] #mass_center(self.model, self.data)
        # print(f"xyz_position_after: {xyz_position_after}")
        xyz_velocity = (self.xyz_position_after - self.xyz_position_before) / self.dt
        # print(f"xyz_velocity: {xyz_velocity}")
        x_velocity, y_velocity, z_velocity = xyz_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(z_velocity, action, z_position=self.data.qpos[2])
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
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _contact_penalty(self):
        left_foot_geoms = [26]                     # 左腳
        right_foot_geoms = [23]                    # 右腳
        left_hand_geoms = [45, 46, 47, 48, 49, 50] # 左手
        right_hand_geoms = [33, 34, 35, 36, 37, 38]# 右手
        floor = [0]
        
        contact_groups = {
            'left_foot': False,
            'right_foot': False,
            'left_hand': False,
            'right_hand': False
        }

        for contact in self.data.contact[:self.data.ncon]:
            geom1, geom2 = contact.geom1, contact.geom2
            if geom1 not in floor and geom2 in left_foot_geoms:
                contact_groups['left_foot'] = True
            if geom1 not in floor and geom2 in right_foot_geoms:
                contact_groups['right_foot'] = True
            if geom2 in left_hand_geoms:
                contact_groups['left_hand'] = True
            if geom2 in right_hand_geoms:
                contact_groups['right_hand'] = True
        
        contact_count = sum(contact_groups.values())
        
        min_required_contacts = 3
        if contact_count >= min_required_contacts:
            return 0  
        else:
            penalty_weight = 3.0  
            penalty = penalty_weight * (min_required_contacts - contact_count)
            return penalty

    
    def _get_rew(self, z_velocity: float, action, z_position: float):
        # if z_position >1.5 and z_velocity<0:
        #     upward_reward = self._upward_reward_weight * z_velocity * 0.1
        # elif z_position > 1.5:
        #     upward_reward = self._upward_reward_weight * z_velocity
        # else:
        #     upward_reward = max(0.0, self._upward_reward_weight * z_velocity)
        if z_velocity > 0:
            upward_reward = self._upward_reward_weight * z_velocity #+ 0.1*z_position
        else:
            upward_reward = self._upward_reward_weight * z_velocity * 0.1 #+ 0.1*z_position
        
        # head_geom_index = self.model.geom_name2id("head")  
        # head_position = self.data.geom_xpos[head_geom_index]
        # print(head_position)
        # geom_names = [self.model.geom_names[i] for i in range(self.model.ngeom)]
        # print('***')
        # print(dir(self.data))
        # print(len(self.data.geom_xpos))
        # print(self.model.ngeom)
        # print(self.data.geom_xpos[17])
        # print(dir(self.model))
        # for i in range(self.model.ngeom):
        #     # print(self.model.geom_names[i])
        #     print(i, self.model.geom(i).name)
        # print(dir(self.model.geom))
        # print(self.model.geom_pos[17])
        # print(self.model.geom_name2id('head'))
        # upward_reward = max(0.0, self._upward_reward_weight * z_velocity)

        # left_foot: 26
        # right_foot: 23

        # finger_mid_l1: 45
        # finger_mid_l2: 46
        # finger_mid_l3: 47
        # finger_tip_l1: 48
        # finger_tip_l2: 49
        # finger_tip_l3: 50

        # finger_mid_r1: 33
        # finger_mid_r2: 34
        # finger_mid_r3: 35
        # finger_tip_r1: 36
        # finger_tip_r2: 37
        # finger_tip_r3: 38

        # floor: 2

        healthy_reward = self.healthy_reward
        contact_penalty = self._contact_penalty()
        rewards = upward_reward + healthy_reward 
        # print('@@@')
        # print(f"upward_reward: {upward_reward}, healthy_reward: {healthy_reward}, ")
        # print(0.1*z_position)
        # print('####')
        # print(self.data.contact)
        # print('***')
        # for contact in self.data.contact[:self.data.ncon]:
        #     geom1 = contact.geom1
        #     geom2 = contact.geom2
        #     print(f"Contact between {self.model.geom(geom1).name} and {self.model.geom(geom2).name}")
        # print('&&&')

        ctrl_cost = self.control_cost(action)
        # contact_cost = self.contact_cost
        # costs = ctrl_cost + contact_cost
        costs = ctrl_cost

        reward = rewards - contact_penalty #- costs

        reward_info = {
            "reward_survive": healthy_reward,
            "reward_upward": upward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_penalty,
            # "reward_contact": -contact_cost,
            # "sensor_data": self.data.sensordata.copy()
        }

        return reward, reward_info

    def reset_model(self):
        if self.keyframe is None:
            return super().reset_model()
        # print(f"Resetting model to keyframe '{self.keyframe}'")
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        key_qpos = self.model.key(self.keyframe).qpos
        key_qvel = self.model.key(self.keyframe).qvel

        qpos = key_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = key_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation


    def render(self):
        return super().render()
