import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv

LEFT_FOOT_GEOMS = [26]                     # left foot
RIGHT_FOOT_GEOMS = [23]                    # right foot
LEFT_HAND_GEOMS = [45, 46, 47, 48, 49, 50] # left hand
RIGHT_HAND_GEOMS = [33, 34, 35, 36, 37, 38]# right hand
FLOOR_GEOMS = [0]

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
        self._upward_reward_weight = 50.0
        self._down_reward_weight = 5.0
        self._invalid_contact_penalty_weight = 3.0
        self._three_contacts_constraint_weight = 3.0
        self.keyframe = keyframe
    
    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2]
        return is_healthy

    def step(self, action):
        self.xyz_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        self.xyz_position_after = mass_center(self.model, self.data)
        
        xyz_velocity = (self.xyz_position_after - self.xyz_position_before) / self.dt
        x_velocity, y_velocity, z_velocity = xyz_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
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

    def _step_on_reward(self):
        reward = 0
        for contact in self.data.contact[:self.data.ncon]:
            geom1, geom2 = contact.geom1, contact.geom2
            if geom1 not in FLOOR_GEOMS and geom2 in LEFT_FOOT_GEOMS:
                # reward +=  self.data.xipos[10][2] #mass_center(self.model, self.data)[2]
                reward +=  mass_center(self.model, self.data)[2]

            if geom1 not in FLOOR_GEOMS and geom2 in RIGHT_FOOT_GEOMS:
                # reward +=  self.data.xipos[7][2] #mass_center(self.model, self.data)[2]
                reward +=  mass_center(self.model, self.data)[2]
        return reward


    def _upward_reward(self, z_velocity):
        sign_z_velocity = np.sign(z_velocity)
        upward_reward = self._upward_reward_weight * z_velocity * max(0, sign_z_velocity) + self._down_reward_weight * z_velocity * (1-max(0, sign_z_velocity))

        return upward_reward
    
    def _contact_penalty(self):
        allowed_contact_geoms = LEFT_FOOT_GEOMS + RIGHT_FOOT_GEOMS + LEFT_HAND_GEOMS + RIGHT_HAND_GEOMS
        num_invalid_contact = 0
    
        contact_groups = {
            'left_foot': False,
            'right_foot': False,
            'left_hand': False,
            'right_hand': False
        }
        valid_contact_count = 0
        for contact in self.data.contact[:self.data.ncon]:
            geom1, geom2 = contact.geom1, contact.geom2
            if geom1 not in FLOOR_GEOMS and geom2 in LEFT_FOOT_GEOMS:
                contact_groups['left_foot'] = True
            if geom1 not in FLOOR_GEOMS and geom2 in RIGHT_FOOT_GEOMS:
                contact_groups['right_foot'] = True
            if geom2 in LEFT_HAND_GEOMS:
                contact_groups['left_hand'] = True
            if geom2 in RIGHT_HAND_GEOMS:
                contact_groups['right_hand'] = True
            if geom2 not in allowed_contact_geoms:
                num_invalid_contact += 1
        
        valid_contact_count = sum(contact_groups.values())
        
        min_required_contacts = 3

        penalty = 0
        # three contacts constraint
        penalty += max(0, min_required_contacts - valid_contact_count) * self._three_contacts_constraint_weight

        # invalid contact penalty
        penalty += num_invalid_contact * self._invalid_contact_penalty_weight

        return penalty

    
    def _get_rew(self, z_velocity: float, action, z_position: float):
        upward_reward = self._upward_reward(z_velocity)
        healthy_reward = self.is_healthy 
        contact_penalty = self._contact_penalty()
        step_on_reward = self._step_on_reward()
        rewards = upward_reward + healthy_reward + step_on_reward
        ctrl_cost = self.control_cost(action)
        reward = rewards - contact_penalty 

        reward_info = {
            "reward_survive": healthy_reward,
            "reward_upward": upward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_penalty,
            "reward_step_on": step_on_reward,
        }

        return reward, reward_info

    def reset_model(self):
        if self.keyframe is None:
            return super().reset_model()
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
