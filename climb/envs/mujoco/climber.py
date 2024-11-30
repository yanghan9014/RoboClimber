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
        # print(f"xyz_position_before: {xyz_position_before}")
        self.do_simulation(action, self.frame_skip)
        xyz_position_after = mass_center(self.model, self.data)

        xyz_velocity = (xyz_position_after - xyz_position_before) / self.dt
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

    def _get_rew(self, z_velocity: float, action, z_position: float):
        # if z_position >1.5 and z_velocity<0:
        #     upward_reward = self._upward_reward_weight * z_velocity * 0.1
        # elif z_position > 1.5:
        #     upward_reward = self._upward_reward_weight * z_velocity
        # else:
        #     upward_reward = max(0.0, self._upward_reward_weight * z_velocity)
        if z_velocity > 0:
            upward_reward = self._upward_reward_weight * z_velocity
        else:
            upward_reward = self._upward_reward_weight * z_velocity * 0.1
        # upward_reward = max(0.0, self._upward_reward_weight * z_velocity)
        healthy_reward = self.healthy_reward
        rewards = upward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        # contact_cost = self.contact_cost
        # costs = ctrl_cost + contact_cost
        costs = ctrl_cost

        reward = rewards #- costs

        reward_info = {
            "reward_survive": healthy_reward,
            "reward_upward": upward_reward,
            "reward_ctrl": -ctrl_cost,
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
