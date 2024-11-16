import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv

def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass)).copy()

class Climber(HumanoidEnv):
    def __init__(self, xml_file=None, **kwargs):
        render_mode = kwargs.get('render_mode', 'rgb_array')
        if xml_file is not None:
            super().__init__(xml_file=xml_file, render_mode=render_mode)
        else:
            super().__init__(render_mode=render_mode)
        self._upward_reward_weight = self._forward_reward_weight
    
    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2]
        return is_healthy

    def step(self, action):
        print(self.data.sensordata)
        xyz_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xyz_position_after = mass_center(self.model, self.data)

        xyz_velocity = (xyz_position_after - xyz_position_before) / self.dt
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
            "sensor_data": self.data.sensordata.copy(), # r1, r2, r3, l1, l2, l3
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_rew(self, z_velocity: float, action):
        upward_reward = self._upward_reward_weight * z_velocity
        healthy_reward = self.healthy_reward
        rewards = upward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        reward_info = {
            "reward_survive": healthy_reward,
            "reward_upward": upward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
        }

        return reward, reward_info

    # def reset(self):
    #     obs = super().reset()
    #     return obs

    def render(self):
        return super().render()
