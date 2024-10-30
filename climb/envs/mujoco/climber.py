import gymnasium as gym
from gymnasium.envs.mujoco import HumanoidEnv
import numpy as np

class CustomHumanoidEnv(HumanoidEnv):
    def __init__(self):
        # Optionally set up a custom model file path for the humanoid
        # Otherwise, it will use the default humanoid.xml
        model_path = "path_to_your_custom_humanoid.xml"
        super().__init__()
        # super().__init__(model_path=model_path)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        
        # Customize the reward if needed
        # For example, adding a custom reward for maintaining an upright position
        upright_bonus = np.clip(self.data.qpos[2], 1.0, 1.2)
        reward += upright_bonus
        
        # Optional: customize the done condition
        done = done or (self.data.qpos[2] < 1.0)  # Done if humanoid falls
        
        return obs, reward, done, truncated, info

    def reset(self):
        obs = super().reset()
        return obs

    def render(self, mode="human"):
        # Render the environment, can be extended or left as default
        return super().render(mode=mode)
