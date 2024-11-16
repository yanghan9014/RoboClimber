import gymnasium as gym
from gymnasium.wrappers import RecordVideo
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
import time
from climb.envs.envs_utils import register_custom_envs
import os
import pdb
import numpy as np

# class RL_Trainer(object):
#     def __init__(self, param):
#         register_custom_envs('Climber-v1')
#         self.env = gym.make('Climber-v1', render_mode="rgb_array")
    # def run_training_loop(self, n_iter, collect_policy, eval_policy,
    #                     initial_expertdata=None, relabel_with_expert=False,
    #                     start_relabel_with_expert=1, expert_policy=None)

env_name = 'Climber-v0'
register_custom_envs(env_name)
xml_file = os.path.abspath('assets/climber_v0.xml')
env = gym.make(env_name, xml_file=xml_file, render_mode="rgb_array")
# env = gym.make("Humanoid-v5", render_mode="rgb_array")
# env = DummyVecEnv([lambda: env])  # Wrap for stable-baselines3 compatibility

# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=1000000)
# model.save("ppo_humanoid_walk")
env.close()



# model = PPO.load("ppo_humanoid_walk")
# env = gym.make("Humanoid-v4", render_mode="rgb_array")

env = RecordVideo(env, "videos/", episode_trigger=lambda x: True)  # Save every episode
obs, _ = env.reset()
done = False
target_joint_angles = np.zeros(env.action_space.shape)
target_joint_angles[0] = -5.0  # 膝蓋輕微彎曲
target_joint_angles[6] = -5.0  # 膝蓋輕微彎曲
target_joint_angles[10] = -5.0  # 膝蓋輕微彎曲

while not done:
    # action, _ = model.predict(obs, deterministic=True)
    # action = env.action_space.sample()
    # print(env.action_space)
    action = target_joint_angles
    obs, reward, done, truncated, info = env.step(action)
    # time.sleep(0.01)  # Delay to make the rendering viewable in real-time
    # env.render()

env.close()