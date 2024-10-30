import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import time

env = gym.make("Humanoid-v4", render_mode="rgb_array")
env = DummyVecEnv([lambda: env])  # Wrap for stable-baselines3 compatibility

model = PPO("MlpPolicy", env, verbose=1)
# model = PPO.load("ppo_humanoid_walk")

model.learn(total_timesteps=5000000)

model.save("ppo_humanoid_walk")
env.close()


model = PPO.load("ppo_humanoid_walk")
# env = gym.make("Humanoid-v4", render_mode="human")

env = RecordVideo(env, "videos/", episode_trigger=lambda x: True)  # Save every episode
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    time.sleep(0.01)  # Delay to make the rendering viewable in real-time

env.close()