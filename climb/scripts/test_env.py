# Register the environment if you want to call it by name
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
# from climb.envs.mujoco.climber import CustomHumanoidEnv

# env = CustomHumanoidEnv()

env = gym.make("Humanoid-v4", render_mode="rgb_array")
env = RecordVideo(env, "videos/", episode_trigger=lambda x: True)  # Save every episode

obs = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # Take a random action
    obs, reward, done, truncated, info = env.step(action)
    print(f"State: {obs}, Reward: {reward}, Done: {done}")

env.close()