import gymnasium as gym

# Create a MuJoCo environment
env = gym.make("Ant-v4")  # You can choose other environments like 'HalfCheetah-v4' or 'Humanoid-v4'

# Reset the environment
observation, info = env.reset()

# Take a random action
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)

print("Observation:", observation)
print("Reward:", reward)
