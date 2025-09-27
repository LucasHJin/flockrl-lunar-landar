import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v2")

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=500_000)

model.save("ppo-LunarLander-v3")

env.close()