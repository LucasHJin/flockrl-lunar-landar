import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

env = gym.make("LunarLander-v3", render_mode="rgb_array") # Use human render mode to visualize, rgb to save video
env = RecordVideo(env, video_folder="videos/", name_prefix="lander_run")

model = PPO.load("ppo-LunarLander-v3")

for episode in range(5):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close()