import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Source -> https://huggingface.co/learn/deep-rl-course/en/unit1/hands-on

eval_env = Monitor(gym.make("LunarLander-v3", render_mode='rgb_array'))
model = PPO.load("ppo-LunarLander-v3")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100, deterministic=True)
print(f"mean_reward = {mean_reward:.2f} +/- {std_reward:.2f}")