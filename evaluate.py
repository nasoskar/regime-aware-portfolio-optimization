from stable_baselines3 import PPO
from environment import PortfolioEnv
import pandas as pd
import torch
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np

def evaluate_model(VERSION, SEED, filename, filepath):

    model = PPO.load(filepath, device='cpu')
    df = pd.read_csv(filename, parse_dates=['Date'])
    env = PortfolioEnv(df, VERSION, 'testing')
    obs, info = env.reset()
    print(env.dates[env.t], env.dates[-2]) # first and last day the agent actually plays
    done = False
    returns = []
    all_weights = []   

    while not done: 
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        all_weights.append(env.weights.copy())
        returns.append(reward) 
        done = terminated or truncated

    all_weights = np.array(all_weights) 
    print("average weight per asset:", all_weights.mean(axis=0).round(3))
    print("average daily turnover:", np.abs(np.diff(all_weights, axis=0)).sum(axis=1).mean().round(3))
    returns = np.array(returns)
    sharpe_ratio = returns.mean()/returns.std() * np.sqrt(252)
    print(len(returns), np.prod(1 + returns), sharpe_ratio)

    return sharpe_ratio, float(np.prod(1 + returns))

def evaluate_model_equal(VERSION, SEED, filename):

    df = pd.read_csv(filename, parse_dates=['Date'])
    env = PortfolioEnv(df, VERSION, 'testing')
    obs, info = env.reset()
    print(env.dates[env.t], env.dates[-2]) # first and last day the agent actually plays
    done = False
    returns = []

    while not done: 
        obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape[0]))
        returns.append(reward) 
        done = terminated or truncated

    returns = np.array(returns)
    sharpe_ratio = returns.mean()/returns.std() * np.sqrt(252)
    print(len(returns), np.prod(1 + returns), sharpe_ratio)

    return sharpe_ratio, float(np.prod(1 + returns))

evaluate_model('vanilla', 0, 'val_dataset.csv', 'length_test_ckpts_logscale05/vanilla_seed0_100000_steps.zip')


