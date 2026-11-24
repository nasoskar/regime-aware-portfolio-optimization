from stable_baselines3 import PPO
from environment import PortfolioEnv
import pandas as pd
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np

VERSION = 'vix_only'
SEED = [0,12,25,38,42]

# environment
df = pd.read_csv('train_dataset.csv', parse_dates=['Date'])
env = PortfolioEnv(df, VERSION, 'training')

config = {
    "version": VERSION,
    "seed": SEED,
    "total_timesteps": 200000, 
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64, 
    "gamma": 0.99,
    "lookback": env.lookback_window,
    "cost_rate": env.cost_rate,
    "episode_length": env.episode_length,
    "target_kl": 0.03,
    "log_scale": 0.5,
    "weight_cap": 0.4
}

for seed in SEED:

    env = PortfolioEnv(df, VERSION, 'training')
    run_config = {**config, "seed": seed}

    run = wandb.init(
        project="regime-ppo-training",
        job_type='training',
        group=f"training-{VERSION}",            
        name=f"{VERSION}-seed{seed}",
        config=run_config,
        sync_tensorboard=True,                 
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        gamma=config['gamma'],
        seed=seed,
        device="cpu",
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        target_kl = config['target_kl']
    )

    model.learn(total_timesteps=config["total_timesteps"], callback=WandbCallback(verbose=2))
    
    path = f"train/final{VERSION}_seed{seed}.zip"
    model.save(path)

    run.finish()