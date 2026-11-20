from stable_baselines3 import PPO
from environment import PortfolioEnv
import pandas as pd
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np
from evaluate import evaluate_model

VERSION = 'vanilla'
SEED = [0,1,2]

# environment
df = pd.read_csv('train_tuning_dataset.csv', parse_dates=['Date'])
env = PortfolioEnv(df, VERSION, 'training')

config = {
    "version": VERSION,
    "seed": SEED,
    "total_timesteps": 200000, 
    "learning_rate": [3e-4, 1e-4],
    "n_steps": 2048,
    "batch_size": 64, 
    "gamma": 0.99,
    "lookback": env.lookback_window,
    "cost_rate": env.cost_rate,
    "episode_length": env.episode_length,
    "target_kl": [None, 0.03],
    "train_end": '2019-12-31'
}

for lr in config['learning_rate']:
    for kl in config['target_kl']:
        for seed in SEED:

            env = PortfolioEnv(df, VERSION, 'training')
            run_config = {**config, "learning_rate": lr, "target_kl": kl, "seed": seed}

            run = wandb.init(
                project="regime-ppo-tuning-v2",
                job_type='tuning',
                group=f"tuning-{VERSION}",            
                name=f"{VERSION}-lr{lr}-kl{kl}-seed{seed}",
                config=run_config,
                sync_tensorboard=True,                 
            )

            model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=lr,
                n_steps=config["n_steps"],
                batch_size=config["batch_size"],
                gamma=config['gamma'],
                seed=seed,
                device="cpu",
                verbose=1,
                tensorboard_log=f"runs/{run.id}",
                target_kl = kl
            )

            model.learn(total_timesteps=config["total_timesteps"], callback=WandbCallback(verbose=2))
            
            path = f"tune_{VERSION}_lr{lr}_kl{kl}_seed{seed}.zip"
            model.save(path)

            sharpe, final_value = evaluate_model(VERSION, seed, 'val_dataset.csv', path)
            run.summary.update({"val/sharpe": float(sharpe), "val/final_value": float(final_value)})

            run.finish()