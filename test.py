import pandas as pd
from stable_baselines3 import PPO
from environment import PortfolioEnv      
import numpy as np

VERSION = ['soft_probabilities', 'hard_labels', 'vix_only']
SEED = [0,12,25,38,42]

rows = []     

for version in VERSION:
    for seed in SEED:

        df = pd.read_csv('test_dataset.csv', parse_dates=['Date'])

        env = PortfolioEnv(df, version, 'testing', regime_window=20)
        model = PPO.load(f'train/final{version}_regimehist_seed{seed}.zip', device='cpu')

        obs, info = env.reset()
        print(env.dates[env.t], env.dates[-1])  # first and last day the agent actually plays
        value = 1.0
        done = False
        returns = []
        turnovers = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            value *= 1 + reward
            done = terminated or truncated
            returns.append(reward)
            turnovers.append(info['turnover'])

        turnover = np.mean(turnovers)
        returns = np.array(returns)
        cum = np.cumprod(1 + returns)
        sharpe = returns.mean()/returns.std()*np.sqrt(252)
        max_drawdown = (cum / np.maximum.accumulate(cum) - 1).min()
        
        rows.append({'name': f'{version}_regimehist', 'seed': seed, 'sharpe': sharpe,
        'final_value': value, 'max_drawdown': max_drawdown, 'turnover': turnover})

        pd.DataFrame(rows).to_csv('test_results_regimehist.csv', index=False)

########### BASELINES #############

# 1. Equal-weight
'''
env = PortfolioEnv(df, 'vanilla', 'testing', regime_window=20)
obs, info = env.reset()
returns = []
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(np.zeros(env.n_assets, dtype=np.float32))
    returns.append(reward)
    done = terminated or truncated
returns = np.array(returns)
cum = np.cumprod(1 + returns)
rows.append({'name': 'equal_weight_hold', 'seed': 0,
             'sharpe': returns.mean() / returns.std() * np.sqrt(252),
             'final_value': cum[-1],
             'max_drawdown': (cum / np.maximum.accumulate(cum) - 1).min(),
             'turnover': np.nan})

# 2. SPY, buy and hold

spy = env.raw_returns[env.lookback_window : env.lookback_window + len(returns), 0]
cum = np.cumprod(1 + spy)
rows.append({'name': 'spy_hold', 'seed': 0,
             'sharpe': spy.mean() / spy.std() * np.sqrt(252),
             'final_value': cum[-1],
             'max_drawdown': (cum / np.maximum.accumulate(cum) - 1).min(),
             'turnover': np.nan})

results = pd.DataFrame(rows)
results.to_csv('test_results.csv', index=False)

summary = results.groupby('name')[['sharpe', 'final_value', 'max_drawdown', 'turnover']].agg(['mean', 'std'])
print(summary)
summary.to_csv('test_summary.csv')
'''