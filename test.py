import pandas as pd
from stable_baselines3 import PPO
from environment import PortfolioEnv      

VERSION = 'vanilla'
df = pd.read_csv('test_dataset.csv', parse_dates=['Date'])

env = PortfolioEnv(df, VERSION, 'testing')
model = PPO.load(f'ppo_{VERSION}', device='cpu')


# 3. Play through the whole test period
obs, info = env.reset()
print(env.dates[env.t], env.dates[-1])     # first and last day the agent actually plays
value = 1.0
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    value *= 1 + reward
    done = terminated or truncated

# 4. Print the result
print('Final value:', value)


import numpy as np

obs, info = env.reset()
eq_value = 1.0
done = False
while not done:
    action = np.zeros(env.n_assets, dtype=np.float32)    # equal weights
    obs, reward, terminated, truncated, info = env.step(action)
    eq_value *= 1 + reward
    done = terminated or truncated

print('Equal-weight final value:', eq_value)