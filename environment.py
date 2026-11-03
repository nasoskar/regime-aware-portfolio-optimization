import gymnasium as gym
import pandas as pd
import numpy as np
from scipy.special import softmax
from stable_baselines3.common.env_checker import check_env

class PortfolioEnv(gym.Env): # inherits from gymnasium
    def __init__(self, df, version, flag):
        super().__init__()
        self.lookback_window = 20
        self.cost_rate = 0.001
        self.raw_returns = df[['SPY', 'EFA', 'VNQ', 'BIL', 'TLT', 'LQD', 'GLD', 'DBC']].to_numpy()
        self.version = version
        self.obs_returns = self.raw_returns * 100 
        self.no_obs = self.lookback_window * self.raw_returns.shape[1] + self.raw_returns.shape[1]  
        self.n_assets = self.raw_returns.shape[1]
        self.weights = np.full(self.n_assets, 1/self.n_assets) # equal weights in initialization
        self.dates = df['Date'].to_numpy()
        self.t = self.lookback_window

        self.flag = flag # either training or testing
        if self.flag == 'training': 
            self.episode_length = 252 # one year
        elif self.flag == 'testing':
            self.episode_length = len(self.dates) 
        else:
            raise ValueError("Flag not supported. Use training or testing.")

        if self.version == 'vix_only':
            self.regime = df[['vix_scaled']].to_numpy()
            self.no_obs += 1
        elif self.version == 'soft_probabilities':
            self.regime = df[['p_state0', 'p_state1', 'p_state2']].to_numpy()
            self.no_obs += 3
        elif self.version == 'hard_labels':
            self.regime = df[['hard_state_0', 'hard_state_1', 'hard_state_2']].to_numpy()
            self.no_obs += 3
        elif self.version == 'vanilla':
            self.regime = np.empty((len(df), 0))
        else:
            raise ValueError("Version not supported. Use: vanilla, vix_only, hard_labels, soft_probabilities")

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32) # continuous values
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.no_obs,), dtype=np.float32)

    def get_observation(self):
        
        r = np.concatenate([self.obs_returns[self.t - self.lookback_window : self.t].flatten(), self.weights, self.regime[self.t]]) 
        r = r.astype(np.float32)
        return r

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.weights = np.full(self.n_assets, 1/self.n_assets)
        if self.flag == 'training':
            self.t = self.np_random.integers(self.lookback_window, (len(self.dates) - self.episode_length - 1))  #for training pick a random start, leaving room for 252 days.
        elif self.flag == 'testing':
            self.t = self.lookback_window
        observation = self.get_observation() 
        self.steps = 0
        info = {}

        return observation, info

    def step(self, action):

        self.steps += 1
        terminated = False # a market never ends
        info = {}
        probabilities = softmax(action * 5) # streching the probabilities 
        traded_volume = np.abs(probabilities - self.weights).sum()
        trading_fee = traded_volume * self.cost_rate
        reward = np.dot(probabilities, self.raw_returns[self.t]) - trading_fee

        self.weights = probabilities
        returns = self.raw_returns[self.t]
        sizes = self.weights * (1 + returns)
        total = sizes.sum()
        self.weights = sizes / total
        self.t += 1 # move to the next day

        observation = self.get_observation() 
        truncated = bool((self.steps >= self.episode_length) or (self.t >= len(self.dates) - 1))

        return observation, reward, terminated, truncated, info



df = pd.read_csv('train_dataset.csv', parse_dates=['Date'])
print(f"Columns: {df.columns.tolist()}")
print(f"Start Date: {df['Date'].iloc[0]}")
print(f"End Date: {df['Date'].iloc[-1]}")
env = PortfolioEnv(df, 'vanilla', 'training')

env.reset(seed=0)

observation, info = env.reset()
print(f"Observation shape: {observation.shape}")

env.reset()
n = 0
truncated = False
while not truncated:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    n += 1
print(n)  # expect exactly 252

check_env(env)
print(f"Null check: {np.isnan(env.raw_returns).any(), np.isnan(env.regime).any()}")
value = 1
counter = 0

while True:
    counter += 1
    action = env.action_space.sample() # take a random action
    observation, reward, terminated, truncated, info = env.step(action)

    value *= (1 + reward)

    if counter % 50 == 0:
        print("Observation:", observation)
        print("Reward:", reward)

    if terminated or truncated:
        print('Episode finished')
        print(f'Total reward: {value}')
        break

env.close()
