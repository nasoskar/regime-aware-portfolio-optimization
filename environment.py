import gymnasium as gym
import pandas as pd
import numpy as np
from scipy.special import softmax
from stable_baselines3.common.env_checker import check_env

class PortfolioEnv(gym.Env): # inherits from gymnasium
    def __init__(self, df, version, flag, regime_window=1):
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
        self.regime_window = regime_window

        self.flag = flag # either training or testing
        if self.flag == 'training': 
            self.episode_length = 252 # one year
        elif self.flag == 'testing':
            self.episode_length = len(self.dates) 
        else:
            raise ValueError("Flag not supported. Use training or testing.")

        if self.version == 'vix_only':
            self.regime = df[['vix_scaled']].to_numpy()
        elif self.version == 'soft_probabilities':
            self.regime = df[['p_state0', 'p_state1', 'p_state2']].to_numpy()
        elif self.version == 'hard_labels':
            self.regime = df[['hard_state_0', 'hard_state_1', 'hard_state_2']].to_numpy()
        elif self.version == 'vanilla':
            self.regime = np.empty((len(df), 0))
        else:
            raise ValueError("Version not supported. Use: vanilla, vix_only, hard_labels, soft_probabilities")

        self.no_obs += self.regime.shape[1] * self.regime_window

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32) # continuous values
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.no_obs,), dtype=np.float32)

    def get_observation(self):
        
        regime_part = self.regime[self.t - self.regime_window + 1 : self.t + 1].flatten()
        r = np.concatenate([self.obs_returns[self.t - self.lookback_window : self.t].flatten(),
        self.weights, regime_part])
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
        probabilities = softmax(np.log(self.weights + 1e-8) + action * 0.5)
        probabilities = np.minimum(probabilities, 0.4)
        probabilities = probabilities / probabilities.sum()
        
        traded_volume = np.abs(probabilities - self.weights).sum()
        info['turnover'] = traded_volume
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

