import gymnasium as gym
import pandas as pd
import numpy as np

class PortfolioEnv(gym.Env): # inherits from gymnasium
    def __init__(self, df, version):
        super().__init__()
        self.lookback_window = 20
        self.cost_rate = 0.001
        self.episode_length = 252 # one year
        self.raw_returns = df[['SPY', 'EFA', 'VNQ', 'BIL', 'TLT', 'LQD', 'GLD', 'DBC']]
        self.version = version
        self.obs_returns = self.raw_returns * 100 
        self.no_obs = self.lookback_window * self.raw_returns.shape[1] + self.raw_returns.shape[1]  

        if self.version == 'vix_only':
            self.regime = df[['Date', 'VIX']]
            self.no_obs += 1
        elif self.version == 'soft_probabilities':
            self.regime = df[['Date', 'p_state0', 'p_state1', 'p_state2']]
            self.no_obs += 3
        elif self.version == 'hard_labels':
            self.regime = df[['Date', 'hard_state_0', 'hard_state_1', 'hard_state_2']]
            self.no_obs += 3
        elif self.version == 'vanilla':
            self.regime = df[['Date']]
        else:
            raise NameError("Version not supported. Use: vanilla, vix_only, hard_labels, soft_probabilities")

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.no_obs,), dtype=np.float32)

    def reset(self, seed):
        super().reset(seed=seed)
        starting_day = df['Date'].sample()
        starting_weights = [1/8 for item in self.raw_returns]
        first_observation = #TODO
        observation = 

        return observation, reward, terminated, truncated, info

    def step(self, action):
        return next_obs, reward

df = pd.read_csv('full_dataset.csv')
env = PortfolioEnv(df, 'vix_only')

observation, info = env.reset()
total_reward = 0

while True:
    action = env.action_space.sample() # take a random action
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    print("Observation:", observation)
    print("Reward:", reward)

    if terminated or truncated:
        print('Episode finished')
        print(f'Total reward: {total_reward}')
        break

env.close()
