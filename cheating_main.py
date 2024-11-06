'''
In this file we will train a TD3 model on the trading environment with the look ahead feature,
in order to test the setup. An almost optimal strategy is implemented in the file "cheating_baseline.py",
and we want this model to achieve similar results (or at least results way superior to random trading).
'''


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

import trading_env

train=False
test=True

# Get data and process it
TICKERS = ['MSFT', 'QCOM']
#TICKERS = ['AAPL', 'MSFT', 'QCOM', 'IBM', 'RTX', 'PG', 'GS', 'NKE', 'DIS', 'AXP']
data = trading_env.get_processed_data(TICKERS, '2010-01-01', '2014-01-01', '1d')
data = trading_env.add_indicators(data)
env = trading_env.TradingEnv(stock_data=data, cash=1000, k_max=10, t0=0)

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", env, action_noise=action_noise, learning_rate=1e-3, verbose=1)

if train:
    model.learn(total_timesteps=10000, log_interval=1)
    model.save("with_look_ahead")
    vec_env = model.get_env()

if test:
    model = TD3.load("with_look_ahead")
    env = trading_env.TradingEnv(stock_data=data, cash=1000, k_max=10, t0=0)

    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, _, info = env.step(action)
        env.render("human")