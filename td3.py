'''
Train a TD3 model using stable-baselines3 library and the environment in TradingEnvContinuousAction.py
'''

import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from TradingEnvContinuousAction import TradingEnvContinuousAction, GraphCallback


def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data

def data2features(data):
    # For now features are closing prices (normalized by initial price) and returns
    prices = data['Adj Close']
    features = pd.DataFrame(index=prices.index)
    for ticker in prices.columns:
        features[f'Price {ticker}'] = prices[ticker] / prices[ticker].iloc[0]
        features[f'Return {ticker}'] = prices[ticker].pct_change()
        
    features.dropna(inplace=True)
    return features

def train(env, total_timesteps=10000):
    action_noise = NormalActionNoise(mean=np.zeros(env.n_stocks), sigma=0.4 * np.ones(env.n_stocks))
    model = TD3('MlpPolicy', env, 
                action_noise=action_noise,
                learning_starts=total_timesteps // 5,
                verbose=1)
    
    # Print updates after every episode, and plot data during training
    callback = GraphCallback(env)
    model.learn(total_timesteps=total_timesteps,                 
                log_interval=1, 
                callback=callback)
    return model

def test(env, model):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, _, _ = env.step(action)
        
    print(env.log)
                
    # Print final return and sharpe ratio
    returns = env.log['Portfolio Value'].drop(index=env.log.index[0]).pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    final_return = env.log['Portfolio Value'].iloc[-1] / env.initial_cash - 1
    print(f'Excess return: {final_return}, Sharpe ratio: {sharpe_ratio}')
            
    # Plot ptf value, actions, effective actions, shares/cash, reward
    fig, ax = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    ax[0].plot(env.prices.index, env.log['Portfolio Value'] / env.initial_cash - 1, label='TD3')
    ax[0].plot(env.prices.index, env.prices.mean(axis=1) / env.prices.iloc[0].mean() - 1, linestyle='--', label='Buy and hold')
    ax[0].set_title('Cumulative Return')
    ax[0].legend()
    
    for ticker in env.prices.columns:
        ax[1].plot(env.prices.index, env.log[f'Action {ticker}'], label=ticker)
        ax[2].plot(env.prices.index, env.log[f'Eff Action {ticker}'], label=ticker)
        ax[3].plot(env.prices.index, env.log[f'Shares {ticker}'], label=ticker)
    cash_ax = ax[3].twinx()
    cash_ax.plot(env.prices.index, env.log['Cash'], color='black', linestyle='--', label='Cash')
    ax[1].set_title('Actions')
    ax[2].set_title('Effective Actions')
    ax[3].set_title('Shares/cash')
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    
    ax[4].plot(env.prices.index, env.log['Reward'])
    ax[4].set_title('Reward')
    
    plt.show()
    return env.log

def main():
    tickers = ['AAPL', 'MSFT']
    start_date = '2011-01-01'
    end_date = '2016-12-31'
    data = download_data(tickers, start_date, end_date)
    features = data2features(data)
    prices = data.loc[features.index, 'Adj Close']
    
    TRAIN = True
    
    env = TradingEnvContinuousAction(prices, features, initial_cash=10000, k_max=100)
    if TRAIN:
        # Train the model
        model = train(env, total_timesteps=75000)
        model.save('td3_trading')
    
    model = TD3.load('td3_trading')
    env = TradingEnvContinuousAction(prices, features, initial_cash=10000, k_max=100)
    test(env, model)
    
if __name__ == '__main__':
    main()