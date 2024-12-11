'''
Train a TD3 model using stable-baselines3 library and the environment in TradingEnvContActionPtf.py.
In this environment, the action directly represents the new portfolio, instead of the stocks to buy/sell from
the previous portfolio.
'''

import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from TradingEnvContActionPtf import TradingEnvContActionPtf, GraphCallback


def download_data(tickers, start_date, end_date):
    '''
    Download data from Yahoo Finance, and return a pandas DataFrame with a MultiIndex
    '''
    data = yf.download(tickers, start=start_date, end=end_date)
    if len(tickers) == 1:
        # Add a subcolumn to each column called stock_name
        # so 'Open' becomes ('Open', '<stock name>'), etc.
        # This way the returned dataframe is similar to when there are multiple tickers.
        data.columns = pd.MultiIndex.from_product([data.columns, tickers], names=['Price', 'Ticker'])       
    return data

def data2features(data):
    # For now features are closing prices (normalized by initial price) and returns
    prices = data['Adj Close']
    features = pd.DataFrame(index=prices.index)
    for ticker in prices.columns:
        #features[f'Price {ticker}'] = prices[ticker] / prices[ticker].iloc[0]
        features[f'Return {ticker}'] = prices[ticker].pct_change()
        #features[f'Return {ticker}'] = prices[ticker].pct_change().shift(-1) # TODO: REMOVE CHEATING INDICATOR
        
    features.dropna(inplace=True)
    print(features.head())
    print(prices.head())
    return features

def cheating_test():
    '''
    Test the environment with a simple strategy that cheats by using future data.
    The strategy buys if the next return is positive, and sells if it is negative.
    This also serves as a baseline for the RL model (in the case where I give future returns
    as inputs to the model).
    '''
    tickers = ['AAPL']
    start_date = '2011-01-01'
    end_date = '2016-12-31'
    data = download_data(tickers, start_date, end_date)
    features = data2features(data)
    prices = data.loc[features.index, 'Adj Close']
    
    env_params = {
        'prices': prices,
        'features': features,
        'initial_cash': 1000,
        'k_max':100,
        'reward': 'RelativeReturn'
    }
    
    env = TradingEnvContActionPtf(**env_params)
    obs, _ = env.reset()
    done = False
    while not done:
        action = np.zeros(env.n_stocks)
        for i, ticker in enumerate(env.prices.columns):
            action[i] = 1 if obs[i] > 0 else 0
        obs, rewards, done, _, _ = env.step(action)
        
    print(env.log)
                
    # Print final return and sharpe ratio
    returns = env.log['Portfolio Value'].drop(index=env.log.index[0]).pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    final_return = env.log['Portfolio Value'].iloc[-1] / env.initial_cash - 1
    print(f'Excess return: {final_return}, Sharpe ratio: {sharpe_ratio}')
            
    # Plot ptf value, actions, effective actions, shares/cash, reward
    fig, ax = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    ax[0].plot(env.prices.index, env.log['Portfolio Value'] / env.initial_cash - 1, label='Cheating')
    ax[0].plot(env.prices.index, env.prices.mean(axis=1) / env.prices.iloc[0].mean() - 1, linestyle='--', label='Buy and hold (No limit on share #)')
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
    fig.tight_layout()
    plt.show()

def train(env, total_timesteps=10000):
    '''
    Train a TD3 model on the environment env.
    '''
    action_noise = NormalActionNoise(mean=np.zeros(env.n_stocks), sigma=0.1 * np.ones(env.n_stocks))
    model = TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "episode"),
        gradient_steps=-1,
        policy_kwargs=dict(net_arch=[256, 256])
    )
    
    # Print updates after every episode, and plot data during training
    model.learn(total_timesteps=total_timesteps,                 
                log_interval=1,
                callback=GraphCallback(env))
    return model

def test(env, model, plot_name=None):
    '''
    Test the model on the environment, and plot the results.
    If plot_name is not None, save the plot to a file with that name.
    '''
    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, _, _ = env.step(action)
        
    print(env.log.head(15))
    print(env.log.tail(15))
                
    # Print final return and sharpe ratio
    returns = env.log['Portfolio Value'].drop(index=env.log.index[0]).pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    final_return = env.log['Portfolio Value'].iloc[-1] / env.initial_cash - 1
    print(f'Excess return: {final_return}, Sharpe ratio: {sharpe_ratio}')
            
    # Plot ptf value, actions, effective actions, shares/cash, reward
    fig, ax = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    ax[0].plot(env.prices.index, env.log['Portfolio Value'] / env.initial_cash - 1, label='TD3')
    ax[0].plot(env.prices.index, env.prices.mean(axis=1) / env.prices.iloc[0].mean() - 1, linestyle='--', label='Buy and hold (No limit on share #)')
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
    
    if plot_name:
        fig.savefig(plot_name)
    return env.log

def main():
    # Download data
    tickers = ['AAPL']
    start_date = '2011-01-01'
    end_date = '2016-01-01'
    data = download_data(tickers, start_date, end_date)
    
    # Create features and make sure prices have the same index
    features = data2features(data)
    prices = data.loc[features.index, 'Adj Close']
    
    TRAIN = True
    
    env_params = {
        'prices': prices,
        'features': features,
        'initial_cash': 1000,
        'k_max':100,
        'reward': 'RelativeReturn',
        'normalize_state': True
    }
    
    env = TradingEnvContActionPtf(**env_params)
    if TRAIN:
        # Train the model
        model = train(env, total_timesteps=50000)
        model.save('td3_trading')
    
    # Test on training data
    model = TD3.load('td3_trading')
    env = TradingEnvContActionPtf(**env_params)
    test(env, model, plot_name='td3_E2_RetOnly_RelRet_AAPL_train.png')
    
    # Test on test data
    start_date = '2016-01-01'
    end_date = '2017-12-31'
    data = download_data(tickers, start_date, end_date)
    features = data2features(data)
    prices = data.loc[features.index, 'Adj Close']
    
    env_params['prices'] = prices
    env_params['features'] = features
    env = TradingEnvContActionPtf(**env_params)
    test(env, model, plot_name='td3_E2_RetOnly_RelRet_AAPL_test.png')
    
if __name__ == '__main__':
    main()