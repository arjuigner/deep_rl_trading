'''
In this file we will create a "perfect" strategy that will be used to compare the performance of the RL agent.
The strategy assumes that the last indicator is the 'LA' indicator, which contains the "future" return of the asset.
This indicator is obviously cheating and would not be available in a real-world scenario, but we add it here for testing purpose,
and the RL agent should be able to get a similar performance to this strategy when given this look-ahead indicator.
'''

from trading_env import *
import numpy as np

if __name__ == '__main__':
    # Get data and process it
    TICKERS = ['MSFT', 'QCOM']
    #TICKERS = ['AAPL', 'MSFT', 'QCOM', 'IBM', 'RTX', 'PG', 'GS', 'NKE', 'DIS', 'AXP']
    data = get_processed_data(TICKERS, '2010-01-01', '2014-01-01', '1d')
    data = add_indicators(data, indicators=['LA'])
 
    # Initialize environment
    env = TradingEnv(data, cash=10000, k_max=100, t0=0) # t0=14 because it is the size of the window for SMA and RSI
    state, info = env.reset()
    done = False

    while not done:
        # Find what stock to buy (highest return) and which ones to sell (negative returns)
        future_returns = state[-env.n_stocks:]
        if np.isnan(future_returns).any():
            print("End of episode: future returns contain NaN values")
            break
        best_performing = np.argmax(future_returns)
        neg_returns = future_returns < 0
        
        # Perform the actions
        action = np.zeros(env.n_stocks, dtype=np.float32)
        action[best_performing] = 1
        action[neg_returns] = -1 # Note: even if the future returns of best_performing was negative, this line will make us sell this stock and effectively "ignore" the previous line
        
        next_state, reward, done, _, info = env.step(action)
        state = next_state
        
    print(f"Final return: {env.get_portfolio_return()}")