'''
In this file we will implement a random trading strategy, and plot its returns as well
as a buy and hold strategy for comparison.
'''


from trading_env import *

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def random_trading(env):
    returns = []
    env.reset()

    done = False
    while not done:
        action = np.random.normal(0, 1/3, size=n_stocks) #99% probability of not buying/selling K_max 
        action = np.clip(action, -1, 1)
         
        state, reward, done, _, info = env.step(action)
        returns.append(env.get_portfolio_return())
    return returns

if __name__ == '__main__':
    tickers = ['MSFT', 'QCOM']
    n_stocks = len(tickers)
    data = get_processed_data(tickers, '2010-01-01', '2014-01-01', '1d')
    
    env = TradingEnv(data, cash=1000, k_max=100, t0=0)
    returns = random_trading(env)

    # Buy and hold cumulative returns:
    buy_and_hold_msft_returns = data[('Adj Close', 'MSFT')] / data[('Adj Close', 'MSFT')].iloc[0]
        
    print(env.log)
    plt.plot(data.index[env.t0+1:], returns, color='red')
    plt.plot(data.index, buy_and_hold_msft_returns, color='blue')
    plt.legend(['Random Trading', 'Buy and Hold MSFT'])
    plt.show()