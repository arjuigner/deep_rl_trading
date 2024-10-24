from env import *

import matplotlib.pyplot as plt

def random_trading(env):
    returns = []
    env.start()

    done = False
    while not done:
        action = np.random.normal(0, 1/3, size=n_stocks) #99% probability of not buying/selling K_max 
        action = np.clip(action, -1, 1)
         
        state, reward, done = env.step(action)
        returns.append(env.get_portfolio_return())
    return returns

if __name__ == '__main__':
    tickers = ['MSFT', 'QCOM']
    n_stocks = len(tickers)
    data = get_stock_data(tickers, '2011-01-01', '2017-01-01', '1d')
    data = timestamps_processing(data)
    data = make_adjusted(data)
    
    env = TradingEnvironment(data, cash=1000, K_max=100, indicators=True, t0=14) #t0=14 because size of window for SMA and RSI
    returns = random_trading(env)

    # Buy and hold cumulative returns:
    buy_and_hold_msft_returns = data[('Adj Close', 'MSFT')] / data[('Adj Close', 'MSFT')].iloc[0]
        
    print(env.log)
    plt.plot(data.index[env.t0+1:], returns, color='red')
    plt.plot(data.index, buy_and_hold_msft_returns, color='blue')
    plt.legend(['Random Trading', 'Buy and Hold MSFT'])
    plt.show()