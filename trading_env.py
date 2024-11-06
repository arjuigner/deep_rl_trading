import exchange_calendars as xcals
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

########### Getting data and preprocessing ###########

def _get_stock_data(ticker, start_date, end_date, granularity):
    '''
    ticker: str, stock ticker
    start_date: str, start date in format 'YYYY-MM-DD'
    end_date: str, end date in format 'YYYY-MM-DD'
    granularity: str, granularity of data, e.g. '1d', '1h', '1m'

    return: pandas DataFrame, stock data. Its columns are: 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
    '''
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=granularity)
    return stock_data


def _timestamps_processing(data):
    '''
    data: pandas DataFrame, stock data. Its index is made of dates
    return: pandas DataFrame, stock data with only days where all stock prices are available, and where the exchange was open.
    '''
    # Get the exchange calendar for the stock data
    calendar = xcals.get_calendar('NASDAQ')
    index_list = data.index.strftime('%Y-%m-%d').tolist()
    open_days = calendar.sessions_in_range(index_list[0], index_list[-1]).strftime('%Y-%m-%d').tolist()
    open_days = [day for day in open_days if day in index_list]

    # Remove days where there are NaN values
    data = data.loc[open_days].dropna()
    return data


def _make_adjusted(data):
    '''
    data: pandas DataFrame, stock data. Its columns are: 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
        with at least two subcolumns each for the stocks
    return: pandas DataFrame, stock data with columns: 'Adj Open', 'Adj High', 'Adj Low', 'Adj Close', 'Volume'

    We don't need non-adjusted data for our trading environment, so we will convert the data to be adjusted, and drop the non-adjusted columns.
    '''
    adj_data = pd.DataFrame(index=data.index)
    for stock in data['Adj Close'].columns:
        adj_data[('Adj Open', stock)]  = data[('Open', stock)] * data[('Adj Close', stock)] / data[('Close', stock)]
        adj_data[('Adj High', stock)]  = data[('High', stock)] * data[('Adj Close', stock)] / data[('Close', stock)]
        adj_data[('Adj Low', stock)]   = data[('Low', stock)]  * data[('Adj Close', stock)] / data[('Close', stock)]
        adj_data[('Adj Close', stock)] = data[('Adj Close', stock)]
        adj_data[('Volume', stock)]    = data[('Volume', stock)]
    adj_data.columns = pd.MultiIndex.from_tuples(adj_data.columns)
    return adj_data


def get_processed_data(ticker, start_date, end_date, granularity):
    '''
    ticker: str, stock ticker
    start_date: str, start date in format 'YYYY-MM-DD'
    end_date: str, end date in format 'YYYY-MM-DD'
    granularity: str, granularity of data, e.g. '1d', '1h', '1m'

    return: pandas DataFrame, stock data. Its columns are: 'Adj Open', 'Adj High', 'Adj Low', 'Adj Close', 'Volume'
    '''
    data = _get_stock_data(ticker, start_date, end_date, granularity)
    data = _timestamps_processing(data)
    data = _make_adjusted(data)
    return data


########### Indicators ###########


N_INDICATORS = 3

def _compute_sma(data, window=14):
    '''
    data: pandas DataFrame, stock data. Its columns must contain at least: 'Adj Close', and it has at least 2 sub-columns for each 
    of them, one for each stock.
    window: int, window size for the simple moving average
    return: pandas DataFrame, stock data with additional column for the simple moving average. 
    This new column is called 'SMA' and has subcolumns for each stock.
    '''
    sma_df = pd.DataFrame(index=data.index)
    for stock in data['Adj Close'].columns:
        sma_df[('SMA', stock)] = data['Adj Close'][stock].rolling(window=window).mean()
    return pd.concat([data, sma_df], axis=1)

def _compute_rsi(data, window=14):
    '''
    data: pandas DataFrame, stock data. Its columns must contain at least: 'Adj Close', 'Adj Open', and it has at least 2 sub-columns for each 
    of them, one for each stock.
    window: int, window size for the relative strength index
    return: pandas DataFrame, stock data with additional column for the simple moving average. 
    This new column is called 'RSI' and has subcolumns for each stock. Note: RSI is in [0, 100]
    '''
    rsi_df = pd.DataFrame(index=data.index)
    for stock in data['Adj Close'].columns:
        delta = data['Adj Close'][stock].diff()
        gain_series = delta.where(delta > 0, 0)
        loss_series = -delta.where(delta < 0, 0)
        # TODO: Is this really correct ? If there was only one up move in the window, avg_gain will not be the value of that up move but the up move divided by window
        avg_gain = gain_series.rolling(window=window).mean()
        avg_loss = loss_series.rolling(window=window).mean()
        rel_strength = avg_gain / avg_loss
        rsi_df[('RSI', stock)] = 100 - (100 / (1 + rel_strength))
    return pd.concat([data, rsi_df], axis=1)

def _compute_look_ahead(data):
    '''
    data: pandas DataFrame, stock data. Its columns must contain at least: 'Adj Close', and it has at least 2 sub-columns, one for each stock.
    
    return: pandas DataFrame, stock data with additional column for the look ahead indicator. 
    
    THIS IS NOT A TRUE INDICATOR, BUT A WAY TO "CHEAT" USING INFORMATION ABOUT THE FUTURE.
    This new column is called 'LA' and has subcolumns for each stock. The value of this indicator at time t is the return of the stock at time t+1.
    In other words, we are purposefully leaking information from the future. This is used to TEST the TD3 implementation, which should have no trouble
    making perfect trades with this indicator.
    '''
    la_df = pd.DataFrame(index=data.index)
    for stock in data['Adj Close'].columns:
        la_df[('LA', stock)] = data['Adj Close'][stock].pct_change().shift(-1)
    return pd.concat([data, la_df], axis=1)

def add_indicators(data, indicators=None, sma_args=(), rsi_args=(), look_ahead_args=()):
    '''
    data: pandas DataFrame, stock data. Its columns are: 'Adj Open', 'Adj High', 'Adj Low', 'Adj Close', 'Volume', and it has at least 2 sub-columns for each 
    of them, one for each stock.
    indicators: list of str or None, list of indicators to compute. If None, computes all indicators.
    sma_args: tuple, arguments for the simple moving average indicator
    rsi_args: tuple, arguments for the relative strength index indicator
    look_ahead_args: tuple, arguments for the look ahead indicator
    
    return: pandas DataFrame, stock data with additional columns for indicators. Its columns are: 'Adj Close', 
    and then one column for each indicator. The indicator columns might have subcolumns for each stock.
    '''
    
    # By default, use all indicators
    if indicators is None:
        indicators = ['SMA', 'RSI', 'LA'] #TODO: Remove LA when done testing
    
    # Compute the indicators
    if 'SMA' in indicators:
        data = _compute_sma(data, *sma_args)
        print(data.head(5))
    if 'RSI' in indicators:
        data = _compute_rsi(data, *rsi_args)
    if 'LA' in indicators:
        data = _compute_look_ahead(data, *look_ahead_args)
        
    # Remove columns 'Adj Open', 'Adj High', 'Adj Low', 'Volume' as we don't need them
    data = data.drop(columns=['Adj Open', 'Adj High', 'Adj Low', 'Volume'], level=0)
    return data.dropna()


############## Trading Environment ##############


class TradingEnv(gym.Env):
    '''
    TODO: write more...
    
    observation space: vector with following components: 
        - cash (divided by initial cash)
        - number of stocks we have for each stock (divided by k_max)
        - stock prices (divided by initial stock price) for each stock
        - indicator #1 for each stock
        - indicator #2 for each stock
        - ...
        
    action space: vector with one component for each stock. Each component is in [-1, 1], and k_max * action is the number of shares to
    buy (if positive) or sell (if negative) for each stock.
    '''
    metadata = {'render.modes': ['human']}
    
    def __init__(self, stock_data, cash=1000, k_max=10, t0=0) -> None:
        
        #TODO : I assume there are multiple stocks, thus self.stock_data['Adj Close'].iloc[self.t] is NOT a scalar value and I can access .values on it
        self.stock_data = stock_data
        
        self.n_stocks = len(self.stock_data['Adj Close'].columns) # Number of stocks
        self.t0 = t0 # Time to start at
        self.initial_cash = cash
        self.K_max = k_max # Maximum number of stocks we can be longor short
        self._max_episode_steps = len(self.stock_data) - 1 - self.t0
        
        self.t = -1 # Current time step. Currently invalid, as we haven't started yet
        self.ptf = np.zeros(self.n_stocks + 1) # Portfolio. Cash, and number of stocks we have for each stock
        self.ptf[0] = cash
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.n_stocks,)) # Action space is the weights of each stock
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1 + self.n_stocks * (2 + N_INDICATORS),))
        
        self.log = pd.DataFrame(columns=['Cash'] + \
                                        [s+' Position' for s in stock_data['Adj Close'].columns] + \
                                        ['Portfolio Value'] + \
                                        [s+' Price' for s in stock_data['Adj Close'].columns] + \
                                        [s+' Submitted Action' for s in stock_data['Adj Close'].columns],
                                index=stock_data.index)

    def _get_obs(self):
        # TODO: when batchnorm will be used, no need to normalize here anymore, so this function can be simplified a lot
        
        # Calculate new state
        state = np.concatenate((
            self.ptf, 
            self.stock_data['Adj Close'].iloc[self.t].values / self.stock_data['Adj Close'].iloc[self.t0].values, # Normalize prices by the first price
        ))
        if 'SMA' in self.stock_data.columns:
            state = np.concatenate((
                state, 
                self.stock_data['SMA'].iloc[self.t].values / self.stock_data['Adj Close'].iloc[self.t0].values, # Normalize SMA by the first price
            ))
        if 'RSI' in self.stock_data.columns:
            state = np.concatenate((
                state, 
                self.stock_data['RSI'].iloc[self.t].values / 100, # RSI (initially in [0, 100])
            ))
        if 'LA' in self.stock_data.columns:
            state = np.concatenate((
                state, 
                self.stock_data['LA'].iloc[self.t].values * 100,
            ))
        
        # Normalize portfolio values
        state[0] /= self.initial_cash           # Cash gained/lost, normalized by initial cash
        state[1:self.n_stocks+1] /= self.K_max  # Normalize number of stocks by the maximum number of stocks
            
        return state
    
    def _get_info(self):
        return {}
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.t = self.t0
        self.ptf = np.zeros(self.n_stocks + 1)
        self.ptf[0] = self.initial_cash
        
        self.log = pd.DataFrame(columns=['Cash'] + \
                                        [s+' Position' for s in self.stock_data['Adj Close'].columns] + \
                                        ['Portfolio Value'] + \
                                        [s+' Price' for s in self.stock_data['Adj Close'].columns] + \
                                        [s+' Submitted Action' for s in self.stock_data['Adj Close'].columns],
                                index=self.stock_data.index)
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        if self.t == -1:
            print("Environment not started yet. Please call reset() first.")
            return None
        
        submitted_action = action.copy()

        # Clip the selling part of the action to be achievable with current portfolio
        action = np.round(self.K_max * action)
        action = np.clip(action, -self.ptf[1:], self.K_max) # sell at most all the stocks you have

        # Cash available for buying and cash needed to do the buying part of the action
        sell_mask = action < 0
        buy_mask = action > 0
        buying_power = self.ptf[0] + np.dot(-action[sell_mask], self.stock_data['Adj Close'].iloc[self.t].values[sell_mask])
        buying_cost = np.dot(action[buy_mask], self.stock_data['Adj Close'].iloc[self.t].values[buy_mask])

        # Scale down how much we buy if we don't have enough cash
        if buying_cost > buying_power:
            action[buy_mask] = np.floor(action[buy_mask] * buying_power / buying_cost)
            buying_cost = np.dot(action[buy_mask], self.stock_data['Adj Close'].iloc[self.t].values[buy_mask])
            
        # TODO: prevent having more than K_max stocks

        # Update portfolio
        self.ptf[0] = buying_power - buying_cost
        self.ptf[1:] += action

        # Move to next time step and calculate reward
        self.t += 1
        # Reward is the relative change in portfolio value
        reward = np.dot(self.ptf[1:], self.stock_data['Adj Close'].iloc[self.t].values - self.stock_data['Adj Close'].iloc[self.t - 1].values)
        reward /= (self.ptf[0] + np.dot(self.ptf[1:], self.stock_data['Adj Close'].iloc[self.t - 1].values))

        # Check if we are at the end of our data
        done = self.t == len(self.stock_data) - 1

        # Calculate new state
        state = self._get_obs()

        # Update log
        self.log.loc[self.log.index[self.t], 'Cash'] = self.ptf[0]
        for i, stock in enumerate(self.stock_data['Adj Close'].columns):
            self.log.loc[self.log.index[self.t], stock + ' Position'] = self.ptf[i+1]
            self.log.loc[self.log.index[self.t], stock + ' Price'] = self.stock_data['Adj Close'].iloc[self.t][stock]
            self.log.loc[self.log.index[self.t], stock + ' Submitted Action'] = submitted_action[i] * self.K_max
        self.log.loc[self.log.index[self.t], 'Portfolio Value'] = self.ptf[0] + np.dot(self.ptf[1:], self.stock_data['Adj Close'].iloc[self.t].values)

        # Check for NaNs for debugging purposes
        if np.isnan(action).any() or np.isnan(state).any():
            print("Nan in action or state")
            print(f"{submitted_action=}")
            print(f"{state=}")
            print(f"{self.ptf=}")
            print(f"{self.t=}")
            print(f"Log at previous timestep:\n{self.log.iloc[self.t - 1]}")
            raise ValueError("Nan in action or state")

        # TODO: do something with log and info
        info = self._get_info()
        return state, reward, done, False, info
    
    def render(self, mode=None):
        if mode == 'human' and self.t == self.stock_data.shape[0] - 1:
            print(self.log.to_string())
            plt.plot(self.log.index, self.log['Portfolio Value'])
            plt.show()
    
    def close(self):
        pass
    
    def get_portfolio_return(self):
        return (self.ptf[0] + np.dot(self.ptf[1:], self.stock_data['Adj Close'].iloc[self.t].values)) / self.initial_cash