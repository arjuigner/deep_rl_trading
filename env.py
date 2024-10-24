import exchange_calendars as xcals
import yfinance as yf

import numpy as np
import pandas as pd


########### Getting data and preprocessing ###########


def get_stock_data(ticker, start_date, end_date, granularity):
    '''
    ticker: str, stock ticker
    start_date: str, start date in format 'YYYY-MM-DD'
    end_date: str, end date in format 'YYYY-MM-DD'
    granularity: str, granularity of data, e.g. '1d', '1h', '1m'

    return: pandas DataFrame, stock data. Its columns are: 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
    '''
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=granularity)
    return stock_data


def timestamps_processing(data):
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


def make_adjusted(data):
    '''
    data: pandas DataFrame, stock data. Its columns are: 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
        with at least two subcolumns each for the stocks
    return: pandas DataFrame, stock data with columns: 'Adj Open', 'Adj High', 'Adj Low', 'Adj Close', 'Volume'

    We don't need non-adjusted data for our trading environment, so we will convert the data to be adjusted, and drop the non-adjusted columns.
    '''
    adj_data = pd.DataFrame(columns=data.columns, index=data.index)
    for stock in data['Adj Close'].columns:
        adj_data[('Adj Open', stock)]  = data[('Open', stock)] * data[('Adj Close', stock)] / data[('Close', stock)]
        adj_data[('Adj High', stock)]  = data[('High', stock)] * data[('Adj Close', stock)] / data[('Close', stock)]
        adj_data[('Adj Low', stock)]   = data[('Low', stock)]  * data[('Adj Close', stock)] / data[('Close', stock)]
        adj_data[('Adj Close', stock)] = data[('Adj Close', stock)]
        adj_data[('Volume', stock)]    = data[('Volume', stock)]
    return adj_data


########### Indicators ###########


def compute_sma(data, window=14):
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

def compute_rsi(data, window=14):
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

def compute_indicators(data):
    '''
    data: pandas DataFrame, stock data. Its columns are: 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', and it has at least 2 sub-columns for each 
    of them, one for each stock.
    return: pandas DataFrame, stock data with additional columns for indicators. Its columns are: 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
    'SMA', 'RSI'
    '''
    data = compute_sma(data)
    data = compute_rsi(data)
    return data


########### Trading Environment ###########


class TradingEnvironment():

    def __init__(self, stock_data, cash=1000, K_max=10, indicators=True, t0=0):
        '''
        stock_data: pandas DataFrame, stock data. It must have columns 'Adj Close', and it has at least 2 sub-columns (one for each stock) for each column.
        cash: float, initial cash you have
        K_max: int, maximum number of stocks you can be long/short
        indicators: bool, whether to compute indicators in the states or not
        t0: int, initial time step. Default is 0, but changeable if using indicators, and they are only available after some time steps.
        '''
    
        #TODO : I assume there are multiple stocks, thus self.stock_data['Adj Close'].iloc[self.t] is NOT a scalar value and I can access .values on it
        self.stock_data = stock_data
        if indicators:
            self.stock_data = compute_indicators(self.stock_data)
        
        self.n_stocks = len(self.stock_data['Adj Close'].columns) # Number of stocks
        self.t = -1 # Current time step. Currently invalid, as we haven't started yet
        self.t0 = t0 # Time to start at
        self.initial_cash = cash
        self.ptf = np.zeros(self.n_stocks + 1) # Portfolio. Number of stocks we have for each stock, and cash
        self.ptf[0] = cash
        self.K_max = K_max # Maximum number of stocks we can be long/short
        self.use_indicators = indicators

        self.log = pd.DataFrame(columns=['Cash'] + \
                                        [s+' Position' for s in stock_data['Adj Close'].columns] + \
                                        ['Portfolio Value'] + \
                                        [s+' Price' for s in stock_data['Adj Close'].columns] + \
                                        [s+' Submitted Action' for s in stock_data['Adj Close'].columns],
                                index=stock_data.index)


    def start(self):
        self.t = self.t0
        # TODO : In paper they use 'Close' price, ask whether 'Adj Close' is fine or not
        return self.get_state()


    def step(self, action):
        '''
        action: np.array, action to take. Its length should be the same as the number of stocks, and all values should be in [-1, 1]. A_t in the paper.
        return: tuple, (state, reward, done), where done is a bool telling you whether the episode is over or not.

        Action will be scaled to [-K_max, K_max], where K_max is the maximum number of stocks we can be long/short.
        Then, we will buy (or sell for negative values) the number of stocks given by the action, for each stock.

        If you don't have enough cash to buy the stocks, it will buy as many as you can (by scaling down your action vector until it is small enough).
        If you try to sell more stocks than you have, it will sell all the stocks you have.

        State are made of the following components:
        * 0: Cash in portfolio
        * 1 to n: Number of stocks we have for each stock
        * n+1 to 2n: Prices of each stock
        * 2n+1 to 3n: Technical indicator #1 of each stock (ex: SMA)
        * 3n+1 to 4n: Technical indicator #2 of each stock (ex: RSI)
        * ...
        '''


        if self.t == -1:
            print("Environment not started yet. Please call start() first.")
            return None

        # Clip the selling part of the action to be achievable with current portfolio
        submitted_action = action.copy()
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

        # Update portfolio
        self.ptf[0] = buying_power - buying_cost
        self.ptf[1:] += action

        # Move to next time step and calculate reward
        self.t += 1
        reward = np.dot(self.ptf[1:], self.stock_data['Adj Close'].iloc[self.t].values - self.stock_data['Adj Close'].iloc[self.t - 1].values)
        # Following line divides reward by the portfolio value at the previous time step (so if new value is 120% of previous one, reward is 0.2)
        reward /= (self.ptf[0] + np.dot(self.ptf[1:], self.stock_data['Adj Close'].iloc[self.t - 1].values))

        # Check if we are at the end of our data
        done = self.t == len(self.stock_data) - 1

        # Calculate new state
        state = self.get_state()

        # Update log
        self.log['Cash'].iloc[self.t] = self.ptf[0]
        for i, stock in enumerate(self.stock_data['Adj Close'].columns):
            self.log[stock + ' Position'].iloc[self.t] = self.ptf[i+1]
            self.log[stock + ' Price'].iloc[self.t] = self.stock_data['Adj Close'].iloc[self.t][stock]
            self.log[stock + ' Submitted Action'].iloc[self.t] = submitted_action[i] * self.K_max
        self.log['Portfolio Value'].iloc[self.t] = self.ptf[0] + np.dot(self.ptf[1:], self.stock_data['Adj Close'].iloc[self.t].values)

        return state, reward, done


    def get_portfolio_return(self):
        return (self.ptf[0] + np.dot(self.ptf[1:], self.stock_data['Adj Close'].iloc[self.t].values)) / self.initial_cash
    
    def get_state(self):
        # Calculate new state
        state = np.concatenate((self.ptf, self.stock_data['Adj Close'].iloc[self.t].values))
        if self.use_indicators:
            state = np.concatenate((
                state, 
                self.stock_data['SMA'].iloc[self.t].values,
                self.stock_data['RSI'].iloc[self.t].values
            ))
        return state