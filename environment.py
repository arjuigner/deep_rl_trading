import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random
from typing import List


class TradingEnv(gym.Env):
    """
    A Gym-compatible trading environment for portfolio rebalancing with a cash account.
    
    The environment expects a pandas DataFrame where each row represents a day.
    Columns not in feature_names are assumed to be stock prices.
    
    When reset, the environment randomly selects a contiguous segment of N days from the data.
    The action is a vector in [0,1]^n_stocks which, when multiplied by K_max, indicates the
    desired target number of shares to hold in each stock.
    
    The environment maintains a cash account (starting at initial_cash). When the target
    positions require more cash than available, the buy orders are scaled down and an overspending
    penalty is applied.
    
    The reward is the change in portfolio value (cash + holdings) from one day to the next,
    plus any overspending penalty.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
        N: int = 20,
        K_max: float = 100.0,
        transaction_fee_rate: float = 0.001,
        initial_cash: float = 10000.0,
        overspending_penalty_coeff: float = 0.001,
        reward_type: str = "relative returns"
    ):
        """
        Parameters:
            df (pd.DataFrame): Historical data with one row per day. Columns not in feature_names
                               are assumed to be stock prices.
            N (int): Number of days in each episode.
            K_max (float): Maximum number of shares (scaling factor for the action output).
            transaction_fee_rate (float): Proportional fee rate for trades.
            initial_cash (float): The starting cash balance.
            overspending_penalty_coeff (float): Coefficient for penalizing attempted overspending.
            reward_type (str): Type of reward to use. Currently only "relative returns", "final returns".
        """
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)  # Ensure a fresh integer index.
        self.N = N
        self.K_max = K_max
        self.transaction_fee_rate = transaction_fee_rate
        self.initial_cash = initial_cash
        self.overspending_penalty_coeff = overspending_penalty_coeff
        self.reward_type = reward_type
        
        # Identify stock price columns (those not in FEATURE_NAMES)
        self.feature_names = feature_names
        self.stock_columns = [col for col in self.df.columns if col not in feature_names]
        self.n_stocks = len(self.stock_columns)
        self.n_features = len(feature_names)
        
        # Action: desired target holdings for each stock, given as a fraction in [0, 1],
        # which is then scaled by K_max.
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_stocks,), dtype=np.float32)
        
        # Observation: current prices, indicator features, current holdings, and cash.
        # Dimension: n_stocks (prices) + n_features + n_stocks (holdings) + 1 (cash)
        obs_dim = self.n_stocks + self.n_features + self.n_stocks + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Episode state variables.
        self.episode_data = None
        self.current_step = None
        self.start_index = None
        
        # Portfolio state: holdings (number of shares) and cash.
        self.current_holdings = None
        self.cash = None
        
    def reset(self, seed=None):
        """Resets the environment and randomly selects a contiguous segment of N days."""
        max_start = len(self.df) - self.N
        self.start_index = random.randint(0, max_start)
        self.end_index = self.start_index + self.N
        self.episode_data = self.df.iloc[self.start_index:self.end_index].reset_index(drop=True)
        self.current_step = 0
        self.current_holdings = np.zeros(self.n_stocks, dtype=np.float32)
        self.cash = self.initial_cash
        return self._get_obs(), {}
    
    def _get_obs(self):
        """
        Constructs the observation from the current day's data.
        Observation consists of:
          - Stock prices (for each stock)
          - Technical indicator features
          - Current portfolio holdings (number of shares held)
          - Current cash balance
        """
        row = self.episode_data.iloc[self.current_step]
        prices = row[self.stock_columns].values.astype(np.float32) / self.df.loc[0, self.stock_columns].values
        features = row[self.feature_names].values.astype(np.float32)
        ptf = self.current_holdings / self.K_max
        cash = np.array([self.cash], dtype=np.float32) / self.initial_cash
        obs = np.concatenate([prices, features, ptf, cash])
        return obs
    
    def step(self, action):
        """
        Executes one time step.
        
        Parameters:
            action (np.ndarray): A vector in [0,1]^n_stocks. This is scaled by K_max to obtain the
                                 desired target holdings.
        
        Returns:
            obs (np.ndarray): The observation for the next time step.
            reward (float): The reward, computed as the change in portfolio value from today to tomorrow,
                            plus any overspending penalty.
            done (bool): Whether the episode is finished.
            info (dict): Additional information.
        """
        done = False
        info = {}
        
        # Get current day's prices.
        current_prices = self.episode_data.iloc[self.current_step][self.stock_columns].values.astype(np.float32)
        
        # Compute current portfolio value.
        V_old = self.cash + np.sum(self.current_holdings * current_prices)
        
        # Scale action to get desired target holdings (in shares).
        desired_holdings = action * self.K_max
        
        # Calculate planned trades: sell amounts and buy amounts.
        sell_amounts = np.maximum(self.current_holdings - desired_holdings, 0)
        buy_amounts = np.maximum(desired_holdings - self.current_holdings, 0)
        
        # Cash available includes current cash plus money from selling.
        cash_from_sales = np.sum(sell_amounts * current_prices)
        available_cash = self.cash + cash_from_sales
        
        # Cash required for the planned buys.
        required_cash = np.sum(buy_amounts * current_prices)
        
        # Initialize overspending penalty.
        overspending_penalty = 0.0
        # If required cash exceeds available cash, scale down the buy orders.
        if required_cash > available_cash and required_cash > 0:
            scale = available_cash / required_cash
            buy_amounts = buy_amounts * scale
            overspending_penalty = - (required_cash - available_cash) * self.overspending_penalty_coeff
        
        # Compute new target holdings after executing trades.
        new_holdings = self.current_holdings - sell_amounts + buy_amounts
        
        # Transaction cost: fee_rate * sum(|change in holdings| * current price)
        transaction_cost = self.transaction_fee_rate * np.sum(np.abs(new_holdings - self.current_holdings) * current_prices)
        
        # Update cash:
        # Money spent on buys = sum(buy_amounts * current_prices)
        new_cash = available_cash - np.sum(buy_amounts * current_prices) - transaction_cost
        
        # Update state.
        self.current_holdings = new_holdings
        self.cash = new_cash
        
        # If not at the final step, compute reward using next day's prices.
        if self.current_step < self.N - 1:
            next_prices = self.episode_data.iloc[self.current_step + 1][self.stock_columns].values.astype(np.float32)
            V_next = self.cash + np.sum(self.current_holdings * next_prices)
            if self.reward_type == "relative returns":
                reward = (V_next / V_old) - 1 + overspending_penalty
            elif self.reward_type == "final returns":
                reward = overspending_penalty
                if self.current_step == self.N - 2:
                    reward = (V_next / self.initial_cash) - 1 + overspending_penalty
            else:
                raise ValueError(f"Unknown reward type: {self.reward_type}")
        else:
            # Last step: no next day data; reward is just the overspending penalty minus transaction cost.
            reward = overspending_penalty - transaction_cost
            done = True
        
        # Move to next time step.
        self.current_step += 1
        if self.current_step >= self.N:
            done = True
        
        # Return the next observation (or dummy observation if done).
        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, False, info
    
    def render(self, mode='human'):
        """Simple rendering: prints out the current step, stock prices, holdings, and cash balance."""
        if self.current_step < self.N:
            row = self.episode_data.iloc[self.current_step]
            prices = row[self.stock_columns].values.astype(np.float32)
            print(f"Step: {self.current_step}")
            print(f"Prices: {prices}")
            print(f"Holdings: {self.current_holdings}")
            print(f"Cash: {self.cash}")
        else:
            print("Episode finished.")
    
    def close(self):
        pass

    
