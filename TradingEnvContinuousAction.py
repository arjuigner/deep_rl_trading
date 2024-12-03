'''
Gym-like environment for trading with continuous action space, meaning an action is a
numpy ndarray of size (n_stocks,) with values in [-1, 1], representing the number of 
shares to buy/sell for each stock in the portfolio.

The stock data should be provided upon initialization, and the features (that will compose
the state) should be provided as well separately. 
* You should make sure that your data doesn't contain NaNs, the environment assumes that the data has been preprocessed beforehand. 
* The environment won't normalize your features any further.
* The index column of "prices" and "features" should be the same

'''

from collections import deque
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stable_baselines3.common.callbacks import BaseCallback

matplotlib.use('TkAgg')

class TradingEnvContinuousAction(gym.Env):
    def __init__(self, prices, features, initial_cash=1e4, k_max=100):
        self.prices = prices
        self.features = features
        self.initial_cash = initial_cash
        self.k_max = k_max
        self.n_stocks = prices.shape[1]
                
        obs_dim = features.shape[1] + self.n_stocks + 1
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.n_stocks,))
        
        self.reset()
        
    def _get_obs(self):
        state = np.concatenate([self.features.iloc[self.t].values, self.shares, [self.cash]])
        return state
        
    def reset(self, seed=None):
        self.t = 0
        self.cash = self.initial_cash
        self.shares = np.zeros(self.n_stocks)
        self.portfolio_value = self.cash
        
        log_columns = [f'Action {ticker}' for ticker in self.prices.columns] + \
                    [f'Eff Action {ticker}' for ticker in self.prices.columns] + \
                    [f'Shares {ticker}' for ticker in self.prices.columns] + \
                    ['Cash', 'Portfolio Value', 'Reward']
        self.log = pd.DataFrame(index=self.prices.index, columns=log_columns)
        return self._get_obs(), {}
    
    def step(self, action):
        action_copy = action.copy()
        action = np.trunc(action * self.k_max)
        action = np.clip(action, np.minimum(-self.shares, 0), self.k_max - self.shares)
        
        eff_action = action.copy()
        cost = np.dot(action, self.prices.iloc[self.t])
        if cost > self.cash: # If cost is too high, scale down the action... maybe do nothing instead ?
            eff_action = self.cash * action / cost
            eff_action = np.trunc(eff_action)
            cost = np.dot(eff_action, self.prices.iloc[self.t])
        
        new_cash = self.cash - cost
        new_shares = self.shares + eff_action
        ptf_value = new_cash + np.dot(new_shares, self.prices.iloc[self.t])
        assert new_cash >= 0, f'Cash is negative: {new_cash}'
        assert np.all(new_shares >= 0), f'Shares are negative: {new_shares}'
        if self.t > 0:
            assert ptf_value == self.portfolio_value, f'Portfolio value is not consistent: {ptf_value} vs {self.portfolio_value}'
        
        # Move to the next day
        self.cash = new_cash
        self.shares = new_shares
        self.t += 1
        self.portfolio_value = self.cash + np.dot(self.shares, self.prices.iloc[self.t])
        
        # Calculate reward
        reward = self.portfolio_value / ptf_value - 1
        
        # Update log
        self.log.iloc[self.t] = np.concatenate([action_copy, eff_action, self.shares, [self.cash, self.portfolio_value, reward]])
        
        done = self.t == self.prices.shape[0] - 1 # Done at the end of the data
        info = {}
        truncated = False
        
        return self._get_obs(), reward, done, truncated, info
    
    def render(self):
        pass
    

class GraphCallback(BaseCallback):

    def __init__(self, env, verbose=0, update_freq=250, memory_len=10000):
        '''
        update_freq: how many steps before each of the graph updates
        '''
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.env = env
        self.update_freq = update_freq
        self.rewards = deque(maxlen=memory_len)
        self.cumulative_returns = deque(maxlen=memory_len)
        self.shares = [deque(maxlen=memory_len) for _ in range(self.env.n_stocks)]
        self.actions = [deque(maxlen=memory_len) for _ in range(self.env.n_stocks)]
        
        self.fig, self.axs = plt.subplots(4, 1, figsize=(12, 10))
        plt.ion()

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        self.rewards.append(self.env.log['Reward'].iloc[self.env.t])
        self.cumulative_returns.append(self.env.log['Portfolio Value'].iloc[self.env.t] / self.env.initial_cash - 1)
        for i, stock in enumerate(self.env.prices.columns):
            self.shares[i].append(self.env.log[f'Shares {stock}'].iloc[self.env.t])
            self.actions[i].append(self.env.log[f'Action {stock}'].iloc[self.env.t])
        
        if self.env.t % self.update_freq == 0:
            self._update_plots()
            self._show_q_value()
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self._update_plots()
        plt.ioff()
    
    def _update_plots(self):
        """
        Update live plots for rewards, returns, holdings, and commission fees during training.
        """
        # Clear plots
        for ax in self.axs:
            ax.clear()

        # Plot episode rewards
        self.axs[0].plot(self.rewards, label="Episode Rewards", color="blue")
        self.axs[0].set_title("Episode Rewards")
        self.axs[0].set_xlabel("Step")
        self.axs[0].set_ylabel("Reward")
        self.axs[0].legend()

        # Plot cumulative returns
        self.axs[1].plot(self.cumulative_returns, label="Cumulative Returns", color="green")
        self.axs[1].set_title("Cumulative Returns")
        self.axs[1].set_xlabel("Step")
        self.axs[1].set_ylabel("Return (%)")
        self.axs[1].legend()

        # Plot shares
        for i, stock in enumerate(self.env.prices.columns):
            self.axs[2].plot(self.shares[i], label=f"Shares {stock}")
        self.axs[2].set_title("Shares")
        self.axs[2].set_xlabel("Step")
        self.axs[2].set_ylabel("# of shares")
        self.axs[2].legend()
        
        # Plot actions
        for i, stock in enumerate(self.env.prices.columns):
            self.axs[3].plot(self.actions[i], label=f"Action {stock}")
        self.axs[3].set_title("Actions")
        self.axs[3].set_xlabel("Step")
        self.axs[3].set_ylabel("Action")
        self.axs[3].legend()
        
        plt.tight_layout()
        plt.pause(0.01)
        
    def _show_q_value(self):
        """
        Show the Q value of the current state for multiple potential actions
        """
        pass #TODO
        