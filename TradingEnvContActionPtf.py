import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Box

# Imports for the callback class
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import torch
matplotlib.use('TkAgg')


class TradingEnvContActionPtf(Env):
    def __init__(self, prices: pd.DataFrame, features: pd.DataFrame, initial_cash: float, k_max: int, reward='RelativeReturn', normalize_state=None):
        super().__init__()

        # Initialization parameters
        self.prices = prices
        self.features = features
        self.initial_cash = initial_cash
        self.k_max = k_max
        self.n_stocks = prices.shape[1]
        self.stock_names = prices.columns
        self.reward_type = reward

        # Spaces
        feature_size = features.shape[1]
        self.state_size = feature_size + self.n_stocks + 1  # Features + portfolio + cash
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32)
        self.action_space = Box(low=0, high=1, shape=(self.n_stocks,), dtype=np.float32)

        # Reset environment
        self.reset()


    def reset(self, seed=None):
        self.t = 0
        self.cash = self.initial_cash
        self.portfolio = np.zeros(self.n_stocks)
        self.done = False
        log_columns = [f'Action {ticker}' for ticker in self.prices.columns] + \
                    [f'Eff Action {ticker}' for ticker in self.prices.columns] + \
                    [f'Shares {ticker}' for ticker in self.prices.columns] + \
                    ['Cash', 'Portfolio Value', 'Reward']
        self.log = pd.DataFrame(index=self.prices.index, columns=log_columns)
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            raise RuntimeError("Cannot step; environment is done. Please reset.")

        # Clip actions
        action = np.clip(action, 0, 1)

        # Calculate desired portfolio
        desired_portfolio = action * self.k_max
        current_prices = self.prices.iloc[self.t].values

        # Calculate old portfolio value
        old_portfolio_value = self._get_portfolio_value(current_prices)

        # Scale action to fit within cash constraints
        effective_portfolio = self._scale_action_to_cash(desired_portfolio, current_prices)

        # Update portfolio and cash
        self.portfolio = effective_portfolio
        self.cash = old_portfolio_value - np.sum(self.portfolio * current_prices)

        # Increment timestep
        self.t += 1
        if self.t + 1 >= len(self.prices):
            self.done = True

        # Calculate new portfolio value and reward
        new_prices = self.prices.iloc[self.t].values
        new_portfolio_value = self._get_portfolio_value(new_prices)
        reward = 0
        if self.reward_type == 'RelativeReturn':
            reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
        elif self.reward_type == 'AbsoluteReturn':
            reward = new_portfolio_value - old_portfolio_value
        elif self.reward_type == 'LogReturn':
            reward = np.log(new_portfolio_value / old_portfolio_value)
        else:
            raise ValueError(f"Invalid reward type: {self.reward_type}")
            

        # Log results
        self._log_step(action, effective_portfolio, reward)

        return self._get_obs(), reward, self.done, False, {}

    def _get_obs(self):
        features = self.features.iloc[self.t].values.astype(np.float32)
        normalized_portfolio = (self.portfolio / self.k_max).astype(np.float32)
        normalized_cash = np.array([self.cash / self.initial_cash], dtype=np.float32)
        return np.concatenate([features, normalized_portfolio, normalized_cash])

    def _scale_action_to_cash(self, desired_portfolio, current_prices):
        portfolio_cost = np.sum(desired_portfolio * current_prices)
        current_portfolio_value = self._get_portfolio_value(current_prices)
        if portfolio_cost <= current_portfolio_value:
            return desired_portfolio

        scale_factor = current_portfolio_value / portfolio_cost
        return desired_portfolio * scale_factor

    def _get_portfolio_value(self, prices):
        return self.cash + np.sum(self.portfolio * prices)

    def _log_step(self, action, effective_portfolio, reward):
        current_prices = self.prices.iloc[self.t - 1].values if self.t > 0 else self.prices.iloc[0].values
        portfolio_value = self._get_portfolio_value(current_prices)

        log_entry = {
            **{f"Action {name}": a for name, a in zip(self.stock_names, action)},
            **{f"Eff Action {name}": eff for name, eff in zip(self.stock_names, effective_portfolio)},
            **{f"Shares {name}": shares for name, shares in zip(self.stock_names, self.portfolio)},
            "Cash": self.cash,
            "Portfolio Value": portfolio_value,
            "Reward": reward
        }
        self.log.loc[self.log.index[self.t]] = log_entry

    def render(self, mode="human"):
        print(self.log.tail(1))


# CALLBACk

class GraphCallback(BaseCallback):
    '''
    This class allows to update a 'live' graph of the rewards and portfolio value during training.
    '''

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
            
            # Look at some Q values, assumes only one stock and one feature
            if self.env.n_stocks == 1 and self.env.features.shape[1] == 1:
                a = torch.tensor([0, 1/4, 1/2, 3/4, 1]).unsqueeze(1)            # shape 5x1
                s = torch.tensor(self.env._get_obs()).unsqueeze(0).repeat(5, 1) # shape 5x3
                q_values = self.model.policy.critic(s, a)[0].detach().numpy()
                print(f"Q values: {q_values}")
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
        