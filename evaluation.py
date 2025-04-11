from __future__ import annotations
import numpy as np
from environment import TradingEnv
from typing import List


def evaluate_agent(env: TradingEnv, agent: TD3Agent, n_iter: int):
    stats = {
        "final return": [],
        "sharpe ratio": [],
        "max drawdown": [],
        "sortino ratio": [],
    }
    
    # Assume 252 trading days per year for annualizing metrics.
    trading_days = 252

    for i in range(n_iter):
        portfolio_values = []
        state, _ = env.reset()
        
        initial_value = env.initial_cash
        portfolio_values.append(initial_value)
        
        while True:
            action = agent.act(state)
            state, reward, done, info, _ = env.step(action)
            if done:
                break
            
            current_price = env.episode_data.iloc[env.current_step][env.stock_columns]
            current_value = np.dot(env.current_holdings, current_price) + env.cash
            portfolio_values.append(current_value)
                    
        # Convert list to NumPy array for vectorized operations.
        portfolio_values = np.array(portfolio_values)
        
        # Compute final return: (final portfolio value / initial value) - 1.
        final_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        
        # Compute daily returns (percentage changes).
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Compute Sharpe ratio (annualized). Here risk-free rate is assumed to be zero.
        if np.std(daily_returns) > 0:
            sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(trading_days)
        else:
            sharpe = 0.0
        
        # Compute maximum drawdown.
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (cumulative_max - portfolio_values) / cumulative_max
        max_drawdown = np.max(drawdowns)
        
        # Compute Sortino ratio (annualized). Only consider the standard deviation of negative returns.
        downside_returns = daily_returns[daily_returns < 0]
        if downside_returns.size > 0:
            sortino = (np.mean(daily_returns) / np.std(downside_returns)) * np.sqrt(trading_days)
        else:
            sortino = np.mean(daily_returns) * np.sqrt(trading_days)
        
        # Save metrics for this episode.
        stats["final return"].append(final_return)
        stats["sharpe ratio"].append(sharpe)
        stats["max drawdown"].append(max_drawdown)
        stats["sortino ratio"].append(sortino)
        
    # Print summary
    print("=== Evaluation Summary over {} episodes ===".format(n_iter))
    print("Final Return:       mean = {:.2%}, min = {:.2%}, max = {:.2%}".format(
        np.mean(stats["final return"]), np.min(stats["final return"]), np.max(stats["final return"])))
    print("Sharpe Ratio:       mean = {:.3f}, min = {:.3f}, max = {:.3f}".format(
        np.mean(stats["sharpe ratio"]), np.min(stats["sharpe ratio"]), np.max(stats["sharpe ratio"])))
    print("Max Drawdown:       mean = {:.2%}, min = {:.2%}, max = {:.2%}".format(
        np.mean(stats["max drawdown"]), np.min(stats["max drawdown"]), np.max(stats["max drawdown"])))
    print("Sortino Ratio:      mean = {:.3f}, min = {:.3f}, max = {:.3f}".format(
        np.mean(stats["sortino ratio"]), np.min(stats["sortino ratio"]), np.max(stats["sortino ratio"])))

    return stats
        
def evaluate_random_agent(env: TradingEnv, n_iter: int):
    stats = {
        "final return": [],
        "sharpe ratio": [],
        "max drawdown": [],
        "sortino ratio": [],
    }
    
    # Assume 252 trading days per year for annualizing metrics.
    trading_days = 252

    for i in range(n_iter):
        portfolio_values = []
        state, _ = env.reset()
        
        initial_value = env.initial_cash
        portfolio_values.append(initial_value)
        
        while True:
            action = env.action_space.sample()
            state, reward, done, info, _ = env.step(action)
            if done:
                break
            
            current_price = env.episode_data.iloc[env.current_step][env.stock_columns]
            current_value = np.dot(env.current_holdings, current_price) + env.cash
            portfolio_values.append(current_value)
        
        # Convert list to NumPy array for vectorized operations.
        portfolio_values = np.array(portfolio_values)
        
        # Compute final return: (final portfolio value / initial value) - 1.
        final_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        
        # Compute daily returns (percentage changes).
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Compute Sharpe ratio (annualized). Here risk-free rate is assumed to be zero.
        if np.std(daily_returns) > 0:
            sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(trading_days)
        else:
            sharpe = 0.0
        
        # Compute maximum drawdown.
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (cumulative_max - portfolio_values) / cumulative_max
        max_drawdown = np.max(drawdowns)
        
        # Compute Sortino ratio (annualized). Only consider the standard deviation of negative returns.
        downside_returns = daily_returns[daily_returns < 0]
        if downside_returns.size > 0:
            sortino = (np.mean(daily_returns) / np.std(downside_returns)) * np.sqrt(trading_days)
        else:
            sortino = np.mean(daily_returns) * np.sqrt(trading_days)
        
        # Save metrics for this episode.
        stats["final return"].append(final_return)
        stats["sharpe ratio"].append(sharpe)
        stats["max drawdown"].append(max_drawdown)
        stats["sortino ratio"].append(sortino)
        
    # Print summary
    print("=== Evaluation Summary of Random Trader over {} episodes ===".format(n_iter))
    print("Final Return:       mean = {:.2%}, min = {:.2%}, max = {:.2%}".format(
        np.mean(stats["final return"]), np.min(stats["final return"]), np.max(stats["final return"])))
    print("Sharpe Ratio:       mean = {:.3f}, min = {:.3f}, max = {:.3f}".format(
        np.mean(stats["sharpe ratio"]), np.min(stats["sharpe ratio"]), np.max(stats["sharpe ratio"])))
    print("Max Drawdown:       mean = {:.2%}, min = {:.2%}, max = {:.2%}".format(
        np.mean(stats["max drawdown"]), np.min(stats["max drawdown"]), np.max(stats["max drawdown"])))
    print("Sortino Ratio:      mean = {:.3f}, min = {:.3f}, max = {:.3f}".format(
        np.mean(stats["sortino ratio"]), np.min(stats["sortino ratio"]), np.max(stats["sortino ratio"])))

    return stats


def evaluate_buy_and_hold_agent(env: TradingEnv, n_iter: int):
    stats = {
        "final return": [],
        "sharpe ratio": [],
        "max drawdown": [],
        "sortino ratio": [],
    }
    
    # Assume 252 trading days per year for annualizing metrics.
    trading_days = 252

    for i in range(n_iter):
        portfolio_values = []
        state, _ = env.reset()
        
        initial_value = env.initial_cash
        portfolio_values.append(initial_value)
        
        # invest the same amount into each stock
        # n_shares * initial_price = initial_cash / n_stocks
        # thus n_shares = initial_cash / (n_stocks * initial_price)
        initial_price = env.episode_data.iloc[0][env.stock_columns]
        action = env.initial_cash / (env.n_stocks * initial_price)
        while True:
            state, reward, done, info, _ = env.step(action)
            if done:
                break
            
            current_price = env.episode_data.iloc[env.current_step][env.stock_columns]
            current_value = np.dot(env.current_holdings, current_price) + env.cash
            portfolio_values.append(current_value)
        
        # Convert list to NumPy array for vectorized operations.
        portfolio_values = np.array(portfolio_values)
        
        # Compute final return: (final portfolio value / initial value) - 1.
        final_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        
        # Compute daily returns (percentage changes).
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Compute Sharpe ratio (annualized). Here risk-free rate is assumed to be zero.
        if np.std(daily_returns) > 0:
            sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(trading_days)
        else:
            sharpe = 0.0
        
        # Compute maximum drawdown.
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (cumulative_max - portfolio_values) / cumulative_max
        max_drawdown = np.max(drawdowns)
        
        # Compute Sortino ratio (annualized). Only consider the standard deviation of negative returns.
        downside_returns = daily_returns[daily_returns < 0]
        if downside_returns.size > 0:
            sortino = (np.mean(daily_returns) / np.std(downside_returns)) * np.sqrt(trading_days)
        else:
            sortino = np.mean(daily_returns) * np.sqrt(trading_days)
        
        # Save metrics for this episode.
        stats["final return"].append(final_return)
        stats["sharpe ratio"].append(sharpe)
        stats["max drawdown"].append(max_drawdown)
        stats["sortino ratio"].append(sortino)
        
    # Print summary
    print("=== Evaluation Summary for Buy&Hold over {} episodes ===".format(n_iter))
    print("Final Return:       mean = {:.2%}, min = {:.2%}, max = {:.2%}".format(
        np.mean(stats["final return"]), np.min(stats["final return"]), np.max(stats["final return"])))
    print("Sharpe Ratio:       mean = {:.3f}, min = {:.3f}, max = {:.3f}".format(
        np.mean(stats["sharpe ratio"]), np.min(stats["sharpe ratio"]), np.max(stats["sharpe ratio"])))
    print("Max Drawdown:       mean = {:.2%}, min = {:.2%}, max = {:.2%}".format(
        np.mean(stats["max drawdown"]), np.min(stats["max drawdown"]), np.max(stats["max drawdown"])))
    print("Sortino Ratio:      mean = {:.3f}, min = {:.3f}, max = {:.3f}".format(
        np.mean(stats["sortino ratio"]), np.min(stats["sortino ratio"]), np.max(stats["sortino ratio"])))

    return stats

def evaluate_agent_during_training(env: TradingEnv, agent: TD3Agent, n_iter: int) -> float:
    sum_cum_reward = 0
    for i in range(n_iter):
        state, _ = env.reset()
        
        cum_reward = 0
        while True:
            action = agent.act(state)
            state, reward, done, info, _ = env.step(action)
            if done:
                break
            
            cum_reward += reward
        
        # to compute the average cumulative reward
        sum_cum_reward += cum_reward
    
    return float(sum_cum_reward / n_iter)