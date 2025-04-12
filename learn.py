import json
import numpy as np
import os
import pandas as pd 
import torch.optim
import yfinance as yf
from data import get_data_with_features, remove_nans, normalize
from environment import TradingEnv
from td3_impl import TD3Agent, create_live_plot_logger
from evaluation import (
    evaluate_agent, evaluate_random_agent, evaluate_buy_and_hold_agent
)
from typing import Dict


def run_experiment(experiment_id: int,
                   data: pd.DataFrame, 
                   eval_data: pd.DataFrame,
                   td3agent_params: Dict, 
                   tradingenv_params: Dict, 
                   training_params: Dict,
                   lr: float=0.001,
                   n_runs: int=3, 
                   T: int=1000000,
                   T_eval: int=1000000):
    """
    Run an experiment with the given parameters and return the results.
    
    Parameters:
        experiment_id (int): Identifier for the experiment.
        data (pd.DataFrame): Stock data, with features.
        eval_data (pd.DataFrame): Stock data, with features, for evaluation.
        td3agent_params (dict): Parameters for the TD3Agent constructor.
        tradingenv_params (dict): Parameters for the TradingEnv constructor.
        n_runs (int): Number of runs to perform.
        T (int): upperbound on the number of days in the dataset. Defaults to 1,000,000.
        T_eval (int): same, for evaluation
    """
    os.makedirs("data/params", exist_ok=True)
    all_params = {
        "td3agent": td3agent_params,
        "tradingenv": tradingenv_params,
        "training": training_params,
        "lr": lr,
        "n_runs": n_runs,
        "T": T,
        "T_eval": T_eval
    }
    with open(f"data/params/exp_{experiment_id}.json", "w") as f:
        json.dump(all_params, f, indent=4)
            
    for i in range(n_runs):
        print(f"=== Run {i+1}/{n_runs} ===")
        env = TradingEnv(data.iloc[:T], **tradingenv_params)
        eval_env = TradingEnv(eval_data.iloc[:T_eval], **tradingenv_params)
        
        agent_optim_ctor = lambda params: torch.optim.Adam(params, lr=lr)
        agent = TD3Agent(min_action=env.action_space.low, 
                         max_action=env.action_space.high, 
                         **td3agent_params, 
                         optim_constructor=agent_optim_ctor)
        
        agent.train(env, eval_env,  **training_params, 
                    save_path=f"models/exp_{experiment_id}_run_{i}",
                    log_fn=create_live_plot_logger())
        
        # reload the agent so that it is the best one and not the one that caused early stopping
        agent = TD3Agent.load(f"models/exp_{experiment_id}_run_{i}", agent_optim_ctor)
        
        # evaluate it
        stats = evaluate_agent(eval_env, agent, 100)
        
        # evaluate the random and buy-and-hold agents for comparison
        evaluate_buy_and_hold_agent(eval_env, 100)
        evaluate_random_agent(eval_env, 100)
        
        pd_results = pd.DataFrame(index=["Final Returns", "Sharpe Ratio", "Max Drawdown", "Sortino Ratio"], columns=["mean", "std", "min", "max"])
        pd_results.loc["Final Returns"] = {
            "mean": np.mean(stats["final return"]),
            "std": np.std(stats["final return"]),
            "min": np.min(stats["final return"]),
            "max": np.max(stats["final return"])
        }
        pd_results.loc["Sharpe Ratio"] = {
            "mean": np.mean(stats["sharpe ratio"]),
            "std": np.std(stats["sharpe ratio"]),
            "min": np.min(stats["sharpe ratio"]),
            "max": np.max(stats["sharpe ratio"])
        }
        pd_results.loc["Max Drawdown"] = {
            "mean": np.mean(stats["max drawdown"]),
            "std": np.std(stats["max drawdown"]),
            "min": np.min(stats["max drawdown"]),
            "max": np.max(stats["max drawdown"])
        }
        pd_results.loc["Sortino Ratio"] = {
            "mean": np.mean(stats["sortino ratio"]),
            "std": np.std(stats["sortino ratio"]),
            "min": np.min(stats["sortino ratio"]),
            "max": np.max(stats["sortino ratio"])
        }
        os.makedirs("data/results", exist_ok=True)
        pd_results.to_csv(f"data/results/exp_{experiment_id}_run_{i}.csv")
    print("=== Experiment complete ===")
        
        
if __name__ == "__main__":
    # data, all_features = get_data_with_features("./data/HistoricalQuotes.csv")
    # data = remove_nans(data)
    # data = normalize(data)
    
    # EXPERIMENT_ID = 2
    # feature_names = [
    #     "AAPL ret_t-0",
    #     "AAPL ret_t-1",
    #     "AAPL ret_t-2",
    #     "AAPL ret_t-3",
    #     "AAPL ret_t-4",
    # ]
    # T=1000
    # T_eval=250
    # n_runs = 5
    
    # assert T + T_eval < len(data), "Not enough data for training and evaluation"
    # assert all([fn in all_features for fn in feature_names]), "Some of your features are not found in the DataFrame"
    # data = data[["AAPL Close"] + feature_names]
    
    # # Split the data into training and evaluation sets
    # eval_data = data.iloc[T:T + T_eval]
    # data = data.iloc[:T]
    # print(data.head())
    # print("NaNs:", data.isna().sum().sum())
    # print("Eval NaNs:", eval_data.isna().sum().sum())
    
    # env_params = {
    #     "N": 20,
    #     "K_max": 100.0,
    #     "transaction_fee_rate": 0.001,
    #     "initial_cash": 100.0,
    #     "overspending_penalty_coeff": 0.001,
    #     "feature_names": feature_names
    # }
    
    # td3_params = {
    #     "state_dim": 3 + len(feature_names),
    #     "action_dim": 1,
    #     "polyak": 0.995
    # }
    
    # training_params = {
    #     "steps": 10000,
    #     "batch_size": 128,
    #     "gamma": 0.99,
    #     "expl_noise_std": 0.1,
    #     "policy_noise_std": 0.2,
    #     "policy_noise_clip": 0.5,
    #     "policy_delay": 2,
    #     "random_steps": 1000,
    #     "memory_size": 100_000,
    #     "patience":20,
    #     "eval_freq":5
    # }
    
    # run_experiment(EXPERIMENT_ID, data, eval_data, td3_params, env_params, training_params, n_runs=n_runs, T=T, T_eval=T_eval)
    
    
    
    # """env = TradingEnv(data, N=20, K_max=100.0, 
    #                  transaction_fee_rate=0 * 0.001,
    #                  initial_cash=100.0,
    #                  overspending_penalty_coeff=0 * 0.001,
    #                  feature_names=["feature1", "feature2"],)
    
    # # action_noise = NormalActionNoise(mean=np.zeros(env.n_stocks), sigma=0.1 * np.ones(env.n_stocks))
    # # model = TD3(
    # #     "MlpPolicy",
    # #     env,
    # #     action_noise=action_noise,
    # #     verbose=1,
    # #     learning_rate=1e-3,
    # #     buffer_size=100_000,
    # #     learning_starts=1000,
    # #     batch_size=128,
    # #     tau=0.005,
    # #     gamma=0.99,
    # #     train_freq=(1, "episode"),
    # #     gradient_steps=-1,
    # #     policy_kwargs=dict(net_arch=[256, 256])
    # # )
    # # model.learn(total_timesteps=10000)
    # DO_TRAIN = False
    # if DO_TRAIN:
    #     model = TD3Agent(state_dim=env.observation_space.shape[0], 
    #                 action_dim=env.action_space.shape[0],
    #                 min_action=env.action_space.low,
    #                 max_action=env.action_space.high,
    #                 optim_constructor=lambda params: torch.optim.Adam(params, lr=0.001),
    #                 polyak=0.995
    #                 )
    #     model.train(env, steps=3000, batch_size=128, gamma=0.99, expl_noise_std=0.1,
    #                 policy_noise_std=0.2, policy_noise_clip=0.5, policy_delay=2,
    #                 random_steps=1000, memory_size=100_000, log_fn=create_live_plot_logger())
    #     model.save("models/toy")
    # else:
    #     model = TD3Agent.load("models/toy", lambda params: torch.optim.Adam(params, lr=0.001))
    # evaluate_agent(env, model, 50)
    # evaluate_random_agent(env, 50)
    # evaluate_buy_and_hold_agent(env, 50)"""
    
    # data, all_features = get_data_with_features("./data/HistoricalQuotes.csv")
    # data = remove_nans(data)
    # data = normalize(data)
    
    # EXPERIMENT_ID = 3
    # feature_names = [
    #     "AAPL ret_t-0",
    #     "AAPL ret_t-1",
    #     "AAPL ret_t-2",
    #     "AAPL ret_t-3",
    #     "AAPL ret_t-4",
    #     "AAPL ret_t-5",
    #     "AAPL ret_t-6",
    #     "AAPL ret_t-7",
    #     "AAPL ret_t-8",
    #     "AAPL ret_t-9",
    # ]
    # T=1000
    # T_eval=250
    # n_runs = 5
    
    # assert T + T_eval < len(data), "Not enough data for training and evaluation"
    # assert all([fn in all_features for fn in feature_names]), "Some of your features are not found in the DataFrame"
    # data = data[["AAPL Close"] + feature_names]
    
    # # Split the data into training and evaluation sets
    # eval_data = data.iloc[T:T + T_eval]
    # data = data.iloc[:T]
    # print(data.head())
    # print("NaNs:", data.isna().sum().sum())
    # print("Eval NaNs:", eval_data.isna().sum().sum())
    
    # env_params = {
    #     "N": 20,
    #     "K_max": 100.0,
    #     "transaction_fee_rate": 0.001,
    #     "initial_cash": 100.0,
    #     "overspending_penalty_coeff": 0.001,
    #     "feature_names": feature_names
    # }
    
    # td3_params = {
    #     "state_dim": 3 + len(feature_names),
    #     "action_dim": 1,
    #     "polyak": 0.995
    # }
    
    # training_params = {
    #     "steps": 10000,
    #     "batch_size": 128,
    #     "gamma": 0.99,
    #     "expl_noise_std": 0.1,
    #     "policy_noise_std": 0.2,
    #     "policy_noise_clip": 0.5,
    #     "policy_delay": 2,
    #     "random_steps": 1000,
    #     "memory_size": 100_000,
    #     "patience":20,
    #     "eval_freq":5
    # }
    
    # run_experiment(EXPERIMENT_ID, data, eval_data, td3_params, env_params, training_params, n_runs=n_runs, T=T, T_eval=T_eval)
    
    ###################################################################
    ###################################################################
    ###################################################################
    
    # data, all_features = get_data_with_features("./data/HistoricalQuotes.csv")
    # data = remove_nans(data)
    # data = normalize(data)
    
    # EXPERIMENT_ID = 4
    # feature_names = [
    #     "AAPL ret_t-0",
    #     "AAPL ret_t-1",
    #     "AAPL ret_t-2",
    #     "AAPL ret_t-3",
    #     "AAPL ret_t-4",
    #     "AAPL ret_t-5",
    #     "AAPL ret_t-6",
    #     "AAPL ret_t-7",
    #     "AAPL ret_t-8",
    #     "AAPL ret_t-9",
    #     "AAPL ret_t-10",
    #     "AAPL ret_t-11",
    #     "AAPL ret_t-12",
    #     "AAPL ret_t-13",
    #     "AAPL ret_t-14",
    #     "AAPL ret_t-15",
    #     "AAPL ret_t-16",
    #     "AAPL ret_t-17",
    #     "AAPL ret_t-18",
    #     "AAPL ret_t-19",
    # ]
    # T=1000
    # T_eval=250
    # n_runs = 5
    
    # assert T + T_eval < len(data), "Not enough data for training and evaluation"
    # assert all([fn in all_features for fn in feature_names]), "Some of your features are not found in the DataFrame"
    # data = data[["AAPL Close"] + feature_names]
    
    # # Split the data into training and evaluation sets
    # eval_data = data.iloc[T:T + T_eval]
    # data = data.iloc[:T]
    # print(data.head())
    # print("NaNs:", data.isna().sum().sum())
    # print("Eval NaNs:", eval_data.isna().sum().sum())
    
    # env_params = {
    #     "N": 20,
    #     "K_max": 100.0,
    #     "transaction_fee_rate": 0.001,
    #     "initial_cash": 100.0,
    #     "overspending_penalty_coeff": 0.001,
    #     "feature_names": feature_names
    # }
    
    # td3_params = {
    #     "state_dim": 3 + len(feature_names),
    #     "action_dim": 1,
    #     "polyak": 0.995
    # }
    
    # training_params = {
    #     "steps": 10000,
    #     "batch_size": 128,
    #     "gamma": 0.99,
    #     "expl_noise_std": 0.1,
    #     "policy_noise_std": 0.2,
    #     "policy_noise_clip": 0.5,
    #     "policy_delay": 2,
    #     "random_steps": 1000,
    #     "memory_size": 100_000,
    #     "patience":20,
    #     "eval_freq":5
    # }
    
    # run_experiment(EXPERIMENT_ID, data, eval_data, td3_params, env_params, training_params, n_runs=n_runs, T=T, T_eval=T_eval)
    
    # ###################################################################
    # ###################################################################
    # ###################################################################
    
    # data, all_features = get_data_with_features("./data/HistoricalQuotes.csv")
    # data = remove_nans(data)
    # data = normalize(data)
    
    # EXPERIMENT_ID = 5
    # feature_names = all_features
    # T=1000
    # T_eval=250
    # n_runs = 5
    
    # assert T + T_eval < len(data), "Not enough data for training and evaluation"
    # assert all([fn in all_features for fn in feature_names]), "Some of your features are not found in the DataFrame"
    # data = data[["AAPL Close"] + feature_names]
    
    # # Split the data into training and evaluation sets
    # eval_data = data.iloc[T:T + T_eval]
    # data = data.iloc[:T]
    # print(data.head())
    # print("NaNs:", data.isna().sum().sum())
    # print("Eval NaNs:", eval_data.isna().sum().sum())
    
    # env_params = {
    #     "N": 20,
    #     "K_max": 100.0,
    #     "transaction_fee_rate": 0.001,
    #     "initial_cash": 100.0,
    #     "overspending_penalty_coeff": 0.001,
    #     "feature_names": feature_names
    # }
    
    # td3_params = {
    #     "state_dim": 3 + len(feature_names),
    #     "action_dim": 1,
    #     "polyak": 0.995
    # }
    
    # training_params = {
    #     "steps": 10000,
    #     "batch_size": 128,
    #     "gamma": 0.99,
    #     "expl_noise_std": 0.1,
    #     "policy_noise_std": 0.2,
    #     "policy_noise_clip": 0.5,
    #     "policy_delay": 2,
    #     "random_steps": 1000,
    #     "memory_size": 100_000,
    #     "patience":20,
    #     "eval_freq":5
    # }
    
    # run_experiment(EXPERIMENT_ID, data, eval_data, td3_params, env_params, training_params, n_runs=n_runs, T=T, T_eval=T_eval)
    
    ###################################################################
    ###################################################################
    ###################################################################
    
    data, all_features = get_data_with_features("./data/HistoricalQuotes.csv")
    data = remove_nans(data)
    data = normalize(data)
    
    EXPERIMENT_ID = 6
    feature_names = [
        "AAPL ret_t-0",
        "AAPL ret_t-1",
        "AAPL ret_t-2",
        "AAPL ret_t-3",
        "AAPL ret_t-4",
        "AAPL ret_t-5",
        "AAPL ret_t-6",
        "AAPL ret_t-7",
        "AAPL ret_t-8",
        "AAPL ret_t-9",
        "AAPL ret_t-10",
        "AAPL ret_t-11",
        "AAPL ret_t-12",
        "AAPL ret_t-13",
        "AAPL ret_t-14",
        "AAPL ret_t-15",
        "AAPL ret_t-16",
        "AAPL ret_t-17",
        "AAPL ret_t-18",
        "AAPL ret_t-19",
    ]
    T=1000
    T_eval=250
    n_runs = 5
    
    assert T + T_eval < len(data), "Not enough data for training and evaluation"
    assert all([fn in all_features for fn in feature_names]), "Some of your features are not found in the DataFrame"
    data = data[["AAPL Close"] + feature_names]
    
    # Split the data into training and evaluation sets
    eval_data = data.iloc[T:T + T_eval]
    data = data.iloc[:T]
    print(data.head())
    print("NaNs:", data.isna().sum().sum())
    print("Eval NaNs:", eval_data.isna().sum().sum())
    
    env_params = {
        "N": 20,
        "K_max": 100.0,
        "transaction_fee_rate": 0.001,
        "initial_cash": 100.0,
        "overspending_penalty_coeff": 0.001,
        "feature_names": feature_names
    }
    
    td3_params = {
        "state_dim": 3 + len(feature_names),
        "action_dim": 1,
        "polyak": 0.995
    }
    
    training_params = {
        "steps": 10000,
        "batch_size": 128,
        "gamma": 0.99,
        "expl_noise_std": 0.1,
        "policy_noise_std": 0.2,
        "policy_noise_clip": 0.5,
        "policy_delay": 2,
        "random_steps": 1000,
        "memory_size": 100_000,
        "patience":20,
        "eval_freq":10
    }
    
    run_experiment(EXPERIMENT_ID, data, eval_data, td3_params, env_params, training_params, n_runs=n_runs, T=T, T_eval=T_eval)
    