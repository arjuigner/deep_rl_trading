import numpy as np
import pandas as pd

from environment import TradingEnv


def test_trading_env_step():
    FEATURE_NAMES = ["feature1", "feature2"]
    # Create a small DataFrame for 3 days and 2 stocks
    data = {
        "stock_A": [100, 110, 120],  # price increases
        "stock_B": [200, 180, 195],  # price decreases then rises
        "feature1": [0.1, 0.2, 0.3],
        "feature2": [1.0, 1.1, 1.2]
    }
    df = pd.DataFrame(data)
    
    # Initialize the environment
    env = TradingEnv(
        df=df,
        feature_names=FEATURE_NAMES,
        N=3,
        K_max=10.0,
        transaction_fee_rate=0.01,  # 1% transaction fee
        initial_cash=10000.0,
        overspending_penalty_coeff=0.0  # turn off overspending penalty for now
    )
    
    # Force a specific episode (no randomness for testing)
    env.episode_data = df
    env.current_step = 0
    env.current_holdings = np.array([0.0, 0.0])
    env.cash = 10000.0

    # Action: allocate 5 shares to each stock (action = [0.5, 0.5])
    action = np.array([0.5, 0.5])  # will be scaled by K_max = 10 => target = [5, 5]
    
    # Perform step
    obs, reward, done, _, info = env.step(action)

    # Extract current prices
    prices_t = np.array([100.0, 200.0])  # day 0
    prices_tp1 = np.array([110.0, 180.0])  # day 1
    target_holdings = np.array([5.0, 5.0])
    cost = np.sum(target_holdings * prices_t)  # 5*100 + 5*200 = 1500
    transaction_cost = 0.01 * cost  # 1% = 15
    expected_cash = 10000.0 - cost - transaction_cost  # 10000 - 1500 - 15 = 8485

    # Expected reward: portfolio change from day 0 to day 1
    value_t = 10000.0  # initial portfolio value
    value_tp1 = np.sum(target_holdings * prices_tp1) + expected_cash
    expected_reward = (value_tp1 - value_t) / value_t # change in portfolio value

    # Assertions
    assert env.n_features == 2, f"Incorrect number of features, expected 2, got {env.n_features}"
    assert env.n_stocks == 2, f"Incorrect number of stocks, expected 2, got {env.n_stocks}"
    assert np.allclose(obs[:2], prices_tp1 / prices_t), "Prices not updated correctly"
    assert np.allclose(obs[2:4], [0.2, 1.1]), "Features not updated correctly"
    assert np.allclose(obs[4:6], target_holdings / 10.0), f"Holdings not updated correctly, got {obs[4:6]}"
    assert np.isclose(obs[6], expected_cash / 10000), "Cash not updated correctly"
    assert np.allclose(env.current_holdings, target_holdings), "Holdings not updated correctly"
    assert np.isclose(env.cash, expected_cash), "Cash not updated correctly"
    assert np.isclose(reward, expected_reward), f"Incorrect reward: expected {expected_reward}, got {reward}"
    assert done is False, "Episode shouldn't be done yet"

    print("✅ Test passed: step() behaves correctly with simple values.")

def test_trading_env_step_no_features():
    # Create a small DataFrame for 3 days and 2 stocks
    data = {
        "stock_A": [100, 110, 120],  # price increases
        "stock_B": [200, 180, 195]  # price decreases then rises
    }
    df = pd.DataFrame(data)
    
    # Initialize the environment
    env = TradingEnv(
        df=df,
        feature_names=[],
        N=3,
        K_max=10.0,
        transaction_fee_rate=0.01,  # 1% transaction fee
        initial_cash=10000.0,
        overspending_penalty_coeff=0.0  # turn off overspending penalty for now
    )
    
    # Force a specific episode (no randomness for testing)
    env.episode_data = df
    env.current_step = 0
    env.current_holdings = np.array([0.0, 0.0])
    env.cash = 10000.0

    # Action: allocate 5 shares to each stock (action = [0.5, 0.5])
    action = np.array([0.5, 0.5])  # will be scaled by K_max = 10 => target = [5, 5]
    
    # Perform step
    obs, reward, done, _, info = env.step(action)

    # Extract current prices
    prices_t = np.array([100.0, 200.0])  # day 0
    prices_tp1 = np.array([110.0, 180.0])  # day 1
    target_holdings = np.array([5.0, 5.0])
    cost = np.sum(target_holdings * prices_t)  # 5*100 + 5*200 = 1500
    transaction_cost = 0.01 * cost  # 1% = 15
    expected_cash = 10000.0 - cost - transaction_cost  # 10000 - 1500 - 15 = 8485

    # Expected reward: portfolio change from day 0 to day 1
    value_t = 10000.0  # initial portfolio value
    value_tp1 = np.sum(target_holdings * prices_tp1) + expected_cash
    expected_reward = (value_tp1 - value_t) / value_t # change in portfolio value

    # Assertions
    assert env.n_features == 0, f"Incorrect number of features, expected 2, got {env.n_features}"
    assert env.n_stocks == 2, f"Incorrect number of stocks, expected 2, got {env.n_stocks}"
    assert env.stock_columns == ["stock_A", "stock_B"], f"Incorrect stock columns, expected ['stock_A', 'stock_B'], got {env.stock_columns}"
    assert np.allclose(obs[:2], prices_tp1 / prices_t), "Prices not updated correctly"
    assert np.allclose(obs[2:4], target_holdings / 10.0), f"Holdings not updated correctly, got {obs[4:6]}"
    assert np.isclose(obs[4], expected_cash / 10000), "Cash not updated correctly"
    assert np.allclose(env.current_holdings, target_holdings), "Holdings not updated correctly"
    assert np.isclose(env.cash, expected_cash), "Cash not updated correctly"
    assert np.isclose(reward, expected_reward), f"Incorrect reward: expected {expected_reward}, got {reward}"
    assert done is False, "Episode shouldn't be done yet"

    print("✅ Test passed: step() behaves correctly with simple values.")

def test_trading_env_step_final_returns():
    FEATURE_NAMES = ["feature1", "feature2"]
    # Create a small DataFrame for 3 days and 2 stocks
    data = {
        "stock_A": [100, 110, 120],  # price increases
        "stock_B": [200, 180, 195],  # price decreases then rises
        "feature1": [0.1, 0.2, 0.3],
        "feature2": [1.0, 1.1, 1.2]
    }
    df = pd.DataFrame(data)
    
    # Initialize the environment
    env = TradingEnv(
        df=df,
        feature_names=FEATURE_NAMES,
        N=3,
        K_max=10.0,
        transaction_fee_rate=0.01,  # 1% transaction fee
        initial_cash=10000.0,
        overspending_penalty_coeff=0.0,  # turn off overspending penalty for now
        reward_type="final returns"
    )
    
    # Force a specific episode (no randomness for testing)
    env.episode_data = df
    env.current_step = 0
    env.current_holdings = np.array([0.0, 0.0])
    env.cash = 10000.0

    # Action: allocate 5 shares to each stock (action = [0.5, 0.5])
    action = np.array([0.5, 0.5])  # will be scaled by K_max = 10 => target = [5, 5]
    
    # Perform step
    obs, reward, done, _, info = env.step(action)

    # Extract current prices
    prices_t = np.array([100.0, 200.0])  # day 0
    prices_tp1 = np.array([110.0, 180.0])  # day 1
    target_holdings = np.array([5.0, 5.0])
    cost = np.sum(target_holdings * prices_t)  # 5*100 + 5*200 = 1500
    transaction_cost = 0.01 * cost  # 1% = 15
    expected_cash = 10000.0 - cost - transaction_cost  # 10000 - 1500 - 15 = 8485

    # Expected reward: portfolio change from day 0 to day 1
    value_t = 10000.0  # initial portfolio value
    value_tp1 = np.sum(target_holdings * prices_tp1) + expected_cash

    # Second step, do nothing
    action = target_holdings / 10.0  # do nothing
    obs2, reward2, done2, _, info = env.step(action)
    prices_tp2 = np.array([120.0, 195.0])  # day 2
    value_tp2 = np.sum(target_holdings * prices_tp2) + expected_cash
    expected_reward = (value_tp2 - value_t) / value_t # change in portfolio value

    # Assertions
    assert env.n_features == 2, f"Incorrect number of features, expected 2, got {env.n_features}"
    assert env.n_stocks == 2, f"Incorrect number of stocks, expected 2, got {env.n_stocks}"
    assert np.allclose(obs[:2], prices_tp1 / prices_t), "Prices not updated correctly"
    assert np.allclose(obs[2:4], [0.2, 1.1]), "Features not updated correctly"
    assert np.allclose(obs[4:6], target_holdings / 10.0), f"Holdings not updated correctly, got {obs[4:6]}"
    assert np.isclose(obs[6], expected_cash / 10000), "Cash not updated correctly"
    assert np.allclose(env.current_holdings, target_holdings), "Holdings not updated correctly"
    assert np.isclose(env.cash, expected_cash), "Cash not updated correctly"
    assert np.isclose(reward, 0.0), f"Incorrect reward: expected 0.00, got {reward}"
    assert np.isclose(reward2, expected_reward), f"Incorrect reward: expected {expected_reward}, got {reward2}"
    assert done is False, "Episode shouldn't be done yet"

    print("✅ Test passed: step() behaves correctly with simple values.")
    
# Run the test
test_trading_env_step()
test_trading_env_step_no_features()
test_trading_env_step_final_returns()