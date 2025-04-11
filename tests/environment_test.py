import numpy as np
import pandas as pd

from environment import TradingEnv, FEATURE_NAMES

# For testing, define the features
FEATURE_NAMES = ["feature1", "feature2"]

def test_trading_env_step():
    # Create a small DataFrame for 3 days and 2 stocks
    data = {
        "stock_A": [100, 110, 120],  # price increases
        "stock_B": [200, 190, 195],  # price decreases then rises
        "feature1": [0.1, 0.2, 0.3],
        "feature2": [1.0, 1.1, 1.2]
    }
    df = pd.DataFrame(data)
    
    # Initialize the environment
    env = TradingEnv(
        df=df,
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
    obs, reward, done, info = env.step(action)

    # Extract current prices
    prices_t = np.array([100.0, 200.0])  # day 0
    prices_tp1 = np.array([110.0, 190.0])  # day 1
    target_holdings = np.array([5.0, 5.0])
    cost = np.sum(target_holdings * prices_t)  # 5*100 + 5*200 = 1500
    transaction_cost = 0.01 * cost  # 1% = 15
    expected_cash = 10000.0 - cost - transaction_cost  # 10000 - 1500 - 15 = 8485

    # Expected reward: portfolio change from day 0 to day 1
    value_t = np.sum(target_holdings * prices_t) + expected_cash
    value_tp1 = np.sum(target_holdings * prices_tp1) + expected_cash
    expected_reward = value_tp1 - value_t  # change in portfolio value

    # Assertions
    assert np.allclose(env.current_holdings, target_holdings), "Holdings not updated correctly"
    assert np.isclose(env.cash, expected_cash), "Cash not updated correctly"
    assert np.isclose(reward, expected_reward), f"Incorrect reward: expected {expected_reward}, got {reward}"
    assert done is False, "Episode shouldn't be done yet"

    print("✅ Test passed: step() behaves correctly with simple values.")

# Run the test
test_trading_env_step()
