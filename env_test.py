from trading_env import *
from trading_env import _get_stock_data, _timestamps_processing, _compute_sma
import pandas as pd
import numpy as np

def _state_normalization(state, cash, k_max, stock_a_price0, stock_b_price0):
    state[0] /= cash
    state[1] /= k_max
    state[2] /= k_max
    state[3] /= stock_a_price0
    state[4] /= stock_b_price0
    return state

def test_env():
    # Create a date range starting from 2023-01-01, for 10 days
    dates = pd.date_range(start="2023-01-01", periods=6, freq="D")

    # Create some arbitrary data for Stock A and Stock B
    stock_a_prices = np.array([2, 4, 3, 5, 4, 4])
    stock_b_prices = np.array([10, 12, 8, 7, 7, 8])

    # Create a DataFrame with MultiIndex columns for 'Adj Close' -> 'Stock A' and 'Stock B'
    df = pd.DataFrame({
        ('Adj Close', 'Stock A'): stock_a_prices,
        ('Adj Close', 'Stock B'): stock_b_prices
    }, index=dates)

    env = TradingEnv(df, cash=50, k_max=10)

    tol = 10e-4
    state = env.reset()

    # Action 1 : legal action after rounding
    action = np.array([2.9, 1.1]) / 10
    state, reward, done, _, _ = env.step(action)
    expected_state = np.array([34, 3, 1, 4, 12], dtype=np.float32)
    expected_state = _state_normalization(expected_state, 50, 10, stock_a_prices[0], stock_b_prices[0])
    expected_reward = (58 - 50)/50
    try:
        assert not done, "Episode should not be over"
        assert np.allclose(state, expected_state, tol), "Incorrect new state"
        assert np.allclose(reward, expected_reward, tol), "Incorrect reward"
    except AssertionError as e:
        print("Action 1 fail ; state=\n", state)
        print("Expected state=\n", expected_state)
        print("Reward=", reward)
        print("Expected reward=", expected_reward)
        raise e
    
    # Action 2 : selling too much and buying not too much
    action = np.array([2.1, -2]) / 10
    state, reward, done, _, _ = env.step(action)
    expected_state = np.array([38, 5, 0, 3, 8], dtype=np.float32)
    expected_state = _state_normalization(expected_state, 50, 10, stock_a_prices[0], stock_b_prices[0])
    expected_reward = (53 - 58)/58
    try:
        assert not done, "Episode should not be over"
        assert np.allclose(state, expected_state, tol), "Incorrect new state"
        assert np.allclose(reward, expected_reward, tol), "Incorrect reward"
    except AssertionError as e:
        print("Action 2 fail; state=\n", state)
        print("Expected state=\n", expected_state)
        print("Reward=", reward)
        print("Expected reward=", expected_reward)
        raise e
    
    # Action 3: selling too much and buying too much
    action = np.array([-28.3, 18.3]) / 10
    state, reward, done, _, _ = env.step(action)
    expected_state = np.array([5, 0, 6, 5, 7], dtype=np.float32)
    expected_state = _state_normalization(expected_state, 50, 10, stock_a_prices[0], stock_b_prices[0])
    expected_reward = (47 - 53)/53
    try:
        assert not done, "Episode should not be over"
        assert np.allclose(state, expected_state, tol), "Incorrect new state"
        assert np.allclose(reward, expected_reward, tol), "Incorrect reward"
    except AssertionError as e:
        print("Action 2 fail; state=\n", state)
        print("Expected state=\n", expected_state)
        print("Reward=", reward)
        print("Expected reward=", expected_reward)
        raise e
    
    # Action 4: selling ok but buying too much
    action = np.array([10, 0.4]) / 10
    state, reward, done, _, _ = env.step(action)
    expected_state = np.array([0, 1, 6, 4, 7], dtype=np.float32)
    expected_state = _state_normalization(expected_state, 50, 10, stock_a_prices[0], stock_b_prices[0])
    expected_reward =  (46 - 47)/47
    try:
        assert not done, "Episode should not be over"
        assert np.allclose(state, expected_state, tol), "Incorrect new state"
        assert np.allclose(reward, expected_reward, tol), "Incorrect reward"
    except AssertionError as e:
        print("Action 2 fail; state=\n", state)
        print("Expected state=\n", expected_state)
        print("Reward=", reward)
        print("Expected reward=", expected_reward)
        raise e
    
    # Action 5: buying when you have 0 cash
    action = np.array([10, 1.4]) / 10
    state, reward, done, _, _ = env.step(action)
    expected_state = np.array([0, 1, 6, 4, 8], dtype=np.float32)
    expected_state = _state_normalization(expected_state, 50, 10, stock_a_prices[0], stock_b_prices[0])
    expected_reward =  (52 - 46)/46
    try:
        assert done, "Episode should be over now (last step of data)"
        assert np.allclose(state, expected_state, tol), "Incorrect new state"
        assert np.allclose(reward, expected_reward, tol), "Incorrect reward"
    except AssertionError as e:
        print("Action 2 fail; state=\n", state)
        print("Expected state=\n", expected_state)
        print("Reward=", reward)
        print("Expected reward=", expected_reward)
        raise e
    
    print("test_env: all tests passed")
    # TODO: add test where we exceed K_max in a direction or another


def test_compute_sma():
    print("\n------------------------------------------------")
    print("--------------- test_compute_sma ---------------")
    print("------------------------------------------------\n")
    stock_data = _get_stock_data(['MSFT', 'AMZN'], '2023-01-01', '2023-02-01', '1d')
    print("test_compute_sma:\n", stock_data.head(5))
    with_sma = _compute_sma(stock_data, 3)
    print(with_sma.columns)
    print(with_sma.head(5))
    print("\n------------------------------------------------")
    print("--------------- end test_compute_sma -----------")
    print("------------------------------------------------\n")

def test_timestamps_processing():
    stock_data = _get_stock_data(['MSFT', 'AMZN'], '2023-01-01', '2023-02-01', '1d')
    stock_data = stock_data.drop('2023-01-13', axis='index') # Drop a row, now calendar will have a date that's not in our data
    stock_data.loc['2023-01-30', ('Adj Close', 'AMZN')] = np.nan # Add a NaN value, so this row should be removed
    print("\n----------------------------------------------------------")
    print("--------------- test_timespamps_processing ---------------")
    print("----------------------------------------------------------\n")
    print(stock_data)
    stock_data = _timestamps_processing(stock_data)
    print(stock_data)
    print("\n----------------------------------------------------------")
    print("--------------- end test_timespamps_processing -----------")
    print("----------------------------------------------------------\n")

test_env()
test_compute_sma()
test_timestamps_processing()