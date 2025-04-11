import pandas as pd
import numpy as np
from typing import Tuple, List

def make_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create features for the trading environment.
    
    Parameters:
        data (pd.DataFrame): Stock data with a column named like "XXX Close".
        
    Returns:
        pd.DataFrame: DataFrame with the new feature columns.
        List[str]: List with the names of the new columns.
    """
    # Identify a column that ends with "Close" and extract the ticker symbol.
    close_cols = [col for col in data.columns if col.endswith("Close")]
    if not close_cols:
        raise ValueError("DataFrame must contain a column ending with 'Close'.")
    close_col = close_cols[0]
    ticker = close_col.split()[0]  # Assume the ticker symbol is the first token.
    
    df = data.copy()
    new_cols = []
    
    # --- Daily Return and Lagged Returns ---
    # Compute daily percentage return.
    df[f'{ticker} Daily Return'] = df[close_col].pct_change()
    # Create lagged return features (using data from past days only)
    for lag in range(20):
        col_name = f'{ticker} ret_t-{lag}'
        # For row t, the return is computed from day t-lag.
        df[col_name] = df[f'{ticker} Daily Return'].shift(lag)
        new_cols.append(col_name)
    
    # --- Simple Moving Averages (SMAs) ---
    # We compute SMAs over different windows and normalize them
    # by comparing to the previous day's close.
    for window in [5, 10, 20]:
        sma_raw = df[close_col].rolling(window=window, min_periods=window).mean()
        sma_shifted = sma_raw.shift(1)  # using only historical data (up to t-1)
        normalized_sma = (sma_shifted - df[close_col].shift(1)) / df[close_col].shift(1)
        col_name = f'{ticker} sma_{window}'
        df[col_name] = normalized_sma
        new_cols.append(col_name)
    
    # --- Relative Strength Index (RSI) ---
    # Use a 14-day period for RSI calculation.
    delta = df[close_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.shift(1)  # Shift so that row t uses past data only
    normalized_rsi = (rsi - 50) / 50   # Transform to roughly range [-1, 1]
    col_name = f'{ticker} rsi_14'
    df[col_name] = normalized_rsi
    new_cols.append(col_name)
    
    # --- MACD ---
    # Compute 12-day and 26-day exponential moving averages (EMAs)
    ema12 = df[close_col].ewm(span=12, adjust=False).mean().shift(1)
    ema26 = df[close_col].ewm(span=26, adjust=False).mean().shift(1)
    macd = ema12 - ema26
    # Signal line is a 9-day EMA of MACD (shifted for causality)
    macd_signal = macd.ewm(span=9, adjust=False).mean().shift(1)
    macd_hist = macd - macd_signal
    # Normalize MACD and its components by the previous day's close.
    norm_macd = macd / df[close_col].shift(1)
    norm_macd_signal = macd_signal / df[close_col].shift(1)
    norm_macd_hist = macd_hist / df[close_col].shift(1)
    for feature, norm_feature, suffix in zip(
         [macd, macd_signal, macd_hist],
         [norm_macd, norm_macd_signal, norm_macd_hist],
         ['macd', 'macd_signal', 'macd_hist']
         ):
        col_name = f'{ticker} {suffix}'
        df[col_name] = norm_feature
        new_cols.append(col_name)
    
    # --- Rolling Volatility ---
    # Compute a 10-day rolling standard deviation of daily returns.
    vol = df[f'{ticker} Daily Return'].shift(1).rolling(window=10, min_periods=10).std()
    col_name = f'{ticker} vol_10'
    df[col_name] = vol
    new_cols.append(col_name)
    
    # Optionally drop temporary columns
    df.drop(columns=[f'{ticker} Daily Return'], inplace=True)
    
    return df, new_cols

def load_and_clean_stock_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Data size: {df.shape}; #NaNs = {df.isna().sum().sum()}")
    
    # Rename columns to remove slashes and spaces
    df.rename(columns={
        ' Close/Last': 'Close',
        ' Open': 'Open',
        ' High': 'High',
        ' Low': 'Low',
        ' Volume': 'Volume',
        'Date': 'Date'
    }, inplace=True)
    
    # Remove '$' and convert price columns to float
    price_cols = ['Close', 'Open', 'High', 'Low']
    for col in price_cols:
        df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)
    
    # Convert Volume to numeric
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    
    # Convert Date to datetime, sort and replace index by integers
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.drop('Date', axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df[["Close",]]

def get_data_with_features(filepath: str) -> pd.DataFrame:
    data = load_and_clean_stock_data(filepath)
    data.rename({"Close": "AAPL Close"}, axis="columns", inplace=True)
    print("Columns of data after selecting Close and renaming it:", data.columns)
    
    data, feature_names = make_features(data)
    print("Columns of data after make_features:", data.columns)
    print("feature_names =", feature_names)
    
    return data, feature_names

def remove_nans(data: pd.DataFrame) -> pd.DataFrame:
    print("Number of NaNs in each column:\n", data.isna().sum(axis=0))
    max_nans_per_col = data.isna().sum(axis=0).max()
    print("max_nans_per_col = %d" % max_nans_per_col)
    data = data.loc[max_nans_per_col:]
    nans_remaining = data.isna().sum().sum()
    print(f"Remaining NaNs after removing the {max_nans_per_col} first rows: {nans_remaining}")
    print(f"Shape: {data.shape}")
    return data

def normalize(data: pd.DataFrame) -> pd.DataFrame:
    # Returns
    ret_cols = [col for col in data.columns if "ret" in col]
    for col in ret_cols:
        data.loc[:,col] *= 10
        
    # SMAs
    sma_cols = [col for col in data.columns if "sma" in col]
    for col in sma_cols:
        data.loc[:,col] *= 10
        
    # MACD
    macd_cols = [col for col in data.columns if "macd" in col]
    for col in macd_cols:
        data.loc[:,col] *= 20
    
    # VOL
    vol_cols = [col for col in data.columns if "vol" in col]
    for col in vol_cols:
        data.loc[:,col] *= 40
        
    # Print stats about the columns, 8 by 8 (to avoid some columns being replaced by ...)
    # Also only print stats for one of the ret_t-X columns because it should be almost the same for all of them
    all_but_ret = [col for col in data.columns if not "ret" in col or "ret_t-0" in col]
    print("Stats about each column after normalization:")
    for i in range(0, len(all_but_ret), 8):
        print(data[all_but_ret[i:i+8]].describe())
    return data
