import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import argparse
from dotenv import load_dotenv
import os

load_dotenv()

# MT5 Credentials - Loaded from .env file
MT5_LOGIN = int(os.getenv('MT5_LOGIN'))
MT5_PASSWORD = os.getenv('MT5_PASSWORD')
MT5_SERVER = os.getenv('MT5_SERVER')

# Trading parameters (default values from the original script)
# SYMBOL = "XAUUSD"  # Gold symbol in MT5
# COUNT = 1000  # Number of historical bars to fetch
# TIMEFRAME = mt5.TIMEFRAME_D1  # D1 timeframe (1 day)

def initialize_mt5(symbol, login=None, password=None, server=None):
    """Initialize MT5 connection."""
    if login is None:
        login = MT5_LOGIN
    if password is None:
        password = MT5_PASSWORD
    if server is None:
        server = MT5_SERVER
    
    if not mt5.initialize():
        print("MT5 initialize() failed")
        return False
    
    if not mt5.login(login, password=password, server=server):
        print(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False
    
    # Select symbol
    if not mt5.symbol_select(symbol, True):
        print(f"MT5 symbol_select({symbol}) failed")
        mt5.shutdown()
        return False
    
    print(f"MT5 connected to login {login} on {server}")
    return True

def get_historical_prices(symbol, count, timeframe):
    """Fetch historical prices from MT5 and return as DataFrame."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        print("Failed to fetch rates")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    # Keep all columns: time, open, high, low, close, tick_volume, spread, real_volume
    return df

def save_to_csv(df, symbol, tf_str, filename=None):
    """Save DataFrame to CSV file."""
    if filename is None:
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtests/{symbol}_{tf_str}_{now}.csv"
    if df is not None and not df.empty:
        df.to_csv(filename, index=False)
        print(f"Historical prices saved to {filename}")
    else:
        print("No data to save")

def main():
    """Main function to fetch and save historical prices."""
    parser = argparse.ArgumentParser(description='Fetch historical prices from MT5')
    parser.add_argument('--symbol', type=str, default='XAUUSD', help='Symbol to fetch (default: XAUUSD)')
    parser.add_argument('--timeframe', type=str, default='M1', help='Timeframe: M1, M5, M15, M30, H1, H4, D1 (default: M1)')
    parser.add_argument('--count', type=int, default=1000, help='Number of bars to fetch (default: 1000)')
    args = parser.parse_args()

    symbol = args.symbol
    tf_str = args.timeframe.upper()
    count = args.count

    timeframe_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1
    }
    if tf_str not in timeframe_map:
        print(f"Invalid timeframe: {tf_str}. Supported: M1, M5, M15, M30, H1, H4, D1")
        return
    timeframe = timeframe_map[tf_str]

    if not initialize_mt5(symbol):
        return
    
    try:
        df = get_historical_prices(symbol, count, timeframe)
        if df is not None:
            print(f"Fetched {len(df)} historical prices for {symbol} on {tf_str}")
            print(df.head())  # Preview the data
            save_to_csv(df, symbol, tf_str)
        else:
            print("Failed to fetch historical prices")
    finally:
        mt5.shutdown()
        print("MT5 connection closed")

if __name__ == "__main__":
    main()
