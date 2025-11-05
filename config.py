# config.py
"""
Ultra-High Win Rate XAUUSD Trading Bot Configuration
"""

class TradingConfig:
    # === MT5 Connection ===
    MT5_LOGIN = YOUR_LOGIN
    MT5_PASSWORD = "YOUR_PASSWORD"
    MT5_SERVER = "YOUR_SERVER"
    
    # === Symbol Settings ===
    SYMBOL = "XAUUSD"
    MAGIC_NUMBER = 234000
    
    # === Risk Management ===
    MAX_RISK_PER_TRADE = 0.01  # 1% of account per trade
    MAX_TRADES_PER_DAY = 5     # Limit overtrading
    MAX_OPEN_POSITIONS = 2
    MIN_CONFIDENCE = 0.80      # Minimum 80% confidence to trade
    
    # === TP/SL Settings ===
    MIN_RR_RATIO = 2.0         # Minimum 1:2 Risk/Reward
    MIN_SL_PIPS = 8.0          # Minimum stop loss
    MIN_TP_PIPS = 16.0         # Minimum take profit
    ATR_SL_MULTIPLIER = 1.2    # SL = ATR * multiplier
    ATR_TP_MULTIPLIER = 3.0    # TP = ATR * multiplier
    
    # === Entry Filters ===
    MIN_ATR = 2.0              # Minimum volatility
    MAX_ATR = 25.0             # Maximum volatility
    MIN_ADX = 20               # Minimum trend strength
    
    # === Time Filters ===
    TRADING_START_HOUR = 8     # London open
    TRADING_END_HOUR = 16      # Before NY close
    
    # === Trend Detection ===
    EMA_FAST = 9
    EMA_MEDIUM = 21
    EMA_SLOW = 50
    EMA_TREND = 200
    MIN_TREND_SCORE = 5        # Minimum score to trade
    
    # === Position Management ===
    BREAKEVEN_PIPS = 12.0      # Move SL to breakeven after X pips
    TRAILING_START_PIPS = 20.0 # Start trailing after X pips
    TRAILING_DISTANCE_PIPS = 10.0  # Trail distance
    
    # === Model Paths ===
    PPO_MODEL_PATH = "models/ppo_xauusd_model.zip"
    REPLAY_BUFFER_PATH = "models/replay_buffer.pkl"
    
    # === Logging ===
    LOG_FILE = "logs/trading_bot.log"
    ENABLE_TELEGRAM = False
    TELEGRAM_TOKEN = "YOUR_TOKEN"
    TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"