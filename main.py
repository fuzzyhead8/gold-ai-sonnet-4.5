#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
"""
Simplified MT5 Gold AI Trader Main Module

This module handles the core trading loop but delegates all trading logic to individual strategies.
Clean separation of concerns for better maintainability and backtest compatibility.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
import threading
import argparse
import signal
import sys
import io
import os
from datetime import datetime

from models.strategy_classifier import StrategyClassifier
from news_module.news_fetcher import NewsFetcher
from news_module.sentiment_analyzer import NewsSentimentHandler
from mt5_connector.account_handler import AccountHandler

# Import strategies
from strategies.scalping import ScalpingStrategy
from strategies.day_trading import DayTradingStrategy
from strategies.swing import SwingTradingStrategy
from strategies.golden_scalping_simplified import GoldenScalpingStrategySimplified
from strategies.goldstorm_strategy import GoldStormStrategy
from strategies.vwap_strategy import VWAPStrategy
from strategies.multi_rsi_ema import MultiRSIEMAStrategy
from strategies.range_oscillator import RangeOscillatorStrategy
from strategies.orb_strategy import ORBStrategy
from strategies.gold_breaker import GoldBreakerStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy

from dotenv import load_dotenv
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
load_dotenv()

@dataclass
class Position:
    """Represents an open position"""
    ticket: int
    symbol: str
    direction: str
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    open_time: datetime
    current_profit: float = 0.0

class ShieldProtocol:
    """Shield Protocol - Risk Management System"""

    def __init__(self, max_risk_per_trade: float = 0.02, max_daily_loss: float = 0.05):
        self.max_risk_per_trade = max_risk_per_trade  # 2% per trade
        self.max_daily_loss = max_daily_loss  # 5% daily loss
        self.daily_loss = 0.0
        self.equity_lock_triggered = False
        self.initial_balance = 0.0

    def calculate_position_size(self, account_balance: float, stop_loss_pips: float,
                              symbol: str, risk_amount: float = None) -> float:
        """Calculate position size based on risk management rules"""
        if risk_amount is None:
            risk_amount = account_balance * self.max_risk_per_trade

        # Dynamic risk limiter - adjust based on volatility
        volatility_multiplier = self._get_volatility_multiplier(symbol)

        adjusted_risk = risk_amount * volatility_multiplier

        # Convert pips to price
        pip_value = self._get_pip_value(symbol, account_balance)

        if pip_value == 0 or stop_loss_pips == 0:
            return 0.01  # Minimum lot size

        position_size = adjusted_risk / (stop_loss_pips * pip_value)

        # Ensure minimum and maximum lot sizes
        position_size = max(0.01, min(position_size, 100.0))

        return round(position_size, 2)

    def _get_volatility_multiplier(self, symbol: str) -> float:
        """Get volatility multiplier for dynamic risk adjustment"""
        # Simplified volatility calculation
        # In real implementation, this would use ATR or similar
        base_volatility = 1.0

        if 'XAU' in symbol:  # Gold specific
            base_volatility = 1.2  # Gold is more volatile

        return base_volatility

    def _get_pip_value(self, symbol: str, account_balance: float) -> float:
        """Calculate pip value for the symbol"""
        # Simplified pip value calculation
        if 'XAU' in symbol:
            return account_balance * 0.00001  # Approximate for gold
        return account_balance * 0.0001  # Standard forex

    def check_equity_lock(self, current_equity: float, initial_balance: float) -> bool:
        """Check if equity lock should be triggered"""
        loss_percentage = (initial_balance - current_equity) / initial_balance

        if loss_percentage >= 0.1:  # 10% loss triggers lock
            self.equity_lock_triggered = True
            logging.warning(f"Equity lock triggered. Loss: {loss_percentage:.2%}")
            return True

        return False

    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed"""
        return not self.equity_lock_triggered

# Fix console encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class TradingBot:
    """Simplified trading bot that delegates all trading logic to strategies"""
    
    def __init__(self, symbol="XAUUSD", bars=200, manual_strategy=None, use_sentiment=False):
        self.symbol = symbol
        self.bars = bars
        self.manual_strategy = manual_strategy
        self.classifier = StrategyClassifier()
        self.sentiment_handler = NewsSentimentHandler()
        self.account_handler = AccountHandler()
        self.shield_protocol = ShieldProtocol()
        self.acc_info = None
        self.initial_balance = 0.0
        self.connected = False
        self.stop_event = threading.Event()
        self.use_sentiment = use_sentiment
        
        # Strategy instances
        self.strategies = {
            'scalping': ScalpingStrategy(symbol),
            'day_trading': DayTradingStrategy(symbol),
            'golden': GoldenScalpingStrategySimplified(symbol),
            'goldstorm': GoldStormStrategy(symbol),
            'vwap': VWAPStrategy(symbol),
            'multi_rsi_ema': MultiRSIEMAStrategy(symbol),
            'range_oscillator': RangeOscillatorStrategy(symbol),
            'orb': ORBStrategy(symbol),
            'gold_breaker': GoldBreakerStrategy(symbol),
            'mean_reversion': MeanReversionStrategy(symbol),
            'swing': SwingTradingStrategy(symbol)
        }

    def initialize_mt5(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            logging.error("MT5 initialization failed")
            return False
        
        # Select the trading symbol
        if not mt5.symbol_select(self.symbol, True):
            logging.error(f"Failed to select symbol {self.symbol}")
            mt5.shutdown()
            return False
        
        self.acc_info = self.account_handler.get_account_info()
        if not self.acc_info:
            logging.error("Account info retrieval failed")
            mt5.shutdown()
            return False
        
        self.initial_balance = self.acc_info['balance']
        self.shield_protocol.initial_balance = self.initial_balance
        
        self.connected = True
        logging.info("MT5 initialized and account info retrieved")
        return True

    def shutdown_mt5(self):
        """Clean shutdown of MT5 connection"""
        if self.connected:
            self.close_all_positions()
            mt5.shutdown()
            self.connected = False
            logging.info("MT5 connection closed and positions cleared")

    def close_all_positions(self):
        """Emergency close all positions"""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            for pos in positions:
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "price": mt5.symbol_info_tick(self.symbol).bid if pos.type == 0 else mt5.symbol_info_tick(self.symbol).ask,
                    "deviation": 20,
                    "magic": 999999,
                    "comment": "Emergency Close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                mt5.order_send(close_request)
            logging.info("All positions closed")

    def load_classifier(self):
        """Load the strategy classification model"""
        try:
            self.classifier.load_model('models/trained_classifier.joblib')
            logging.info("Classifier loaded from trained model")
        except Exception as e:
            logging.error(f"Failed to load classifier: {e}")

    def get_sentiment_score(self) -> float:
        """Get current market sentiment score"""
        try:
            fetcher = NewsFetcher(api_key=os.getenv('NEWS_API_KEY'), cache_duration_hours=4)
            
            # Try cached first to avoid API limits
            articles = fetcher.get_cached_or_fallback()
            if not articles:
                logging.info("No cached news, attempting fresh fetch...")
                articles = fetcher.fetch_news(page_size=10)
            
            sentiment_results = self.sentiment_handler.process_news(articles)
            sent_score = np.mean([item['score'] for item in sentiment_results]) if sentiment_results else 0
            
            logging.info(f"Using {len(articles)} news articles for sentiment analysis (score: {sent_score:.3f})")
            return sent_score
            
        except Exception as e:
            logging.warning(f"News processing failed: {e}")
            return 0  # Neutral sentiment as fallback

    def select_strategy(self, df: pd.DataFrame = None, sentiment_score: float = 0) -> str:
        """Select trading strategy based on market conditions"""
        if self.manual_strategy:
            return self.manual_strategy
        
        if df is None or len(df) < 10:
            # Fallback to sentiment-based selection
            if sentiment_score > 0.1:
                return 'golden'
            elif sentiment_score < -0.1:
                return 'goldstorm'
            else:
                return 'multi_rsi_ema'  # Neutral default
        
        # Use classifier to predict best strategy
        try:
            volatility = df['close'].rolling(window=10).std().iloc[-1]
            volume = df['tick_volume'].iloc[-1]
            momentum = df['close'].pct_change().rolling(5).mean().iloc[-1]
            
            input_df = pd.DataFrame([[volatility, volume, momentum, sentiment_score]], 
                                  columns=['volatility', 'volume', 'momentum', 'sentiment_score'])
            
            strategy_type = self.classifier.predict(input_df)[0]
            logging.info(f"AI predicted strategy: {strategy_type}")
            return strategy_type
            
        except Exception as e:
            logging.error(f"Strategy prediction failed: {e}")
            return 'goldstorm'  # Default fallback


    def get_market_data(self, timeframe: int, bars: int = None) -> pd.DataFrame:
        """Fetch market data with retry logic"""
        bars = bars or self.bars
        data = None
        retry_count = 0
        max_retries = 3
        
        # Check if symbol is available
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            raise Exception(f"Symbol {self.symbol} not found or not selected")
        
        if not symbol_info.visible:
            if not mt5.symbol_select(self.symbol, True):
                raise Exception(f"Failed to select symbol {self.symbol}")
        
        while retry_count < max_retries and data is None:
            try:
                data = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, bars)
                if data is None:
                    error = mt5.last_error()
                    logging.error(f"MT5 data fetch failed (attempt {retry_count + 1}/{max_retries}): Error {error[0]} - {error[1]}")
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(2 ** retry_count)
                else:
                    break
            except Exception as e:
                logging.error(f"Unexpected error in MT5 data fetch (attempt {retry_count + 1}/{max_retries}): {e}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2 ** retry_count)
        
        if data is None:
            error = mt5.last_error()
            raise Exception(f"Failed to fetch market data after {max_retries} attempts. Last error: {error[0]} - {error[1]}")
        
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        logging.info(f"Successfully fetched {len(df)} bars of {self.symbol} on timeframe {timeframe}")
        return df

    def trading_iteration(self):
        """Single trading iteration"""
        if self.stop_event.is_set():
            return
        
        try:
            # Get sentiment score
            if self.use_sentiment:
                sentiment_score = self.get_sentiment_score()
            else:
                sentiment_score = 0.0
            sentiment_str = "bullish" if sentiment_score > 0 else "bearish" if sentiment_score < 0 else "neutral"
            
            # Select strategy using sentiment (no data dependency to avoid timeframe mismatch)
            strategy_name = self.select_strategy(sentiment_score=sentiment_score)

            
            if strategy_name not in self.strategies:
                logging.error(f"Unknown strategy: {strategy_name}")
                return
            
            strategy = self.strategies[strategy_name]
            config = strategy.get_strategy_config()
            
            logging.info(f"Using {strategy_name} strategy")
            
            # Get appropriate timeframe data for the strategy
            timeframe = config.get('timeframe', mt5.TIMEFRAME_M15)
                
            df = self.get_market_data(timeframe)
            
            # Check data sufficiency
            if len(df) < 20:
                logging.warning(f"Insufficient data: {len(df)} bars. Skipping iteration.")
                return
            
            # Fetch fresh account info for dynamic balance
            acc_info = self.account_handler.get_account_info()
            if not acc_info:
                logging.warning("Failed to fetch account info, skipping iteration")
                return
            
            # Check shield protocol
            if self.shield_protocol.check_equity_lock(acc_info['equity'], self.initial_balance):
                logging.warning("Trading stopped by Shield Protocol - equity lock triggered")
                return
            
            lot_scale = float(os.getenv('LOT_SCALE', '1.0'))
            effective_balance = acc_info['balance'] * lot_scale
            logging.info(f"Using effective balance for lot sizing: {effective_balance} (real_balance={acc_info['balance']} * LOT_SCALE={lot_scale})")
            
            # Execute strategy
            positions_before = mt5.positions_get(symbol=self.symbol)
            logging.info(f"Positions before execution: {len(positions_before) if positions_before else 0}")
            
            if hasattr(strategy, 'execute_strategy'):
                # New style strategy with built-in execution
                result = strategy.execute_strategy(df, sentiment_str, effective_balance)
                logging.info(f"{strategy_name} execution result: {result}")
            else:
                # Legacy strategy - just generate signals
                signals = strategy.generate_signals(df)
                latest_signal = signals['signal'].iloc[-1]
                logging.warning(f"{strategy_name} has NO execute_strategy() - won't trade! Signal: {latest_signal} (legacy mode)")
            
            positions_after = mt5.positions_get(symbol=self.symbol)
            logging.info(f"Positions after execution: {len(positions_after) if positions_after else 0}")
            
            # Get sleep time from strategy config
            sleep_time = config.get('sleep_time', 300)
            
            logging.info(f"Adjusted sleep time for {strategy_name}: {sleep_time} seconds")

            
            logging.info(f"{strategy_name} iteration completed. Sleeping for {sleep_time} seconds...")
            
            # Sleep with interruption check
            start = time.time()
            while time.time() - start < sleep_time and not self.stop_event.is_set():
                time.sleep(1)
                
        except Exception as e:
            logging.error(f"Trading iteration failed: {e}")
            time.sleep(60)  # Wait 1 minute before retry

    def trading_loop(self):
        """Main trading loop"""
        while not self.stop_event.is_set():
            try:
                self.trading_iteration()
            except Exception as e:
                logging.error(f"Unexpected error in trading loop: {e}")
                time.sleep(60)

    def run(self):
        """Main entry point"""
        def signal_handler(sig, frame):
            logging.info("SIGINT received, setting stop event")
            self.stop_event.set()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Initialize MT5
        if not self.initialize_mt5():
            return

        # Load classifier
        self.load_classifier()

        # Start trading thread
        trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
        trading_thread.start()

        try:
            while trading_thread.is_alive() and not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            logging.info("Stopping... Ctrl+C received")
        finally:
            self.stop_event.set()
            self.shutdown_mt5()


if __name__ == "__main__":
    import sys
    
    test_mode = False
    if len(sys.argv) > 1 and sys.argv[1] == 'test_consistency':
        test_mode = True
        # For test mode, use default symbol and strategy
        sys.argv = [sys.argv[0]]  # Remove the test_consistency arg so parser doesn't see it
    
    parser = argparse.ArgumentParser(description='Clean MT5 Gold AI Trader')
    parser.add_argument('symbol', nargs='?', type=str, default='XAUUSD', 
                       help='Trading symbol (default: XAUUSD)')
    parser.add_argument('bars', nargs='?', type=int, default=200, 
                       help='Number of historical bars (default: 200)')
    parser.add_argument('strategy', nargs='?', type=str, default='auto',
                       choices=['auto', 'scalping', 'swing', 'day_trading', 'golden', 'goldstorm', 'vwap', 'multi_rsi_ema', 'range_oscillator', 'orb', 'gold_breaker', 'mean_reversion'],
                       help='Trading strategy (default: auto)')
    parser.add_argument('--with_sentiment', action='store_true', 
                       help='Enable sentiment analysis in strategy selection and execution')
    
    args = parser.parse_args()
    
    bot = TradingBot(
        symbol=args.symbol, 
        bars=args.bars, 
        manual_strategy=args.strategy if args.strategy != 'auto' else None,
        use_sentiment=args.with_sentiment
    )
    
    if test_mode:
        # Run single iteration for testing
        def signal_handler(sig, frame):
            logging.info("SIGINT received, setting stop event")
            bot.stop_event.set()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Initialize MT5
        if not bot.initialize_mt5():
            sys.exit(1)

        # Load classifier
        bot.load_classifier()

        # Run single trading iteration
        try:
            bot.trading_iteration()
            logging.info("Test consistency iteration completed successfully")
        except Exception as e:
            logging.error(f"Test iteration failed: {e}")
        finally:
            bot.shutdown_mt5()
    else:
        bot.run()
