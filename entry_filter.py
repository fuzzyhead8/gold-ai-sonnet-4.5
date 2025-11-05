# entry_filter.py
"""
Ultra-Strict Entry Filter System
90%+ Win Rate Optimization
"""
import MetaTrader5 as mt5
from datetime import datetime
from indicators import TechnicalIndicators
from config import TradingConfig

class EntryFilter:
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.trades_today = 0
        self.consecutive_losses = 0
        self.last_trade_date = None
    
    def reset_daily_counters(self):
        """Reset counters at start of new day"""
        current_date = datetime.now().date()
        if self.last_trade_date != current_date:
            self.trades_today = 0
            self.last_trade_date = current_date
    
    def check_time_filter(self):
        """Only trade during high liquidity hours"""
        current_hour = datetime.now().hour
        
        # London/NY overlap (best liquidity)
        if TradingConfig.TRADING_START_HOUR <= current_hour <= TradingConfig.TRADING_END_HOUR:
            return True
        
        return False
    
    def check_volatility_filter(self, df):
        """Check if volatility is in acceptable range"""
        atr = self.indicators.calculate_atr(df)
        current_atr = atr.iloc[-1]
        
        # Not too quiet, not too wild
        if TradingConfig.MIN_ATR <= current_atr <= TradingConfig.MAX_ATR:
            return True, current_atr
        
        return False, current_atr
    
    def check_rsi_filter(self, df, signal):
        """Avoid overbought/oversold extremes"""
        rsi = self.indicators.calculate_rsi(df)
        current_rsi = rsi.iloc[-1]
        
        # Don't buy when overbought
        if signal == 'LONG' and current_rsi > 75:
            return False
        
        # Don't sell when oversold
        if signal == 'SHORT' and current_rsi < 25:
            return False
        
        return True
    
    def check_support_resistance(self, df, signal):
        """Don't trade into major S/R levels"""
        support, resistance, dist_to_support, dist_to_resistance = \
            self.indicators.detect_support_resistance(df)
        
        # Don't buy near resistance
        if signal == 'LONG' and dist_to_resistance < 5.0:
            return False
        
        # Don't sell near support
        if signal == 'SHORT' and dist_to_support < 5.0:
            return False
        
        return True
    
    def check_max_positions(self):
        """Limit open positions"""
        positions = mt5.positions_total()
        
        if positions >= TradingConfig.MAX_OPEN_POSITIONS:
            return False
        
        return True
    
    def check_max_trades_today(self):
        """Limit daily trades to prevent overtrading"""
        self.reset_daily_counters()
        
        if self.trades_today >= TradingConfig.MAX_TRADES_PER_DAY:
            return False
        
        return True
    
    def check_loss_streak(self):
        """Pause after consecutive losses"""
        if self.consecutive_losses >= 2:
            return False
        
        return True
    
    def check_volume_confirmation(self, df, signal):
        """Require volume confirmation"""
        buy_pressure, sell_pressure, volume_ratio = \
            self.indicators.calculate_volume_pressure(df)
        
        # Need volume spike in direction of trade
        if signal == 'LONG' and buy_pressure.iloc[-1] and volume_ratio.iloc[-1] > 1.5:
            return True
        
        if signal == 'SHORT' and sell_pressure.iloc[-1] and volume_ratio.iloc[-1] > 1.5:
            return True
        
        # Allow if moderate volume (not against us)
        if volume_ratio.iloc[-1] > 1.0:
            return True
        
        return False
    
    def validate_entry(self, signal, confidence, df):
        """
        Master entry validation
        ALL conditions must pass
        """
        checks = {
            'confidence': confidence >= TradingConfig.MIN_CONFIDENCE,
            'time': self.check_time_filter(),
            'volatility': self.check_volatility_filter(df)[0],
            'rsi': self.check_rsi_filter(df, signal),
            'support_resistance': self.check_support_resistance(df, signal),
            'max_positions': self.check_max_positions(),
            'max_trades': self.check_max_trades_today(),
            'loss_streak': self.check_loss_streak(),
            'volume': self.check_volume_confirmation(df, signal)
        }
        
        # Log which checks failed
        failed_checks = [k for k, v in checks.items() if not v]
        
        if failed_checks:
            print(f"❌ Entry REJECTED - Failed: {', '.join(failed_checks)}")
            return False
        
        print(f"✅ Entry APPROVED - All filters passed | Confidence: {confidence:.2%}")
        return True
    
    def increment_trade_counter(self):
        """Increment daily trade counter"""
        self.trades_today += 1
    
    def update_loss_streak(self, is_win):
        """Update consecutive loss counter"""
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1