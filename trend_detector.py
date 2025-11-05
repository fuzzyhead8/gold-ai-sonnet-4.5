# trend_detector.py
"""
Multi-Timeframe Trend Detection
"""
import MetaTrader5 as mt5
import pandas as pd
from indicators import TechnicalIndicators
from config import TradingConfig

class TrendDetector:
    
    def __init__(self, symbol=TradingConfig.SYMBOL):
        self.symbol = symbol
        self.indicators = TechnicalIndicators()
    
    def get_ohlc_data(self, timeframe, bars=200):
        """Fetch OHLC data from MT5"""
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    def analyze_single_timeframe(self, df):
        """Analyze trend on single timeframe"""
        if df is None or len(df) < 200:
            return 'NEUTRAL', 0.0
        
        # Calculate EMAs
        ema_9 = self.indicators.calculate_ema(df, TradingConfig.EMA_FAST)
        ema_21 = self.indicators.calculate_ema(df, TradingConfig.EMA_MEDIUM)
        ema_50 = self.indicators.calculate_ema(df, TradingConfig.EMA_SLOW)
        ema_200 = self.indicators.calculate_ema(df, TradingConfig.EMA_TREND)
        
        current_price = df['close'].iloc[-1]
        
        # Trend scoring system
        score = 0
        
        # 1. EMA alignment
        if ema_9.iloc[-1] > ema_21.iloc[-1]:
            score += 1
        else:
            score -= 1
        
        if ema_21.iloc[-1] > ema_50.iloc[-1]:
            score += 2
        else:
            score -= 2
        
        if ema_50.iloc[-1] > ema_200.iloc[-1]:
            score += 3
        else:
            score -= 3
        
        # 2. Price position
        if current_price > ema_200.iloc[-1]:
            score += 2
        else:
            score -= 2
        
        # 3. EMA slope
        ema_50_slope = (ema_50.iloc[-1] - ema_50.iloc[-10]) / 10
        if ema_50_slope > 0:
            score += 1
        else:
            score -= 1
        
        # Calculate confidence (0-1)
        confidence = abs(score) / 9.0  # Max score = 9
        
        # Determine direction
        if score >= 5:
            return 'STRONG_LONG', confidence
        elif score >= 3:
            return 'LONG', confidence
        elif score <= -5:
            return 'STRONG_SHORT', confidence
        elif score <= -3:
            return 'SHORT', confidence
        else:
            return 'NEUTRAL', confidence
    
    def multi_timeframe_analysis(self):
        """
        Analyze M5, M15, H1 and require alignment
        """
        # Get data for multiple timeframes
        m5_data = self.get_ohlc_data(mt5.TIMEFRAME_M5, 200)
        m15_data = self.get_ohlc_data(mt5.TIMEFRAME_M15, 200)
        h1_data = self.get_ohlc_data(mt5.TIMEFRAME_H1, 200)
        
        if m5_data is None or m15_data is None or h1_data is None:
            return 'NO_TRADE', 0.0
        
        # Analyze each timeframe
        m5_trend, m5_conf = self.analyze_single_timeframe(m5_data)
        m15_trend, m15_conf = self.analyze_single_timeframe(m15_data)
        h1_trend, h1_conf = self.analyze_single_timeframe(h1_data)
        
        # Check for alignment
        trends = [m5_trend, m15_trend, h1_trend]
        
        # Count bullish vs bearish signals
        bullish_count = sum(1 for t in trends if 'LONG' in t)
        bearish_count = sum(1 for t in trends if 'SHORT' in t)
        
        # Perfect alignment (all 3 timeframes agree)
        if bullish_count == 3:
            avg_confidence = (m5_conf + m15_conf + h1_conf) / 3
            return 'LONG', min(avg_confidence * 1.2, 1.0)  # Boost confidence
        
        elif bearish_count == 3:
            avg_confidence = (m5_conf + m15_conf + h1_conf) / 3
            return 'SHORT', min(avg_confidence * 1.2, 1.0)
        
        # 2 out of 3 alignment (H1 must agree)
        elif bullish_count >= 2 and 'LONG' in h1_trend:
            avg_confidence = (m5_conf + m15_conf + h1_conf) / 3
            return 'LONG', avg_confidence * 0.8
        
        elif bearish_count >= 2 and 'SHORT' in h1_trend:
            avg_confidence = (m5_conf + m15_conf + h1_conf) / 3
            return 'SHORT', avg_confidence * 0.8
        
        else:
            return 'NO_TRADE', 0.0
    
    def get_trend_strength(self, df):
        """Calculate ADX trend strength"""
        adx, plus_di, minus_di = self.indicators.calculate_adx(df)
        
        current_adx = adx.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]
        
        if current_adx > TradingConfig.MIN_ADX:
            if current_plus_di > current_minus_di:
                return 'STRONG_LONG', current_adx / 100
            else:
                return 'STRONG_SHORT', current_adx / 100
        
        return 'WEAK', current_adx / 100