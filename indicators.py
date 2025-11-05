# indicators.py
"""
Technical Indicators for XAUUSD Analysis
"""
import pandas as pd
import numpy as np

class TechnicalIndicators:
    
    @staticmethod
    def calculate_ema(data, period):
        """Exponential Moving Average"""
        return data['close'].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_sma(data, period):
        """Simple Moving Average"""
        return data['close'].rolling(window=period).mean()
    
    @staticmethod
    def calculate_atr(data, period=14):
        """Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_rsi(data, period=14):
        """Relative Strength Index"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_adx(data, period=14):
        """Average Directional Index"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        atr = TechnicalIndicators.calculate_atr(data, period)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        """Moving Average Convergence Divergence"""
        ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(data, period=20, std_dev=2):
        """Bollinger Bands"""
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_volume_pressure(data):
        """Volume Buy/Sell Pressure"""
        volume = data['tick_volume']
        close = data['close']
        
        avg_volume = volume.rolling(20).mean()
        volume_ratio = volume / avg_volume
        
        price_change = close.diff()
        
        # Positive price change with high volume = buy pressure
        buy_pressure = (price_change > 0) & (volume_ratio > 1.5)
        sell_pressure = (price_change < 0) & (volume_ratio > 1.5)
        
        return buy_pressure, sell_pressure, volume_ratio
    
    @staticmethod
    def detect_support_resistance(data, lookback=50):
        """Support and Resistance Levels"""
        highs = data['high'].rolling(window=lookback).max()
        lows = data['low'].rolling(window=lookback).min()
        
        current_price = data['close'].iloc[-1]
        resistance = highs.iloc[-1]
        support = lows.iloc[-1]
        
        # Distance to levels
        dist_to_resistance = resistance - current_price
        dist_to_support = current_price - support
        
        return support, resistance, dist_to_support, dist_to_resistance