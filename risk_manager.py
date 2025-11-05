# risk_manager.py
"""
Advanced Risk Management System
Dynamic TP/SL based on ATR and Confidence
"""
import MetaTrader5 as mt5
from indicators import TechnicalIndicators
from config import TradingConfig

class RiskManager:
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def calculate_position_size(self, account_balance, sl_pips, confidence):
        """
        Calculate lot size based on risk percentage
        Higher confidence = slightly larger position (but still within limits)
        """
        # Base risk per trade
        risk_amount = account_balance * TradingConfig.MAX_RISK_PER_TRADE
        
        # Adjust for confidence (0.8-1.2x multiplier)
        confidence_multiplier = 0.8 + (confidence * 0.4)  # 0.8 to 1.2
        risk_amount *= confidence_multiplier
        
        # XAUUSD: 1 pip = $0.01 per micro lot (0.01)
        # Example: $100 risk / 10 pip SL = $10 per pip = 1.0 lot
        pip_value = 0.01  # For XAUUSD micro lot
        
        lot_size = (risk_amount / sl_pips) / pip_value
        
        # Round to 0.01 lots
        lot_size = round(lot_size, 2)
        
        # Minimum and maximum limits
        lot_size = max(0.01, min(lot_size, 1.0))
        
        return lot_size
    
    def calculate_dynamic_tp_sl(self, df, signal, confidence):
        """
        ATR-based TP/SL calculation
        Higher confidence = tighter SL, wider TP
        """
        atr = self.indicators.calculate_atr(df)
        current_atr = atr.iloc[-1]
        
        # Base SL and TP on ATR
        if confidence >= 0.90:
            # Very high confidence: aggressive
            sl_multiplier = 1.0
            tp_multiplier = 3.5
        elif confidence >= 0.85:
            # High confidence
            sl_multiplier = 1.2
            tp_multiplier = 3.0
        elif confidence >= 0.80:
            # Good confidence
            sl_multiplier = 1.5
            tp_multiplier = 3.0
        else:
            # Medium confidence
            sl_multiplier = 1.8
            tp_multiplier = 2.5
        
        sl_pips = current_atr * sl_multiplier
        tp_pips = current_atr * tp_multiplier
        
        # Apply minimum requirements
        sl_pips = max(sl_pips, TradingConfig.MIN_SL_PIPS)
        tp_pips = max(tp_pips, TradingConfig.MIN_TP_PIPS)
        
        # Ensure minimum R/R ratio
        if tp_pips / sl_pips < TradingConfig.MIN_RR_RATIO:
            tp_pips = sl_pips * TradingConfig.MIN_RR_RATIO
        
        return round(sl_pips, 1), round(tp_pips, 1)
    
    def get_account_info(self):
        """Get MT5 account info"""
        account_info = mt5.account_info()
        
        if account_info is None:
            return None
        
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'profit': account_info.profit
        }