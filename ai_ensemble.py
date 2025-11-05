# ai_ensemble.py
"""
AI Ensemble System
Combines multiple models for higher accuracy
"""
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from config import TradingConfig
import os

class AIEnsemble:
    
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load trained AI models"""
        # Load PPO model
        if os.path.exists(TradingConfig.PPO_MODEL_PATH):
            try:
                self.models['ppo'] = PPO.load(TradingConfig.PPO_MODEL_PATH)
                print("✅ PPO model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load PPO model: {e}")
                self.models['ppo'] = None
        else:
            print("⚠️ PPO model not found, will use rule-based only")
            self.models['ppo'] = None
        
        # Here you can add more models (LSTM, Transformer, XGBoost)
        # self.models['lstm'] = load_lstm_model()
        # self.models['transformer'] = load_transformer_model()
    
    def prepare_observation(self, df):
        """
        Prepare observation state for AI models
        Must match training shape (22 features from original code)
        """
        if len(df) < 50:
            return None
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['tick_volume'].values
        
        # Calculate features (match original 22 shape)
        features = []
        
        # 1. Price features
        features.append(close[-1])  # Current price
        features.append((close[-1] - close[-2]) / close[-2])  # Price change %
        features.append((high[-1] - low[-1]) / close[-1])  # Range %
        
        # 2. Moving averages
        ma_5 = np.mean(close[-5:])
        ma_10 = np.mean(close[-10:])
        ma_20 = np.mean(close[-20:])
        ma_50 = np.mean(close[-50:]) if len(close) >= 50 else ma_20
        
        features.append((close[-1] - ma_5) / close[-1])
        features.append((close[-1] - ma_10) / close[-1])
        features.append((close[-1] - ma_20) / close[-1])
        features.append((close[-1] - ma_50) / close[-1])
        
        # 3. RSI
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi.iloc[-1] / 100)  # Normalize
        
        # 4. ATR
        tr = np.maximum(high[-14:] - low[-14:], 
                       np.maximum(np.abs(high[-14:] - np.roll(close[-14:], 1)),
                                 np.abs(low[-14:] - np.roll(close[-14:], 1))))
        atr = np.mean(tr)
        features.append(atr / close[-1])  # Normalize
        
        # 5. Volume
        avg_volume = np.mean(volume[-20:])
        features.append(volume[-1] / avg_volume if avg_volume > 0 else 1.0)
        
        # 6. MACD
        ema_12 = pd.Series(close).ewm(span=12).mean()
        ema_26 = pd.Series(close).ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        features.append((macd.iloc[-1] - signal.iloc[-1]) / close[-1])
        
        # 7. Bollinger Bands
        bb_middle = ma_20
        bb_std = np.std(close[-20:])
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        features.append((close[-1] - bb_lower) / (bb_upper - bb_lower))
        
        # 8. Support/Resistance distance
        support = np.min(low[-50:])
        resistance = np.max(high[-50:])
        features.append((close[-1] - support) / close[-1])
        features.append((resistance - close[-1]) / close[-1])
        
        # 9. Trend indicators
        features.append(1.0 if close[-1] > ma_50 else 0.0)
        features.append(1.0 if ma_5 > ma_20 else 0.0)
        
        # 10. Momentum
        momentum_5 = (close[-1] - close[-5]) / close[-5]
        momentum_10 = (close[-1] - close[-10]) / close[-10]
        features.append(momentum_5)
        features.append(momentum_10)
        
        # 11. Volatility
        volatility = np.std(close[-20:]) / np.mean(close[-20:])
        features.append(volatility)
        
        # 12. Time of day (normalized 0-1)
        from datetime import datetime
        current_hour = datetime.now().hour
        features.append(current_hour / 24.0)
        
        # 13. Additional feature to reach 22
        high_low_ratio = (high[-1] - low[-1]) / low[-1]
        features.append(high_low_ratio)
        
        return np.array(features, dtype=np.float32)
    
    def get_ppo_prediction(self, observation):
        """Get prediction from PPO model"""
        if self.models['ppo'] is None:
            return 0, 0.5  # Neutral if no model
        
        try:
            action, _states = self.models['ppo'].predict(observation, deterministic=True)
            
            # Action: 0=hold, 1=buy, 2=sell (depends on training)
            # Convert to our format
            if action == 1:
                return 'LONG', 0.75
            elif action == 2:
                return 'SHORT', 0.75
            else:
                return 'HOLD', 0.5
        except Exception as e:
            print(f"⚠️ PPO prediction error: {e}")
            return 'HOLD', 0.5
    
    def get_rule_based_prediction(self, df):
        """
        Rule-based prediction as fallback/confirmation
        Simple but robust strategy
        """
        close = df['close'].values
        
        # EMAs
        ema_9 = pd.Series(close).ewm(span=9).mean().iloc[-1]
        ema_21 = pd.Series(close).ewm(span=21).mean().iloc[-1]
        ema_50 = pd.Series(close).ewm(span=50).mean().iloc[-1]
        
        # RSI
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        current_price = close[-1]
        
        # Simple rules
        score = 0
        
        if ema_9 > ema_21 > ema_50:
            score += 2
        elif ema_9 < ema_21 < ema_50:
            score -= 2
        
        if current_price > ema_50:
            score += 1
        else:
            score -= 1
        
        if 30 < rsi < 70:  # Not extreme
            confidence = 0.8
        else:
            confidence = 0.6
        
        if score >= 2:
            return 'LONG', confidence
        elif score <= -2:
            return 'SHORT', confidence
        else:
            return 'HOLD', 0.5
    
    def get_ensemble_prediction(self, df):
        """
        Combine predictions from all models
        Weighted voting system
        """
        observation = self.prepare_observation(df)
        
        if observation is None:
            return 'NO_TRADE', 0.0
        
        predictions = []
        confidences = []
        weights = []
        
        # 1. PPO model prediction (weight: 0.5)
        ppo_signal, ppo_conf = self.get_ppo_prediction(observation)
        predictions.append(ppo_signal)
        confidences.append(ppo_conf)
        weights.append(0.5)
        
        # 2. Rule-based prediction (weight: 0.5)
        rule_signal, rule_conf = self.get_rule_based_prediction(df)
        predictions.append(rule_signal)
        confidences.append(rule_conf)
        weights.append(0.5)
        
        # Add more models here with their weights
        # Example:
        # lstm_signal, lstm_conf = self.get_lstm_prediction(observation)
        # predictions.append(lstm_signal)
        # confidences.append(lstm_conf)
        # weights.append(0.3)
        
        # Voting logic
        long_votes = sum(w for p, w in zip(predictions, weights) if p == 'LONG')
        short_votes = sum(w for p, w in zip(predictions, weights) if p == 'SHORT')
        hold_votes = sum(w for p, w in zip(predictions, weights) if p == 'HOLD')
        
        total_weight = sum(weights)
        
        # Calculate weighted confidence
        avg_confidence = sum(c * w for c, w in zip(confidences, weights)) / total_weight
        
        # Determine final signal
        if long_votes > short_votes and long_votes > hold_votes:
            final_signal = 'LONG'
        elif short_votes > long_votes and short_votes > hold_votes:
            final_signal = 'SHORT'
        else:
            final_signal = 'NO_TRADE'
            avg_confidence = 0.0
        
        # Boost confidence if unanimous
        if len(set([p for p in predictions if p != 'HOLD'])) == 1:
            avg_confidence = min(avg_confidence * 1.15, 1.0)
        
        return final_signal, avg_confidence