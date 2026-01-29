import unittest
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.strategy_classifier import StrategyClassifier
from models.sentiment_model import SentimentAnalyzer
from lot_optimizer.optimizer import LotSizeOptimizer
from strategies.swing import SwingTradingStrategy

class TestAITradingBot(unittest.TestCase):

    def test_strategy_classifier_prediction(self):
        clf = StrategyClassifier()
        clf.train(pd.DataFrame({
            "volatility": [0.5, 1.2, 0.3, 0.8, 1.5, 0.2, 1.0, 0.6, 1.3, 0.4],
            "volume": [100000, 300000, 50000, 200000, 400000, 150000, 250000, 350000, 450000, 55000],
            "momentum": [1.1, -0.8, 0.4, 0.9, -1.2, 0.7, 1.4, -0.5, 0.2, 1.0],
            "sentiment_score": [0.9, -0.6, 0.2, 0.8, -0.9, 0.3, 0.7, -0.4, 0.1, 0.5],
            "strategy": ['daily', 'swing', 'scalping', 'golden', 'goldstorm', 'vwap', 'multi_rsi_ema', 'range_oscillator', 'orb', 'gold_breaker']
        }))
        pred = clf.predict([0.4, 150000, 0.5, 0.3])
        self.assertIn(pred, ['daily', 'swing', 'scalping', 'golden', 'goldstorm', 'vwap', 'multi_rsi_ema', 'range_oscillator', 'orb', 'gold_breaker'])

    def test_sentiment_model_output(self):
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_sentiment(["Gold is rising rapidly on global uncertainty."])
        self.assertTrue(len(result) > 0)
        self.assertIn(result[0]['sentiment'], ["bullish", "bearish"])

    def test_lot_size_optimizer(self):
        optimizer = LotSizeOptimizer(max_risk_percent=2)
        lot = optimizer.optimize(balance=10000, entry_price=1950, stop_price=1940)
        self.assertGreater(lot, 0)

    def test_swing_strategy_signals(self):
        data = pd.DataFrame({"close": [1940, 1945, 1950, 1955, 1952, 1958, 1960]})
        strategy = SwingTradingStrategy("XAUUSD")
        result = strategy.generate_signals(data)
        self.assertIn("signal", result.columns)

if __name__ == '__main__':
    unittest.main()
