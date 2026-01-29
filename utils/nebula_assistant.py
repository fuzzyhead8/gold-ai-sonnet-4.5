"""
Nebula Assistant - AI Interaction Layer with GPT/Claude Integration

This module provides AI-powered confirmation for trading signals.
Used in indicator-first, AI-confirmation trading approach.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import random
import os

# AI imports
try:
    import openai
    import anthropic
    from dotenv import load_dotenv
    load_dotenv()
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logging.warning("AI libraries not available, using mock AI confirmation")

logger = logging.getLogger(__name__)

class NebulaAssistant:
    """AI Interaction Layer with GPT/Claude Integration"""

    def __init__(self):
        self.conversation_history = []
        self.market_context = {}

        # Initialize AI clients if available
        if AI_AVAILABLE:
            self.openai_client = None
            self.anthropic_client = None

            openai_key = os.getenv('OPENAI_API_KEY')
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')

            if openai_key:
                self.openai_client = openai.OpenAI(api_key=openai_key)
            if anthropic_key:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
        else:
            self.openai_client = None
            self.anthropic_client = None

    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(data) < period + 1:
            return 50.0

        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0

    def _calculate_indicator_signals(self, market_data: pd.DataFrame) -> str:
        """Calculate indicator-based signals (simplified golden scalping logic)"""
        try:
            if len(market_data) < 30:
                return 'hold'

            # Calculate indicators
            ema_fast = market_data['close'].ewm(span=8).mean().iloc[-1]
            ema_slow = market_data['close'].ewm(span=21).mean().iloc[-1]
            rsi = self._calculate_rsi(market_data, 14)

            # MACD calculation
            exp1 = market_data['close'].ewm(span=12, adjust=False).mean()
            exp2 = market_data['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal_line

            macd_hist = histogram.iloc[-1]
            macd_hist_prev = histogram.iloc[-2] if len(histogram) > 1 else 0

            # Price momentum
            momentum = market_data['close'].pct_change(3).iloc[-1]

            # Volume check (simplified)
            volume_avg = market_data['volume'].rolling(20).mean().iloc[-1] if 'volume' in market_data.columns else 100
            current_volume = market_data['volume'].iloc[-1] if 'volume' in market_data.columns else 100

            # BUY conditions (simplified)
            buy_trend = ema_fast > ema_slow and ema_fast > market_data['close'].ewm(span=8).mean().iloc[-2]
            buy_rsi = 25 < rsi < 70
            buy_macd = macd_hist > macd_hist_prev and macd_hist > -0.5
            buy_momentum = momentum > 0.0001 and current_volume > volume_avg * 0.8

            if buy_trend and buy_rsi and buy_macd and buy_momentum:
                return 'buy'

            # SELL conditions (simplified)
            sell_trend = ema_fast < ema_slow and ema_fast < market_data['close'].ewm(span=8).mean().iloc[-2]
            sell_rsi = 30 < rsi < 75
            sell_macd = macd_hist < macd_hist_prev and macd_hist < 0.5
            sell_momentum = momentum < -0.0001 and current_volume > volume_avg * 0.8

            if sell_trend and sell_rsi and sell_macd and sell_momentum:
                return 'sell'

            return 'hold'

        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            return 'hold'

    def confirm_indicator_signal(self, indicator_signal: str, market_data: pd.DataFrame, use_real_ai: bool = False) -> Dict:
        """AI confirmation for indicator signals (only called when indicators suggest a trade)"""
        try:
            if indicator_signal == 'hold':
                return {
                    'confirmed': False,
                    'confidence': 0.0,
                    'reasoning': 'No indicator signal to confirm'
                }

            # For backtesting or when AI is not available, use mock confirmation
            if not use_real_ai or not AI_AVAILABLE or (not self.openai_client and not self.anthropic_client):
                return self._mock_ai_confirmation(indicator_signal, market_data)

            # Calculate additional context for AI analysis
            rsi = self._calculate_rsi(market_data, 14)
            current_price = market_data['close'].iloc[-1]

            # Trend strength
            sma_20 = market_data['close'].rolling(20).mean().iloc[-1]
            trend_strength = abs(current_price - sma_20) / sma_20

            # Volatility assessment
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.1

            # Try real AI analysis
            try:
                if self.openai_client:
                    return self._get_gpt_confirmation(indicator_signal, rsi, trend_strength, volatility)
                elif self.anthropic_client:
                    return self._get_claude_confirmation(indicator_signal, rsi, trend_strength, volatility)
            except Exception as e:
                logger.warning(f"Real AI confirmation failed: {e}, falling back to mock")
                return self._mock_ai_confirmation(indicator_signal, market_data)

        except Exception as e:
            logger.error(f"AI confirmation error: {e}")
            return {
                'confirmed': False,
                'confidence': 0.0,
                'reasoning': 'AI confirmation failed',
                'ai_called': True
            }

    def _mock_ai_confirmation(self, indicator_signal: str, market_data: pd.DataFrame) -> Dict:
        """Mock AI confirmation for indicator signals (used in backtesting)"""
        try:
            if indicator_signal == 'hold':
                return {
                    'confirmed': False,
                    'confidence': 0.0,
                    'reasoning': 'No indicator signal to confirm'
                }

            # Calculate additional context for AI analysis
            rsi = self._calculate_rsi(market_data, 14)
            current_price = market_data['close'].iloc[-1]

            # Trend strength
            sma_20 = market_data['close'].rolling(20).mean().iloc[-1]
            trend_strength = abs(current_price - sma_20) / sma_20

            # Volatility assessment
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.1

            # AI confirmation logic
            confirmed = False
            confidence = 0.0
            reasoning = ""

            if indicator_signal == 'buy':
                # Confirm buy signal
                if rsi < 65 and trend_strength > 0.005 and volatility < 0.25:
                    confirmed = True
                    confidence = 0.8
                    reasoning = f"AI confirms BUY: RSI {rsi:.1f} not overbought, strong trend ({trend_strength:.1%}), moderate volatility"
                elif rsi < 75 and trend_strength > 0.002:
                    confirmed = True
                    confidence = 0.6
                    reasoning = f"AI confirms BUY: Acceptable RSI {rsi:.1f}, moderate trend strength"
                else:
                    reasoning = f"AI rejects BUY: RSI {rsi:.1f} too high or weak trend ({trend_strength:.1%})"

            elif indicator_signal == 'sell':
                # Confirm sell signal
                if rsi > 35 and trend_strength > 0.005 and volatility < 0.25:
                    confirmed = True
                    confidence = 0.8
                    reasoning = f"AI confirms SELL: RSI {rsi:.1f} not oversold, strong trend ({trend_strength:.1f}), moderate volatility"
                elif rsi > 25 and trend_strength > 0.002:
                    confirmed = True
                    confidence = 0.6
                    reasoning = f"AI confirms SELL: Acceptable RSI {rsi:.1f}, moderate trend strength"
                else:
                    reasoning = f"AI rejects SELL: RSI {rsi:.1f} too low or weak trend ({trend_strength:.1f})"

            # Add some realistic rejection rate
            random.seed(int(datetime.now().timestamp()))

            if confirmed and random.random() < 0.15:  # 15% chance of AI rejecting good signals
                confirmed = False
                confidence = 0.3
                reasoning += " - AI detects additional uncertainty"

            return {
                'confirmed': confirmed,
                'confidence': confidence,
                'reasoning': reasoning,
                'ai_called': True
            }

        except Exception as e:
            logger.error(f"Mock AI confirmation error: {e}")
            return {
                'confirmed': False,
                'confidence': 0.0,
                'reasoning': 'Mock AI confirmation failed',
                'ai_called': True
            }

    def _get_gpt_confirmation(self, indicator_signal: str, rsi: float, trend_strength: float, volatility: float) -> Dict:
        """Get confirmation from GPT"""
        try:
            prompt = f"""
            You are an expert forex trader analyzing a potential {indicator_signal.upper()} signal for XAUUSD.

            Market Conditions:
            - RSI(14): {rsi:.1f}
            - Trend Strength: {trend_strength:.3f}
            - Volatility: {volatility:.3f}

            Based on these conditions, should I confirm this {indicator_signal.upper()} signal?

            Respond with JSON in this exact format:
            {{
                "confirmed": true/false,
                "confidence": 0.0-1.0,
                "reasoning": "brief explanation"
            }}

            Be conservative - only confirm if conditions are clearly favorable.
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )

            result = response.choices[0].message.content.strip()
            # Extract JSON
            import json
            start_idx = result.find('{')
            end_idx = result.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = result[start_idx:end_idx]
                analysis = json.loads(json_str)
                analysis['ai_called'] = True
                return analysis
            else:
                logger.warning("GPT response not in expected JSON format")
                return self._mock_ai_confirmation(indicator_signal, pd.DataFrame())

        except Exception as e:
            logger.error(f"GPT confirmation error: {e}")
            return self._mock_ai_confirmation(indicator_signal, pd.DataFrame())

    def _get_claude_confirmation(self, indicator_signal: str, rsi: float, trend_strength: float, volatility: float) -> Dict:
        """Get confirmation from Claude"""
        try:
            prompt = f"""
            You are a professional forex trader specializing in gold (XAUUSD) analysis.

            Analyze this potential {indicator_signal.upper()} signal:

            Technical Indicators:
            - RSI(14): {rsi:.1f}
            - Trend Strength: {trend_strength:.3f}
            - Market Volatility: {volatility:.3f}

            Should I confirm this {indicator_signal.upper()} trade?

            Provide your analysis in this exact JSON format:
            {{
                "confirmed": true/false,
                "confidence": 0.0-1.0,
                "reasoning": "detailed analysis explanation"
            }}

            Focus on risk management and only confirm when conditions are clearly favorable.
            """

            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            result = response.content[0].text.strip()
            # Extract JSON
            import json
            start_idx = result.find('{')
            end_idx = result.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = result[start_idx:end_idx]
                analysis = json.loads(json_str)
                analysis['ai_called'] = True
                return analysis
            else:
                logger.warning("Claude response not in expected JSON format")
                return self._mock_ai_confirmation(indicator_signal, pd.DataFrame())

        except Exception as e:
            logger.error(f"Claude confirmation error: {e}")
            return self._mock_ai_confirmation(indicator_signal, pd.DataFrame())

    def confirm_position_close(self, direction: str, market_data: pd.DataFrame, use_real_ai: bool = False) -> Dict:
        """AI confirmation for closing positions at take profit"""
        try:
            # For backtesting or when AI is not available, use mock confirmation
            if not use_real_ai or not AI_AVAILABLE:
                return self._mock_ai_confirm_close(direction, market_data)

            # Try real AI analysis for position closing
            try:
                if self.openai_client:
                    return self._get_gpt_close_confirmation(direction, market_data)
                elif self.anthropic_client:
                    return self._get_claude_close_confirmation(direction, market_data)
            except Exception as e:
                logger.warning(f"Real AI close confirmation failed: {e}, falling back to mock")
                return self._mock_ai_confirm_close(direction, market_data)

        except Exception as e:
            logger.error(f"AI close confirmation error: {e}")
            return {
                'confirmed': False,
                'confidence': 0.0,
                'reasoning': 'AI confirmation failed',
                'ai_called': True
            }

    def _mock_ai_confirm_close(self, direction: str, market_data: pd.DataFrame) -> Dict:
        """Mock AI confirmation for closing positions at take profit"""
        try:
            # Simple logic: AI confirms closing if profit target is reached
            # In reality, AI might analyze if market conditions suggest locking in profits

            # Basic confirmation logic - AI confirms most take profit scenarios
            random.seed(int(datetime.now().timestamp()))

            # AI confirms 85% of take profit opportunities
            confirmed = random.random() < 0.85

            if confirmed:
                confidence = 0.75
                reasoning = f"AI confirms closing {direction} position - profit target reached, market conditions favorable"
            else:
                confidence = 0.4
                reasoning = f"AI suggests holding {direction} position - potential for further gains"

            return {
                'confirmed': confirmed,
                'confidence': confidence,
                'reasoning': reasoning,
                'ai_called': True
            }

        except Exception as e:
            logger.error(f"Mock AI close confirmation error: {e}")
            return {
                'confirmed': False,
                'confidence': 0.0,
                'reasoning': 'Mock AI confirmation failed',
                'ai_called': True
            }

    def _get_gpt_close_confirmation(self, direction: str, market_data: pd.DataFrame) -> Dict:
        """Get GPT confirmation for closing position"""
        try:
            current_price = market_data['close'].iloc[-1]
            rsi = self._calculate_rsi(market_data, 14)

            prompt = f"""
            You are an expert forex trader deciding whether to close a {direction.upper()} position at take profit.

            Current Market Conditions:
            - Current Price: {current_price:.5f}
            - RSI(14): {rsi:.1f}
            - Position Direction: {direction.upper()}

            Should I close this profitable {direction.upper()} position now?

            Respond with JSON in this exact format:
            {{
                "confirmed": true/false,
                "confidence": 0.0-1.0,
                "reasoning": "brief explanation"
            }}

            Consider if market momentum suggests locking in profits vs. potential for more gains.
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )

            result = response.choices[0].message.content.strip()
            # Extract JSON
            import json
            start_idx = result.find('{')
            end_idx = result.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = result[start_idx:end_idx]
                analysis = json.loads(json_str)
                analysis['ai_called'] = True
                return analysis
            else:
                return self._mock_ai_confirm_close(direction, market_data)

        except Exception as e:
            logger.error(f"GPT close confirmation error: {e}")
            return self._mock_ai_confirm_close(direction, market_data)

    def _get_claude_close_confirmation(self, direction: str, market_data: pd.DataFrame) -> Dict:
        """Get Claude confirmation for closing position"""
        try:
            current_price = market_data['close'].iloc[-1]
            rsi = self._calculate_rsi(market_data, 14)

            prompt = f"""
            You are a professional forex trader deciding whether to close a {direction.upper()} position at take profit.

            Current Situation:
            - Position Type: {direction.upper()}
            - Current Price: {current_price:.5f}
            - RSI(14): {rsi:.1f}

            Should I close this position to lock in profits, or hold for potential further gains?

            Provide your analysis in this exact JSON format:
            {{
                "confirmed": true/false,
                "confidence": 0.0-1.0,
                "reasoning": "detailed analysis explanation"
            }}

            Consider market conditions and risk of profit reversal.
            """

            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            result = response.content[0].text.strip()
            # Extract JSON
            import json
            start_idx = result.find('{')
            end_idx = result.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = result[start_idx:end_idx]
                analysis = json.loads(json_str)
                analysis['ai_called'] = True
                return analysis
            else:
                return self._mock_ai_confirm_close(direction, market_data)

        except Exception as e:
            logger.error(f"Claude close confirmation error: {e}")
            return self._mock_ai_confirm_close(direction, market_data)
