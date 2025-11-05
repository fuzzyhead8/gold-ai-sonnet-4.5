# main_bot.py
"""
Main Trading Bot - Ultra High Win Rate XAUUSD System
90%+ Target Win Rate
"""
import MetaTrader5 as mt5
import time
import pandas as pd
from datetime import datetime
import logging

from config import TradingConfig
from trend_detector import TrendDetector
from entry_filter import EntryFilter
from risk_manager import RiskManager
from position_manager import PositionManager
from ai_ensemble import AIEnsemble

class XAUUSDTradingBot:
    
    def __init__(self):
        self.config = TradingConfig()
        self.trend_detector = TrendDetector()
        self.entry_filter = EntryFilter()
        self.risk_manager = RiskManager()
        self.position_manager = PositionManager()
        self.ai_ensemble = AIEnsemble()
        
        self.is_running = False
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(TradingConfig.LOG_FILE),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_mt5(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            self.logger.error("MT5 initialization failed")
            return False
        
        # Login (if needed)
        # authorized = mt5.login(TradingConfig.MT5_LOGIN, 
        #                        password=TradingConfig.MT5_PASSWORD,
        #                        server=TradingConfig.MT5_SERVER)
        
        # if not authorized:
        #     self.logger.error("MT5 login failed")
        #     return False
        
        self.logger.info("✅ MT5 initialized successfully")
        return True
    
    def get_market_data(self):
        """Fetch current market data"""
        df = self.trend_detector.get_ohlc_data(mt5.TIMEFRAME_M5, 500)
        return df
    
    def open_position(self, signal, lot_size, sl_pips, tp_pips):
        """Open a new position"""
        symbol_info = mt5.symbol_info(TradingConfig.SYMBOL)
        
        if symbol_info is None:
            self.logger.error(f"Symbol {TradingConfig.SYMBOL} not found")
            return False
        
        if not symbol_info.visible:
            if not mt5.symbol_select(TradingConfig.SYMBOL, True):
                self.logger.error(f"Failed to select {TradingConfig.SYMBOL}")
                return False
        
        # Get current price
        tick = mt5.symbol_info_tick(TradingConfig.SYMBOL)
        if tick is None:
            self.logger.error("Failed to get tick data")
            return False
        
        point = symbol_info.point
        
        # Determine order type and price
        if signal == 'LONG':
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            sl = price - (sl_pips * point)
            tp = price + (tp_pips * point)
        else:  # SHORT
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            sl = price + (sl_pips * point)
            tp = price - (tp_pips * point)
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": TradingConfig.SYMBOL,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": TradingConfig.MAGIC_NUMBER,
            "comment": f"AI Bot {signal}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return False
        
        self.logger.info(f"""
        ╔════════════════════════════════════════╗
        ║       🚀 NEW POSITION OPENED 🚀       ║
        ╠════════════════════════════════════════╣
        ║ Ticket:     #{result.order}
        ║ Signal:     {signal}
        ║ Lot Size:   {lot_size}
        ║ Entry:      {price:.2f}
        ║ Stop Loss:  {sl:.2f} ({sl_pips:.1f} pips)
        ║ Take Profit: {tp:.2f} ({tp_pips:.1f} pips)
        ║ R/R Ratio:  1:{tp_pips/sl_pips:.2f}
        ╚════════════════════════════════════════╝
        """)
        
        # Update counter
        self.entry_filter.increment_trade_counter()
        
        return True
    
    def analyze_and_trade(self):
        """Main analysis and trading logic"""
        try:
            # 1. Get market data
            df = self.get_market_data()
            
            if df is None or len(df) < 200:
                self.logger.warning("Insufficient data")
                return
            
            # 2. Multi-timeframe trend analysis
            trend_signal, trend_confidence = self.trend_detector.multi_timeframe_analysis()
            
            if trend_signal == 'NO_TRADE':
                self.logger.info("⏸️ No clear trend alignment - Standing by...")
                return
            
            self.logger.info(f"📊 Trend: {trend_signal} | Confidence: {trend_confidence:.2%}")
            
            # 3. AI Ensemble prediction
            ai_signal, ai_confidence = self.ai_ensemble.get_ensemble_prediction(df)
            
            if ai_signal == 'NO_TRADE':
                self.logger.info("🤖 AI: No trade signal")
                return
            
            self.logger.info(f"🤖 AI Signal: {ai_signal} | Confidence: {ai_confidence:.2%}")
            
            # 4. Check if trend and AI agree
            if trend_signal != ai_signal:
                self.logger.info("⚠️ Trend and AI disagree - No trade")
                return
            
            final_signal = trend_signal
            
            # 5. Combined confidence (weighted average)
            combined_confidence = (trend_confidence * 0.6 + ai_confidence * 0.4)
            
            self.logger.info(f"🎯 Combined Confidence: {combined_confidence:.2%}")
            
            # 6. Ultra-strict entry filter
            if not self.entry_filter.validate_entry(final_signal, combined_confidence, df):
                return
            
            # 7. Calculate position size and TP/SL
            account_info = self.risk_manager.get_account_info()
            
            if account_info is None:
                self.logger.error("Failed to get account info")
                return
            
            sl_pips, tp_pips = self.risk_manager.calculate_dynamic_tp_sl(
                df, final_signal, combined_confidence
            )
            
            lot_size = self.risk_manager.calculate_position_size(
                account_info['balance'], sl_pips, combined_confidence
            )
            
            self.logger.info(f"💼 Position: {lot_size} lots | SL: {sl_pips:.1f} | TP: {tp_pips:.1f}")
            
            # 8. Execute trade
            success = self.open_position(final_signal, lot_size, sl_pips, tp_pips)
            
            if success:
                self.logger.info("✅ Trade executed successfully!")
            else:
                self.logger.error("❌ Trade execution failed")
        
        except Exception as e:
            self.logger.error(f"Error in analyze_and_trade: {e}", exc_info=True)
    
    def run(self):
        """Main bot loop"""
        if not self.initialize_mt5():
            return
        
        self.is_running = True
        self.logger.info("""
        ╔════════════════════════════════════════════════╗
        ║   🤖 XAUUSD ULTRA-HIGH WIN RATE BOT STARTED   ║
        ║          Target: 90%+ Win Rate                ║
        ╚════════════════════════════════════════════════╝
        """)
        
        iteration = 0
        
        try:
            while self.is_running:
                iteration += 1
                
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Iteration #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.logger.info(f"{'='*60}")
                
                # 1. Manage existing positions
                self.position_manager.manage_all_positions()
                
                # 2. Analyze and potentially open new trades
                self.analyze_and_trade()
                
                # 3. Display account status
                account_info = self.risk_manager.get_account_info()
                if account_info:
                    self.logger.info(f"""
                    💰 Account Status:
                       Balance: ${account_info['balance']:.2f}
                       Equity: ${account_info['equity']:.2f}
                       Profit: ${account_info['profit']:.2f}
                       Positions: {mt5.positions_total()}
                    """)
                
                # 4. Wait before next iteration
                self.logger.info(f"⏳ Waiting 60 seconds before next check...")
                time.sleep(60)  # Check every minute
        
        except KeyboardInterrupt:
            self.logger.info("\n⚠️ Bot stopped by user")
        
        except Exception as e:
            self.logger.error(f"Critical error: {e}", exc_info=True)
        
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("🛑 Shutting down bot...")
        
        # Optional: Close all positions on shutdown
        # self.position_manager.close_all_positions("Bot shutdown")
        
        mt5.shutdown()
        self.logger.info("✅ MT5 connection closed")
        self.logger.info("👋 Bot shutdown complete")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    bot = XAUUSDTradingBot()
    bot.run()