# position_manager.py
"""
Advanced Position Management
- Trailing Stop
- Breakeven Protection
- Partial Profit Taking
"""
import MetaTrader5 as mt5
from datetime import datetime
from config import TradingConfig

class PositionManager:
    
    def __init__(self):
        self.symbol = TradingConfig.SYMBOL
        self.breakeven_triggered = {}  # Track which positions moved to BE
        self.trailing_activated = {}   # Track trailing stop status
    
    def get_open_positions(self):
        """Get all open positions for our symbol"""
        positions = mt5.positions_get(symbol=self.symbol)
        
        if positions is None:
            return []
        
        return list(positions)
    
    def calculate_profit_pips(self, position):
        """Calculate current profit in pips"""
        if position.type == mt5.ORDER_TYPE_BUY:
            profit_pips = (position.price_current - position.price_open)
        else:  # SELL
            profit_pips = (position.price_open - position.price_current)
        
        return profit_pips
    
    def move_to_breakeven(self, position):
        """Move stop loss to entry price (no-loss protection)"""
        ticket = position.ticket
        
        # Check if already moved
        if ticket in self.breakeven_triggered:
            return False
        
        profit_pips = self.calculate_profit_pips(position)
        
        # If profit >= threshold, move SL to breakeven + spread
        if profit_pips >= TradingConfig.BREAKEVEN_PIPS:
            point = mt5.symbol_info(self.symbol).point
            
            if position.type == mt5.ORDER_TYPE_BUY:
                new_sl = position.price_open + (2.0 * point)  # BE + 2 pips
            else:
                new_sl = position.price_open - (2.0 * point)
            
            # Modify position
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": self.symbol,
                "position": ticket,
                "sl": new_sl,
                "tp": position.tp,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.breakeven_triggered[ticket] = True
                print(f"✅ Position #{ticket} moved to BREAKEVEN at {new_sl:.2f}")
                return True
            else:
                print(f"❌ Failed to move to breakeven: {result.comment}")
                return False
        
        return False
    
    def apply_trailing_stop(self, position):
        """Apply trailing stop logic"""
        ticket = position.ticket
        profit_pips = self.calculate_profit_pips(position)
        
        # Only trail if profit is significant
        if profit_pips < TradingConfig.TRAILING_START_PIPS:
            return False
        
        point = mt5.symbol_info(self.symbol).point
        trail_distance = TradingConfig.TRAILING_DISTANCE_PIPS * point
        
        if position.type == mt5.ORDER_TYPE_BUY:
            # For BUY: trail below current price
            new_sl = position.price_current - trail_distance
            
            # Only move SL up, never down
            if new_sl > position.sl:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": self.symbol,
                    "position": ticket,
                    "sl": new_sl,
                    "tp": position.tp,
                }
                
                result = mt5.order_send(request)
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    if ticket not in self.trailing_activated:
                        self.trailing_activated[ticket] = True
                        print(f"🎯 Trailing STARTED for #{ticket}")
                    
                    print(f"📈 Trailing SL updated: {new_sl:.2f} (Profit: {profit_pips:.1f} pips)")
                    return True
        
        else:  # SELL
            # For SELL: trail above current price
            new_sl = position.price_current + trail_distance
            
            # Only move SL down, never up
            if new_sl < position.sl or position.sl == 0:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": self.symbol,
                    "position": ticket,
                    "sl": new_sl,
                    "tp": position.tp,
                }
                
                result = mt5.order_send(request)
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    if ticket not in self.trailing_activated:
                        self.trailing_activated[ticket] = True
                        print(f"🎯 Trailing STARTED for #{ticket}")
                    
                    print(f"📉 Trailing SL updated: {new_sl:.2f} (Profit: {profit_pips:.1f} pips)")
                    return True
        
        return False
    
    def partial_profit_take(self, position, percentage=0.5):
        """
        Close partial position (optional advanced feature)
        Close 50% at 2x risk, let rest run
        """
        ticket = position.ticket
        profit_pips = self.calculate_profit_pips(position)
        
        # Calculate initial risk
        if position.type == mt5.ORDER_TYPE_BUY:
            risk_pips = position.price_open - position.sl
        else:
            risk_pips = position.sl - position.price_open
        
        # If profit >= 2x risk, close 50%
        if profit_pips >= (risk_pips * 2.0):
            partial_volume = position.volume * percentage
            partial_volume = round(partial_volume, 2)
            
            # Close partial
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "position": ticket,
                "volume": partial_volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": position.price_current,
                "deviation": 20,
                "magic": TradingConfig.MAGIC_NUMBER,
                "comment": "Partial profit take",
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"💰 Partial profit taken: {percentage*100}% of position #{ticket}")
                return True
        
        return False
    
    def manage_all_positions(self):
        """
        Main position management loop
        Apply breakeven, trailing, partial profits
        """
        positions = self.get_open_positions()
        
        if not positions:
            return
        
        for position in positions:
            # 1. Move to breakeven first
            if position.ticket not in self.breakeven_triggered:
                self.move_to_breakeven(position)
            
            # 2. Apply trailing stop
            self.apply_trailing_stop(position)
            
            # 3. Optional: Partial profit (commented out by default)
            # self.partial_profit_take(position, percentage=0.5)
    
    def close_position(self, ticket, reason="Manual close"):
        """Manually close a position"""
        position = mt5.positions_get(ticket=ticket)
        
        if not position:
            print(f"Position #{ticket} not found")
            return False
        
        position = position[0]
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "position": ticket,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": position.price_current,
            "deviation": 20,
            "magic": TradingConfig.MAGIC_NUMBER,
            "comment": reason,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ Position #{ticket} closed: {reason}")
            
            # Clean up tracking
            if ticket in self.breakeven_triggered:
                del self.breakeven_triggered[ticket]
            if ticket in self.trailing_activated:
                del self.trailing_activated[ticket]
            
            return True
        else:
            print(f"❌ Failed to close #{ticket}: {result.comment}")
            return False
    
    def close_all_positions(self, reason="Close all"):
        """Emergency close all positions"""
        positions = self.get_open_positions()
        
        for position in positions:
            self.close_position(position.ticket, reason)