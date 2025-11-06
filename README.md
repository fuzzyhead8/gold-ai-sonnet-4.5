# 🚀 XAUUSD Ultra-High Win Rate Trading Bot v2.0

**90%+ Target Win Rate** - Production-Ready AI Trading System

## ⚡ Features

- ✅ Multi-timeframe trend analysis (M5, M15, H1)
- ✅ AI Ensemble (PPO + Rule-based)
- ✅ Ultra-strict 9-point entry filter
- ✅ Dynamic ATR-based TP/SL
- ✅ Automatic trailing stop & breakeven
- ✅ Maximum 5 trades per day (anti-overtrading)
- ✅ Risk management: 1% per trade
- ✅ Real-time position monitoring

## 📋 Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/gold-ai-sonnet-4.5
cd gold-ai-sonnet-4.5

# Install dependencies
pip install -r requirements.txt

```

## ⚙️ Configuration

Edit `config.py`:

```python
# MT5 Settings
MT5_LOGIN = YOUR_LOGIN
MT5_PASSWORD = "YOUR_PASSWORD"
MT5_SERVER = "YOUR_SERVER"

# Risk Settings
MAX_RISK_PER_TRADE = 0.01  # 1%
MAX_TRADES_PER_DAY = 5
MIN_CONFIDENCE = 0.80      # 80% minimum
```

## 🚀 Usage

### 1. Train PPO Model (First Time)
```bash
# Use original training script or your own
python train_ppo_model.py
```

### 2. Run Bot
```bash
python main_bot.py
```

### 3. Monitor
- Bot logs to console + `logs/trading_bot.log`
- Check MT5 terminal for positions
- Trades are managed automatically

## 📊 Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Win Rate | 90%+ | With strict filters |
| Risk/Reward | 1:2.5+ | Dynamic ATR-based |
| Max Trades/Day | 5 | Prevents overtrading |
| Max Drawdown | <5% | Position sizing |

## ⚠️ Important Notes

1. **Backtest First!** - Test on demo account minimum 1 month
2. **Never Risk More Than 1-2%** per trade
3. **Monitor Daily** - AI is not 100% autonomous
4. **News Events** - Bot pauses during high-impact news
5. **VPS Recommended** - For 24/7 operation

## 🛡️ Risk Warning

**Trading involves substantial risk of loss. Past performance is not indicative of future results. Only trade with capital you can afford to lose.**

## 📝 License

MIT License - Free to use, modify, distribute