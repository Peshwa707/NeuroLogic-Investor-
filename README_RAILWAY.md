# 📈 ML Trading Bot - Quick Start (Railway)

This is the simplified README for deploying to Railway. For full documentation, see [README.md](README.md).

## 🚀 Deploy to Railway in 3 Steps

### Step 1: Click Deploy Button

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

### Step 2: Wait for Build

Railway will automatically:
- Detect the configuration
- Install dependencies
- Build the application
- Deploy to a public URL (2-5 minutes)

### Step 3: Configure API Keys

1. Open your deployed app URL
2. Click **"⚙️ Settings"** in the sidebar
3. Add your API keys (see below)
4. Click **"💾 Save Configuration"**
5. Go to **"📈 Trading Dashboard"**
6. Start predicting!

## 🔑 Getting Free API Keys

### Alpha Vantage (Required for Stocks)
1. Visit: https://www.alphavantage.co/support/#api-key
2. Enter your email → Get API key
3. Free: 5 requests/min, 500/day

### Binance (Optional for Crypto)
1. Visit: https://www.binance.com
2. Create account → API Management
3. Create read-only API key
4. Copy both Key and Secret

### News API (Optional)
1. Visit: https://newsapi.org/register
2. Register → Get API key
3. Free: 100 requests/day

## ✨ Features

- 📊 **Stock Predictions** - Yahoo Finance data
- 💰 **Crypto Predictions** - Binance integration
- 📈 **Technical Analysis** - 20+ indicators
- 🤖 **AI Forecasting** - LSTM neural networks
- ⚡ **Day Trading Signals** - Real-time BUY/SELL/HOLD
- 🎨 **Interactive Charts** - Beautiful visualizations
- ⚙️ **Web Configuration** - No redeployment needed

## 📱 How to Use

1. **Configure APIs** (Settings page)
2. **Enter Symbol** (e.g., AAPL, BTCUSDT)
3. **Select Date Range**
4. **Click "Load Data"**
5. **View Charts & Predictions**

## 💡 Tips for Railway

- Use **smaller date ranges** (3-6 months) for faster predictions
- **Cache results** - predictions are saved temporarily
- **Free tier** is sufficient for personal use
- **Upgrade to Pro** ($20/month) for more compute power

## ⚠️ Important Notes

- **Educational purposes only** - Not financial advice
- **Paper trade first** - Test before real money
- **No data persistence** - Models don't save across restarts (free tier)
- **Rate limits** - Respect API quotas

## 🆘 Troubleshooting

### "API key not configured"
→ Go to Settings page and add your API keys

### "No data found"
→ Check symbol spelling (AAPL not Apple)

### Predictions timeout
→ Use smaller date range or fewer epochs

### Build fails
→ Check Railway logs in dashboard

## 📚 Full Documentation

For complete documentation, examples, and advanced features:
- [Full README](README.md)
- [Railway Deployment Guide](RAILWAY_DEPLOYMENT.md)
- [API Documentation](ml_trading_bot/)

## 🤝 Support

- **Railway Issues**: [Railway Discord](https://discord.gg/railway)
- **App Issues**: Check [GitHub Issues](https://github.com/YOUR_USERNAME/NeuroLogic-Investor-/issues)
- **Documentation**: See [README.md](README.md)

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

**Enjoy your AI-powered trading predictions!** 🎉

*Remember: Always do your own research before making investment decisions.*
