# 🚂 Railway Deployment Guide

This guide will help you deploy the ML Trading Bot to Railway in just a few minutes.

## 📋 Prerequisites

- A [Railway](https://railway.app) account (free tier available)
- A GitHub account
- Your API keys (can be added later through the web interface)

## 🚀 Deployment Steps

### Option 1: One-Click Deploy (Recommended)

1. **Click the Deploy Button:**

   [![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/YOUR_USERNAME/NeuroLogic-Investor-)

2. **Configure the Deployment:**
   - Railway will automatically detect the configuration
   - Click "Deploy Now"
   - Wait for the build to complete (2-5 minutes)

3. **Access Your App:**
   - Once deployed, Railway will provide a public URL
   - Click on the URL to access your ML Trading Bot dashboard
   - Go to the Settings page to configure your API keys

### Option 2: Manual Deployment

#### Step 1: Fork/Clone Repository

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/NeuroLogic-Investor-.git
cd NeuroLogic-Investor-

# Or fork it on GitHub and clone your fork
```

#### Step 2: Push to Your GitHub

```bash
# If you cloned, push to your own repository
git remote set-url origin https://github.com/YOUR_USERNAME/NeuroLogic-Investor-.git
git push -u origin main
```

#### Step 3: Create New Project on Railway

1. Go to [Railway Dashboard](https://railway.app/dashboard)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your forked repository
5. Railway will automatically detect the configuration

#### Step 4: Configure Environment (Optional)

Railway will use the configuration from `railway.json`, `Procfile`, and `nixpacks.toml`.

No environment variables are required initially - you can add API keys through the web interface!

#### Step 5: Deploy

1. Railway will automatically build and deploy
2. Wait for deployment to complete (2-5 minutes)
3. Click on the generated URL to access your app

## ⚙️ Configuration

### Adding API Keys via Web Interface

1. Access your deployed Railway app
2. Click on **"⚙️ Settings"** in the sidebar
3. Enter your API keys:
   - **Alpha Vantage** (required for stocks): Get from [alphavantage.co](https://www.alphavantage.co/support/#api-key)
   - **Binance** (optional for crypto): Get from [binance.com](https://www.binance.com/en/my/settings/api-management)
   - **News API** (optional): Get from [newsapi.org](https://newsapi.org/register)
4. Click "💾 Save Configuration"
5. Test your APIs using the "🧪 Test APIs" button

### Alternative: Environment Variables (Advanced)

If you prefer to set API keys as Railway environment variables:

1. Go to your Railway project dashboard
2. Click on your service
3. Go to "Variables" tab
4. Add these variables:
   ```
   ALPHA_VANTAGE_API_KEY=your_key_here
   BINANCE_API_KEY=your_key_here
   BINANCE_API_SECRET=your_secret_here
   NEWS_API_KEY=your_key_here
   ```
5. Redeploy the service

## 🎯 Features on Railway

### What Works:
✅ **Web-based API configuration** - No need to redeploy to change keys
✅ **Stock market predictions** - Yahoo Finance integration
✅ **Cryptocurrency predictions** - Binance integration
✅ **Technical analysis** - All indicators work
✅ **Real-time charts** - Plotly visualizations
✅ **Persistent settings** - API keys are saved securely

### Limitations (Free Tier):
⚠️ **No persistent storage** - Trained models won't persist across restarts
⚠️ **Limited compute** - Training large models may timeout
⚠️ **Memory limits** - Simplified models recommended

### Recommendations for Railway:
1. **Use pre-fetched data** instead of real-time streaming
2. **Smaller date ranges** for faster predictions
3. **Simplified models** with fewer layers/epochs
4. **Cache predictions** to reduce computation

## 🔧 Troubleshooting

### Build Failures

**Issue:** Build fails with dependency errors
**Solution:**
```bash
# Ensure requirements.txt is up to date
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update dependencies"
git push
```

### Application Won't Start

**Issue:** Application crashes on startup
**Solution:** Check Railway logs:
1. Go to Railway dashboard
2. Click on your service
3. Check "Deployments" tab
4. Click on latest deployment
5. View logs for errors

### API Key Not Working

**Issue:** "API key not configured" message
**Solution:**
1. Go to Settings page in your app
2. Re-enter your API keys
3. Click "Test APIs" to verify
4. If using environment variables, check they're set correctly in Railway

### Port Issues

**Issue:** Application not accessible
**Solution:** Railway automatically sets `$PORT` - the app is configured to use it. No action needed.

### Memory/Timeout Issues

**Issue:** Training times out or crashes
**Solution:**
1. Use smaller date ranges (3-6 months instead of years)
2. Reduce model complexity (fewer LSTM units, fewer layers)
3. Use fewer epochs (10-20 instead of 50+)
4. Consider using pre-trained models

## 📊 Monitoring

### View Logs
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Link to your project
railway link

# View logs
railway logs
```

### Resource Usage

Check your Railway dashboard for:
- CPU usage
- Memory usage
- Build times
- Request metrics

## 🔒 Security Best Practices

1. **Never commit API keys** to the repository
2. **Use the web interface** to store sensitive keys
3. **Rotate keys regularly** (every 3-6 months)
4. **Use read-only permissions** for exchange APIs
5. **Monitor usage** to detect unauthorized access

## 💰 Costs

### Railway Free Tier:
- **$5 free credit** per month
- **500 hours** of usage
- **Perfect for** personal projects and demos
- **Sufficient for** this application

### Upgrade Considerations:
Upgrade to Pro ($20/month) if you need:
- More compute resources
- Persistent storage
- Higher memory limits
- Priority support

## 🔄 Updates

### Update Your Deployment:

```bash
# Pull latest changes
git pull origin main

# Push to trigger redeploy
git push

# Railway will automatically rebuild and redeploy
```

### Manual Redeploy:

1. Go to Railway dashboard
2. Click on your service
3. Click "Deployments"
4. Click "Redeploy" on the latest deployment

## 📞 Support

### Railway Support:
- [Railway Docs](https://docs.railway.app/)
- [Railway Discord](https://discord.gg/railway)
- [Railway Twitter](https://twitter.com/Railway)

### Application Issues:
- Check application logs in Railway
- Review the main README.md for usage
- Check GitHub issues for known problems

## 🎉 Success!

Once deployed, you'll have a fully functional ML Trading Bot accessible from anywhere!

**Your Railway URL will look like:**
```
https://your-app-name.railway.app
```

Share it, use it, and enjoy AI-powered trading predictions!

---

**Note:** This application is for educational purposes only. Always do your own research before making any trading decisions.
