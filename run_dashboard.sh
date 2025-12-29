#!/bin/bash

# ML Trading Bot Dashboard Launcher
# This script starts the Streamlit dashboard

echo "🚀 Starting ML Trading Bot Dashboard..."
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✓ Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your API keys before running the dashboard!"
    echo ""
fi

# Start dashboard
echo "🌐 Launching dashboard at http://localhost:8501"
echo ""
streamlit run ml_trading_bot/app/streamlit_dashboard.py
