@echo off
REM ML Trading Bot Dashboard Launcher (Windows)
REM This script starts the Streamlit dashboard

echo 🚀 Starting ML Trading Bot Dashboard...
echo.

REM Check if virtual environment exists
if exist "venv\" (
    echo ✓ Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo ⚠️  Virtual environment not found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo 📦 Installing dependencies...
    pip install -r requirements.txt
)

REM Check if .env file exists
if not exist ".env" (
    echo ⚠️  .env file not found. Copying from .env.example...
    copy .env.example .env
    echo ⚠️  Please edit .env and add your API keys before running the dashboard!
    echo.
)

REM Start dashboard
echo 🌐 Launching dashboard at http://localhost:8501
echo.
streamlit run ml_trading_bot\app\streamlit_dashboard.py
