@echo off
REM Integrated Crypto Trading System (Paper 1 + Paper 2)
REM Run script for Windows users

echo ===================================================
echo   Integrated Crypto Trading System Execution
echo ===================================================

REM Check Python version
python --version

REM Install requirements if needed
if "%1"=="--install" (
  echo Installing dependencies...
  pip install -r requirements.txt
)

REM Create results directory
if not exist results mkdir results

REM Run the main script
echo Starting integrated simulation...
python main.py

echo Simulation completed!
echo Check results directory for outputs. 