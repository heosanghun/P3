#!/bin/bash

# Integrated Crypto Trading System (Paper 1 + Paper 2)
# Run script for Linux and macOS users

echo "==================================================="
echo "  Integrated Crypto Trading System Execution"
echo "==================================================="

# Check Python version
python --version

# Install requirements if needed
if [ "$1" == "--install" ]; then
  echo "Installing dependencies..."
  pip install -r requirements.txt
fi

# Create results directory
mkdir -p results

# Run the main script
echo "Starting integrated simulation..."
python main.py

echo "Simulation completed!"
echo "Check results directory for outputs." 