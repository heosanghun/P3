"""
Adaptive Ensemble Controller - Paper 2
=====================================

This package contains the implementation of the adaptive ensemble controller
described in Paper 2. The system dynamically combines multiple trading strategies
to optimize performance across different market regimes.

Components:
- run_paper2_ensemble.py: Runner script for the Paper 2 implementation
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Version
__version__ = '1.0.0' 