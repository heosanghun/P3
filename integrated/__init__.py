"""
Integrated Analysis Module for Paper 1 and Paper 2
=================================================

This package contains tools for comparing and analyzing results from
both the multimodal trading system (Paper 1) and the ensemble controller 
system (Paper 2).

Components:
- comparative_analyzer.py: Contains the ComparativeAnalyzer class that
  loads results from both papers and generates comparative visualizations
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Version
__version__ = '1.0.0' 