#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comparative Analyzer Module
===========================

This module provides tools for comparing and analyzing results from 
Paper 1 (Multimodal Trading System) and Paper 2 (Adaptive Ensemble Controller).
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ComparativeAnalyzer:
    """
    A class for comparative analysis of results from Paper 1 and Paper 2.
    
    This analyzer loads the results from both papers, performs comparative
    analysis, and generates visualizations for decision making.
    """
    
    def __init__(self, paper1_results_dir, paper2_results_dir, output_dir):
        """
        Initialize the ComparativeAnalyzer.
        
        Args:
            paper1_results_dir (str): Directory containing Paper 1 results
            paper2_results_dir (str): Directory containing Paper 2 results
            output_dir (str): Directory to save comparative analysis results
        """
        self.paper1_results_dir = paper1_results_dir
        self.paper2_results_dir = paper2_results_dir
        self.output_dir = output_dir
        
        # Create visualizations directory
        self.viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Results containers
        self.paper1_data = None
        self.paper2_data = None
        self.comparative_metrics = {}
        
        # Set style for visualizations
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("viridis")
    
    def load_data(self):
        """Load data from both Paper 1 and Paper 2 results directories"""
        # Load Paper 1 data
        paper1_portfolio = pd.read_csv(os.path.join(self.paper1_results_dir, 'portfolio_history.csv'))
        with open(os.path.join(self.paper1_results_dir, 'performance_metrics.json'), 'r') as f:
            paper1_metrics = json.load(f)
        
        self.paper1_data = {
            'portfolio': paper1_portfolio,
            'metrics': paper1_metrics
        }
        
        # Load Paper 2 data
        paper2_portfolio = pd.read_csv(os.path.join(self.paper2_results_dir, 'portfolio_history.csv'))
        with open(os.path.join(self.paper2_results_dir, 'performance_metrics.json'), 'r') as f:
            paper2_metrics = json.load(f)
        
        self.paper2_data = {
            'portfolio': paper2_portfolio,
            'metrics': paper2_metrics
        }
        
        print(f"Loaded data from Paper 1 and Paper 2 results directories")
    
    def analyze(self):
        """Perform comparative analysis of Paper 1 and Paper 2 results"""
        if self.paper1_data is None or self.paper2_data is None:
            self.load_data()
        
        # Calculate comparative metrics
        for metric in self.paper1_data['metrics'].keys():
            if metric in self.paper2_data['metrics']:
                p1_value = self.paper1_data['metrics'][metric]
                p2_value = self.paper2_data['metrics'][metric]
                
                if isinstance(p1_value, (int, float)) and isinstance(p2_value, (int, float)):
                    diff = p2_value - p1_value
                    pct_diff = (diff / abs(p1_value)) * 100 if p1_value != 0 else float('inf')
                    
                    self.comparative_metrics[metric] = {
                        'paper1': p1_value,
                        'paper2': p2_value,
                        'difference': diff,
                        'pct_difference': pct_diff
                    }
        
        # Save comparative metrics to JSON
        metrics_path = os.path.join(self.output_dir, 'comparative_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.comparative_metrics, f, indent=4)
        
        # Create summary table as CSV
        summary_df = pd.DataFrame({
            'Metric': list(self.comparative_metrics.keys()),
            'Paper1_Value': [m['paper1'] for m in self.comparative_metrics.values()],
            'Paper2_Value': [m['paper2'] for m in self.comparative_metrics.values()],
            'Difference': [m['difference'] for m in self.comparative_metrics.values()],
            'Pct_Difference': [m['pct_difference'] for m in self.comparative_metrics.values()]
        })
        
        summary_path = os.path.join(self.output_dir, 'comparative_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Comparative analysis completed and saved to {self.output_dir}")
        
        return self.comparative_metrics
    
    def visualize(self):
        """Generate visualizations for the comparative analysis"""
        if not self.comparative_metrics:
            self.analyze()
        
        # 1. Portfolio Value Comparison Plot
        self._plot_portfolio_comparison()
        
        # 2. Performance Metrics Comparison
        self._plot_metrics_comparison()
        
        # 3. Drawdown Comparison
        self._plot_drawdown_comparison()
        
        # 4. Monthly Returns Heatmap
        self._plot_monthly_returns()
        
        # 5. Cumulative Returns Plot
        self._plot_cumulative_returns()
        
        print(f"Visualizations generated and saved to {self.viz_dir}")
    
    def _plot_portfolio_comparison(self):
        """Plot portfolio value comparison between Paper 1 and Paper 2"""
        plt.figure(figsize=(12, 6))
        
        # Ensure datetime format
        p1_portfolio = self.paper1_data['portfolio'].copy()
        p2_portfolio = self.paper2_data['portfolio'].copy()
        
        p1_portfolio['date'] = pd.to_datetime(p1_portfolio['date'])
        p2_portfolio['date'] = pd.to_datetime(p2_portfolio['date'])
        
        # Plot
        plt.plot(p1_portfolio['date'], p1_portfolio['portfolio_value'], 
                 label='Paper 1 - Multimodal Trading', linewidth=2)
        plt.plot(p2_portfolio['date'], p2_portfolio['portfolio_value'], 
                 label='Paper 2 - Adaptive Ensemble', linewidth=2)
        
        plt.title('Portfolio Value Comparison', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'portfolio_comparison.png'), dpi=300)
        plt.close()
    
    def _plot_metrics_comparison(self):
        """Plot performance metrics comparison as a bar chart"""
        # Select key metrics for visualization
        key_metrics = ['total_return_pct', 'sharpe_ratio', 'win_rate', 'profit_factor']
        metric_labels = {
            'total_return_pct': 'Total Return (%)',
            'sharpe_ratio': 'Sharpe Ratio',
            'win_rate': 'Win Rate (%)',
            'profit_factor': 'Profit Factor'
        }
        
        # Filter metrics that exist in both datasets
        available_metrics = [m for m in key_metrics if m in self.comparative_metrics]
        
        if not available_metrics:
            return
        
        # Prepare data for plotting
        paper1_values = [self.comparative_metrics[m]['paper1'] for m in available_metrics]
        paper2_values = [self.comparative_metrics[m]['paper2'] for m in available_metrics]
        labels = [metric_labels.get(m, m) for m in available_metrics]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(labels))
        width = 0.35
        
        bar1 = ax.bar(x - width/2, paper1_values, width, label='Paper 1 - Multimodal Trading')
        bar2 = ax.bar(x + width/2, paper2_values, width, label='Paper 2 - Adaptive Ensemble')
        
        ax.set_title('Performance Metrics Comparison', fontsize=16)
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        # Add value labels on bars
        for bars in [bar1, bar2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'metrics_comparison.png'), dpi=300)
        plt.close()
    
    def _plot_drawdown_comparison(self):
        """Plot drawdown comparison between Paper 1 and Paper 2"""
        p1_portfolio = self.paper1_data['portfolio'].copy()
        p2_portfolio = self.paper2_data['portfolio'].copy()
        
        # Calculate drawdowns
        def calculate_drawdown(portfolio_df):
            portfolio_df = portfolio_df.copy()
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            portfolio_df.set_index('date', inplace=True)
            portfolio_df['peak'] = portfolio_df['portfolio_value'].cummax()
            portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] / portfolio_df['peak'] - 1) * 100
            return portfolio_df
        
        p1_dd = calculate_drawdown(p1_portfolio)
        p2_dd = calculate_drawdown(p2_portfolio)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        plt.plot(p1_dd.index, p1_dd['drawdown'], 
                 label='Paper 1 - Multimodal Trading', linewidth=2)
        plt.plot(p2_dd.index, p2_dd['drawdown'], 
                 label='Paper 2 - Adaptive Ensemble', linewidth=2)
        
        plt.title('Drawdown Comparison', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'drawdown_comparison.png'), dpi=300)
        plt.close()
    
    def _plot_monthly_returns(self):
        """Plot monthly returns as heatmaps for both strategies"""
        for paper_num, data in [(1, self.paper1_data), (2, self.paper2_data)]:
            portfolio_df = data['portfolio'].copy()
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            
            # Calculate daily returns
            portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change() * 100
            
            # Create month and year columns
            portfolio_df['year'] = portfolio_df['date'].dt.year
            portfolio_df['month'] = portfolio_df['date'].dt.month
            
            # Calculate monthly returns
            monthly_returns = portfolio_df.groupby(['year', 'month'])['daily_return'].sum().unstack()
            
            # Plot heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(monthly_returns, annot=True, fmt=".2f", cmap="RdYlGn", 
                        center=0, linewidths=.5, cbar_kws={"label": "Monthly Return (%)"})
            
            plt.title(f'Monthly Returns Heatmap - Paper {paper_num}', fontsize=16)
            plt.xlabel('Month', fontsize=12)
            plt.ylabel('Year', fontsize=12)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, f'monthly_returns_paper{paper_num}.png'), dpi=300)
            plt.close()
    
    def _plot_cumulative_returns(self):
        """Plot cumulative returns comparison"""
        # Calculate cumulative returns for both papers
        p1_portfolio = self.paper1_data['portfolio'].copy()
        p2_portfolio = self.paper2_data['portfolio'].copy()
        
        p1_portfolio['date'] = pd.to_datetime(p1_portfolio['date'])
        p2_portfolio['date'] = pd.to_datetime(p2_portfolio['date'])
        
        p1_portfolio['daily_return'] = p1_portfolio['portfolio_value'].pct_change()
        p2_portfolio['daily_return'] = p2_portfolio['portfolio_value'].pct_change()
        
        p1_portfolio['cum_return'] = (1 + p1_portfolio['daily_return']).cumprod() - 1
        p2_portfolio['cum_return'] = (1 + p2_portfolio['daily_return']).cumprod() - 1
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        plt.plot(p1_portfolio['date'], p1_portfolio['cum_return'] * 100, 
                 label='Paper 1 - Multimodal Trading', linewidth=2)
        plt.plot(p2_portfolio['date'], p2_portfolio['cum_return'] * 100, 
                 label='Paper 2 - Adaptive Ensemble', linewidth=2)
        
        plt.title('Cumulative Returns Comparison', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return (%)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'cumulative_returns.png'), dpi=300)
        plt.close() 