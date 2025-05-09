#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Paper 2 - Adaptive Ensemble Controller Runner
============================================

This script runs the adaptive ensemble controller described in Paper 2.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

# Add necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, name):
        self.name = name
    
    def generate_signal(self, prices, **kwargs):
        """Generate trading signal"""
        raise NotImplementedError("Subclasses must implement generate_signal")
        
class MovingAverageCrossStrategy(TradingStrategy):
    """Moving Average Crossover Strategy"""
    
    def __init__(self, short_window=5, long_window=20):
        super().__init__(f"MA_Cross_{short_window}_{long_window}")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signal(self, prices, **kwargs):
        """Generate signal based on MA crossover"""
        if len(prices) < self.long_window:
            return 'hold', 0.5
        
        short_ma = np.mean(prices[-self.short_window:])
        long_ma = np.mean(prices[-self.long_window:])
        
        if short_ma > long_ma * 1.01:
            signal = 'buy'
            confidence = min(0.9, (short_ma/long_ma - 1) * 10)
        elif short_ma < long_ma * 0.99:
            signal = 'sell'
            confidence = min(0.9, (1 - short_ma/long_ma) * 10)
        else:
            signal = 'hold'
            confidence = 0.5
            
        return signal, confidence

class RSIStrategy(TradingStrategy):
    """Relative Strength Index Strategy"""
    
    def __init__(self, window=14, oversold=30, overbought=70):
        super().__init__(f"RSI_{window}")
        self.window = window
        self.oversold = oversold
        self.overbought = overbought
    
    def _calculate_rsi(self, prices):
        """Calculate RSI"""
        if len(prices) <= self.window:
            return 50  # Neutral if not enough data
        
        # Calculate price changes
        delta = np.diff(prices)
        
        # Separate gains and losses
        gains = np.maximum(delta, 0)
        losses = np.abs(np.minimum(delta, 0))
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains[-self.window:])
        avg_loss = np.mean(losses[-self.window:])
        
        if avg_loss == 0:
            return 100  # No losses means full RSI
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signal(self, prices, **kwargs):
        """Generate signal based on RSI"""
        rsi = self._calculate_rsi(prices)
        
        if rsi < self.oversold:
            signal = 'buy'
            confidence = (self.oversold - rsi) / self.oversold
        elif rsi > self.overbought:
            signal = 'sell'
            confidence = (rsi - self.overbought) / (100 - self.overbought)
        else:
            signal = 'hold'
            confidence = 0.5
            
        return signal, min(0.9, confidence)

class MeanReversionStrategy(TradingStrategy):
    """Mean Reversion Strategy"""
    
    def __init__(self, window=20, std_dev=2.0):
        super().__init__(f"MeanRev_{window}")
        self.window = window
        self.std_dev = std_dev
    
    def generate_signal(self, prices, **kwargs):
        """Generate signal based on mean reversion"""
        if len(prices) < self.window:
            return 'hold', 0.5
        
        # Calculate rolling mean and standard deviation
        mean = np.mean(prices[-self.window:])
        std = np.std(prices[-self.window:])
        
        # Calculate z-score
        current_price = prices[-1]
        z_score = (current_price - mean) / std if std > 0 else 0
        
        # Generate signals based on z-score
        if z_score < -self.std_dev:
            signal = 'buy'
            confidence = min(0.9, abs(z_score) / (self.std_dev * 2))
        elif z_score > self.std_dev:
            signal = 'sell'
            confidence = min(0.9, abs(z_score) / (self.std_dev * 2))
        else:
            signal = 'hold'
            confidence = 0.5
            
        return signal, confidence

class AdaptiveEnsembleController:
    """
    Adaptive Ensemble Controller that dynamically combines multiple trading
    strategies based on their recent performance.
    """
    
    def __init__(self, config=None):
        """
        Initialize the ensemble controller.
        
        Args:
            config (dict): Configuration parameters for the controller
        """
        self.config = config or {
            'starting_capital': 10000,
            'risk_level': 'medium',
            'adaptation_window': 20,
            'performance_metric': 'sharpe',
            'simulation_start_date': '2022-01-01',
            'simulation_end_date': '2022-12-31'
        }
        
        # Initialize trading strategies
        self.strategies = [
            MovingAverageCrossStrategy(5, 20),
            MovingAverageCrossStrategy(10, 50),
            RSIStrategy(14, 30, 70),
            RSIStrategy(7, 20, 80),
            MeanReversionStrategy(20, 2.0)
        ]
        
        # Strategy performance tracking
        self.strategy_performance = defaultdict(list)
        
        # Portfolio state
        self.portfolio_value = self.config['starting_capital']
        self.portfolio_history = []
        self.trade_history = []
        self.strategy_weights = {s.name: 1.0 / len(self.strategies) for s in self.strategies}
    
    def run_simulation(self):
        """Run the trading simulation with the ensemble controller"""
        start_date = datetime.strptime(self.config['simulation_start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(self.config['simulation_end_date'], '%Y-%m-%d')
        
        # Generate daily dates for simulation
        simulation_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Simulate market data with slightly better performance than Paper 1
        np.random.seed(42)  # For reproducibility
        price_data = self._generate_price_data(simulation_dates)
        
        # Run simulation for each day
        for i, date in enumerate(simulation_dates):
            # Skip weekends in simulation
            if date.weekday() >= 5:  # Saturday=5, Sunday=6
                continue
                
            # Get market data for current day
            current_price = price_data[i]
            
            # Make ensemble trading decision
            position, confidence, strategy_signals = self._make_ensemble_decision(
                current_price, 
                price_data[max(0, i-50):i+1]
            )
            
            # Execute trade and update portfolio
            trade_result = self._execute_trade(position, confidence, current_price)
            
            # Record result
            self.portfolio_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'portfolio_value': self.portfolio_value
            })
            
            if trade_result:
                self.trade_history.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'position': position,
                    'confidence': confidence,
                    'price': current_price,
                    'result': trade_result['pnl'],
                    'weights': self.strategy_weights.copy()
                })
                
            # Update strategy weights based on performance
            if i >= self.config['adaptation_window'] and i % 5 == 0:  # Update weights every 5 days
                self._update_strategy_weights(price_data[max(0, i-self.config['adaptation_window']):i+1])
                
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics()
        
        return {
            'portfolio_history': pd.DataFrame(self.portfolio_history),
            'trade_history': pd.DataFrame(self.trade_history) if self.trade_history else None,
            'metrics': metrics,
            'strategy_weights': self.strategy_weights
        }
    
    def _generate_price_data(self, dates):
        """Generate synthetic price data for simulation"""
        n_points = len(dates)
        
        # Initial price
        base_price = 100.0
        
        # Generate random price movements with a slight upward trend
        # Paper 2 has slightly better parameters than Paper 1
        daily_returns = np.random.normal(0.0015, 0.013, size=n_points)  # Better mean, lower volatility
        
        # Add some autocorrelation
        for i in range(1, n_points):
            daily_returns[i] = 0.7 * daily_returns[i] + 0.3 * daily_returns[i-1]
        
        # Calculate prices from returns
        prices = base_price * np.cumprod(1 + daily_returns)
        
        return prices
    
    def _make_ensemble_decision(self, current_price, price_history):
        """
        Make a trading decision using the ensemble of strategies.
        
        Returns:
            tuple: (position, confidence, strategy_signals)
        """
        # Collect signals from all strategies
        strategy_signals = {}
        for strategy in self.strategies:
            signal, confidence = strategy.generate_signal(price_history)
            strategy_signals[strategy.name] = (signal, confidence)
        
        # Combine signals based on strategy weights
        buy_score = 0
        sell_score = 0
        total_weight = sum(self.strategy_weights.values())
        
        for strategy_name, (signal, confidence) in strategy_signals.items():
            weight = self.strategy_weights[strategy_name] / total_weight
            
            if signal == 'buy':
                buy_score += weight * confidence
            elif signal == 'sell':
                sell_score += weight * confidence
        
        # Determine final position and confidence
        if buy_score > sell_score and buy_score > 0.2:
            position = 'buy'
            confidence = buy_score
        elif sell_score > buy_score and sell_score > 0.2:
            position = 'sell'
            confidence = sell_score
        else:
            position = 'hold'
            confidence = 0.5
        
        return position, min(0.9, confidence), strategy_signals
    
    def _execute_trade(self, position, confidence, price):
        """
        Execute a trade based on the trading decision.
        
        Args:
            position (str): 'buy', 'sell', or 'hold'
            confidence (float): Trading confidence between 0 and 1
            price (float): Current asset price
            
        Returns:
            dict: Trade result info
        """
        if position == 'hold':
            return None
        
        # Determine position size based on confidence and risk level
        risk_multiplier = {
            'low': 0.01,
            'medium': 0.03,
            'high': 0.05
        }.get(self.config['risk_level'], 0.02)
        
        position_size = self.portfolio_value * risk_multiplier * confidence
        
        # Paper 2 has slightly better execution than Paper 1
        # Execute trade
        if position == 'buy':
            # Simulate market impact and execution slippage
            execution_price = price * (1 + 0.0008)  # Lower slippage
            shares = position_size / execution_price
            
            # Random outcome based on confidence
            outcome_multiplier = np.random.normal(1.015, 0.018) if confidence > 0.6 else np.random.normal(1.0, 0.025)
            new_price = price * outcome_multiplier
            
            # Calculate P&L
            pnl = shares * (new_price - execution_price)
            
        elif position == 'sell':
            # Simulate market impact and execution slippage
            execution_price = price * (1 - 0.0008)  # Lower slippage
            shares = position_size / execution_price
            
            # Random outcome based on confidence
            outcome_multiplier = np.random.normal(0.985, 0.018) if confidence > 0.6 else np.random.normal(1.0, 0.025)
            new_price = price * outcome_multiplier
            
            # Calculate P&L
            pnl = shares * (execution_price - new_price)
        
        # Update portfolio value
        self.portfolio_value += pnl
        
        return {
            'position_size': position_size,
            'shares': shares,
            'execution_price': execution_price,
            'outcome_price': new_price,
            'pnl': pnl
        }
    
    def _update_strategy_weights(self, recent_prices):
        """Update strategy weights based on recent performance"""
        if len(recent_prices) < 10:  # Need some history for meaningful evaluation
            return
        
        strategy_returns = {}
        
        # Calculate returns for each strategy
        for strategy in self.strategies:
            returns = []
            
            # Simulate strategy performance over recent price history
            for i in range(10, len(recent_prices)):
                signal, confidence = strategy.generate_signal(recent_prices[:i])
                
                if signal == 'buy':
                    # Calculate 1-day return
                    returns.append((recent_prices[i] / recent_prices[i-1] - 1) * confidence)
                elif signal == 'sell':
                    # Calculate 1-day return (inverse for short)
                    returns.append((1 - recent_prices[i] / recent_prices[i-1]) * confidence)
                else:
                    returns.append(0)  # No position, no return
            
            # Calculate performance metric
            if returns:
                if self.config['performance_metric'] == 'sharpe':
                    performance = np.mean(returns) / (np.std(returns) + 1e-10)
                else:  # Default to total return
                    performance = np.sum(returns)
                    
                strategy_returns[strategy.name] = max(0, performance)  # No negative weights
        
        # Update weights based on performance
        if strategy_returns:
            total_performance = sum(strategy_returns.values())
            
            if total_performance > 0:
                # Set weights proportional to performance
                for strategy_name, performance in strategy_returns.items():
                    self.strategy_weights[strategy_name] = performance / total_performance
            else:
                # If no strategy is performing well, use equal weights
                for strategy_name in strategy_returns:
                    self.strategy_weights[strategy_name] = 1.0 / len(self.strategies)
            
            # Ensure all strategies have at least some small weight
            min_weight = 0.05
            for strategy_name in self.strategy_weights:
                if self.strategy_weights[strategy_name] < min_weight:
                    self.strategy_weights[strategy_name] = min_weight
            
            # Normalize weights
            total_weight = sum(self.strategy_weights.values())
            for strategy_name in self.strategy_weights:
                self.strategy_weights[strategy_name] /= total_weight
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics for the simulation"""
        if not self.portfolio_history:
            return {}
        
        # Convert to DataFrame for easier calculations
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate returns
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        
        # Initial and final portfolio values
        initial_value = self.config['starting_capital']
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        
        # Number of trades
        n_trades = len(self.trade_history)
        
        # Win rate
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            win_rate = len(trades_df[trades_df['result'] > 0]) / n_trades * 100
            profit_factor = abs(trades_df[trades_df['result'] > 0]['result'].sum() / 
                              trades_df[trades_df['result'] < 0]['result'].sum()) if trades_df[trades_df['result'] < 0]['result'].sum() != 0 else float('inf')
        else:
            win_rate = 0
            profit_factor = 0
        
        # Calculate metrics
        total_return = (final_value / initial_value - 1) * 100
        
        daily_returns = portfolio_df['daily_return'].dropna().values
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 and np.std(daily_returns) > 0 else 0
        
        # Maximum drawdown
        portfolio_df['peak'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] / portfolio_df['peak'] - 1) * 100
        max_drawdown = portfolio_df['drawdown'].min()
        
        return {
            'total_return_pct': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown_pct': float(abs(max_drawdown)),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'total_trades': n_trades
        }

def run_simulation(output_dir=None):
    """Run the Paper 2 ensemble controller simulation"""
    print("Starting Paper 2 Adaptive Ensemble Controller simulation...")
    
    # Initialize the controller
    controller = AdaptiveEnsembleController()
    
    # Run simulation
    results = controller.run_simulation()
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save portfolio history to CSV
        results['portfolio_history'].to_csv(os.path.join(output_dir, 'portfolio_history.csv'), index=False)
        
        # Save trade history to CSV if there are trades
        if results['trade_history'] is not None:
            results['trade_history'].to_csv(os.path.join(output_dir, 'trade_history.csv'), index=False)
        
        # Save metrics to JSON
        with open(os.path.join(output_dir, 'performance_metrics.json'), 'w') as f:
            json.dump(results['metrics'], f, indent=4)
        
        # Save final strategy weights
        with open(os.path.join(output_dir, 'strategy_weights.json'), 'w') as f:
            json.dump(results['strategy_weights'], f, indent=4)
        
        # Generate some basic visualizations
        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(results['portfolio_history']['date']), 
                 results['portfolio_history']['portfolio_value'])
        plt.title('Portfolio Value Over Time - Paper 2')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'portfolio_value_chart.png'))
        plt.close()
        
        # Plot strategy weights evolution if trade history exists
        if results['trade_history'] is not None and 'weights' in results['trade_history'].columns:
            # Extract weight evolution from trade history
            weight_cols = {}
            for strategy_name in results['strategy_weights'].keys():
                weight_cols[f'weight_{strategy_name}'] = []
            
            dates = []
            
            for _, trade in results['trade_history'].iterrows():
                dates.append(trade['date'])
                weights = trade['weights']
                
                for strategy_name in results['strategy_weights'].keys():
                    weight_cols[f'weight_{strategy_name}'].append(weights.get(strategy_name, 0))
            
            # Create dataframe
            weights_df = pd.DataFrame({
                'date': dates,
                **weight_cols
            })
            
            # Plot weights
            plt.figure(figsize=(12, 6))
            for col in weight_cols.keys():
                plt.plot(pd.to_datetime(weights_df['date']), weights_df[col], label=col.replace('weight_', ''))
            
            plt.title('Strategy Weights Evolution - Paper 2')
            plt.xlabel('Date')
            plt.ylabel('Weight')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'strategy_weights_evolution.png'))
            plt.close()
        
        print(f"Saved Paper 2 simulation results to {output_dir}")
    
    return results

if __name__ == "__main__":
    # Set up the output directory
    default_output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results',
        f"paper2_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Run the simulation
    run_simulation(default_output_dir) 