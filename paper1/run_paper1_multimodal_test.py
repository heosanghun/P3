#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Paper 1 - Multimodal Trading System Runner
==========================================

This script runs the multimodal trading system described in Paper 1.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Add necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# 루트 디렉토리 경로 설정
paper3_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(paper3_dir)
data_root_dir = os.path.join(root_dir, 'data')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# DeepSeek_r1(32b) 모델 기반 감성 분석기 import
from ollama_sentiment_analyzer import OllamaSentimentAnalyzer

class MultimodalTradingModel:
    """
    Multimodal Trading System that combines multiple data sources and modalities
    to make trading decisions.
    """
    
    def __init__(self, config=None):
        """
        Initialize the multimodal trading model.
        
        Args:
            config (dict): Configuration parameters for the model
        """
        self.config = config or {
            'starting_capital': 10000,
            'lookback_window': 30,
            'risk_level': 'medium',
            'data_sources': ['price', 'sentiment', 'news', 'technical'],
            'trading_frequency': 'daily',
            'simulation_start_date': '2022-01-01',
            'simulation_end_date': '2022-12-31',
            'use_deepseek': True,  # DeepSeek_r1(32b) 모델 사용 여부
            'data_dir': data_root_dir  # 루트 데이터 디렉토리 설정
        }
        
        # Model state
        self.portfolio_value = self.config['starting_capital']
        self.portfolio_history = []
        self.trade_history = []
        
        # DeepSeek_r1(32b) 모델 기반 감성 분석기 초기화
        self.sentiment_analyzer = OllamaSentimentAnalyzer(config={
            'offline_mode': not self.config.get('use_deepseek', True),  # 설정에 따라 온라인/오프라인 모드 결정
            'model_name': 'deepseek-llm:latest',  # 기본 모델명 설정
            'use_deepseek_r1': True,  # 딥시크r1(32b) 모델 명시적 사용 설정
            'output': {
                'save_dir': os.path.join(os.path.dirname(current_dir), 'results', 'paper1_ollama')
            }
        })
        
        logging.info(f"MultimodalTradingModel 초기화 완료 (DeepSeek_r1 모델 사용: {self.config.get('use_deepseek', True)})")
    
    def run_simulation(self):
        """Run the trading simulation for the specified period"""
        start_date = datetime.strptime(self.config['simulation_start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(self.config['simulation_end_date'], '%Y-%m-%d')
        
        # Generate daily dates for simulation
        simulation_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Simulate market data
        np.random.seed(42)  # For reproducibility
        price_data = self._generate_price_data(simulation_dates)
        
        # 뉴스 데이터 생성 및 감성 분석
        news_data = self._generate_news_data(simulation_dates)
        sentiment_results = {}
        
        if not news_data.empty and self.config.get('use_deepseek', True):
            logging.info("DeepSeek_r1 모델을 사용한 뉴스 감성 분석 수행 중...")
            try:
                # 뉴스 감성 분석 실행
                sentiment_analysis = self.sentiment_analyzer.analyze_sentiment(news_data)
                
                # 날짜별 감성 점수 매핑
                for i, date in enumerate(news_data['date']):
                    date_str = date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date
                    idx = news_data.index[i]
                    if idx < len(sentiment_analysis.get('detailed_scores', [])):
                        sentiment_results[date_str] = sentiment_analysis['detailed_scores'][idx]
                
                logging.info(f"뉴스 감성 분석 완료: {len(sentiment_results)} 항목")
            except Exception as e:
                logging.error(f"뉴스 감성 분석 중 오류 발생: {str(e)}")
        
        # Run simulation for each day
        for i, date in enumerate(simulation_dates):
            # Skip weekends in simulation
            if date.weekday() >= 5:  # Saturday=5, Sunday=6
                continue
                
            # Get market data for current day
            current_price = price_data[i]
            
            # Get sentiment data for current day
            date_str = date.strftime('%Y-%m-%d')
            sentiment_score = sentiment_results.get(date_str, 0.0)
            
            # Make trading decision with sentiment
            position, confidence = self._make_trading_decision(
                current_price, 
                price_history=price_data[max(0, i-self.config['lookback_window']):i+1],
                sentiment_score=sentiment_score
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
                    'sentiment_score': sentiment_score,
                    'result': trade_result['pnl']
                })
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics()
        
        return {
            'portfolio_history': pd.DataFrame(self.portfolio_history),
            'trade_history': pd.DataFrame(self.trade_history) if self.trade_history else None,
            'metrics': metrics
        }
    
    def _generate_price_data(self, dates):
        """Generate synthetic price data for simulation"""
        n_points = len(dates)
        
        # Initial price
        base_price = 100.0
        
        # Generate random price movements with a slight upward trend
        daily_returns = np.random.normal(0.0005, 0.015, size=n_points)
        
        # Add some autocorrelation
        for i in range(1, n_points):
            daily_returns[i] = 0.7 * daily_returns[i] + 0.3 * daily_returns[i-1]
        
        # Calculate prices from returns
        prices = base_price * np.cumprod(1 + daily_returns)
        
        return prices
    
    def _generate_news_data(self, dates):
        """샘플 뉴스 데이터 생성"""
        try:
            # DeepSeek_r1 감성 분석기가 있는 경우 해당 모듈의 샘플 생성 함수 사용
            if hasattr(self, 'sentiment_analyzer'):
                df = self.sentiment_analyzer.fetch_news(symbol='BTC', days_back=len(dates))
                return df
            else:
                # 간단한 샘플 뉴스 데이터 생성
                news_data = []
                # 일부 날짜에만 뉴스 생성 (3일에 1개 정도)
                for i, date in enumerate(dates):
                    if i % 3 == 0:
                        news_data.append({
                            'date': date,
                            'title': f"Sample news for {date.strftime('%Y-%m-%d')}",
                            'content': "Sample content."
                        })
                
                return pd.DataFrame(news_data)
        except Exception as e:
            logging.error(f"뉴스 데이터 생성 중 오류: {str(e)}")
            return pd.DataFrame()
    
    def _make_trading_decision(self, current_price, price_history, sentiment_score=0.0):
        """
        Make a trading decision based on price and other data sources.
        
        Args:
            current_price (float): Current price
            price_history (list): Historical price data
            sentiment_score (float): News sentiment score (-1 to 1)
            
        Returns:
            tuple: (position, confidence)
                position: 'buy', 'sell', or 'hold'
                confidence: value between 0 and 1
        """
        # Calculate some simple indicators
        if len(price_history) < 2:
            return 'hold', 0.5
        
        # Simple trend following
        short_ma = np.mean(price_history[-5:])
        long_ma = np.mean(price_history[-20:]) if len(price_history) >= 20 else short_ma
        
        # Volatility
        volatility = np.std(price_history[-10:]) / np.mean(price_history[-10:]) if len(price_history) >= 10 else 0.01
        
        # Simple decision rule based on moving averages
        if short_ma > long_ma * 1.01:
            position = 'buy'
            confidence = min(0.9, (short_ma/long_ma - 1) * 10)
        elif short_ma < long_ma * 0.99:
            position = 'sell'
            confidence = min(0.9, (1 - short_ma/long_ma) * 10)
        else:
            position = 'hold'
            confidence = 0.5
        
        # 뉴스 감성 점수 반영 - DeepSeek_r1 감성 분석 결과 통합
        if abs(sentiment_score) > 0.1:  # 감성 점수가 유의미할 경우
            if sentiment_score > 0.3:  # 매우 긍정적 뉴스
                if position == 'sell':  # 기술적 분석이 매도 신호일 때
                    position = 'hold'  # 상충하는 신호는 홀드로 중화
                    confidence = 0.5
                elif position == 'buy':  # 기술적 분석도 매수 신호일 때
                    confidence += 0.2  # 신뢰도 증가
            elif sentiment_score > 0.1:  # 약간 긍정적 뉴스
                if position == 'buy':  # 기술적 분석이 매수 신호일 때
                    confidence += 0.1  # 신뢰도 약간 증가
            elif sentiment_score < -0.3:  # 매우 부정적 뉴스
                if position == 'buy':  # 기술적 분석이 매수 신호일 때
                    position = 'hold'  # 상충하는 신호는 홀드로 중화
                    confidence = 0.5
                elif position == 'sell':  # 기술적 분석도 매도 신호일 때
                    confidence += 0.2  # 신뢰도 증가
            elif sentiment_score < -0.1:  # 약간 부정적 뉴스
                if position == 'sell':  # 기술적 분석이 매도 신호일 때
                    confidence += 0.1  # 신뢰도 약간 증가
        
        # Adjust confidence based on volatility
        confidence = confidence * (1 - min(volatility * 10, 0.5))
        
        # 최종 신뢰도 값 범위 제한
        confidence = min(0.95, max(0.05, confidence))
        
        return position, confidence
    
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
        
        # Execute trade
        if position == 'buy':
            # Simulate market impact and execution slippage
            execution_price = price * (1 + 0.001)
            shares = position_size / execution_price
            
            # Random outcome based on confidence
            outcome_multiplier = np.random.normal(1.01, 0.02) if confidence > 0.6 else np.random.normal(0.99, 0.03)
            new_price = price * outcome_multiplier
            
            # Calculate P&L
            pnl = shares * (new_price - execution_price)
            
        elif position == 'sell':
            # Simulate market impact and execution slippage
            execution_price = price * (1 - 0.001)
            shares = position_size / execution_price
            
            # Random outcome based on confidence
            outcome_multiplier = np.random.normal(0.99, 0.02) if confidence > 0.6 else np.random.normal(1.01, 0.03)
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
    """Run the Paper 1 multimodal trading simulation"""
    print("Starting Paper 1 Multimodal Trading simulation...")
    
    # Initialize the model
    model = MultimodalTradingModel(config={
        'starting_capital': 10000,
        'lookback_window': 30,
        'risk_level': 'medium',
        'data_sources': ['price', 'sentiment', 'news', 'technical'],
        'trading_frequency': 'daily',
        'simulation_start_date': '2022-01-01',
        'simulation_end_date': '2022-12-31',
        'use_deepseek': True  # DeepSeek_r1(32b) 모델 사용 활성화
    })
    
    # Run simulation
    results = model.run_simulation()
    
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
        
        # Generate some basic visualizations
        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(results['portfolio_history']['date']), 
                 results['portfolio_history']['portfolio_value'])
        plt.title('Portfolio Value Over Time - Paper 1')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'portfolio_value_chart.png'))
        plt.close()
        
        print(f"Saved Paper 1 simulation results to {output_dir}")
    
    return results

if __name__ == "__main__":
    # Set up the output directory
    default_output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results',
        f"paper1_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Run the simulation
    run_simulation(default_output_dir) 