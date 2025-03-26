"""
回测引擎测试
"""
import sys
import unittest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime, timedelta

from autofactor.data.dolphindb_client import DolphinDBClient
from autofactor.backtest.engine import BacktestEngine


class TestBacktestEngine(unittest.TestCase):
    """回测引擎测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 初始化回测引擎
        self.engine = BacktestEngine()
        
        # 创建测试数据
        self.create_test_data()
        
    def create_test_data(self):
        """创建测试数据"""
        # 创建因子测试数据
        dates = pd.date_range('2022-01-01', periods=60)
        symbols = ['000001.SZ', '600000.SH', '000002.SZ', '600036.SH', '601318.SH']
        
        factor_data = []
        for date in dates:
            for symbol in symbols:
                factor_data.append({
                    'date': date,
                    'symbol': symbol,
                    'factor1': np.random.randn(),
                    'factor2': np.random.randn(),
                    'industry': np.random.choice(['金融', '科技', '消费', '医药', '能源'])
                })
                
        self.factor_data = pd.DataFrame(factor_data)
        
        # 创建价格测试数据
        price_data = []
        for symbol in symbols:
            # 为每只股票生成一个起始价格
            base_price = np.random.uniform(10, 50)
            
            for i, date in enumerate(dates):
                # 生成价格序列（随机游走）
                if i == 0:
                    price = base_price
                else:
                    price = price_data[-1]['close'] * (1 + np.random.normal(0.0005, 0.015))
                    
                price_data.append({
                    'date': date,
                    'symbol': symbol,
                    'open': price * (1 + np.random.normal(0, 0.005)),
                    'high': price * (1 + np.random.normal(0, 0.01)),
                    'low': price * (1 - np.random.normal(0, 0.01)),
                    'close': price,
                    'volume': np.random.randint(1000000, 10000000),
                    'amount': np.random.randint(10000000, 100000000),
                    'adj_factor': 1.0
                })
                
        self.price_data = pd.DataFrame(price_data)
        
        # 添加一日收益率
        self.price_data['return_1d'] = self.price_data.groupby('symbol')['close'].pct_change()
        
        # 添加前视收益率（用于IC计算）
        for days in [1, 5, 10, 20]:
            self.price_data[f'forward_return_{days}d'] = self.price_data.groupby('symbol')['close'].pct_change(days).shift(-days)
    
    @patch.object(BacktestEngine, '_get_rebalance_dates')
    @patch.object(BacktestEngine, '_update_portfolio_value')
    @patch.object(BacktestEngine, '_generate_target_portfolio')
    @patch.object(BacktestEngine, '_execute_trades')
    @patch.object(BacktestEngine, '_calculate_performance_metrics')
    @patch.object(BacktestEngine, '_analyze_factor')
    def test_backtest_factor(self, mock_analyze_factor, mock_calculate_performance, 
                          mock_execute_trades, mock_generate_target, 
                          mock_update_portfolio, mock_get_rebalance_dates):
        """测试因子回测"""
        # 准备测试数据
        factor_data = self.factor_data.copy()
        
        # 配置mock
        mock_get_rebalance_dates.return_value = factor_data['date'].unique()
        mock_calculate_performance.return_value = {'annualized_return': 0.15, 'sharpe_ratio': 1.2}
        mock_analyze_factor.return_value = {'ic': {'1d': {'IC Mean': 0.05}}, 'quantile': {'1d': {'Spread (Q5-Q1)': 0.01}}}
        
        # Mock日线处理器的方法
        self.engine.daily_processor.get_trade_dates = MagicMock(return_value=factor_data['date'].unique())
        self.engine.daily_processor.get_data = MagicMock(return_value=self.price_data)
        
        # 执行回测
        result = self.engine.backtest_factor(
            factor_data=factor_data,
            start_date='2022-01-01',
            end_date='2022-03-01',
            quantiles=5,
            holding_period=1,
            long_short=True,
            factor_col='factor1'
        )
        
        # 验证调用
        mock_get_rebalance_dates.assert_called_once()
        mock_calculate_performance.assert_called_once()
        mock_analyze_factor.assert_called_once()
        
        # 验证执行交易和更新组合价值被调用多次
        self.assertTrue(mock_update_portfolio.call_count > 0)
        self.assertTrue(mock_execute_trades.call_count > 0)
        
        # 验证返回结果
        self.assertIsInstance(result, dict)
        self.assertIn('returns', result)
        self.assertIn('positions', result)
        self.assertIn('trades', result)
        self.assertIn('performance', result)
        self.assertIn('factor_analysis', result)
        self.assertIn('config', result)
        
    def test_get_rebalance_dates(self):
        """测试获取再平衡日期"""
        # 准备测试数据
        trade_dates = pd.date_range('2022-01-01', periods=60)
        
        # 测试日频再平衡
        rebalance_dates = self.engine._get_rebalance_dates(trade_dates, "daily")
        self.assertEqual(len(rebalance_dates), len(trade_dates))
        
        # 测试周频再平衡
        rebalance_dates = self.engine._get_rebalance_dates(trade_dates, "weekly")
        self.assertLess(len(rebalance_dates), len(trade_dates))
        self.assertGreater(len(rebalance_dates), 0)
        
        # 测试月频再平衡
        rebalance_dates = self.engine._get_rebalance_dates(trade_dates, "monthly")
        self.assertLess(len(rebalance_dates), len(rebalance_dates))
        self.assertGreater(len(rebalance_dates), 0)
        
    def test_calculate_performance_metrics(self):
        """测试计算绩效指标"""
        # 准备测试数据
        returns = pd.DataFrame({
            'date': pd.date_range('2022-01-01', periods=60),
            'portfolio_value': np.cumprod(1 + np.random.normal(0.001, 0.01, 60)),
            'daily_return': np.random.normal(0.001, 0.01, 60),
            'benchmark_return': np.random.normal(0.0005, 0.01, 60)
        })
        
        # 测试计算绩效指标
        metrics = self.engine._calculate_performance_metrics(returns)
        
        # 验证返回结果
        self.assertIsInstance(metrics, dict)
        self.assertIn('annualized_return', metrics)
        self.assertIn('volatility', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('beta', metrics)
        self.assertIn('alpha', metrics)
        self.assertIn('information_ratio', metrics)
        self.assertIn('win_rate', metrics)
        
    def test_generate_target_portfolio(self):
        """测试生成目标投资组合"""
        # 准备测试数据
        factor_data = self.factor_data.copy()
        factor_data['quantile'] = np.random.randint(0, 5, len(factor_data))
        
        price_data = self.price_data.copy()
        current_date = factor_data['date'].iloc[0]
        
        # 测试多空策略
        target_portfolio = self.engine._generate_target_portfolio(
            factor_data[factor_data['date'] == current_date],
            price_data,
            current_date,
            quantiles=5,
            long_short=True
        )
        
        # 验证返回结果
        self.assertIsInstance(target_portfolio, dict)
        
        # 应该有做多和做空的持仓
        has_long = False
        has_short = False
        for symbol, weight in target_portfolio.items():
            if weight > 0:
                has_long = True
            if weight < 0:
                has_short = True
                
        self.assertTrue(has_long)
        self.assertTrue(has_short)
        
        # 测试纯多头策略
        target_portfolio = self.engine._generate_target_portfolio(
            factor_data[factor_data['date'] == current_date],
            price_data,
            current_date,
            quantiles=5,
            long_short=False
        )
        
        # 验证返回结果
        self.assertIsInstance(target_portfolio, dict)
        
        # 应该只有做多的持仓
        for symbol, weight in target_portfolio.items():
            self.assertGreaterEqual(weight, 0)
    
    def test_execute_trades(self):
        """测试执行交易"""
        # 初始化投资组合状态
        self.engine.portfolio_value = 1000000
        self.engine.cash = 1000000
        self.engine.current_positions = {}
        
        # 准备测试数据
        target_portfolio = {
            '000001.SZ': 0.3,
            '600000.SH': 0.3,
            '000002.SZ': 0.4
        }
        
        price_data = self.price_data.copy()
        current_date = price_data['date'].iloc[0]
        
        # 执行交易
        self.engine._execute_trades(
            target_portfolio,
            price_data,
            current_date,
            commission_rate=0.0003,
            slippage=0.0001
        )
        
        # 验证交易结果
        self.assertIsInstance(self.engine.current_positions, dict)
        self.assertEqual(len(self.engine.current_positions), 3)
        
        # 检查现金是否更新
        self.assertLess(self.engine.cash, 1000000)
        
        # 检查是否记录了交易
        self.assertGreater(len(self.engine.trades), 0)
        
        # 检查投资组合总价值是否正确
        portfolio_value = self.engine.cash
        for symbol, position in self.engine.current_positions.items():
            portfolio_value += position['value']
            
        self.assertAlmostEqual(portfolio_value, self.engine.portfolio_value, delta=10)
    
    def test_analyze_factor(self):
        """测试因子分析"""
        # 准备测试数据
        factor_data = self.factor_data.copy()
        price_data = self.price_data.copy()
        
        # 测试因子分析
        analysis = self.engine._analyze_factor(
            factor_data,
            price_data,
            factor_col='factor1',
            date_col='date',
            symbol_col='symbol'
        )
        
        # 验证返回结果
        self.assertIsInstance(analysis, dict)
        self.assertIn('ic', analysis)
        self.assertIn('quantile', analysis)
        
    def test_generate_report(self):
        """测试生成回测报告"""
        # 准备mock数据
        self.engine.results = {
            'returns': pd.DataFrame({
                'date': pd.date_range('2022-01-01', periods=60),
                'portfolio_value': np.cumprod(1 + np.random.normal(0.001, 0.01, 60)),
                'daily_return': np.random.normal(0.001, 0.01, 60),
                'cumulative_return': np.cumsum(np.random.normal(0.001, 0.01, 60))
            }),
            'positions': pd.DataFrame({
                'date': pd.date_range('2022-01-01', periods=60).repeat(3),
                'symbol': ['000001.SZ', '600000.SH', '000002.SZ'] * 60,
                'shares': np.random.randint(100, 1000, 60 * 3),
                'cost': np.random.uniform(10000, 50000, 60 * 3),
                'value': np.random.uniform(10000, 50000, 60 * 3)
            }),
            'trades': pd.DataFrame({
                'date': pd.date_range('2022-01-01', periods=20),
                'symbol': ['000001.SZ', '600000.SH'] * 10,
                'shares': np.random.randint(-1000, 1000, 20),
                'price': np.random.uniform(10, 50, 20),
                'cost': np.random.uniform(10, 100, 20),
                'type': ['buy', 'sell'] * 10
            }),
            'performance': {
                'annualized_return': 0.15,
                'volatility': 0.2,
                'sharpe_ratio': 0.75,
                'max_drawdown': -0.1,
                'win_rate': 0.55
            },
            'factor_analysis': {
                'ic': {'1d': {'IC Mean': 0.05}},
                'quantile': {'1d': {'Spread (Q5-Q1)': 0.01}}
            },
            'config': {
                'start_date': '2022-01-01',
                'end_date': '2022-03-01',
                'quantiles': 5,
                'holding_period': 1,
                'long_short': True,
                'commission_rate': 0.0003,
                'slippage': 0.0001,
                'capital': 1000000,
                'benchmark': '000300.SH',
                'rebalance_freq': 'daily'
            }
        }
        
        # 测试生成报告
        report = self.engine.generate_report()
        
        # 验证返回结果
        self.assertIsInstance(report, dict)
        self.assertIn('summary', report)
        self.assertIn('performance', report)
        self.assertIn('factor_analysis', report)
        self.assertIn('daily_returns', report)
        self.assertIn('monthly_returns', report)
        self.assertIn('drawdowns', report)
        
        # 测试保存报告到文件
        temp_file = "test_report.json"
        report = self.engine.generate_report(temp_file)
        
        # 验证文件已创建
        self.assertTrue(os.path.exists(temp_file))
        
        # 删除测试文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    def test_calculate_monthly_returns(self):
        """测试计算月度收益率"""
        # 准备测试数据
        self.engine.results = {
            'returns': pd.DataFrame({
                'date': pd.date_range('2022-01-01', periods=90),
                'portfolio_value': np.cumprod(1 + np.random.normal(0.001, 0.01, 90))
            })
        }
        
        # 测试计算月度收益率
        monthly_returns = self.engine._calculate_monthly_returns()
        
        # 验证返回结果
        self.assertIsInstance(monthly_returns, list)
        self.assertGreater(len(monthly_returns), 0)
        for mr in monthly_returns:
            self.assertIn('year_month', mr)
            self.assertIn('return', mr)
            
    def test_calculate_drawdowns(self):
        """测试计算回撤"""
        # 准备测试数据
        returns = pd.DataFrame({
            'date': pd.date_range('2022-01-01', periods=90),
            'portfolio_value': np.cumprod(1 + np.random.normal(0.001, 0.01, 90)),
            'daily_return': np.random.normal(0.001, 0.01, 90)
        })
        
        # 计算累积收益
        returns['cumulative_return'] = (1 + returns['daily_return']).cumprod() - 1
        
        self.engine.results = {'returns': returns}
        
        # 测试计算回撤
        drawdowns = self.engine._calculate_drawdowns()
        
        # 验证返回结果
        self.assertIsInstance(drawdowns, list)
        for dd in drawdowns:
            self.assertIn('start_date', dd)
            self.assertIn('max_drawdown', dd)
            self.assertIn('max_drawdown_date', dd)


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # 运行测试
    unittest.main()