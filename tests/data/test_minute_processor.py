"""
分钟数据处理器测试
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime, time

from autofactor.data.dolphindb_client import DolphinDBClient
from autofactor.data.processor.frequency.minute_processor import MinuteDataProcessor


class TestMinuteProcessor(unittest.TestCase):
    """分钟数据处理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 初始化DolphinDB客户端(测试模式)
        self.db_client = DolphinDBClient(test_mode=True)
        
        # 初始化分钟数据处理器
        self.processor = MinuteDataProcessor()
        self.processor.set_db_client(self.db_client)

        # 测试开始前清除缓存
        if hasattr(self.processor, 'cache') and hasattr(self.processor.cache, 'clear'):
            self.processor.cache.clear()
            print("已清除数据处理器缓存")
        
        # 创建测试数据
        self.create_test_data()
        
    def create_test_data(self):
        """创建测试数据"""
        # 创建分钟级别测试数据
        dates = pd.date_range('2022-01-01', periods=2)
        symbols = ['000001.SZ', '600000.SH']
        
        # 生成时间序列 (9:30-11:30, 13:00-15:00)
        times = []
        for h in range(9, 16):
            if h == 9:
                for m in range(30, 60):
                    times.append(time(h, m))
            elif h == 11:
                for m in range(0, 31):
                    times.append(time(h, m))
            elif h == 12:
                continue  # 午休
            else:
                for m in range(0, 60):
                    times.append(time(h, m))
        
        # 创建DataFrame
        data = []
        for symbol in symbols:
            for date in dates:
                base_price = np.random.rand() * 50 + 10
                # 基于随机游走生成价格
                price_series = np.cumsum(np.random.normal(0, 0.01, len(times))) + base_price
                
                for i, t in enumerate(times):
                    price = price_series[i]
                    data.append({
                        'date': date.date(),
                        'time': t,
                        'symbol': symbol,
                        'open': price * (1 + np.random.normal(0, 0.001)),
                        'high': price * (1 + np.random.normal(0, 0.002)),
                        'low': price * (1 - np.random.normal(0, 0.002)),
                        'close': price,
                        'volume': np.random.randint(1000, 10000),
                        'amount': np.random.randint(10000, 100000),
                        'adj_factor': 1.0
                    })
        
        self.test_data = pd.DataFrame(data)
        
    @patch('autofactor.data.processor.frequency.minute_processor.MinuteDataProcessor._query_minute_data')
    def test_get_data(self, mock_query_minute_data):
        """测试获取分钟数据"""
        # 设置mock返回值
        mock_query_minute_data.return_value = self.test_data
        
        # 调用get_data方法
        result = self.processor.get_data(
            symbols=['000001.SZ', '600000.SH'],
            start_date='2022-01-01',
            end_date='2022-01-02',
            fields=['symbol', 'date', 'time', 'open', 'high', 'low', 'close', 'volume'],
            freq="1min",
            adjust=True
        )
        
        # 验证调用了_query_minute_data方法
        self.assertTrue(mock_query_minute_data.called)
        
        # 验证返回的数据
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], len(self.test_data))
        
    @patch('autofactor.data.processor.frequency.minute_processor.MinuteDataProcessor._query_minute_data')
    def test_get_data_with_freq_conversion(self, mock_query_minute_data):
        """测试获取不同频率的分钟数据"""
        # 设置mock返回值
        mock_query_minute_data.return_value = self.test_data
        
        # 调用get_data方法获取5分钟数据
        result = self.processor.get_data(
            symbols=['000001.SZ', '600000.SH'],
            start_date='2022-01-01',
            end_date='2022-01-02',
            fields=['symbol', 'date', 'time', 'open', 'high', 'low', 'close', 'volume'],
            freq="5min",
            adjust=True
        )
        
        # 验证调用了_query_minute_data方法
        self.assertTrue(mock_query_minute_data.called)
        
        # 5分钟数据应该比1分钟数据少
        if not result.empty:
            self.assertLess(result.shape[0], len(self.test_data))
        
    def test_need_chunking(self):
        """测试分块查询判断"""
        # 测试小数据集
        need_chunking = self.processor._need_chunking(
            symbols=['000001.SZ', '600000.SH'],
            start_date='2022-01-01',
            end_date='2022-01-02'
        )
        
        # 小数据集可能不需要分块
        self.assertIsInstance(need_chunking, bool)
        
        # 测试大数据集
        need_chunking = self.processor._need_chunking(
            symbols=['000001.SZ', '600000.SH', '000002.SZ', '600036.SH', '601318.SH'] * 10,
            start_date='2022-01-01',
            end_date='2022-04-01'  # 3个月的数据
        )
        
        # 大数据集应该需要分块
        self.assertTrue(need_chunking)
        
    def test_split_date_range(self):
        """测试日期范围分割"""
        # 测试短时间范围
        date_chunks = self.processor._split_date_range('2022-01-01', '2022-01-05')
        
        # 短时间范围应该只有一个分块
        self.assertEqual(len(date_chunks), 1)
        self.assertEqual(date_chunks[0], ('2022-01-01', '2022-01-05'))
        
        # 测试长时间范围
        date_chunks = self.processor._split_date_range('2022-01-01', '2022-02-01')
        
        # 应该有多个分块
        self.assertGreater(len(date_chunks), 1)
        self.assertEqual(date_chunks[0][0], '2022-01-01')
        
    def test_convert_minute_freq(self):
        """测试分钟频率转换"""
        # 准备测试数据
        test_data = self.test_data.copy()
        
        # 转换时间格式
        test_data['datetime'] = pd.to_datetime(
            test_data['date'].astype(str) + ' ' + 
            test_data['time'].apply(lambda x: f"{x.hour:02d}:{x.minute:02d}:{x.second:02d}")
        )
        
        # 测试转换到5分钟频率
        result = self.processor._convert_minute_freq(test_data, "5min")
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        if not result.empty:
            self.assertLess(result.shape[0], test_data.shape[0])
            self.assertIn('date', result.columns)
            self.assertIn('time', result.columns)
            self.assertIn('open', result.columns)
            self.assertIn('high', result.columns)
            self.assertIn('low', result.columns)
            self.assertIn('close', result.columns)
        
    def test_adjust_price(self):
        """测试价格复权"""
        # 准备测试数据（添加复权因子）
        test_data = self.test_data.copy()
        test_data['adj_factor'] = 1.1
        
        # 测试复权
        result = self.processor._adjust_price(test_data)
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        
        # 检查是否生成了复权价格列
        if 'adj_factor' in result.columns and 'close' in result.columns:
            self.assertIn('close_adj', result.columns)
            
            # 验证复权价格计算
            expected_value = test_data['close'].iloc[0] * test_data['adj_factor'].iloc[0]
            self.assertAlmostEqual(result['close_adj'].iloc[0], expected_value)
    
    def test_filter_trading_hours(self):
        """测试交易时间过滤"""
        # 准备测试数据
        test_data = self.test_data.copy()
        
        # 添加非交易时间的记录
        non_trading_records = []
        for symbol in test_data['symbol'].unique():
            for date in test_data['date'].unique():
                # 添加一个中午12:00的记录（非交易时段）
                non_trading_records.append({
                    'date': date,
                    'time': time(12, 0),
                    'symbol': symbol,
                    'open': 10.0,
                    'high': 10.1,
                    'low': 9.9,
                    'close': 10.0,
                    'volume': 100,
                    'amount': 1000,
                    'adj_factor': 1.0
                })
                
        # 合并数据
        test_data_with_non_trading = pd.concat([test_data, pd.DataFrame(non_trading_records)])
        
        # 测试过滤
        result = self.processor.filter_trading_hours(test_data_with_non_trading)
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        
        # 检查是否正确过滤了非交易时段
        if not result.empty:
            # 验证没有12:00的记录
            self.assertEqual(len(result[result['time'].apply(lambda x: x.hour == 12)]), 0)
            
    @patch('autofactor.data.processor.frequency.minute_processor.MinuteDataProcessor.get_data')
    def test_compute_minute_factor(self, mock_get_data):
        """测试计算分钟因子"""
        # 设置mock返回值
        mock_get_data.return_value = self.test_data
        
        # 测试计算日内动量因子
        result = self.processor.compute_minute_factor(
            factor_name="intraday_momentum",
            symbols=['000001.SZ', '600000.SH'],
            start_date='2022-01-01',
            end_date='2022-01-02',
            params={"window": 5},
            freq="1min"
        )
        
        # 验证调用了get_data方法
        self.assertTrue(mock_get_data.called)
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        if not result.empty:
            self.assertIn('intraday_momentum', result.columns)
            
    def test_resample_to_daily(self):
        """测试分钟数据聚合到日频"""
        # 测试OHLC聚合
        result = self.processor.resample_to_daily(self.test_data, method="ohlc")
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        
        # 检查聚合后的行数
        if not result.empty:
            # 应该每只股票每天只有一行数据
            expected_rows = len(self.test_data['symbol'].unique()) * len(self.test_data['date'].unique())
            self.assertLessEqual(len(result), expected_rows)
            
    def test_get_trade_sessions(self):
        """测试获取交易时段信息"""
        # 获取交易时段
        trade_sessions = self.processor.get_trade_sessions(['2022-01-01', '2022-01-02'])
        
        # 验证结果
        self.assertIsInstance(trade_sessions, pd.DataFrame)
        self.assertIn('date', trade_sessions.columns)
        self.assertIn('session_start', trade_sessions.columns)
        self.assertIn('session_end', trade_sessions.columns)
        
        # 检查时段数量
        # 每天应该有2个交易时段（上午和下午）
        self.assertEqual(len(trade_sessions), 4)  # 2天 * 2个时段


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # 运行测试
    unittest.main()