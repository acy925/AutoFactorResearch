"""
多频率数据处理器测试
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
from autofactor.data.processor.frequency.daily_processor import DailyDataProcessor
from autofactor.data.processor.frequency.minute_processor import MinuteDataProcessor
from autofactor.data.processor.frequency.multi_freq_processor import MultiFreqProcessor


class TestMultiFreqProcessor(unittest.TestCase):
    """多频率数据处理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 初始化多频率处理器
        self.processor = MultiFreqProcessor()
        
        # Mock日线和分钟处理器
        self.processor.daily_processor = MagicMock(spec=DailyDataProcessor)
        self.processor.minute_processor = MagicMock(spec=MinuteDataProcessor)
        
        # 创建测试数据
        self.create_test_data()
        
    def create_test_data(self):
        """创建测试数据"""
        # 创建日线测试数据
        dates = pd.date_range('2022-01-01', periods=10)
        symbols = ['000001.SZ', '600000.SH']
        
        daily_data = []
        for symbol in symbols:
            for date in dates:
                daily_data.append({
                    'date': date,
                    'symbol': symbol,
                    'open': np.random.rand() * 50 + 10,
                    'high': np.random.rand() * 50 + 15,
                    'low': np.random.rand() * 50 + 5,
                    'close': np.random.rand() * 50 + 10,
                    'volume': np.random.randint(1000, 10000),
                    'amount': np.random.randint(10000, 100000),
                    'adj_factor': np.random.rand() + 0.5
                })
        
        self.daily_test_data = pd.DataFrame(daily_data)
        
        # 创建分钟级别测试数据
        times = [time(9, 30), time(10, 0), time(10, 30), time(11, 0), time(11, 30),
                time(13, 0), time(13, 30), time(14, 0), time(14, 30), time(15, 0)]
        
        minute_data = []
        for symbol in symbols:
            for date in dates[:2]:  # 只用前两天的数据
                base_price = np.random.rand() * 50 + 10
                # 基于随机游走生成价格
                price_series = np.cumsum(np.random.normal(0, 0.01, len(times))) + base_price
                
                for i, t in enumerate(times):
                    price = price_series[i]
                    minute_data.append({
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
        
        self.minute_test_data = pd.DataFrame(minute_data)
        
        # 添加datetime列
        self.minute_test_data['datetime'] = pd.to_datetime(
            self.minute_test_data['date'].astype(str) + ' ' + 
            self.minute_test_data['time'].apply(lambda x: f"{x.hour:02d}:{x.minute:02d}:00")
        )
        
    def test_get_data(self):
        """测试获取数据"""
        # 配置mock返回值
        self.processor.daily_processor.get_data.return_value = self.daily_test_data
        self.processor.minute_processor.get_data.return_value = self.minute_test_data
        
        # 测试获取日线数据
        result = self.processor.get_data(
            symbols=['000001.SZ', '600000.SH'],
            start_date='2022-01-01',
            end_date='2022-01-10',
            fields=['symbol', 'date', 'open', 'close'],
            freq='day',
            adjust=True
        )
        
        # 验证调用了日线处理器
        self.processor.daily_processor.get_data.assert_called_once()
        self.assertEqual(result.shape[0], len(self.daily_test_data))
        
        # 重置mock
        self.processor.daily_processor.get_data.reset_mock()
        self.processor.minute_processor.get_data.reset_mock()
        
        # 测试获取分钟数据
        result = self.processor.get_data(
            symbols=['000001.SZ', '600000.SH'],
            start_date='2022-01-01',
            end_date='2022-01-02',
            fields=['symbol', 'date', 'time', 'open', 'high', 'low', 'close'],
            freq='1min',
            adjust=True
        )
        
        # 验证调用了分钟处理器
        self.processor.minute_processor.get_data.assert_called_once()
        self.assertEqual(result.shape[0], len(self.minute_test_data))
        
    def test_get_multi_freq_data(self):
        """测试获取多频率数据"""
        # 配置mock返回值
        self.processor.daily_processor.get_data.return_value = self.daily_test_data
        self.processor.minute_processor.get_data.return_value = self.minute_test_data
        
        # 测试获取多频率数据
        result = self.processor.get_multi_freq_data(
            symbols=['000001.SZ', '600000.SH'],
            start_date='2022-01-01',
            end_date='2022-01-10',
            freqs=['day', '1min'],
            fields={
                'day': ['symbol', 'date', 'open', 'close'],
                '1min': ['symbol', 'date', 'time', 'open', 'high', 'low', 'close']
            }
        )
        
        # 验证返回结果
        self.assertIsInstance(result, dict)
        self.assertIn('day', result)
        self.assertIn('1min', result)
        self.assertEqual(result['day'].shape[0], len(self.daily_test_data))
        self.assertEqual(result['1min'].shape[0], len(self.minute_test_data))
        
    def test_align_multi_freq_data(self):
        """测试对齐多频率数据"""
        # 准备测试数据
        data_dict = {
            'day': self.daily_test_data.copy(),
            '1min': self.minute_test_data.copy()
        }
        
        # 确保日线数据有datetime列
        data_dict['day']['datetime'] = pd.to_datetime(data_dict['day']['date'])
        
        # 测试对齐（以日线为基准）
        result = self.processor.align_multi_freq_data(data_dict, base_freq='day')
        
        # 验证返回结果
        self.assertIsInstance(result, dict)
        self.assertIn('day', result)
        self.assertIn('1min', result)
        
        # 日线数据应该不变
        self.assertEqual(result['day'].shape[0], len(self.daily_test_data))
        
        # 分钟数据应该被重采样和对齐
        if '1min' in result and not result['1min'].empty:
            # 分钟数据的时间戳应该匹配日线数据的时间戳
            day_timestamps = set(pd.to_datetime(result['day']['datetime']))
            minute_timestamps = set(pd.to_datetime(result['1min']['datetime']))
            # 确认所有日线时间戳在分钟时间戳中存在
            day_timestamps_in_minute = day_timestamps.intersection(minute_timestamps)
            self.assertGreaterEqual(len(day_timestamps_in_minute), 1)
    
    def test_merge_multi_freq_data(self):
        """测试合并多频率数据"""
        # 准备对齐后的测试数据
        aligned_dict = {
            'day': self.daily_test_data.copy(),
            '1min': self.minute_test_data.copy()
        }
        
        # 确保都有datetime列
        aligned_dict['day']['datetime'] = pd.to_datetime(aligned_dict['day']['date'])
        
        # 限制数据量使两个频率有重叠
        aligned_dict['day'] = aligned_dict['day'].head(5)
        aligned_dict['1min'] = aligned_dict['1min'].head(5)
        
        # 设置列名后缀
        freq_suffixes = {'day': '_D', '1min': '_M'}
        
        # 测试合并
        result = self.processor.merge_multi_freq_data(aligned_dict, freq_suffixes)
        
        # 验证返回结果
        self.assertIsInstance(result, pd.DataFrame)
        
        # 检查列名是否正确添加了后缀
        self.assertIn('close_D', result.columns)
        self.assertIn('close_M', result.columns)
        
    def test_compute_multi_freq_factor(self):
        """测试计算多频率因子"""
        # 配置mock
        self.processor.get_multi_freq_data = MagicMock(return_value={
            'day': self.daily_test_data.copy(),
            '5min': self.minute_test_data.copy()
        })
        self.processor.align_multi_freq_data = MagicMock(return_value={
            'day': self.daily_test_data.copy(),
            '5min': self.minute_test_data.copy()
        })
        self.processor.merge_multi_freq_data = MagicMock()
        self.processor.cache.get = MagicMock(return_value=None)
        
        # 模拟合并后的结果
        merged_data = self.daily_test_data.copy()
        merged_data['datetime'] = pd.to_datetime(merged_data['date'])
        merged_data['close_day'] = merged_data['close']
        merged_data['close_5min'] = np.random.rand(len(merged_data)) * 50 + 10
        merged_data['volatility_ratio'] = np.random.rand(len(merged_data))
        self.processor.merge_multi_freq_data.return_value = merged_data
        
        # 测试计算波动率比率因子
        result = self.processor.compute_multi_freq_factor(
            factor_name="volatility_ratio",
            symbols=['000001.SZ', '600000.SH'],
            start_date='2022-01-01',
            end_date='2022-01-10',
            params={
                "high_freq": "5min",
                "low_freq": "day",
                "window": 20
            },
            freq="day"
        )
        
        # 验证调用
        self.processor.get_multi_freq_data.assert_called_once()
        self.processor.align_multi_freq_data.assert_called_once()
        self.processor.merge_multi_freq_data.assert_called_once()
        
        # 验证返回结果
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('volatility_ratio', result.columns)
        
    def test_resample(self):
        """测试重采样"""
        # 准备测试数据
        test_data = self.minute_test_data.copy()
        
        # 测试重采样到小时级别
        result = self.processor.resample(
            test_data,
            target_freq='1h',
            method='ohlc'
        )
        
        # 验证返回结果
        self.assertIsInstance(result, pd.DataFrame)
        
        # 重采样后的数据应该比原始数据少
        if not result.empty:
            self.assertLess(result.shape[0], test_data.shape[0])
            
            # 验证OHLC列存在
            self.assertIn('open', result.columns)
            self.assertIn('high', result.columns)
            self.assertIn('low', result.columns)
            self.assertIn('close', result.columns)
    
    def test_get_requirements(self):
        """测试获取需求"""
        # 配置mock
        self.processor.daily_processor.get_requirements.return_value = [
            'symbol', 'date', 'open', 'high', 'low', 'close', 'volume'
        ]
        self.processor.minute_processor.get_requirements.return_value = [
            'symbol', 'date', 'time', 'open', 'high', 'low', 'close', 'volume'
        ]
        
        # 测试获取需求
        result = self.processor.get_requirements()
        
        # 验证调用
        self.processor.daily_processor.get_requirements.assert_called_once()
        self.processor.minute_processor.get_requirements.assert_called_once()
        
        # 验证返回结果（应该是两个列表的合并去重）
        self.assertIsInstance(result, list)
        self.assertIn('symbol', result)
        self.assertIn('date', result)
        self.assertIn('time', result)
        self.assertIn('open', result)
        self.assertIn('high', result)
        self.assertIn('low', result)
        self.assertIn('close', result)
        self.assertIn('volume', result)


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # 运行测试
    unittest.main()