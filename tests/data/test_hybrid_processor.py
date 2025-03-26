"""
混合处理器测试
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

from autofactor.data.processor.base import DataProcessor
from autofactor.data.processor.location.local_processor import LocalDataProcessor
from autofactor.data.processor.location.db_processor import DBProcessor
from autofactor.data.processor.location.hybrid_processor import HybridProcessor
from autofactor.data.utils.cache import CacheManager


class TestHybridProcessor(unittest.TestCase):
    """混合处理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # Mock本地处理器和DB处理器
        self.mock_local_processor = MagicMock(spec=LocalDataProcessor)
        self.mock_db_processor = MagicMock(spec=DBProcessor)
        self.mock_cache = MagicMock(spec=CacheManager)
        
        # 使用patch来避免实例化抽象类
        with patch('autofactor.data.processor.location.hybrid_processor.LocalDataProcessor', 
                  return_value=self.mock_local_processor), \
             patch('autofactor.data.processor.location.hybrid_processor.DBProcessor', 
                  return_value=self.mock_db_processor), \
             patch('autofactor.data.processor.location.hybrid_processor.CacheManager', 
                  return_value=self.mock_cache):
            # 现在可以安全地实例化HybridProcessor
            self.processor = HybridProcessor()
            
            # 手动设置一些内部属性
            self.processor.stats = {
                "local_tasks": 0,
                "db_tasks": 0,
                "cache_hits": 0,
                "errors": 0,
                "fallbacks": 0
            }
            
            self.processor.task_history = []
            
        # 创建测试数据
        self.create_test_data()
        
    def create_test_data(self):
        """创建测试数据"""
        # 股票日线数据
        dates = pd.date_range('2022-01-01', periods=10)
        symbols = ['000001.SZ', '600000.SH']
        
        data = []
        for symbol in symbols:
            for date in dates:
                data.append({
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
        
        self.test_data = pd.DataFrame(data)
        
    def test_decide_processor(self):
        """测试处理器决策逻辑"""
        # 测试任务类型决策
        # 本地优先任务
        processor_type = self.processor._decide_processor(
            task_type="handle_outliers", 
            data_size={"rows": 1000, "columns": 10, "symbols": 2, "days": 10}
        )
        self.assertEqual(processor_type, "local")
        
        # DB优先任务
        processor_type = self.processor._decide_processor(
            task_type="resample", 
            data_size={"rows": 1000, "columns": 10, "symbols": 2, "days": 10}
        )
        self.assertEqual(processor_type, "db")
        
        # 根据数据规模决策
        # 大规模数据
        processor_type = self.processor._decide_processor(
            task_type="normalize", 
            data_size={"rows": 200000, "columns": 10, "symbols": 2, "days": 10}
        )
        self.assertEqual(processor_type, "db")
        
        # 多股票
        processor_type = self.processor._decide_processor(
            task_type="normalize", 
            data_size={"rows": 1000, "columns": 10, "symbols": 100, "days": 10}
        )
        self.assertEqual(processor_type, "db")
        
        # 默认本地处理
        processor_type = self.processor._decide_processor(
            task_type="normalize", 
            data_size={"rows": 1000, "columns": 10, "symbols": 2, "days": 10}
        )
        self.assertEqual(processor_type, "local")
        
    def test_get_data_size(self):
        """测试数据规模估算"""
        # 使用DataFrame估算
        data_size = self.processor._get_data_size(data=self.test_data)
        self.assertEqual(data_size["rows"], len(self.test_data))
        self.assertEqual(data_size["columns"], len(self.test_data.columns))
        self.assertEqual(data_size["symbols"], 2)  # 两只股票
        
        # 使用参数估算
        data_size = self.processor._get_data_size(
            symbols=['000001.SZ', '600000.SH', '000002.SZ'],
            start_date='2022-01-01',
            end_date='2022-01-31'
        )
        self.assertEqual(data_size["symbols"], 3)
        self.assertTrue(data_size["days"] > 20)  # 约一个月的交易日
        
    def test_get_data(self):
        """测试获取数据"""
        # 替换_handle_task_with_fallback方法
        self.processor._handle_task_with_fallback = MagicMock(return_value=self.test_data)
        self.processor.cache.get = MagicMock(return_value=None)
        
        # 测试获取数据
        symbols = ['000001.SZ', '600000.SH']
        result = self.processor.get_data(
            symbols=symbols,
            start_date='2022-01-01',
            end_date='2022-01-10',
            fields=['symbol', 'date', 'open', 'close'],
            freq='day',
            adjust=True
        )
        
        # 验证_handle_task_with_fallback被调用
        self.processor._handle_task_with_fallback.assert_called_once()
        
        # 验证返回的数据
        self.assertEqual(result.shape[0], len(self.test_data))
        
    def test_handle_task_with_fallback(self):
        """测试任务处理和降级"""
        # 创建专门的受控环境用于测试此方法
        processor = HybridProcessor()
        processor._execute_task = MagicMock()
        processor._decide_processor = MagicMock(return_value="db")
        processor._get_data_size = MagicMock(return_value={})
        
        # 测试正常执行
        processor._execute_task.return_value = self.test_data
        
        # 使用_handle_task_with_fallback方法
        result = processor._handle_task_with_fallback(
            "get_data",
            ['000001.SZ', '600000.SH'],
            '2022-01-01',
            '2022-01-10',
            fields=['symbol', 'date', 'open', 'close']
        )
        
        # 验证调用
        processor._decide_processor.assert_called_once()
        processor._get_data_size.assert_called_once()
        processor._execute_task.assert_called_once_with(
            "db", "get_data", 
            ['000001.SZ', '600000.SH'], '2022-01-01', '2022-01-10', 
            fields=['symbol', 'date', 'open', 'close']
        )
        
        # 验证返回的数据
        self.assertEqual(result.shape[0], len(self.test_data))
        
        # 重置mock
        processor._execute_task.reset_mock()
        processor._decide_processor.reset_mock()
        processor._get_data_size.reset_mock()
        
        # 测试降级处理
        # 第一次调用失败，第二次调用成功
        processor._execute_task.side_effect = [Exception("DB处理失败"), self.test_data]
        
        result = processor._handle_task_with_fallback(
            "get_data",
            ['000001.SZ', '600000.SH'],
            '2022-01-01',
            '2022-01-10',
            fields=['symbol', 'date', 'open', 'close']
        )
        
        # 验证返回的数据
        self.assertEqual(result.shape[0], len(self.test_data))
        
        # 验证调用
        self.assertEqual(processor._execute_task.call_count, 2)
        processor._execute_task.assert_any_call(
            "db", "get_data", 
            ['000001.SZ', '600000.SH'], '2022-01-01', '2022-01-10', 
            fields=['symbol', 'date', 'open', 'close']
        )
        processor._execute_task.assert_any_call(
            "local", "get_data", 
            ['000001.SZ', '600000.SH'], '2022-01-01', '2022-01-10', 
            fields=['symbol', 'date', 'open', 'close']
        )
        
        # 验证统计信息
        self.assertEqual(processor.stats["fallbacks"], 1)

    def test_normalize(self):
        """测试标准化方法"""
        # 替换_handle_task_with_fallback方法
        self.processor._handle_task_with_fallback = MagicMock(return_value=self.test_data)
        
        # 调用标准化方法
        result = self.processor.normalize(
            self.test_data,
            method='zscore',
            by_cross_section=True
        )
        
        # 验证_handle_task_with_fallback被调用
        self.processor._handle_task_with_fallback.assert_called_once_with(
            "normalize", 
            self.test_data, 
            method='zscore', 
            by_cross_section=True,
            date_col='date',
            symbol_col='symbol'
        )
        
        # 验证返回的数据
        self.assertEqual(result.shape[0], len(self.test_data))
        
    def test_compute_factor(self):
        """测试因子计算"""
        # 为本地处理器添加compute_factor方法的mock
        self.mock_local_processor.compute_factor.return_value = self.test_data
        self.mock_cache.get.return_value = None
        
        # 调用因子计算方法
        result = self.processor.compute_factor(
            factor_name="momentum",
            symbols=['000001.SZ', '600000.SH'],
            start_date='2022-01-01',
            end_date='2022-01-10',
            params={"window": 5}
        )
        
        # 验证本地处理器的compute_factor被调用
        self.mock_local_processor.compute_factor.assert_called_once_with(
            "momentum", 
            ['000001.SZ', '600000.SH'], 
            '2022-01-01', 
            '2022-01-10', 
            {"window": 5}, 
            "day"
        )
        
        # 验证返回的数据
        self.assertEqual(result.shape[0], len(self.test_data))
        
    def test_get_stats(self):
        """测试获取统计信息"""
        # 设置一些统计信息
        self.processor.stats = {
            "local_tasks": 5,
            "db_tasks": 3,
            "cache_hits": 2,
            "errors": 1,
            "fallbacks": 1
        }
        
        # 添加一些任务历史
        self.processor.task_history = [
            {"task_type": "get_data", "processor": "db", "time": 0.5, "error": None},
            {"task_type": "normalize", "processor": "local", "time": 0.2, "error": None}
        ]
        
        # 获取统计信息
        stats = self.processor.get_stats()
        
        # 验证返回的信息
        self.assertIn("task_stats", stats)
        self.assertIn("task_history", stats)
        self.assertIn("processor_preference", stats)
        self.assertEqual(stats["task_stats"]["local_tasks"], 5)
        self.assertEqual(stats["task_stats"]["db_tasks"], 3)
        self.assertEqual(len(stats["task_history"]), 2)


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # 运行测试
    unittest.main()