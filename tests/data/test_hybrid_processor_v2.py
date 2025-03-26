"""
混合处理器测试v2, 测试的功能更多
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import time
import numpy as np
import pandas as pd
from loguru import logger

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from autofactor.data.processor.location.hybrid_processor import HybridProcessor
from autofactor.data.utils.cache import CacheManager


class TestHybridProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = HybridProcessor(config={"rows_threshold": 1000})
        self.processor.cache = CacheManager(config={})
        self.processor.cache.clear()
        self.processor.stats = {"local_tasks": 0, "db_tasks": 0, "cache_hits": 0, "errors": 0, "fallbacks": 0}
        self.create_test_data()
    
    def create_test_data(self):
        dates = pd.date_range('2022-01-01', periods=10)
        symbols = ['000001.SZ', '600000.SH']
        data = []
        for symbol in symbols:
            for date in dates:
                data.append({
                    'date': date, 'symbol': symbol, 'open': np.random.rand() * 50 + 10,
                    'high': np.random.rand() * 50 + 15, 'low': np.random.rand() * 50 + 5,
                    'close': np.random.rand() * 50 + 10, 'volume': np.random.randint(1000, 10000),
                    'amount': np.random.randint(10000, 100000), 'adj_factor': np.random.rand() + 0.5
                })
        self.test_data = pd.DataFrame(data)

    def test_decide_processor(self):
        processor_type = self.processor._decide_processor("handle_outliers", {"rows": 1000})
        self.assertEqual(processor_type, "local")
        processor_type = self.processor._decide_processor("resample", {"rows": 1000})
        self.assertEqual(processor_type, "db")
        processor_type = self.processor._decide_processor("normalize", {"rows": 2000})
        self.assertEqual(processor_type, "db")

    def test_get_data_size(self):
        data_size = self.processor._get_data_size(data=self.test_data)
        self.assertEqual(data_size["rows"], 20)
        data_size = self.processor._get_data_size(symbols=['000001.SZ'], start_date='2022-01-01', end_date='2022-01-31')
        self.assertEqual(data_size["symbols"], 1)

    def test_hybrid_processor_with_real_data(self):
        logger.info("开始测试混合处理器真实数据处理...")
        symbols = ["000001.SZ", "600000.SH", "000002.SZ"]
        start_date = "2022-01-01"
        end_date = "2022-01-31"
        data = self.processor.get_data(symbols, start_date, end_date)
        logger.info(f"获取数据成功，形状: {data.shape}")
        if data.empty:
            self.fail("获取的数据为空，无法进行后续处理")
        data_with_na = data.copy()
        data_with_na.loc[data_with_na.sample(frac=0.1).index, "close"] = np.nan
        data_filled = self.processor.handle_missing_values(data_with_na, method="ffill")
        logger.info(f"缺失值处理前: {data_with_na.isna().sum().sum()}, 处理后: {data_filled.isna().sum().sum()}")
        data_normalized = self.processor.normalize(data_filled, method="zscore")
        logger.info(f"标准化后数据前5行:\n{data_normalized.head()}")
        stats = self.processor.get_stats()
        logger.info(f"统计信息: {stats}")

    def test_boundary_conditions(self):
        large_data = pd.DataFrame(np.random.rand(10000, 5), columns=["date", "symbol", "open", "high", "close"])
        result = self.processor.normalize(large_data)
        self.assertGreaterEqual(self.processor.stats["db_tasks"], 1, "大规模数据应使用DB处理器")
        with self.assertRaises(ValueError):
            self.processor._execute_task("local", "unsupported_task", large_data)

    def test_performance(self):
        start_time = time.time()
        self.processor.get_data(["000001.SZ"], "2022-01-01", "2022-01-02")
        small_time = time.time() - start_time
        logger.info(f"小规模数据执行时间: {small_time:.3f}秒")
        large_data = pd.DataFrame(np.random.rand(2000, 5), columns=["date", "symbol", "open", "high", "close"])
        start_time = time.time()
        self.processor.normalize(large_data)
        large_time = time.time() - start_time
        logger.info(f"大规模数据执行时间: {large_time:.3f}秒")
        self.assertTrue(small_time < large_time)

    def test_fallback_and_error_handling(self):
        self.processor._get_data_size = MagicMock(return_value={"rows": 100})
        self.processor._decide_processor = MagicMock(return_value="db")
        self.processor._execute_task = MagicMock(side_effect=[Exception("DB失败"), Exception("本地失败")])
        with self.assertRaises(Exception) as context:
            self.processor._handle_task_with_fallback("get_data", ["000001.SZ"], "2022-01-01", "2022-01-02")
        self.assertTrue("本地失败" in str(context.exception))
        self.assertEqual(self.processor.stats["fallbacks"], 1)
        self.assertEqual(self.processor.stats["errors"], 2)

    def test_cache_functionality(self):
        result1 = self.processor.get_data(["000001.SZ"], "2022-01-01", "2022-01-02")
        self.assertEqual(self.processor.stats["cache_hits"], 0)
        self.processor.cache.get = MagicMock(return_value=self.test_data)
        result2 = self.processor.get_data(["000001.SZ"], "2022-01-01", "2022-01-02")
        self.assertEqual(self.processor.stats["cache_hits"], 1)

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")  # 改为 DEBUG
    unittest.main()