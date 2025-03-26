"""
DolphinDB数据处理器测试 - 修复调用次数检查
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import pandas as pd
import numpy as np
from loguru import logger

from autofactor.data.dolphindb_client import DolphinDBClient
from autofactor.data.processor.location.db_processor import DBProcessor


class TestDBProcessor(unittest.TestCase):
    """DolphinDB数据处理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建conn属性的mock
        self.mock_conn = MagicMock()
        
        # 使用测试模式初始化DolphinDB客户端
        self.db_client = MagicMock(spec=DolphinDBClient)
        
        # 显式设置conn属性
        self.db_client.conn = self.mock_conn
        
        # 初始化DB处理器
        self.processor = DBProcessor()
        self.processor.db_client = self.db_client
        
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
        
    def test_get_data(self):
        """测试获取数据方法"""
        # 设置mock返回值
        self.db_client.execute.return_value = self.test_data
        
        # 调用get_data方法
        result = self.processor.get_data(
            symbols=['000001.SZ', '600000.SH'],
            start_date='2022-01-01',
            end_date='2022-01-10',
            fields=['symbol', 'date', 'open', 'close'],
            freq='day',
            adjust=True
        )
        
        # 验证execute被调用，且参数包含了正确的脚本内容
        self.db_client.execute.assert_called_once()
        script_arg = self.db_client.execute.call_args[0][0]
        
        # 验证脚本包含关键信息
        self.assertIn("getStockData", script_arg)
        self.assertIn("2022-01-01", script_arg)
        self.assertIn("2022-01-10", script_arg)
        self.assertIn("['000001.SZ', '600000.SH']", script_arg)
        
        # 验证返回的数据
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], len(self.test_data))
        
    def test_resample(self):
        """测试重采样方法"""
        # 准备测试数据（分钟数据）
        minute_data = self.test_data.copy()
        minute_data['time'] = pd.Series([f"{h:02d}:{m:02d}:00" for h, m in 
                                       [(9+i//6, 30+i%6*10) for i in range(10)]])
        
        # 设置mock返回值
        self.db_client.execute.return_value = self.test_data
        
        # 调用resample方法
        result = self.processor.resample(
            minute_data,
            target_freq='day',
            method='ohlc'
        )
        
        # 验证数据上传和执行
        # 1. 验证执行了数据上传
        self.mock_conn.upload.assert_called_once()
        upload_args = self.mock_conn.upload.call_args[0][0]
        self.assertIn("data", upload_args)
        
        # 2. 验证执行了临时表创建、重采样脚本和清理操作
        self.assertEqual(self.db_client.execute.call_count, 3)
        
        # 验证临时表创建调用
        temp_table_creation = self.db_client.execute.call_args_list[0][0][0]
        self.assertIn("table(data)", temp_table_creation)
        self.assertIn("share t", temp_table_creation)
        
        # 验证重采样脚本调用
        resample_script = self.db_client.execute.call_args_list[1][0][0]
        self.assertIn("resampleData", resample_script)
        
        # 验证清理临时表调用
        cleanup_script = self.db_client.execute.call_args_list[2][0][0]
        self.assertIn("undef(", cleanup_script)
        
        # 验证返回的数据
        self.assertIsInstance(result, pd.DataFrame)
        
    def test_handle_missing_values(self):
        """测试处理缺失值方法"""
        # 准备含有缺失值的测试数据
        test_data_with_na = self.test_data.copy()
        test_data_with_na.loc[0:5, 'close'] = np.nan
        
        # 设置mock返回值
        self.db_client.execute.return_value = self.test_data
        
        # 调用handle_missing_values方法
        result = self.processor.handle_missing_values(
            test_data_with_na,
            method='ffill'
        )
        
        # 验证数据上传和执行
        # 1. 验证执行了数据上传
        self.mock_conn.upload.assert_called_once()
        upload_args = self.mock_conn.upload.call_args[0][0]
        self.assertIn("data", upload_args)
        
        # 2. 验证执行了临时表创建、处理缺失值脚本和清理操作
        self.assertEqual(self.db_client.execute.call_count, 3)
        
        # 验证临时表创建调用
        temp_table_creation = self.db_client.execute.call_args_list[0][0][0]
        self.assertIn("table(data)", temp_table_creation)
        self.assertIn("share t", temp_table_creation)
        
        # 验证处理缺失值脚本调用
        missing_values_script = self.db_client.execute.call_args_list[1][0][0]
        self.assertIn("handleMissingValues", missing_values_script)
        self.assertIn("'ffill'", missing_values_script)
        
        # 验证清理临时表调用
        cleanup_script = self.db_client.execute.call_args_list[2][0][0]
        self.assertIn("undef(", cleanup_script)
        
        # 验证返回的数据
        self.assertIsInstance(result, pd.DataFrame)
        
    def test_normalize(self):
        """测试标准化方法"""
        # 设置mock返回值
        normalized_data = self.test_data.copy()
        normalized_data['close'] = (normalized_data['close'] - normalized_data['close'].mean()) / normalized_data['close'].std()
        self.db_client.execute.return_value = normalized_data
        
        # 调用normalize方法
        result = self.processor.normalize(
            self.test_data,
            method='zscore',
            by_cross_section=True
        )
        
        # 验证数据上传和执行
        # 1. 验证执行了数据上传
        self.mock_conn.upload.assert_called_once()
        upload_args = self.mock_conn.upload.call_args[0][0]
        self.assertIn("data", upload_args)
        
        # 2. 验证执行了临时表创建、标准化脚本和清理操作
        self.assertEqual(self.db_client.execute.call_count, 3)
        
        # 验证临时表创建调用
        temp_table_creation = self.db_client.execute.call_args_list[0][0][0]
        self.assertIn("table(data)", temp_table_creation)
        self.assertIn("share t", temp_table_creation)
        
        # 验证标准化脚本调用
        normalize_script = self.db_client.execute.call_args_list[1][0][0]
        self.assertIn("normalizeData", normalize_script)
        self.assertIn("'zscore'", normalize_script)
        self.assertIn("true", normalize_script)  # by_cross_section=True
        
        # 验证清理临时表调用
        cleanup_script = self.db_client.execute.call_args_list[2][0][0]
        self.assertIn("undef(", cleanup_script)
        
        # 验证返回的数据
        self.assertIsInstance(result, pd.DataFrame)
        
    def test_neutralize(self):
        """测试中性化方法"""
        # 准备测试数据（添加行业列）
        test_data_with_industry = self.test_data.copy()
        test_data_with_industry['industry'] = np.random.choice(['A', 'B', 'C'], size=len(test_data_with_industry))
        test_data_with_industry['factor'] = np.random.randn(len(test_data_with_industry))
        
        # 设置mock返回值
        neutralized_data = test_data_with_industry.copy()
        neutralized_data['factor_neutral'] = np.random.randn(len(neutralized_data))
        self.db_client.execute.return_value = neutralized_data
        
        # 调用neutralize方法
        result = self.processor.neutralize(
            test_data_with_industry,
            factor_col='factor',
            industry_col='industry'
        )
        
        # 验证数据上传和执行
        # 1. 验证执行了数据上传
        self.mock_conn.upload.assert_called_once()
        upload_args = self.mock_conn.upload.call_args[0][0]
        self.assertIn("factor_data", upload_args)
        
        # 2. 验证执行了临时表创建、中性化脚本和清理操作
        self.assertEqual(self.db_client.execute.call_count, 3)
        
        # 验证临时表创建调用
        temp_table_creation = self.db_client.execute.call_args_list[0][0][0]
        self.assertIn("table(factor_data)", temp_table_creation)
        self.assertIn("share t", temp_table_creation)
        
        # 验证中性化脚本调用
        neutralize_script = self.db_client.execute.call_args_list[1][0][0]
        self.assertIn("industryNeutralize", neutralize_script)
        self.assertIn("'factor'", neutralize_script)
        self.assertIn("'industry'", neutralize_script)
        
        # 验证清理临时表调用
        cleanup_script = self.db_client.execute.call_args_list[2][0][0]
        self.assertIn("undef(", cleanup_script)
        
        # 验证返回的数据
        self.assertIsInstance(result, pd.DataFrame)
        
    def test_generate_script(self):
        """测试脚本生成方法"""
        # 测试生成get_data脚本
        script = self.processor.generate_script(
            'get_data',
            symbols="['000001.SZ', '600000.SH']",
            start_date='2022-01-01',
            end_date='2022-01-10',
            fields='close, open',
            adjust='true',
            db_path='dfs://quantdb',
            table_name='daily_quote'
        )
        
        # 验证脚本内容
        self.assertIsInstance(script, str)
        self.assertIn("getStockData", script)
        self.assertIn("2022-01-01", script)
        self.assertIn("2022-01-10", script)
        
        # 测试生成normalize脚本
        script = self.processor.generate_script(
            'normalize',
            table='tmp_data',
            method='zscore',
            by_cross_section='true',
            date_col='date',
            symbol_col='symbol'
        )
        
        # 验证脚本内容
        self.assertIsInstance(script, str)
        self.assertIn("normalizeData", script)
        self.assertIn("zscore", script)


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # 运行测试
    unittest.main()