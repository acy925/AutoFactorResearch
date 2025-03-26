"""
本地数据处理器测试
"""
import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import pandas as pd
from loguru import logger

from autofactor.data.dolphindb_client import DolphinDBClient
from autofactor.data.processor.location.local_processor import LocalDataProcessor


def test_local_processor():
    """测试本地数据处理器"""
    logger.info("开始测试本地数据处理器...")
    
    # 初始化DolphinDB客户端(测试模式)
    db_client = DolphinDBClient(test_mode=True)
    
    # 初始化本地数据处理器
    processor = LocalDataProcessor()
    processor.set_db_client(db_client)
    
    # 测试数据获取
    symbols = ["000001.SZ", "600000.SH", "000002.SZ"]  # 增加更多股票
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    
    logger.info(f"获取数据: {symbols}, {start_date} - {end_date}")
    data = processor.get_data(symbols, start_date, end_date)
    data = data.rename(columns={"stock_code": "symbol"})
    
    # 模拟一些缺失值
    data.loc[data.sample(frac=0.05).index, "close"] = np.nan
    
    logger.info(f"获取数据成功，形状: {data.shape}")
    logger.info(f"数据前5行:\n{data.head()}")
    
    # 测试缺失值处理
    logger.info("测试缺失值处理...")
    data_filled = processor.handle_missing_values(data, method="ffill")
    missing_before = data.isna().sum().sum()
    missing_after = data_filled.isna().sum().sum()
    logger.info(f"缺失值处理前: {missing_before}, 处理后: {missing_after}")
    
    # 测试异常值处理
    logger.info("测试异常值处理...")
    data_outliers_handled = processor.handle_outliers(data_filled, method="winsorize", limits=(0.05, 0.95))
    logger.info(f"异常值处理后数据前5行:\n{data_outliers_handled.head()}")
    
    # 测试重采样
    logger.info("测试数据重采样...")
    data_resampled = processor.resample(data_filled, target_freq="W", method="ohlc")
    logger.info(f"重采样后数据形状: {data_resampled.shape}")
    logger.info(f"重采样后数据前5行:\n{data_resampled.head()}")
    
    # 测试标准化
    logger.info("测试标准化...")
    data_normalized = processor.normalize(data_filled, method="zscore", symbol_col="symbol")
    logger.info(f"标准化后数据前5行:\n{data_normalized.head()}")
    
    # 测试因子计算
    logger.info("测试因子计算...")
    factor_params = {"window": 20, "fields": ["date", "symbol", "close"]}
    momentum_data = processor.compute_factor(
        "volatility", symbols, start_date, end_date, params=factor_params
    )
    logger.info(f"波动率因子数据20-25行:\n{momentum_data.iloc[20:25]}")
    
    # 测试中性化处理
    logger.info("测试中性化处理...")
    factor_data = momentum_data.copy()
    factor_data["factor"] = factor_data["volatility"]
    factor_data["industry"] = ["A", "B", "C"] * (len(factor_data) // 3)  # 增加更多行业
    factor_data["market_cap"] = np.log(factor_data["close"] * 1000) * 1000  # 更真实的市值
    neutralized_data = processor.neutralize(
        factor_data,
        date_col="date",
        symbol_col="symbol",
        factor_col="factor",
        industry_col="industry",
        market_cap_col="market_cap"
    )
    logger.info(f"中性化后数据20-25行:\n{neutralized_data.iloc[20:25]}")
    
    logger.info("本地数据处理器测试完成")


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # 运行测试
    test_local_processor()