"""
日线数据处理器测试
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import pandas as pd
from loguru import logger

from autofactor.data.dolphindb_client import DolphinDBClient
from autofactor.data.processor.frequency.daily_processor import DailyDataProcessor


def test_daily_processor():
    """测试日线数据处理器"""
    logger.info("开始测试日线数据处理器...")
    
    # 初始化DolphinDB客户端(测试模式)
    db_client = DolphinDBClient(test_mode=True)
    
    # 初始化日线数据处理器
    processor = DailyDataProcessor()
    processor.set_db_client(db_client)

    # 测试开始前清除缓存
    if hasattr(processor, 'cache') and hasattr(processor.cache, 'clear'):
        processor.cache.clear()
        print("已清除数据处理器缓存")
    
    # 测试数据获取
    symbols = ["000001.SZ", "600000.SH"]
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    
    logger.info(f"获取数据: {symbols}, {start_date} - {end_date}")
    data = processor.get_data(symbols, start_date, end_date)

    # 重要: 将stock_code列重命名为symbol
    data = data.rename(columns={"stock_code": "symbol"})

    logger.info(f"获取数据成功，形状: {data.shape}")
    logger.info(f"数据前5行:\n{data.head()}")
    
    # 测试缺失值处理
    logger.info("测试缺失值处理...")
    data_filled = processor.handle_missing_values(data)
    missing_before = data.isna().sum().sum()
    missing_after = data_filled.isna().sum().sum()
    logger.info(f"缺失值处理前: {missing_before}, 处理后: {missing_after}")
    
    # 测试标准化
    logger.info("测试标准化...")
    data_normalized = processor.normalize(data, method="zscore", symbol_col="symbol")
    logger.info(f"标准化后数据前5行:\n{data_normalized.head()}")
    
    # 测试因子计算
    logger.info("测试因子计算...")
    # 因子计算前获取数据，并重命名列
    data = processor.get_data(symbols, start_date, end_date, fields=["symbol", "open", "high", "low", "close", "volume", "amount"])
    print("数据列名:", data.columns.tolist())  # 打印列名，确认实际列名
    data = data.rename(columns={"stock_code": "symbol"})
    
    # 为避免再次从数据库加载数据，我们可以直接在本地计算因子
    window = 20  # 动量窗口
    data["momentum"] = data.groupby("symbol")["close"].pct_change(window)
    logger.info(f"动量因子数据前5行:\n{data.head()}")
    
    logger.info("日线数据处理器测试完成")


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # 运行测试
    test_daily_processor()