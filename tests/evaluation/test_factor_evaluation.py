"""
因子评价模块测试
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from autofactor.data.dolphindb_client import DolphinDBClient
from autofactor.data.processor.frequency.daily_processor import DailyDataProcessor
from autofactor.factor.base import Factor
from autofactor.factor.registry import FactorRegistry
from autofactor.factor.categories.style.momentum import MomentumFactor
from autofactor.evaluation.ic_analysis import ICAnalyzer
from autofactor.evaluation.return_analysis import QuantileAnalyzer


def test_factor_evaluation():
    """测试因子评价功能"""
    logger.info("开始测试因子评价...")
    
    # 初始化DolphinDB客户端(测试模式)
    from config.settings import DOLPHINDB
    DOLPHINDB["test_mode"] = True
    db_client = DolphinDBClient()
    
    # 初始化日线数据处理器
    processor = DailyDataProcessor()
    processor.set_db_client(db_client)
    
    # 创建测试股票和时间范围
    symbols = ["000001.SZ", "600000.SH", "000002.SZ", "600001.SH", "000568.SZ", "600519.SH"]
    start_date = "2020-01-01"
    end_date = "2022-12-31"
    
    # 获取测试数据
    logger.info(f"获取测试数据: {symbols}, {start_date} - {end_date}")
    fields = ["symbol", "date", "open", "high", "low", "close", "volume"]
    data = processor.get_data(symbols, start_date, end_date, fields=fields)
    logger.info(f"获取数据成功，形状: {data.shape}")
    
    # 1. 创建测试因子
    momentum_factor = MomentumFactor(window=20)
    logger.info(f"创建测试因子: {momentum_factor.name}")
    
    # 2. 计算因子值
    factor_data = momentum_factor.compute(data)
    logger.info(f"因子计算成功，结果形状: {factor_data.shape}")
    
    # 3. 测试IC分析
    logger.info("开始IC分析...")
    ic_analyzer = ICAnalyzer()
    
    # 计算多个周期的IC
    ic_results = ic_analyzer.calculate_ic(
        factor_data,
        factor_col=momentum_factor.name,
        forward_periods=[1, 5, 10, 20],
        return_col='close',
        date_col='date',
        symbol_col='symbol'
    )
    
    # 打印IC统计信息
    logger.info(f"IC分析结果:")
    for period, result in ic_results.items():
        logger.info(f"周期 {period} 天:")
        for key, value in result['ic_stats'].items():
            logger.info(f"  {key}: {value:.4f}")
    
    # 计算IC衰减
    ic_decay = ic_analyzer.get_ic_decay(
        factor_data,
        factor_col=momentum_factor.name,
        max_periods=20
    )
    logger.info(f"IC衰减结果: {ic_decay}")
    
    # 4. 测试分层回测
    logger.info("开始分层回测分析...")
    quantile_analyzer = QuantileAnalyzer()

    # 计算分位数收益
    quantile_results = quantile_analyzer.calculate_quantile_returns(
        factor_data,
        factor_col=momentum_factor.name,
        n_quantiles=5,
        forward_periods=[5, 10, 20],
        price_col='close',
        date_col='date',
        symbol_col='symbol'
    )

    # 打印分层回测统计信息
    logger.info(f"分层回测分析结果:")
    for period, result in quantile_results.items():
        if result:
            logger.info(f"周期 {period} 天:")
            # 使用同一个 quantile_analyzer 实例调用 summary()
            summary = quantile_analyzer.summary()
            for key, value in summary.items():
                if isinstance(value, dict):
                    logger.info(f"  {key}:")
                    for k, v in value.items():
                        logger.info(f"    {k}: {v:.4f}")
                else:
                    logger.info(f"  {key}: {value:.4f}")
    
    # 5. 绘制图表
    try:
        logger.info("绘制IC分析图表...")
        
        # 绘制IC时间序列
        ic_fig = ic_analyzer.plot_ic_series()
        ic_fig.savefig("ic_series.png")
        logger.info("IC时间序列图已保存至: ic_series.png")
        
        # 绘制IC分布
        ic_dist_fig = ic_analyzer.plot_ic_distribution()
        ic_dist_fig.savefig("ic_distribution.png")
        logger.info("IC分布图已保存至: ic_distribution.png")
        
        # 绘制IC衰减
        ic_decay_fig = ic_analyzer.plot_ic_decay(ic_decay)
        ic_decay_fig.savefig("ic_decay.png")
        logger.info("IC衰减图已保存至: ic_decay.png")
        
        logger.info("绘制分层回测图表...")
        
        # 绘制累积收益
        returns_fig = quantile_analyzer.plot_cumulative_returns()
        returns_fig.savefig("quantile_returns.png")
        logger.info("分位数累积收益图已保存至: quantile_returns.png")
        
        # 绘制多空收益
        ls_fig = quantile_analyzer.plot_long_short_returns()
        ls_fig.savefig("long_short_returns.png")
        logger.info("多空收益图已保存至: long_short_returns.png")
        
        # 绘制分位数收益分布
        dist_fig = quantile_analyzer.plot_quantile_returns_distribution()
        dist_fig.savefig("returns_distribution.png")
        logger.info("收益分布图已保存至: returns_distribution.png")
        
    except Exception as e:
        logger.warning(f"绘制图表失败: {e}")
    
    logger.info("因子评价测试完成")


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    
    # 运行测试
    test_factor_evaluation()

