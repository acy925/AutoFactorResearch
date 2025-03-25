"""
因子库使用测试 - 演示如何使用因子库计算和组合因子
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
from autofactor.factor.pipeline import FactorPipeline


class MomentumFactor(Factor):
    """动量因子"""
    
    def __init__(self, window=20):
        """初始化动量因子"""
        name = f"momentum_{window}d"
        super().__init__(
            name=name,
            description=f"{window}日价格动量因子",
            category="Technical",
            subcategory="Momentum",
            parameters={"window": window}
        )
        self.window = window
        
    def compute(self, data, processor=None, **kwargs):
        """计算动量因子"""
        symbol_col = kwargs.get("symbol_col", "symbol")
        close_col = kwargs.get("close_col", "close")
        
        # 计算动量
        result = data.copy()
        result[self.name] = result.groupby(symbol_col)[close_col].pct_change(self.window)
        
        # 可选标准化
        if processor is not None and kwargs.get("normalize", False):
            norm_method = kwargs.get("normalize_method", "zscore")
            norm_data = result.copy()
            norm_data = processor.normalize(
                norm_data, 
                method=norm_method,
                by_cross_section=kwargs.get("by_cross_section", True)
            )
            result[f"{self.name}_norm"] = norm_data[self.name]
            
        return result
        
    def get_requirements(self):
        """获取计算因子所需字段"""
        return ["symbol", "date", "close"]


class RSIFactor(Factor):
    """相对强弱指数因子"""
    
    def __init__(self, window=14):
        """初始化RSI因子"""
        name = f"rsi_{window}d"
        super().__init__(
            name=name,
            description=f"{window}日相对强弱指数",
            category="Technical",
            subcategory="Momentum",
            parameters={"window": window}
        )
        self.window = window
        
    def compute(self, data, processor=None, **kwargs):
        """计算RSI因子"""
        symbol_col = kwargs.get("symbol_col", "symbol")
        close_col = kwargs.get("close_col", "close")
        
        result = data.copy()

        # 确保symbol列存在
        if symbol_col not in result.columns:
            raise ValueError(f"数据中无{symbol_col}列。可用列: {result.columns.tolist()}")
        
        # 计算价格变化
        price_diff = result.groupby(symbol_col)[close_col].diff()
        
        # 将 price_diff 转换为 DataFrame，并加入 symbol 列
        price_diff_df = pd.DataFrame({
            symbol_col: result[symbol_col],
            'price_diff': price_diff
        })
        
        # 分离涨跌幅
        gain = price_diff_df['price_diff'].copy()
        loss = price_diff_df['price_diff'].copy()
        
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = -loss  # 使loss为正值
        
        # 使用 price_diff_df 进行 groupby 操作
        avg_gain = price_diff_df.groupby(symbol_col)['price_diff'].rolling(window=self.window).mean().reset_index(level=0, drop=True)
        avg_loss = price_diff_df.groupby(symbol_col)['price_diff'].rolling(window=self.window).mean().reset_index(level=0, drop=True)
        
        # 计算相对强度
        rs = avg_gain / avg_loss
        
        # 计算RSI
        result[self.name] = 100 - (100 / (1 + rs))
        
        # 处理缺失值
        result[self.name] = result[self.name].fillna(50)  # 用 50 填充（中性值）
        
        # 可选标准化
        if processor is not None and kwargs.get("normalize", False):
            norm_method = kwargs.get("normalize_method", "zscore")
            norm_data = result.copy()
            norm_data = processor.normalize(
                norm_data, 
                method=norm_method,
                by_cross_section=kwargs.get("by_cross_section", True)
            )
            result[f"{self.name}_norm"] = norm_data[self.name]
            
        return result
        
    def get_requirements(self):
        """获取计算因子所需字段"""
        return ["symbol", 'date', "close"]


def test_factor_library():
    """测试因子库的使用"""
    logger.info("开始测试因子库...")
    
    # 初始化DolphinDB客户端(测试模式)
    from config.settings import DOLPHINDB
    DOLPHINDB["test_mode"] = True
    db_client = DolphinDBClient()
    
    # 初始化日线数据处理器
    processor = DailyDataProcessor()
    processor.set_db_client(db_client)
    
    # 创建测试股票和时间范围
    symbols = ["000001.SZ", "600000.SH"]
    start_date = "2022-01-01"
    end_date = "2022-03-31"
    
    # 获取测试数据
    logger.info(f"获取测试数据: {symbols}, {start_date} - {end_date}")
    fields = ["symbol", "date", "open", "high", "low", "close", "volume"]
    data = processor.get_data(symbols, start_date, end_date, fields=fields)
    logger.info(f"获取数据成功，形状: {data.shape}")
    
    # 1. 测试单个因子计算
    logger.info("测试单个因子计算...")
    
    # 创建动量因子
    momentum_factor = MomentumFactor(window=10)
    logger.info(f"创建因子: {momentum_factor.name}")
    
    # 计算动量因子
    momentum_data = momentum_factor.compute(data, processor, normalize=True)
    logger.info(f"计算因子成功，结果示例:\n{momentum_data[['symbol', 'date', 'close', momentum_factor.name, momentum_factor.name+'_norm']].head()}")
    
    # 2. 测试因子注册表
    logger.info("测试因子注册表...")
    
    # 创建因子注册表
    registry = FactorRegistry()
    
    # 注册多个因子
    registry.register(MomentumFactor(window=5))
    registry.register(MomentumFactor(window=10))
    registry.register(MomentumFactor(window=20))
    registry.register(RSIFactor(window=14))
    
    logger.info(f"注册因子数量: {len(registry.factors)}")
    logger.info(f"可用因子列表: {registry.list_factors()}")
    logger.info(f"动量类因子: {registry.list_factors(subcategory='Momentum')}")
    
    # 获取特定因子
    factor = registry.get_factor("momentum_10d")
    logger.info(f"获取因子 momentum_10d: {factor.name}, {factor.description}")
    
    # 3. 测试因子管道
    logger.info("测试因子管道...")
    
    # 创建因子管道
    pipeline = FactorPipeline(processor, registry=registry)  # 传递registry参数
    
    # 添加因子计算步骤
    pipeline.add_factor(MomentumFactor(window=5), normalize=True)
    pipeline.add_factor(RSIFactor(window=14), normalize=True)
    
    # 执行管道
    result = pipeline.execute(symbols, start_date, end_date)
    
    logger.info(f"管道执行成功，结果形状: {result.shape}")
    logger.info(f"结果包含的列: {result.columns.tolist()}")
    logger.info(f"因子计算结果示例:\n{result[['symbol', 'date', 'momentum_5d', 'momentum_5d_norm', 'rsi_14d', 'rsi_14d_norm']].head()}")
    
    # 4. 可视化因子
    try:
        logger.info("绘制因子图表...")
        plt.figure(figsize=(12, 8))

        # 设置 Matplotlib 支持中文
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        plt.figure(figsize=(12, 8))
        
        # 选择一只股票的数据
        stock_data = result[result['symbol'] == symbols[0]].sort_values('date')
        
        # 创建子图
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # 绘制价格
        axes[0].plot(stock_data['date'], stock_data['close'])
        axes[0].set_title(f"{symbols[0]} 价格")
        axes[0].grid(True)
        
        # 绘制动量因子
        axes[1].plot(stock_data['date'], stock_data['momentum_5d'])
        axes[1].set_title("5日动量因子")
        axes[1].grid(True)
        
        # 绘制RSI因子
        axes[2].plot(stock_data['date'], stock_data['rsi_14d'])
        axes[2].axhline(y=70, color='r', linestyle='-')
        axes[2].axhline(y=30, color='g', linestyle='-')
        axes[2].set_title("14日RSI")
        axes[2].grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = "factor_chart.png"
        plt.savefig(chart_path)
        logger.info(f"图表已保存至: {chart_path}")
    except Exception as e:
        logger.warning(f"绘制图表失败: {e}")
    
    logger.info("因子库测试完成")


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # 运行测试
    test_factor_library()