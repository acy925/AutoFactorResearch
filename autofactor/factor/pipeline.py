"""
因子计算管道模块 - 用于顺序执行多个因子计算和转换
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from loguru import logger

from autofactor.factor.base import Factor


class FactorPipeline:
    """因子计算管道，用于顺序执行多个因子计算和转换"""
    
    def __init__(self, data_processor, registry=None):
        """初始化因子管道
        
        Args:
            data_processor: 数据处理器实例
            registry: 可选的因子注册表实例
        """
        self.data_processor = data_processor
        self.registry = registry  # 存储注册表
        self.steps = []  # 步骤列表
        
    def add_factor(self, factor, **params):
        """添加因子计算步骤
        
        Args:
            factor: Factor实例或名称
            **params: 计算参数
            
        Returns:
            self: 支持链式调用
        """
        self.steps.append(("factor", factor, params))
        return self
        
    def add_transform(self, transform_type, **params):
        """添加转换步骤
        
        Args:
            transform_type: 转换类型(如"normalize", "neutralize")
            **params: 转换参数
            
        Returns:
            self: 支持链式调用
        """
        self.steps.append(("transform", transform_type, params))
        return self
        
    def execute(self, symbols, start_date, end_date, freq="day"):
        """执行因子管道
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            freq: 数据频率
            
        Returns:
            pandas DataFrame: 计算结果数据
        """
        # 收集所有所需字段
        required_fields = ["symbol", "date"]
        
        for step_type, step, params in self.steps:
            if step_type == "factor":
                # 如果step是字符串，从注册表中获取因子
                if isinstance(step, str):
                    if self.registry is None:
                        raise ValueError("必须提供factor_registry才能通过名称引用因子")
                    factor = self.registry.get_factor(step)
                    if factor is None:
                        raise ValueError(f"未知因子: {step}")
                else:
                    factor = step
                    
                # 收集因子需要的字段
                for field in factor.get_requirements():
                    if field not in required_fields:
                        required_fields.append(field)
        
        logger.info(f"加载字段: {required_fields}")
        
        # 加载数据
        data = self.data_processor.get_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            fields=required_fields,
            freq=freq
        )
        
        logger.info(f"加载数据列: {data.columns.tolist()}")
        
        # 执行每个步骤
        for step_type, step, params in self.steps:
            if step_type == "factor":
                # 获取因子实例
                if isinstance(step, str):
                    factor = self.registry.get_factor(step)
                else:
                    factor = step
                    
                # 计算因子
                data = factor.compute(data, processor=self.data_processor, **params)
                
            elif step_type == "transform":
                if step == "normalize":
                    # 标准化
                    factor_cols = params.get("columns")
                    data = self.data_processor.normalize(
                        data,
                        method=params.get("method", "zscore"),
                        by_cross_section=params.get("by_cross_section", True),
                        symbol_col=params.get("symbol_col", "symbol"),
                        date_col=params.get("date_col", "date")
                    )
                    
                elif step == "neutralize":
                    # 中性化
                    data = self.data_processor.neutralize(
                        data,
                        factor_col=params.get("factor_col"),
                        industry_col=params.get("industry_col"),
                        market_cap_col=params.get("market_cap_col"),
                        symbol_col=params.get("symbol_col", "symbol"),
                        date_col=params.get("date_col", "date")
                    )
                    
        return data