"""
日线数据处理器 - 专门处理日线级别数据
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from autofactor.data.processor.location.local_processor import LocalDataProcessor
from autofactor.data.utils.cache import CacheManager


class DailyDataProcessor(LocalDataProcessor):
    """日线数据处理器，专门处理日线级别数据"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化日线数据处理器"""
        super().__init__(config)
        # 初始化缓存
        self.cache = CacheManager(config)
        # 初始化行业信息
        self._init_industry_info()
        
    def _init_industry_info(self):
        """初始化行业分类信息"""
        self.industry_info = None
        # 将来实现：从数据库加载行业分类信息
        
    def get_data(self, symbols: Union[str, List[str]], 
                start_date: str, 
                end_date: str,
                fields: Optional[List[str]] = None,
                freq: str = "day",
                adjust: bool = True) -> pd.DataFrame:
        """获取日线数据，优先使用缓存
        
        首先尝试从缓存获取数据，如果缓存不存在则从数据库加载
        并存入缓存。支持全量加载和增量更新。
        
        Args:
            symbols: 股票代码或代码列表
            start_date: 开始日期，格式为 "YYYY-MM-DD"
            end_date: 结束日期，格式为 "YYYY-MM-DD"
            fields: 需要获取的字段列表，默认为所有字段
            freq: 数据频率，固定为 "day"
            adjust: 是否进行复权处理
            
        Returns:
            pd.DataFrame: 股票日线数据
        """
        if freq != "day":
            raise ValueError("DailyDataProcessor只支持日线数据")
            
        # 生成缓存键
        cache_key = self._generate_cache_key(symbols, start_date, end_date, fields, adjust)
        
        # 尝试从缓存获取
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logger.info(f"从缓存获取数据: {cache_key}")
            return cached_data
            
        # 从数据库加载
        logger.info(f"从数据库加载数据: {start_date} 到 {end_date}")
        data = super().get_data(symbols, start_date, end_date, fields, freq, adjust)
        
        # 存入缓存
        if data is not None and not data.empty:
            self.cache.set(cache_key, data)
            
        return data
    
    def _generate_cache_key(self, symbols, start_date, end_date, fields, adjust):
        """生成缓存键"""
        if isinstance(symbols, list):
            symbols_str = "_".join(sorted(symbols)) if len(symbols) <= 5 else f"{len(symbols)}stocks"
        else:
            symbols_str = symbols
            
        fields_str = "_".join(sorted(fields)) if fields else "all"
        adjust_str = "adj" if adjust else "raw"
        
        return f"daily_{symbols_str}_{start_date}_{end_date}_{fields_str}_{adjust_str}"
    
    def industry_neutralize(self, factor_data: pd.DataFrame,
                           date_col: str = "date",
                           symbol_col: str = "symbol",
                           factor_col: str = "factor",
                           industry_col: str = "industry") -> pd.DataFrame:
        """行业中性化处理
        
        对因子进行行业中性化处理，消除行业效应
        
        Args:
            factor_data: 因子数据
            date_col: 日期列名
            symbol_col: 股票代码列名
            factor_col: 因子值列名
            industry_col: 行业列名
            
        Returns:
            pd.DataFrame: 行业中性化后的因子数据
        """
        # 确保行业信息存在
        if industry_col not in factor_data.columns:
            if self.industry_info is None:
                raise ValueError("行业信息未初始化，无法进行行业中性化")
                
            # 合并行业信息
            factor_data = pd.merge(
                factor_data, 
                self.industry_info[[symbol_col, industry_col]],
                on=symbol_col,
                how="left"
            )
            
        result = factor_data.copy()
        
        # 按日期分组进行行业中性化处理
        for date, date_data in result.groupby(date_col):
            # 使用行业哑变量回归
            industry_dummies = pd.get_dummies(date_data[industry_col], drop_first=True)
            X = industry_dummies
            y = date_data[factor_col]
            
            try:
                # 使用最小二乘法回归
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X, y)
                
                # 计算残差(中性化后的因子)
                y_pred = model.predict(X)
                result.loc[date_data.index, f"{factor_col}_neutral"] = y - y_pred
            except:
                # 如果回归失败，使用简单的减均值方法
                result.loc[date_data.index, f"{factor_col}_neutral"] = date_data.groupby(industry_col)[factor_col].transform(lambda x: x - x.mean())
                
        return result
    
    def market_cap_neutralize(self, factor_data: pd.DataFrame,
                             date_col: str = "date",
                             symbol_col: str = "symbol",
                             factor_col: str = "factor",
                             market_cap_col: str = "market_cap") -> pd.DataFrame:
        """市值中性化处理
        
        对因子进行市值中性化处理，消除规模效应
        
        Args:
            factor_data: 因子数据
            date_col: 日期列名
            symbol_col: 股票代码列名
            factor_col: 因子值列名
            market_cap_col: 市值列名
            
        Returns:
            pd.DataFrame: 市值中性化后的因子数据
        """
        # 确保市值信息存在
        if market_cap_col not in factor_data.columns:
            raise ValueError(f"市值列 {market_cap_col} 不存在，无法进行市值中性化")
            
        result = factor_data.copy()
        
        # 对市值取对数
        result[f"log_{market_cap_col}"] = np.log(result[market_cap_col])
        
        # 按日期分组进行市值中性化处理
        for date, date_data in result.groupby(date_col):
            # 准备回归变量
            X = date_data[[f"log_{market_cap_col}"]]
            y = date_data[factor_col]
            
            try:
                # 使用最小二乘法回归
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X, y)
                
                # 计算残差(中性化后的因子)
                y_pred = model.predict(X)
                result.loc[date_data.index, f"{factor_col}_size_neutral"] = y - y_pred
            except:
                # 如果回归失败，使用简单方法
                result.loc[date_data.index, f"{factor_col}_size_neutral"] = y - y.mean()
                
        return result
    
    def neutralize(self, factor_data: pd.DataFrame,
                date_col: str = "date",
                symbol_col: str = "symbol",
                factor_col: str = "factor",
                industry_col: Optional[str] = None,
                market_cap_col: Optional[str] = None) -> pd.DataFrame:
        """因子中性化处理
        
        根据提供的参数，整合行业中性化和市值中性化功能
        
        Args:
            factor_data: 因子数据
            date_col: 日期列名
            symbol_col: 股票代码列名
            factor_col: 因子值列名
            industry_col: 行业列名，如果提供则进行行业中性化
            market_cap_col: 市值列名，如果提供则进行市值中性化
            
        Returns:
            pd.DataFrame: 中性化后的因子数据
        """
        result = factor_data.copy()
        
        # 如果提供了行业列，进行行业中性化
        if industry_col is not None:
            result = self.industry_neutralize(
                result, 
                date_col=date_col, 
                symbol_col=symbol_col, 
                factor_col=factor_col, 
                industry_col=industry_col
            )
            # 更新因子列名，使后续中性化可以基于行业中性化后的结果
            factor_col = f"{factor_col}_neutral"
            
        # 如果提供了市值列，进行市值中性化
        if market_cap_col is not None:
            result = self.market_cap_neutralize(
                result,
                date_col=date_col,
                symbol_col=symbol_col,
                factor_col=factor_col,  # 可能是原始因子列或行业中性化后的列
                market_cap_col=market_cap_col
            )
            
        return result
    
    def compute_factor(self, factor_name: str,
                      symbols: Union[str, List[str]],
                      start_date: str,
                      end_date: str,
                      params: Optional[Dict[str, Any]] = None,
                      freq: str = "day") -> pd.DataFrame:
        """计算日线因子
        
        扩展基础因子计算方法，添加更多日线特定的因子
        
        Args:
            factor_name: 因子名称
            symbols: 股票代码或代码列表
            start_date: 开始日期
            end_date: 结束日期
            params: 因子参数
            freq: 数据频率
            
        Returns:
            pd.DataFrame: 因子数据
        """
        params = params or {}
        
        # 日线特定因子计算
        if factor_name in ["momentum", "reversal", "turnover", "liquidity"]:
            # 获取需要的字段
            fields = ["open", "high", "low", "close", "volume", "amount"]
            data = self.get_data(symbols, start_date, end_date, fields, freq="day")
            
            if factor_name == "momentum":
                # 计算动量因子
                window = params.get("window", 20)
                data["momentum"] = data.groupby("symbol")["close"].pct_change(window)
                
            elif factor_name == "reversal":
                # 计算反转因子
                window = params.get("window", 5)
                data["reversal"] = -data.groupby("symbol")["close"].pct_change(window)
                
            elif factor_name == "turnover":
                # 计算换手率因子
                window = params.get("window", 20)
                data["turnover"] = data["volume"] / data.groupby("symbol")["volume"].rolling(window).mean().reset_index(0, drop=True)
                
            elif factor_name == "liquidity":
                # 计算流动性因子 (Amihud非流动性指标)
                window = params.get("window", 20)
                data["daily_return"] = data.groupby("symbol")["close"].pct_change()
                data["illiquidity"] = abs(data["daily_return"]) / data["amount"]
                data["liquidity"] = 1 / data.groupby("symbol")["illiquidity"].rolling(window).mean().reset_index(0, drop=True)
                data = data.replace([np.inf, -np.inf], np.nan)
                
            # 处理缺失值
            data = self.handle_missing_values(data)
            
            # 处理异常值
            data = self.handle_outliers(data)
            
            return data
        else:
            # 调用基类方法处理其他因子
            return super().compute_factor(factor_name, symbols, start_date, end_date, params, freq)