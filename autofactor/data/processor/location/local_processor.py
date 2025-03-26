"""
本地数据处理器 - 在Python环境中处理数据
"""
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from autofactor.data.processor.base import DataProcessor


class LocalDataProcessor(DataProcessor):
    """本地数据处理器，在Python环境中处理数据"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化本地数据处理器"""
        super().__init__(config)
        self.db_client = None  # 初始化时不依赖DolphinDB客户端
        
    def set_db_client(self, db_client):
        """设置数据库客户端，用于数据加载
        
        Args:
            db_client: DolphinDB客户端实例
        """
        self.db_client = db_client
        
    def get_data(self, symbols: Union[str, List[str]], 
                start_date: str, 
                end_date: str,
                fields: Optional[List[str]] = None,
                freq: str = "day",
                adjust: bool = True) -> pd.DataFrame:
        """从数据库获取数据，在本地处理
        
        Args:
            symbols: 股票代码或代码列表
            start_date: 开始日期，格式为 "YYYY-MM-DD"
            end_date: 结束日期，格式为 "YYYY-MM-DD"
            fields: 需要获取的字段列表，默认为所有字段
            freq: 数据频率，可选值为 "tick", "minute", "day"
            adjust: 是否进行复权处理
            
        Returns:
            pd.DataFrame: 股票数据
        """
        if self.db_client is None:
            raise ValueError("数据库客户端未设置，请先调用set_db_client方法")
            
        # 使用DolphinDB客户端获取数据
        if freq == "day":
            data = self.db_client.get_stock_data(
                symbols, start_date, end_date, fields, adjust
            )
        elif freq == "minute":
            # 分钟数据加载逻辑，可能需要分块处理
            # 这里是简化的示例
            data = self.db_client.get_stock_data(
                symbols, start_date, end_date, fields, adjust, table_name="minute_quote"
            )
        else:
            raise ValueError(f"不支持的数据频率: {freq}")
            
        return data
    
    def resample(self, data: pd.DataFrame, 
                target_freq: str = "day", 
                method: str = "ohlc") -> pd.DataFrame:
        """将数据重采样到指定频率
        
        Args:
            data: 输入数据，必须包含datetime类型的索引或'date'列
            target_freq: 目标频率，例如'D'(日),'W'(周),'M'(月)
            method: 重采样方法，'ohlc'(开高低收),'vwap'(成交量加权),'last'(最后值)
            
        Returns:
            pd.DataFrame: 重采样后的数据
        """
        # 确保数据有日期索引
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data = data.set_index('date')
            else:
                raise ValueError("数据必须包含datetime类型的索引或'date'列")
                
        # 根据不同方法进行重采样
        if method == "ohlc":
            # 针对OHLC数据的特殊处理
            resampled = pd.DataFrame()
            if 'open' in data.columns:
                resampled['open'] = data['open'].resample(target_freq).first()
            if 'high' in data.columns:
                resampled['high'] = data['high'].resample(target_freq).max()
            if 'low' in data.columns:
                resampled['low'] = data['low'].resample(target_freq).min()
            if 'close' in data.columns:
                resampled['close'] = data['close'].resample(target_freq).last()
            if 'volume' in data.columns:
                resampled['volume'] = data['volume'].resample(target_freq).sum()
            if 'amount' in data.columns:
                resampled['amount'] = data['amount'].resample(target_freq).sum()
                
        elif method == "vwap":
            # 成交量加权平均价格
            if 'volume' not in data.columns or 'close' not in data.columns:
                raise ValueError("VWAP计算需要'volume'和'close'列")
                
            # 计算每个区间的成交量加权价格
            grouped = data.resample(target_freq)
            resampled = pd.DataFrame()
            resampled['vwap'] = (grouped['close'] * grouped['volume']).sum() / grouped['volume'].sum()
            
        elif method == "last":
            # 使用最后一个值
            resampled = data.resample(target_freq).last()
            
        else:
            raise ValueError(f"不支持的重采样方法: {method}")
            
        return resampled
    
    def handle_missing_values(self, data: pd.DataFrame, 
                             method: str = "ffill",
                             limit: Optional[int] = None) -> pd.DataFrame:
        """处理缺失值
        
        Args:
            data: 输入数据
            method: 处理方法，可选值为 "ffill", "bfill", "zero", "mean", "median", "drop"
            limit: 最大填充长度
            
        Returns:
            pd.DataFrame: 处理缺失值后的数据
        """
        result = data.copy()
        
        if method == "ffill":
            result = result.ffill(limit=limit)
        elif method == "bfill":
            result = result.bfill(limit=limit)
        elif method == "zero":
            result = result.fillna(0)
        elif method == "mean":
            result = result.fillna(result.mean())
        elif method == "median":
            result = result.fillna(result.median())
        elif method == "drop":
            result = result.dropna()
        else:
            raise ValueError(f"不支持的缺失值处理方法: {method}")
            
        return result
    
    def handle_outliers(self, data: pd.DataFrame,
                       method: str = "winsorize",
                       limits: Tuple[float, float] = (0.01, 0.99),
                       by_group: bool = False,
                       group_col: Optional[str] = None) -> pd.DataFrame:
        """处理异常值
        
        Args:
            data: 输入数据
            method: 处理方法，可选值为 "winsorize", "clip", "z_score", "mad"
            limits: 限制范围，用于winsorize和clip方法
            by_group: 是否按组处理
            group_col: 分组列名，当by_group为True时必须提供
            
        Returns:
            pd.DataFrame: 处理异常值后的数据
        """
        result = data.copy()
        
        # 确定要处理的列
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        
        if by_group and group_col is None:
            raise ValueError("当by_group为True时，必须提供group_col")
            
        if method == "winsorize":
            if by_group:
                for group_name, group_data in result.groupby(group_col):
                    for col in numeric_cols:
                        if col in group_data.columns:
                            lower = group_data[col].quantile(limits[0])
                            upper = group_data[col].quantile(limits[1])
                            result.loc[group_data.index, col] = result.loc[group_data.index, col].clip(lower, upper)
            else:
                for col in numeric_cols:
                    if col in result.columns:
                        lower = result[col].quantile(limits[0])
                        upper = result[col].quantile(limits[1])
                        result[col] = result[col].clip(lower, upper)
                        
        elif method == "clip":
            if by_group:
                for group_name, group_data in result.groupby(group_col):
                    for col in numeric_cols:
                        if col in group_data.columns:
                            lower = group_data[col].quantile(limits[0])
                            upper = group_data[col].quantile(limits[1])
                            result.loc[group_data.index, col] = result.loc[group_data.index, col].clip(lower, upper)
            else:
                for col in numeric_cols:
                    if col in result.columns:
                        lower = result[col].quantile(limits[0])
                        upper = result[col].quantile(limits[1])
                        result[col] = result[col].clip(lower, upper)
                        
        elif method == "z_score":
            # 使用Z分数识别异常值，通常|z|>3视为异常
            z_threshold = 3
            if by_group:
                for group_name, group_data in result.groupby(group_col):
                    for col in numeric_cols:
                        if col in group_data.columns:
                            z_scores = np.abs(stats.zscore(group_data[col], nan_policy='omit'))
                            mask = z_scores > z_threshold
                            result.loc[group_data.index[mask], col] = np.nan
            else:
                for col in numeric_cols:
                    if col in result.columns:
                        z_scores = np.abs(stats.zscore(result[col], nan_policy='omit'))
                        mask = z_scores > z_threshold
                        result.loc[result.index[mask], col] = np.nan
                        
        elif method == "mad":
            # 使用中位数绝对偏差识别异常值
            mad_threshold = 3
            if by_group:
                for group_name, group_data in result.groupby(group_col):
                    for col in numeric_cols:
                        if col in group_data.columns:
                            median = group_data[col].median()
                            mad = np.median(np.abs(group_data[col] - median))
                            mask = np.abs(group_data[col] - median) > mad_threshold * mad
                            result.loc[group_data.index[mask], col] = np.nan
            else:
                for col in numeric_cols:
                    if col in result.columns:
                        median = result[col].median()
                        mad = np.median(np.abs(result[col] - median))
                        mask = np.abs(result[col] - median) > mad_threshold * mad
                        result.loc[result.index[mask], col] = np.nan
        else:
            raise ValueError(f"不支持的异常值处理方法: {method}")
            
        return result
    
    def normalize(self, data: pd.DataFrame, 
                 method: str = "zscore",
                 by_cross_section: bool = True,
                 date_col: str = "date",
                 symbol_col: str = "symbol") -> pd.DataFrame:
        """标准化处理
        
        Args:
            data: 输入数据
            method: 标准化方法，可选值为 "zscore", "rank", "min_max", "robust"
            by_cross_section: 是否按横截面处理
            date_col: 日期列名，当by_cross_section为True时使用
            symbol_col: 股票代码列名，用于区分不同股票
            
        Returns:
            pd.DataFrame: 标准化后的数据
        """
        result = data.copy()
        
        # 确定要处理的列
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        # 排除日期和股票代码列
        if date_col in numeric_cols:
            numeric_cols.remove(date_col)
        if symbol_col in numeric_cols:
            numeric_cols.remove(symbol_col)
            
        if by_cross_section:
            # 按横截面处理
            if date_col not in result.columns:
                raise ValueError(f"按横截面处理需要{date_col}列")
                
            for date, date_data in result.groupby(date_col):
                if method == "zscore":
                    for col in numeric_cols:
                        if col in date_data.columns:
                            mean = date_data[col].mean()
                            std = date_data[col].std()
                            if std != 0:
                                # 显式地将整个列转换为 float 类型
                                result[col] = result[col].astype(float)
                                # 然后进行 Z-score 计算和赋值
                                result.loc[date_data.index, col] = (date_data[col] - mean) / std
                                
                elif method == "rank":
                    for col in numeric_cols:
                        if col in date_data.columns:
                            result.loc[date_data.index, col] = date_data[col].rank(pct=True)
                            
                elif method == "min_max":
                    for col in numeric_cols:
                        if col in date_data.columns:
                            min_val = date_data[col].min()
                            max_val = date_data[col].max()
                            if max_val > min_val:
                                result.loc[date_data.index, col] = (date_data[col] - min_val) / (max_val - min_val)
                                
                elif method == "robust":
                    for col in numeric_cols:
                        if col in date_data.columns:
                            median = date_data[col].median()
                            iqr = date_data[col].quantile(0.75) - date_data[col].quantile(0.25)
                            if iqr != 0:
                                result.loc[date_data.index, col] = (date_data[col] - median) / iqr
                                
                else:
                    raise ValueError(f"不支持的标准化方法: {method}")
        else:
            # 按时间序列处理
            if symbol_col not in result.columns:
                # 如果没有股票代码列，则视为单一时间序列
                if method == "zscore":
                    for col in numeric_cols:
                        if col in result.columns:
                            mean = result[col].mean()
                            std = result[col].std()
                            if std != 0:
                                result[col] = (result[col] - mean) / std
                                
                elif method == "rank":
                    for col in numeric_cols:
                        if col in result.columns:
                            result[col] = result[col].rank(pct=True)
                            
                elif method == "min_max":
                    for col in numeric_cols:
                        if col in result.columns:
                            min_val = result[col].min()
                            max_val = result[col].max()
                            if max_val > min_val:
                                result[col] = (result[col] - min_val) / (max_val - min_val)
                                
                elif method == "robust":
                    for col in numeric_cols:
                        if col in result.columns:
                            median = result[col].median()
                            iqr = result[col].quantile(0.75) - result[col].quantile(0.25)
                            if iqr != 0:
                                result[col] = (result[col] - median) / iqr
                                
                else:
                    raise ValueError(f"不支持的标准化方法: {method}")
            else:
                # 按股票分组处理
                for symbol, symbol_data in result.groupby(symbol_col):
                    if method == "zscore":
                        for col in numeric_cols:
                            if col in symbol_data.columns:
                                mean = symbol_data[col].mean()
                                std = symbol_data[col].std()
                                if std != 0:
                                    result.loc[symbol_data.index, col] = (symbol_data[col] - mean) / std
                                    
                    elif method == "rank":
                        for col in numeric_cols:
                            if col in symbol_data.columns:
                                result.loc[symbol_data.index, col] = symbol_data[col].rank(pct=True)
                                
                    elif method == "min_max":
                        for col in numeric_cols:
                            if col in symbol_data.columns:
                                min_val = symbol_data[col].min()
                                max_val = symbol_data[col].max()
                                if max_val > min_val:
                                    result.loc[symbol_data.index, col] = (symbol_data[col] - min_val) / (max_val - min_val)
                                    
                    elif method == "robust":
                        for col in numeric_cols:
                            if col in symbol_data.columns:
                                median = symbol_data[col].median()
                                iqr = symbol_data[col].quantile(0.75) - symbol_data[col].quantile(0.25)
                                if iqr != 0:
                                    result.loc[symbol_data.index, col] = (symbol_data[col] - median) / iqr
                                    
                    else:
                        raise ValueError(f"不支持的标准化方法: {method}")
                        
        return result
    
    def industry_neutralize(self, factor_data: pd.DataFrame,
                          date_col: str = "date",
                          symbol_col: str = "symbol",
                          factor_col: str = "factor",
                          industry_col: str = "industry") -> pd.DataFrame:
        """行业中性化处理
        
        Args:
            factor_data: 因子数据
            date_col: 日期列名
            symbol_col: 股票代码列名
            factor_col: 因子值列名
            industry_col: 行业列名
            
        Returns:
            pd.DataFrame: 行业中性化后的因子数据
        """
        if industry_col not in factor_data.columns:
            raise ValueError(f"行业列 {industry_col} 不存在，无法进行行业中性化")
            
        result = factor_data.copy()
        
        # 按日期分组进行行业中性化处理
        for date, date_data in result.groupby(date_col):
            # 准备回归变量（行业哑变量）
            X = pd.get_dummies(date_data[industry_col], drop_first=True)
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
                result.loc[date_data.index, f"{factor_col}_neutral"] = date_data.groupby(
                    industry_col
                )[factor_col].transform(lambda x: x - x.mean())
                
        return result
    
    def market_cap_neutralize(self, factor_data: pd.DataFrame,
                            date_col: str = "date",
                            symbol_col: str = "symbol",
                            factor_col: str = "factor",
                            market_cap_col: str = "market_cap") -> pd.DataFrame:
        """市值中性化处理
        
        Args:
            factor_data: 因子数据
            date_col: 日期列名
            symbol_col: 股票代码列名
            factor_col: 因子值列名
            market_cap_col: 市值列名
            
        Returns:
            pd.DataFrame: 市值中性化后的因子数据
        """
        if market_cap_col not in factor_data.columns:
            raise ValueError(f"市值列 {market_cap_col} 不存在，无法进行市值中性化")
            
        result = factor_data.copy()
        
        # 对市值取对数
        result[f"log_{market_cap_col}"] = np.log(result[market_cap_col])
        
        # 按日期分组进行市值中性化处理
        for date, date_data in result.groupby(date_col):
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
        """整合中性化处理
        
        Args:
            factor_data: 因子数据
            date_col: 日期列名
            symbol_col: 股票代码列名
            factor_col: 因子值列名
            industry_col: 行业列名（可选）
            market_cap_col: 市值列名（可选）
            
        Returns:
            pd.DataFrame: 中性化后的因子数据
        """
        result = factor_data.copy()
        
        # 行业中性化
        if industry_col is not None:
            result = self.industry_neutralize(
                result,
                date_col=date_col,
                symbol_col=symbol_col,
                factor_col=factor_col,
                industry_col=industry_col
            )
            factor_col = f"{factor_col}_neutral"  # 更新列名
            
        # 市值中性化
        if market_cap_col is not None:
            result = self.market_cap_neutralize(
                result,
                date_col=date_col,
                symbol_col=symbol_col,
                factor_col=factor_col,
                market_cap_col=market_cap_col
            )
            
        return result
    
    def compute_factor(self, factor_name: str,
                      symbols: Union[str, List[str]],
                      start_date: str,
                      end_date: str,
                      params: Optional[Dict[str, Any]] = None,
                      freq: str = "day") -> pd.DataFrame:
        """计算指定因子
        
        根据因子名称和参数，计算并返回因子值。这是一个示例实现，
        实际使用时需要根据具体因子扩展此方法。
        
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
        
        # 获取基础数据
        fields = params.get("fields", None)
        data = self.get_data(symbols, start_date, end_date, fields, freq)
        
        # 处理缺失值
        data = self.handle_missing_values(data, method=params.get("na_method", "ffill"))
        
        # 简单示例：计算几个基本因子
        if factor_name == "return":
            # 计算收益率
            window = params.get("window", 1)
            if "close" in data.columns:
                data["return"] = data.groupby("symbol")["close"].pct_change(window)
                
        elif factor_name == "ma":
            # 计算移动平均
            window = params.get("window", 5)
            if "close" in data.columns:
                data["ma"] = data.groupby("symbol")["close"].rolling(window).mean().reset_index(0, drop=True)
                
        elif factor_name == "volatility":
            # 计算波动率
            window = params.get("window", 20)
            if "return" not in data.columns and "close" in data.columns:
                data["return"] = data.groupby("symbol")["close"].pct_change()
            data["volatility"] = data.groupby("symbol")["return"].rolling(window).std().reset_index(0, drop=True)
            
        elif factor_name == "rsi":
            # 计算相对强弱指标(RSI)
            window = params.get("window", 14)
            if "close" in data.columns:
                delta = data.groupby("symbol")["close"].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.groupby("symbol").rolling(window).mean().reset_index(0, drop=True)
                avg_loss = loss.groupby("symbol").rolling(window).mean().reset_index(0, drop=True)
                rs = avg_gain / avg_loss
                data["rsi"] = 100 - (100 / (1 + rs))
                
        else:
            raise ValueError(f"不支持的因子: {factor_name}")
            
        # 标准化处理
        if params.get("normalize", False):
            normalize_cols = [col for col in data.columns if col not in ["date", "symbol"]]
            data[normalize_cols] = self.normalize(
                data, 
                method=params.get("normalize_method", "zscore"),
                by_cross_section=params.get("by_cross_section", True)
            )
            
        return data