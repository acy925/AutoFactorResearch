"""
多频处理器 - 处理混合频率的数据，支持高低频数据合并和对齐
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from autofactor.data.processor.base import DataProcessor
from autofactor.data.processor.frequency.daily_processor import DailyDataProcessor
from autofactor.data.processor.frequency.minute_processor import MinuteDataProcessor
from autofactor.data.utils.cache import CacheManager


class MultiFreqProcessor(DataProcessor):
    """多频数据处理器，处理混合频率的数据，支持高低频数据合并和对齐"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化多频数据处理器
        
        Args:
            config: 配置参数，包含各类处理器的配置
        """
        super().__init__(config)
        
        # 初始化日线和分钟处理器
        self.daily_processor = DailyDataProcessor(config)
        self.minute_processor = MinuteDataProcessor(config)
        
        # 初始化缓存
        self.cache = CacheManager(config)
        
        # 频率映射表
        self.freq_map = {
            # 日频相关
            "day": self.daily_processor,
            "D": self.daily_processor,
            "daily": self.daily_processor,
            "1d": self.daily_processor,
            
            # 分钟频相关
            "minute": self.minute_processor,
            "min": self.minute_processor,
            "1min": self.minute_processor,
            "5min": self.minute_processor,
            "15min": self.minute_processor,
            "30min": self.minute_processor,
            "60min": self.minute_processor,
            "1m": self.minute_processor,
            "5m": self.minute_processor,
            "15m": self.minute_processor,
            "30m": self.minute_processor,
            "60m": self.minute_processor,
        }
        
        # 设置DB客户端
        self._connect_db_client()
        
    def _connect_db_client(self):
        """连接数据库客户端"""
        # 检查是否已存在DB客户端
        if hasattr(self.daily_processor, 'db_client') and self.daily_processor.db_client is not None:
            db_client = self.daily_processor.db_client
        else:
            # 尝试创建新的DB客户端
            try:
                from autofactor.data.dolphindb_client import DolphinDBClient
                db_client = DolphinDBClient()
            except Exception as e:
                logger.warning(f"创建DolphinDB客户端失败: {e}")
                db_client = None
                
        # 设置各处理器的DB客户端
        if db_client is not None:
            self.daily_processor.set_db_client(db_client)
            self.minute_processor.set_db_client(db_client)
            
    def get_data(self, symbols: Union[str, List[str]], 
                start_date: str, 
                end_date: str,
                fields: Optional[List[str]] = None,
                freq: str = "day",
                adjust: bool = True) -> pd.DataFrame:
        """获取特定频率的数据
        
        根据freq参数选择合适的处理器获取数据
        
        Args:
            symbols: 股票代码或代码列表
            start_date: 开始日期，格式为 "YYYY-MM-DD"
            end_date: 结束日期，格式为 "YYYY-MM-DD"
            fields: 需要获取的字段列表，默认为所有字段
            freq: 数据频率，如 "day", "1min", "5min" 等
            adjust: 是否进行复权处理
            
        Returns:
            pd.DataFrame: 股票数据
        """
        # 检查频率是否支持
        if freq not in self.freq_map:
            raise ValueError(f"不支持的数据频率: {freq}")
            
        # 选择合适的处理器
        processor = self.freq_map[freq]
        
        # 使用选定的处理器获取数据
        return processor.get_data(symbols, start_date, end_date, fields, freq, adjust)
    
    def get_multi_freq_data(self, symbols: Union[str, List[str]],
                           start_date: str,
                           end_date: str,
                           freqs: List[str],
                           fields: Optional[Dict[str, List[str]]] = None,
                           adjust: bool = True) -> Dict[str, pd.DataFrame]:
        """获取多个频率的数据
        
        Args:
            symbols: 股票代码或代码列表
            start_date: 开始日期
            end_date: 结束日期
            freqs: 频率列表，如 ["day", "5min"]
            fields: 各频率需要的字段字典，例如 {"day": ["open", "close"], "5min": ["open", "high", "low", "close"]}
            adjust: 是否进行复权处理
            
        Returns:
            Dict[str, pd.DataFrame]: 频率到数据的映射字典
        """
        results = {}
        
        # 为每个频率获取数据
        for freq in freqs:
            # 获取该频率的字段
            freq_fields = fields.get(freq) if fields else None
            
            # 获取数据
            data = self.get_data(symbols, start_date, end_date, freq_fields, freq, adjust)
            
            # 存储结果
            results[freq] = data
            
        return results
    
    def align_multi_freq_data(self, data_dict: Dict[str, pd.DataFrame],
                            base_freq: str,
                            method: str = "ffill") -> Dict[str, pd.DataFrame]:
        """对齐多频率数据的时间戳
        
        将不同频率的数据对齐到基准频率的时间戳
        
        Args:
            data_dict: 频率到数据的映射字典
            base_freq: 基准频率
            method: 对齐方法，可选值为 "ffill", "bfill", "nearest"
            
        Returns:
            Dict[str, pd.DataFrame]: 对齐后的数据字典
        """
        if base_freq not in data_dict:
            raise ValueError(f"基准频率 {base_freq} 不在数据字典中")
            
        # 获取基准频率的时间戳
        base_data = data_dict[base_freq]
        
        # 创建完整的datetime索引
        if "time" in base_data.columns:
            # 分钟级数据
            base_data["datetime"] = pd.to_datetime(
                base_data["date"].astype(str) + " " + base_data["time"].astype(str)
            )
            base_timestamps = base_data["datetime"].unique()
        else:
            # 日频数据
            base_timestamps = pd.to_datetime(base_data["date"].unique())
            
        # 对齐每个频率的数据
        aligned_dict = {}
        aligned_dict[base_freq] = base_data  # 基准频率数据不需要对齐
        
        for freq, data in data_dict.items():
            if freq == base_freq:
                continue
                
            # 创建该频率的datetime索引
            if "time" in data.columns:
                # 分钟级数据
                data["datetime"] = pd.to_datetime(
                    data["date"].astype(str) + " " + data["time"].astype(str)
                )
            else:
                # 日频数据
                data["datetime"] = pd.to_datetime(data["date"])
                
            # 按股票分组对齐
            aligned_data_list = []
            
            for symbol, group in data.groupby("symbol"):
                # 设置datetime为索引
                group = group.set_index("datetime")
                
                # 对于每个基准时间戳，找到最近的数据点
                aligned_group = pd.DataFrame(index=base_timestamps)
                
                # 合并数据
                merged = aligned_group.merge(
                    group, left_index=True, right_index=True, how="left"
                )
                
                # 填充缺失值
                if method == "ffill":
                    merged = merged.ffill()
                elif method == "bfill":
                    merged = merged.bfill()
                elif method == "nearest":
                    # 使用前后两个时间点中较近的一个
                    merged = merged.fillna(method="ffill").combine_first(
                        merged.fillna(method="bfill")
                    )
                    
                # 添加symbol列
                merged["symbol"] = symbol
                
                # 将datetime移回列
                merged = merged.reset_index()
                merged = merged.rename(columns={"index": "datetime"})
                
                # 拆分datetime为date和time
                if freq.startswith(("1min", "5min", "15min", "30min", "60min")):
                    merged["date"] = merged["datetime"].dt.date
                    merged["time"] = merged["datetime"].dt.time
                else:
                    merged["date"] = merged["datetime"].dt.date
                    if "time" in merged.columns:
                        merged = merged.drop("time", axis=1)
                        
                aligned_data_list.append(merged)
                
            # 合并所有股票的对齐数据
            if aligned_data_list:
                aligned_dict[freq] = pd.concat(aligned_data_list, ignore_index=True)
                
        return aligned_dict
    
    def merge_multi_freq_data(self, data_dict: Dict[str, pd.DataFrame],
                             freq_suffixes: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """合并多频率数据
        
        将不同频率的数据合并为一个DataFrame
        
        Args:
            data_dict: 频率到数据的映射字典
            freq_suffixes: 各频率的列名后缀，例如 {"day": "_D", "5min": "_5m"}
            
        Returns:
            pd.DataFrame: 合并后的数据
        """
        if not data_dict:
            return pd.DataFrame()
            
        # 默认后缀
        if freq_suffixes is None:
            freq_suffixes = {freq: f"_{freq}" for freq in data_dict.keys()}
            
        # 确保所有数据包含datetime列
        processed_dict = {}
        
        for freq, data in data_dict.items():
            # 复制数据
            data_copy = data.copy()
            
            # 创建datetime列
            if "datetime" not in data_copy.columns:
                if "time" in data_copy.columns:
                    # 分钟级数据
                    data_copy["datetime"] = pd.to_datetime(
                        data_copy["date"].astype(str) + " " + data_copy["time"].astype(str)
                    )
                else:
                    # 日频数据
                    data_copy["datetime"] = pd.to_datetime(data_copy["date"])
                    
            # 添加后缀到列名
            suffix = freq_suffixes.get(freq, f"_{freq}")
            renamed_cols = {}
            
            for col in data_copy.columns:
                if col not in ["symbol", "datetime", "date", "time"]:
                    renamed_cols[col] = f"{col}{suffix}"
                    
            data_copy = data_copy.rename(columns=renamed_cols)
            
            processed_dict[freq] = data_copy
            
        # 选择第一个频率作为基础
        base_freq = list(processed_dict.keys())[0]
        merged = processed_dict[base_freq]
        
        # 合并其他频率
        for freq, data in processed_dict.items():
            if freq == base_freq:
                continue
                
            # 按symbol和datetime合并
            merged = pd.merge(
                merged, 
                data,
                on=["symbol", "datetime"],
                how="outer",
                suffixes=("", "_drop")
            )
            
            # 删除重复列
            drop_cols = [col for col in merged.columns if col.endswith("_drop")]
            merged = merged.drop(columns=drop_cols)
            
        # 排序
        merged = merged.sort_values(["symbol", "datetime"])
        
        return merged
    
    def compute_multi_freq_factor(self, factor_name: str,
                                symbols: Union[str, List[str]],
                                start_date: str,
                                end_date: str,
                                params: Optional[Dict[str, Any]] = None,
                                freq: str = "day") -> pd.DataFrame:
        """计算多频因子
        
        利用多频数据计算复合因子
        
        Args:
            factor_name: 因子名称
            symbols: 股票代码或代码列表
            start_date: 开始日期
            end_date: 结束日期
            params: 因子参数
            freq: 结果因子频率
            
        Returns:
            pd.DataFrame: 因子数据
        """
        params = params or {}
        
        # 获取需要的频率数据
        freqs = params.get("freqs", ["day", "5min"])
        fields = params.get("fields", {
            "day": ["open", "high", "low", "close", "volume", "amount"],
            "5min": ["open", "high", "low", "close", "volume", "amount"]
        })
        
        # 生成缓存键
        if isinstance(symbols, list):
            symbols_str = "_".join(sorted(symbols)) if len(symbols) <= 5 else f"{len(symbols)}stocks"
        else:
            symbols_str = symbols
            
        freqs_str = "_".join(sorted(freqs))
        params_str = ""
        if params:
            params_items = sorted(params.items())
            params_str = "_".join(f"{k}_{v}" for k, v in params_items if k not in ["freqs", "fields"])
            
        cache_key = f"multi_freq_factor_{factor_name}_{symbols_str}_{start_date}_{end_date}_{freqs_str}_{params_str}"
        
        # 尝试从缓存获取
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logger.debug(f"从缓存获取多频因子: {cache_key}")
            return cached_data
            
        # 获取多频数据
        data_dict = self.get_multi_freq_data(symbols, start_date, end_date, freqs, fields)
        
        # 对齐数据
        aligned_dict = self.align_multi_freq_data(data_dict, base_freq=freq)
        
        # 合并数据
        merged_data = self.merge_multi_freq_data(aligned_dict)
        
        # 根据不同因子类型，计算多频因子
        if factor_name == "volatility_ratio":
            # 波动率比率因子：高频波动率与低频波动率的比值
            high_freq = params.get("high_freq", "5min")
            low_freq = params.get("low_freq", "day")
            window = params.get("window", 20)
            
            # 计算各频率收益率
            high_suffix = f"_{high_freq}"
            low_suffix = f"_{low_freq}"
            
            # 计算高频收益率
            merged_data[f"return{high_suffix}"] = merged_data.groupby("symbol")[f"close{high_suffix}"].pct_change()
            
            # 计算低频收益率
            merged_data[f"return{low_suffix}"] = merged_data.groupby("symbol")[f"close{low_suffix}"].pct_change()
            
            # 计算波动率
            high_vol = merged_data.groupby("symbol")[f"return{high_suffix}"].rolling(window).std().reset_index(0, drop=True)
            low_vol = merged_data.groupby("symbol")[f"return{low_suffix}"].rolling(window).std().reset_index(0, drop=True)
            
            # 计算波动率比率
            merged_data[factor_name] = high_vol / low_vol
            
        elif factor_name == "momentum_divergence":
            # 动量分歧因子：高频动量与低频动量的背离
            high_freq = params.get("high_freq", "5min")
            low_freq = params.get("low_freq", "day")
            window_high = params.get("window_high", 12)
            window_low = params.get("window_low", 5)
            
            # 计算各频率动量
            high_suffix = f"_{high_freq}"
            low_suffix = f"_{low_freq}"
            
            # 高频动量
            merged_data[f"momentum{high_suffix}"] = merged_data.groupby("symbol")[f"close{high_suffix}"].pct_change(window_high)
            
            # 低频动量
            merged_data[f"momentum{low_suffix}"] = merged_data.groupby("symbol")[f"close{low_suffix}"].pct_change(window_low)
            
            # 计算动量分歧
            merged_data[factor_name] = merged_data[f"momentum{high_suffix}"] - merged_data[f"momentum{low_suffix}"]
            
        elif factor_name == "volume_price_divergence":
            # 成交量价格分歧因子：高频成交量变化与价格变化的背离
            high_freq = params.get("high_freq", "5min")
            window = params.get("window", 10)
            
            high_suffix = f"_{high_freq}"
            
            # 计算价格变化
            merged_data[f"price_change{high_suffix}"] = merged_data.groupby("symbol")[f"close{high_suffix}"].pct_change(window)
            
            # 计算成交量变化
            merged_data[f"volume_change{high_suffix}"] = merged_data.groupby("symbol")[f"volume{high_suffix}"].pct_change(window)
            
            # 计算Z分数
            price_z = (merged_data[f"price_change{high_suffix}"] - merged_data.groupby("symbol")[f"price_change{high_suffix}"].rolling(window).mean().reset_index(0, drop=True)) / merged_data.groupby("symbol")[f"price_change{high_suffix}"].rolling(window).std().reset_index(0, drop=True)
            
            volume_z = (merged_data[f"volume_change{high_suffix}"] - merged_data.groupby("symbol")[f"volume_change{high_suffix}"].rolling(window).mean().reset_index(0, drop=True)) / merged_data.groupby("symbol")[f"volume_change{high_suffix}"].rolling(window).std().reset_index(0, drop=True)
            
            # 计算分歧
            merged_data[factor_name] = price_z - volume_z
            
        else:
            raise ValueError(f"不支持的多频因子: {factor_name}")
            
        # 处理缺失值和异常值
        result = merged_data.copy()
        result[factor_name] = result[factor_name].replace([np.inf, -np.inf], np.nan)
        
        # 选择需要的列
        result = result[["symbol", "datetime", "date", factor_name]]
        
        # 添加时间列（如果是分钟频率）
        if freq.endswith(("min", "m")):
            result["time"] = result["datetime"].dt.time
            
        # 按照时间排序
        result = result.sort_values(["symbol", "datetime"])
        
        # 缓存结果
        self.cache.set(cache_key, result)
        
        return result
    
    def resample(self, data: pd.DataFrame, 
                target_freq: str = "day", 
                method: str = "ohlc") -> pd.DataFrame:
        """将数据重采样到指定频率
        
        支持各种频率之间的转换，例如分钟到小时、分钟到日等
        
        Args:
            data: 输入数据
            target_freq: 目标频率
            method: 重采样方法
            
        Returns:
            pd.DataFrame: 重采样后的数据
        """
        # 检查数据是否包含必要列
        required_cols = ["symbol", "date"]
        if target_freq.endswith(("min", "m", "h", "H")):
            # 如果目标是分钟或小时频率，需要时间列
            if "time" not in data.columns and "datetime" not in data.columns:
                raise ValueError("重采样到分钟/小时频率需要'time'列或'datetime'列")
                
        # 确保datetime列存在
        if "datetime" not in data.columns:
            if "time" in data.columns:
                # 创建datetime列
                data["datetime"] = pd.to_datetime(
                    data["date"].astype(str) + " " + data["time"].astype(str)
                )
            else:
                # 只有日期，创建日期的datetime
                data["datetime"] = pd.to_datetime(data["date"])
                
        # 确定pandas重采样频率
        freq_map = {
            "1min": "1T",
            "5min": "5T",
            "15min": "15T",
            "30min": "30T",
            "60min": "60T",
            "hour": "1H",
            "1h": "1H",
            "day": "1D",
            "1d": "1D",
            "week": "1W",
            "1w": "1W",
            "month": "1M",
            "1m": "1M"
        }
        
        pandas_freq = freq_map.get(target_freq)
        if not pandas_freq:
            raise ValueError(f"不支持的目标频率: {target_freq}")
            
        # 按股票分组重采样
        result_chunks = []
        
        for symbol, group in data.groupby("symbol"):
            # 设置datetime为索引
            group = group.set_index("datetime")
            
            # 根据不同方法进行重采样
            if method == "ohlc":
                # OHLC重采样
                resampled = pd.DataFrame()
                
                if "open" in group.columns:
                    resampled["open"] = group["open"].resample(pandas_freq).first()
                if "high" in group.columns:
                    resampled["high"] = group["high"].resample(pandas_freq).max()
                if "low" in group.columns:
                    resampled["low"] = group["low"].resample(pandas_freq).min()
                if "close" in group.columns:
                    resampled["close"] = group["close"].resample(pandas_freq).last()
                if "volume" in group.columns:
                    resampled["volume"] = group["volume"].resample(pandas_freq).sum()
                if "amount" in group.columns:
                    resampled["amount"] = group["amount"].resample(pandas_freq).sum()
                    
                # 添加其他非OHLCV列（如因子列），使用last方法
                for col in group.columns:
                    if col not in ["open", "high", "low", "close", "volume", "amount", "symbol", "date", "time"]:
                        resampled[col] = group[col].resample(pandas_freq).last()
                
            elif method == "last":
                # 使用最后一个值
                resampled = group.resample(pandas_freq).last()
                
                # 特殊处理成交量和成交额
                if "volume" in group.columns:
                    resampled["volume"] = group["volume"].resample(pandas_freq).sum()
                if "amount" in group.columns:
                    resampled["amount"] = group["amount"].resample(pandas_freq).sum()
                    
            elif method == "mean":
                # 使用平均值
                resampled = group.resample(pandas_freq).mean()
                
                # 特殊处理成交量和成交额
                if "volume" in group.columns:
                    resampled["volume"] = group["volume"].resample(pandas_freq).sum()
                if "amount" in group.columns:
                    resampled["amount"] = group["amount"].resample(pandas_freq).sum()
                    
            else:
                raise ValueError(f"不支持的重采样方法: {method}")
                
            # 重置索引
            resampled = resampled.reset_index()
            
            # 添加symbol列
            if "symbol" not in resampled.columns:
                resampled["symbol"] = symbol
                
            # 处理日期和时间列
            resampled["date"] = resampled["datetime"].dt.date
            
            if target_freq.endswith(("min", "m", "h", "H")):
                # 保留时间列
                resampled["time"] = resampled["datetime"].dt.time
                
            result_chunks.append(resampled)
            
        # 合并结果
        if result_chunks:
            result = pd.concat(result_chunks, ignore_index=True)
            return result
        else:
            return pd.DataFrame()
    
    def handle_missing_values(self, data: pd.DataFrame, 
                             method: str = "ffill",
                             limit: Optional[int] = None) -> pd.DataFrame:
        """处理缺失值
        
        调用默认处理器的方法
        
        Args:
            data: 输入数据
            method: 处理方法
            limit: 最大填充长度
            
        Returns:
            pd.DataFrame: 处理缺失值后的数据
        """
        return self.daily_processor.handle_missing_values(data, method, limit)
    
    def handle_outliers(self, data: pd.DataFrame,
                       method: str = "winsorize") -> pd.DataFrame:
        """处理异常值
        
        调用默认处理器的方法
        
        Args:
            data: 输入数据
            method: 处理方法
            
        Returns:
            pd.DataFrame: 处理异常值后的数据
        """
        return self.daily_processor.handle_outliers(data, method)
    
    def normalize(self, data: pd.DataFrame, 
                 method: str = "zscore",
                 by_cross_section: bool = True,
                 date_col: str = "date",
                 symbol_col: str = "symbol") -> pd.DataFrame:
        """标准化处理
        
        调用默认处理器的方法
        
        Args:
            data: 输入数据
            method: 标准化方法
            by_cross_section: 是否按横截面处理
            date_col: 日期列名
            symbol_col: 股票代码列名
            
        Returns:
            pd.DataFrame: 标准化后的数据
        """
        return self.daily_processor.normalize(data, method, by_cross_section, date_col, symbol_col)
    
    def neutralize(self, factor_data: pd.DataFrame,
                  date_col: str = "date",
                  symbol_col: str = "symbol",
                  factor_col: str = "factor",
                  industry_col: str = "industry") -> pd.DataFrame:
        """因子中性化处理
        
        调用默认处理器的方法
        
        Args:
            factor_data: 因子数据
            date_col: 日期列名
            symbol_col: 股票代码列名
            factor_col: 因子值列名
            industry_col: 行业列名
            
        Returns:
            pd.DataFrame: 中性化后的因子数据
        """
        return self.daily_processor.neutralize(factor_data, date_col, symbol_col, factor_col, industry_col)
    
    def compute_factor(self, factor_name: str,
                      symbols: Union[str, List[str]],
                      start_date: str,
                      end_date: str,
                      params: Optional[Dict[str, Any]] = None,
                      freq: str = "day") -> pd.DataFrame:
        """计算指定因子
        
        根据频率选择合适的处理器计算因子
        
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
        
        # 检查是否是多频因子
        multi_freq_factors = ["volatility_ratio", "momentum_divergence", "volume_price_divergence"]
        
        if factor_name in multi_freq_factors:
            return self.compute_multi_freq_factor(factor_name, symbols, start_date, end_date, params, freq)
            
        # 如果是普通因子，根据频率选择处理器
        if freq in self.freq_map:
            processor = self.freq_map[freq]
            
            if freq.endswith(("min", "m")):
                # 使用分钟处理器的特殊方法
                if hasattr(processor, "compute_minute_factor"):
                    return processor.compute_minute_factor(factor_name, symbols, start_date, end_date, params, freq)
                    
            # 使用通用方法
            return processor.compute_factor(factor_name, symbols, start_date, end_date, params, freq)
        else:
            raise ValueError(f"不支持的数据频率: {freq}")
            
    def get_requirements(self):
        """获取计算因子所需字段"""
        # 合并日线和分钟处理器的需求
        daily_reqs = self.daily_processor.get_requirements()
        minute_reqs = self.minute_processor.get_requirements()
        
        # 合并去重
        all_reqs = list(set(daily_reqs + minute_reqs))
        return all_reqs