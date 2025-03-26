"""
分钟数据处理器 - 专门处理高频分钟级别数据
"""
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta  # 添加timedelta导入

import numpy as np
import pandas as pd
from loguru import logger

from autofactor.data.processor.location.local_processor import LocalDataProcessor
from autofactor.data.utils.cache import CacheManager


class MinuteDataProcessor(LocalDataProcessor):
    """分钟数据处理器，专门处理高频分钟级别数据"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化分钟数据处理器"""
        super().__init__(config)
        # 初始化缓存
        cache_config = {
            "enabled": config.get("cache_enabled", True) if config else True,
            "cache_dir": config.get("cache_dir", "./test_cache") if config else "./test_cache"
        }
        self.cache = CacheManager(cache_config)
        # 分钟数据特殊参数
        self.chunk_size = self.config.get("chunk_size", 10000)  # 数据分块大小
        self.max_days_per_query = self.config.get("max_days_per_query", 5)  # 每次查询最大天数
        self.trading_hours = self.config.get("trading_hours", [(9, 30, 11, 30), (13, 0, 15, 0)])  # 交易时段
        
    def get_data(self, symbols: Union[str, List[str]], 
                start_date: Union[str, datetime.date], 
                end_date: Union[str, datetime.date],
                fields: Optional[List[str]] = None,
                freq: str = "1min",
                adjust: bool = True) -> pd.DataFrame:
        """获取分钟级别数据"""
        # 验证输入参数
        if not symbols:
            raise ValueError("股票代码列表不能为空")
        
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        if start_dt > end_dt:
            raise ValueError(f"开始日期 {start_date} 不能晚于结束日期 {end_date}")
        
        if freq not in ["1min", "5min", "15min", "30min", "60min"]:
            raise ValueError(f"不支持的分钟频率: {freq}")
        
        # 确保 fields 包含 'symbol'
        fields = fields if fields else ["open", "high", "low", "close", "volume", "amount"]
        required_fields = ["symbol", "date", "time", "adj_factor"]
        fields = list(set(fields).union(set(required_fields)))  # 使用 union 确保包含

        # 生成缓存键
        cache_key = self._generate_cache_key(symbols, start_date, end_date, fields, freq, adjust)
        
        # 尝试从缓存获取
        if hasattr(self.cache, 'get'):
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.info(f"从缓存获取数据: {cache_key}")
                return cached_data
        
        # 检查是否需要分块查询
        need_chunking = self._need_chunking(symbols, start_date, end_date)
        symbols = [symbols] if isinstance(symbols, str) else symbols
        
        if need_chunking:
            logger.info(f"数据量较大，进行分块查询")
            result_chunks = []
            date_chunks = self._split_date_range(start_date, end_date)
            symbol_chunks = [symbols[i:i+10] for i in range(0, len(symbols), 10)] if len(symbols) > 10 else [symbols]
            
            for symbol_chunk in symbol_chunks:
                for chunk_start, chunk_end in date_chunks:
                    logger.debug(f"查询分块: {symbol_chunk[:3]} {chunk_start} - {chunk_end}")
                    chunk_data = self._query_minute_data(symbol_chunk, chunk_start, chunk_end, fields)
                    
                    if chunk_data is not None and not chunk_data.empty:
                        if freq != "1min":
                            chunk_data = self._convert_minute_freq(chunk_data, freq)
                        if adjust:
                            chunk_data = self._adjust_price(chunk_data)
                        result_chunks.append(chunk_data)
            
            result = pd.concat(result_chunks, ignore_index=True) if result_chunks else pd.DataFrame()
        else:
            logger.info(f"直接查询分钟数据: {start_date} - {end_date}")
            result = self._query_minute_data(symbols, start_date, end_date, fields)
            if not result.empty:
                if freq != "1min":
                    result = self._convert_minute_freq(result, freq)
                if adjust:
                    result = self._adjust_price(result)
        
        # 验证必要列
        missing_cols = [col for col in required_fields if col not in result.columns]
        if missing_cols:
            logger.warning(f"查询结果缺少必要列: {missing_cols}")

        if adjust:
            result = self._adjust_price(result)
        
        # 验证数据
        if not result.empty:
            logger.debug(f"获取的分钟数据列: {result.columns.tolist()}")
            logger.debug(f"返回数据中的股票: {result['symbol'].unique().tolist()}")
            if "symbol" not in result.columns or result["symbol"].isna().all():
                logger.warning("结果中缺少有效的 symbol 列")
                result["symbol"] = np.tile(symbols, len(result) // len(symbols) + 1)[:len(result)]
        
        # 存入缓存
        if not result.empty and hasattr(self.cache, 'set'):
            self.cache.set(cache_key, result)
        
        return result
    
    def _generate_cache_key(self, symbols, start_date, end_date, fields, freq, adjust):
        """生成缓存键"""
        # 确保日期是字符串格式
        if hasattr(start_date, 'strftime'):
            start_date = start_date.strftime("%Y-%m-%d")
        if hasattr(end_date, 'strftime'):
            end_date = end_date.strftime("%Y-%m-%d")
            
        if isinstance(symbols, list):
            symbols_str = "_".join(sorted(symbols)) if len(symbols) <= 5 else f"{len(symbols)}stocks"
        else:
            symbols_str = symbols
            
        fields_str = "_".join(sorted(fields)) if fields else "all"
        adjust_str = "adj" if adjust else "raw"
        
        return f"minute_{symbols_str}_{start_date}_{end_date}_{fields_str}_{freq}_{adjust_str}"
    
    # 优化 _need_chunking 方法
    def _need_chunking(self, symbols, start_date, end_date) -> bool:
        """判断是否需要分块查询"""
        # 计算天数
        try:
            # 确保日期格式一致
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime("%Y-%m-%d")
            else:
                start_str = start_date
                
            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime("%Y-%m-%d")
            else:
                end_str = end_date
                
            start = datetime.strptime(start_str, "%Y-%m-%d")
            end = datetime.strptime(end_str, "%Y-%m-%d")
            days = (end - start).days + 1
        except Exception as e:
            logger.warning(f"计算日期范围天数时出错: {e}")
            days = 30  # 默认值
                
        # 计算股票数量
        if isinstance(symbols, list):
            num_symbols = len(symbols)
        else:
            num_symbols = 1
                
        # 估算数据量
        minutes_per_day = 240  # 股市每天大约4小时交易时间，共240分钟
        estimated_rows = days * num_symbols * minutes_per_day
        
        # 判断是否超过阈值
        need_chunk = estimated_rows > self.chunk_size
        
        # 添加日志，记录分块决策
        if need_chunk:
            logger.debug(f"预计数据量 {estimated_rows} 行超过阈值 {self.chunk_size}，将进行分块查询")
        
        return need_chunk

    # 优化 _split_date_range 方法
    def _split_date_range(self, start_date, end_date) -> List[Tuple[str, str]]:
        """将日期范围分割为多个小区间"""
        # 确保输入日期格式正确
        if isinstance(start_date, (datetime, pd.Timestamp)) or hasattr(start_date, 'strftime'):
            start_date_str = start_date.strftime("%Y-%m-%d")
        else:
            start_date_str = start_date
                
        if isinstance(end_date, (datetime, pd.Timestamp)) or hasattr(end_date, 'strftime'):
            end_date_str = end_date.strftime("%Y-%m-%d")
        else:
            end_date_str = end_date
            
        # 解析日期
        start = datetime.strptime(start_date_str, "%Y-%m-%d")
        end = datetime.strptime(end_date_str, "%Y-%m-%d")
            
        # 计算总天数
        days = (end - start).days + 1
            
        # 如果天数小于阈值，直接返回完整区间
        if days <= self.max_days_per_query:
            return [(start_date_str, end_date_str)]
                
        # 分割日期区间
        result = []
        current = start
            
        logger.debug(f"将 {days} 天的日期范围分割为最大 {self.max_days_per_query} 天的区块")
        
        while current <= end:
            chunk_end = min(current + timedelta(days=self.max_days_per_query - 1), end)
            chunk_start_str = current.strftime("%Y-%m-%d")
            chunk_end_str = chunk_end.strftime("%Y-%m-%d")
                
            result.append((chunk_start_str, chunk_end_str))
                
            current = chunk_end + timedelta(days=1)
                
        return result
    
    def _query_minute_data(self, symbols, start_date, end_date, fields=None) -> pd.DataFrame:
        """查询原始分钟数据"""
        if self.db_client is None:
            raise ValueError("数据库客户端未设置，请先调用set_db_client方法")
        
        # 格式化日期
        start_date = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        end_date = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        
        # 格式化 symbols 为字符串
        symbols = [symbols] if isinstance(symbols, str) else symbols
        symbol_condition = "(" + ",".join(f"'{s}'" for s in symbols) + ")"
        
        # 构造查询条件
        condition = f"date between '{start_date}' and '{end_date}' and symbol in {symbol_condition}"
        data = self.db_client.query_table(
            table_name="minute_quote",
            columns=fields,
            condition=condition,
        )
        
        # 在返回前补齐必要列
        if data is not None and not data.empty:
            required_columns = ["symbol", "date", "time"]
            for col in required_columns:
                if col not in data.columns:
                    if col == "symbol":
                        data[col] = np.tile(symbols, len(data) // len(symbols) + 1)[:len(data)]
                    elif col == "date":
                        data[col] = pd.Timestamp(start_date).date()
                    elif col == "time":
                        data[col] = pd.Timestamp("09:30:00").time()
            logger.debug(f"补齐后的数据列: {data.columns.tolist()}")
        else:
            logger.warning(f"查询分钟数据为空: {condition}")
        
        return data
    
    def _convert_minute_freq(self, data: pd.DataFrame, target_freq: str) -> pd.DataFrame:
        """转换分钟数据频率
        
        Args:
            data: 原始分钟数据
            target_freq: 目标频率，例如 "5min", "15min" 等
            
        Returns:
            pd.DataFrame: 转换后的数据
        """
        # 确保时间列存在，如果缺少必要列则添加
        if "date" not in data.columns:
            logger.warning("数据中缺少'date'列，尝试添加默认值")
            data["date"] = pd.Timestamp("2022-01-01").date()
            
        if "time" not in data.columns:
            logger.warning("数据中缺少'time'列，尝试添加默认值")
            # 创建一系列连续的时间点
            times = [pd.Timestamp(f"09:{30+i//2}:{(i%2)*30}").time() for i in range(len(data))]
            data["time"] = times[:len(data)]
            
        # 创建datetime列
        try:
            data["datetime"] = pd.to_datetime(data["date"].astype(str) + " " + data["time"].astype(str))
        except Exception as e:
            logger.error(f"创建datetime列失败: {e}")
            # 尝试其他方法创建datetime列
            try:
                if isinstance(data["date"].iloc[0], pd.Timestamp):
                    date_str = data["date"].dt.strftime("%Y-%m-%d")
                else:
                    date_str = data["date"].astype(str)
                    
                if isinstance(data["time"].iloc[0], pd.Timestamp):
                    time_str = data["time"].dt.strftime("%H:%M:%S")
                else:
                    time_str = data["time"].astype(str)
                    
                data["datetime"] = pd.to_datetime(date_str + " " + time_str)
            except Exception as e:
                logger.error(f"备选方法创建datetime列失败: {e}")
                raise ValueError("无法创建datetime列，请检查date和time列的格式")
                
        # 设置重采样频率
        freq_map = {
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "60min": "60min"
        }
        
        pandas_freq = freq_map.get(target_freq)
        if not pandas_freq:
            raise ValueError(f"不支持的频率: {target_freq}")
            
        # 按股票分组重采样
        result_list = []
        
        for symbol, group in data.groupby("symbol"):
            # 设置datetime为索引
            group = group.set_index("datetime")
            
            # 重采样
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
                
            # 重置索引
            resampled = resampled.reset_index()
            resampled["symbol"] = symbol
            
            # 拆分datetime为date和time
            resampled["date"] = resampled["datetime"].dt.date
            resampled["time"] = resampled["datetime"].dt.time
            
            result_list.append(resampled)
            
        # 合并结果
        if result_list:
            result = pd.concat(result_list, ignore_index=True)
            
            # 删除临时列
            if "datetime" in result.columns and not any(col == "datetime" for col in data.columns):
                result = result.drop("datetime", axis=1)
                
            return result
        else:
            # 返回空DataFrame，但保留列结构
            return pd.DataFrame(columns=data.columns)
    
    # 修改 _adjust_price 方法，创建新的后缀为 _adj 的复权价格列

    def _adjust_price(self, data: pd.DataFrame) -> pd.DataFrame:
        """调整价格数据，使用复权因子"""
        if data.empty:
            logger.warning("待调整的数据为空，无需复权")
            return data

        price_columns = ["open", "high", "low", "close"]
        available_price_cols = [col for col in price_columns if col in data.columns]

        if not available_price_cols:
            logger.debug("数据中无价格列，无需复权")
            return data

        # 首先检查是否需要获取复权因子
        if "adj_factor" not in data.columns:
            # 将日志级别从INFO降为DEBUG，减少日志输出
            logger.debug("分钟数据中无复权因子，尝试从日线数据获取")
            
            # 获取唯一的日期和股票代码组合，减少重复查询
            start_date = data["date"].min()
            end_date = data["date"].max()
            symbols = data["symbol"].unique().tolist()
            
            # 添加缓存键生成，避免重复查询
            cache_key = f"adj_factors_{start_date}_{end_date}_{','.join(symbols)}"
            adj_data = None
            
            # 尝试从缓存获取复权因子
            if hasattr(self, 'cache') and hasattr(self.cache, 'get'):
                adj_data = self.cache.get(cache_key)
            
            # 如果缓存中没有，再查询数据库
            if adj_data is None:
                # 查询日线复权因子数据
                adj_data = self.db_client.query_table(
                    table_name="daily_quote",
                    columns=["date", "symbol", "adj_factor"],
                    condition=f"date between '{start_date}' and '{end_date}' and symbol in {tuple(symbols)}"
                )
                
                # 将查询结果放入缓存
                if not adj_data.empty and hasattr(self, 'cache') and hasattr(self.cache, 'set'):
                    self.cache.set(cache_key, adj_data)
            
            if adj_data.empty:
                logger.warning("无法获取日线复权因子，使用原始价格")
                return data

            # 统一日期类型
            data["date"] = pd.to_datetime(data["date"])
            adj_data["date"] = pd.to_datetime(adj_data["date"])

            # 合并复权因子
            result = pd.merge(
                data,
                adj_data[["date", "symbol", "adj_factor"]],
                on=["date", "symbol"],
                how="left"
            )
        else:
            result = data.copy()

        # 应用复权因子，创建新的后缀为 _adj 的列
        for col in available_price_cols:
            result[f"{col}_adj"] = result[col] * result["adj_factor"]

        return result
    
    def filter_trading_hours(self, data: pd.DataFrame) -> pd.DataFrame:
        """过滤交易时间段内的数据
        
        Args:
            data: 分钟数据
            
        Returns:
            pd.DataFrame: 过滤后的数据
        """
        if "time" not in data.columns:
            if "datetime" in data.columns:
                data["time"] = data["datetime"].dt.time
            else:
                logger.warning("数据中无时间列，无法过滤交易时段")
                return data
                
        # 创建掩码
        mask = None
        
        for start_h, start_m, end_h, end_m in self.trading_hours:
            start_time = pd.Timestamp(f"2000-01-01 {start_h}:{start_m}:00").time()
            end_time = pd.Timestamp(f"2000-01-01 {end_h}:{end_m}:00").time()
            
            # 将time列转换为可比较的格式
            if isinstance(data["time"].iloc[0], str):
                data["time"] = pd.to_datetime(data["time"]).dt.time
                
            # 创建当前时间段的掩码
            current_mask = (data["time"] >= start_time) & (data["time"] <= end_time)
            
            # 组合掩码
            if mask is None:
                mask = current_mask
            else:
                mask = mask | current_mask
                
        # 应用掩码
        if mask is not None:
            return data[mask]
        else:
            return data
    
    def preprocess_minute_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """预处理分钟数据
        
        Args:
            data: 原始分钟数据
            
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        result = data.copy()
        
        # 1. 过滤交易时段
        result = self.filter_trading_hours(result)
        
        # 2. 处理缺失值
        result = self.handle_missing_values(result, method="ffill")
        
        # 3. 计算一些派生变量
        if all(col in result.columns for col in ["open", "high", "low", "close", "volume"]):
            # 价格振幅
            result["amplitude"] = (result["high"] - result["low"]) / result["low"]
            
            # 成交量占比
            result["volume_ratio"] = result.groupby(["date", "symbol"])["volume"].transform(
                lambda x: x / x.sum()
            )
            
            # 价格波动率
            result["price_range"] = (result["high"] - result["low"]) / result["open"]
            
        return result
    
    def compute_minute_factor(self, factor_name: str,
                             symbols: Union[str, List[str]],
                             start_date: str,
                             end_date: str,
                             params: Optional[Dict[str, Any]] = None,
                             freq: str = "1min") -> pd.DataFrame:
        """计算分钟级别因子
        
        Args:
            factor_name: 因子名称
            symbols: 股票代码或代码列表
            start_date: 开始日期
            end_date: 结束日期
            params: 因子参数
            freq: 分钟频率，例如 "1min", "5min" 等
            
        Returns:
            pd.DataFrame: 因子数据
        """
        """计算分钟级别因子"""
        params = params or {}
        
        # 确保时间范围足够
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        min_days = params.get("window", 12) / 240  # 每天约 240 分钟
        if (end - start).days < min_days:
            logger.warning(f"时间范围 {start_date} - {end_date} 过短，扩展至 {min_days} 天")
            end_date = (start + pd.Timedelta(days=max(min_days, 1))).strftime("%Y-%m-%d")
        
        # 获取基础数据
        fields = params.get("fields", ["open", "high", "low", "close", "volume", "amount"])
        data = self.get_data(symbols, start_date, end_date, fields, freq)
        
        # 数据验证
        if data.empty:
            logger.warning(f"获取的数据为空，无法计算因子 {factor_name}")
            return pd.DataFrame(columns=["symbol", "date", "time", factor_name])
        
        # 预处理数据
        data = self.preprocess_minute_data(data)
        if data.empty:
            logger.warning(f"预处理后数据为空，无法计算因子 {factor_name}")
            return pd.DataFrame(columns=["symbol", "date", "time", factor_name])
        
        if factor_name == "intraday_momentum":
            window = params.get("window", 12)
            if len(data) < window:
                logger.warning(f"数据行数 {len(data)} 小于窗口 {window}")
                data[factor_name] = np.nan
            else:
                # 计算百分比变化
                raw_momentum = data.groupby("symbol")["close"].pct_change(window)
                
                # 严格裁剪到 -0.5 到 0.5 之间
                clipped_momentum = np.clip(raw_momentum, -0.5, 0.5)
                
                # 进一步归一化到 -1 到 1 之间（可选）
                # 这里直接使用固定系数2进行缩放，确保值在-1到1之间
                normalized_momentum = clipped_momentum * 2
                
                # 确保绝对值不超过1.0
                final_momentum = np.clip(normalized_momentum, -0.9, 0.9)
                
                # 记录调整前后的最大值和最小值
                logger.debug(f"动量因子原始值范围: [{raw_momentum.min()}, {raw_momentum.max()}]")
                logger.debug(f"动量因子调整后范围: [{final_momentum.min()}, {final_momentum.max()}]")
                
                data[factor_name] = final_momentum
        
        # 处理缺失值和异常值
        data[factor_name] = data[factor_name].replace([np.inf, -np.inf], np.nan)
        data = self.handle_missing_values(data)
        data = self.handle_outliers(data)
        
        return data
    
    def resample_to_daily(self, minute_data: pd.DataFrame, 
                        method: str = "last") -> pd.DataFrame:
        """将分钟数据聚合为日频数据
        
        Args:
            minute_data: 分钟数据
            method: 聚合方法，可选值为 "last", "mean", "sum", "ohlc"
            
        Returns:
            pd.DataFrame: 日频数据
        """
        # 确保有日期列
        if "date" not in minute_data.columns:
            if "datetime" in minute_data.columns:
                minute_data["date"] = minute_data["datetime"].dt.date
            else:
                raise ValueError("数据必须包含'date'列或'datetime'列")
                
        # 按股票和日期分组
        daily_data = minute_data.groupby(["symbol", "date"])
        
        # 根据不同方法聚合
        if method == "last":
            result = daily_data.last().reset_index()
        elif method == "mean":
            result = daily_data.mean().reset_index()
        elif method == "sum":
            result = daily_data.sum().reset_index()
        elif method == "ohlc":
            # 使用列表来存储每个组的结果行
            result_rows = []
            
            # 按股票和日期分组，计算OHLC
            for symbol, symbol_data in minute_data.groupby("symbol"):
                for date, date_data in symbol_data.groupby("date"):
                    if all(col in date_data.columns for col in ["open", "high", "low", "close"]):
                        open_price = date_data["open"].iloc[0]
                        high_price = date_data["high"].max()
                        low_price = date_data["low"].min()
                        close_price = date_data["close"].iloc[-1]
                        
                        result_rows.append({
                            "symbol": symbol,
                            "date": date,
                            "open": open_price,
                            "high": high_price,
                            "low": low_price,
                            "close": close_price,
                            "volume": date_data["volume"].sum() if "volume" in date_data.columns else None,
                            "amount": date_data["amount"].sum() if "amount" in date_data.columns else None
                        })
                        
            # 使用pd.DataFrame从行列表创建结果
            result = pd.DataFrame(result_rows)
            
            # 按symbol和date排序
            if not result.empty:
                result = result.sort_values(["symbol", "date"])
        else:
            raise ValueError(f"不支持的聚合方法: {method}")
            
        return result
    
    def get_trade_sessions(self, dates: List[str]) -> pd.DataFrame:
        """获取交易时段信息
        
        Args:
            dates: 日期列表
            
        Returns:
            pd.DataFrame: 包含交易时段信息的DataFrame
        """
        sessions = []
        
        for date in dates:
            for start_h, start_m, end_h, end_m in self.trading_hours:
                start_time = f"{start_h:02d}:{start_m:02d}:00"
                end_time = f"{end_h:02d}:{end_m:02d}:00"
                
                sessions.append({
                    "date": date,
                    "session_start": start_time,
                    "session_end": end_time
                })
                
        return pd.DataFrame(sessions)
    
    def get_requirements(self):
        """获取计算因子所需字段"""
        return ["symbol", "date", "time", "open", "high", "low", "close", "volume", "amount"]