"""
混合处理器 - 智能选择在本地或DolphinDB服务器上处理数据
"""
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from autofactor.data.processor.base import DataProcessor
from autofactor.data.processor.location.local_processor import LocalDataProcessor
from autofactor.data.processor.location.db_processor import DBProcessor
from autofactor.data.utils.cache import CacheManager


class HybridProcessor(DataProcessor):
    """混合数据处理器，根据数据规模和任务类型自动选择最适合的处理位置"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化混合数据处理器
        
        Args:
            config: 配置参数，例如决策阈值等
        """
        super().__init__(config)
        
        # 初始化本地处理器和DB处理器
        self.local_processor = LocalDataProcessor(config)
        self.db_processor = DBProcessor(config)
        
        # 初始化缓存
        self.cache = CacheManager(config)
        
        # 设置决策阈值
        self.thresholds = {
            "rows_threshold": self.config.get("rows_threshold", 100000),  # 行数阈值
            "symbols_threshold": self.config.get("symbols_threshold", 50),  # 股票数量阈值
            "days_threshold": self.config.get("days_threshold", 60),  # 时间跨度阈值（天）
            "columns_threshold": self.config.get("columns_threshold", 20),  # 列数阈值
        }
        
        # 统计信息
        self.stats = {
            "local_tasks": 0,
            "db_tasks": 0,
            "cache_hits": 0,
            "errors": 0,
            "fallbacks": 0
        }
        
        # 任务历史，用于记录任务执行情况
        self.task_history = []
        
    def _decide_processor(self, task_type: str, data_size: Dict[str, int]) -> str:
        """决定使用哪个处理器
        
        Args:
            task_type: 任务类型，如"get_data", "normalize", "handle_missing_values"等
            data_size: 数据规模信息，包含行数、列数、股票数量等
            
        Returns:
            str: 处理器类型，"local"或"db"
        """
        # 一些任务更适合在本地处理
        local_preferred_tasks = [
            "handle_outliers",  # 异常值处理在本地更灵活
            "compute_factor",   # 复杂因子计算通常在本地
        ]
        
        if task_type in local_preferred_tasks:
            return "local"
            
        # 一些任务更适合在DB处理
        db_preferred_tasks = [
            "resample",        # 重采样在DB更高效
            "get_data",        # 数据获取直接在DB
        ]
        
        if task_type in db_preferred_tasks:
            return "db"
            
        # 根据数据规模决定
        if data_size.get("rows", 0) > self.thresholds["rows_threshold"]:
            return "db"
            
        if data_size.get("symbols", 0) > self.thresholds["symbols_threshold"]:
            return "db"
            
        if data_size.get("days", 0) > self.thresholds["days_threshold"]:
            return "db"
            
        if data_size.get("columns", 0) > self.thresholds["columns_threshold"]:
            return "db"
            
        # 默认使用本地处理
        return "local"
    
    def _get_data_size(self, data=None, symbols=None, start_date=None, end_date=None) -> Dict[str, int]:
        """获取数据规模信息
        
        Args:
            data: 输入数据
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict: 数据规模信息
        """
        result = {"rows": 0, "columns": 0, "symbols": 0, "days": 0}
        
        if isinstance(data, pd.DataFrame):
            result["rows"] = len(data)
            result["columns"] = len(data.columns)
            
            if "symbol" in data.columns:
                result["symbols"] = len(data["symbol"].unique())
                
            if "date" in data.columns:
                result["days"] = len(data["date"].unique())
                
        else:
            # 估算规模
            if symbols:
                if isinstance(symbols, list):
                    result["symbols"] = len(symbols)
                else:
                    result["symbols"] = 1
                    
            if start_date and end_date:
                try:
                    from datetime import datetime
                    start = datetime.strptime(start_date, "%Y-%m-%d")
                    end = datetime.strptime(end_date, "%Y-%m-%d")
                    result["days"] = (end - start).days + 1
                except:
                    result["days"] = 30  # 默认值
                    
        return result
    
    def _execute_task(self, processor_type: str, task_type: str, *args, **kwargs) -> Any:
        """执行任务
        
        Args:
            processor_type: 处理器类型，"local"或"db"
            task_type: 任务类型
            *args, **kwargs: 任务参数
            
        Returns:
            Any: 任务执行结果
        """
        start_time = time.time()
        error = None
        
        # 更新统计信息（即使后续失败）
        if processor_type == "local":
            self.stats["local_tasks"] += 1
        else:
            self.stats["db_tasks"] += 1

        try:
            # 根据处理器类型选择处理器
            processor = self.local_processor if processor_type == "local" else self.db_processor
            
            # 根据任务类型调用相应方法
            if task_type == "get_data":
                result = processor.get_data(*args, **kwargs)
            elif task_type == "resample":
                result = processor.resample(*args, **kwargs)
            elif task_type == "handle_missing_values":
                result = processor.handle_missing_values(*args, **kwargs)
            elif task_type == "handle_outliers":
                result = processor.handle_outliers(*args, **kwargs)
            elif task_type == "normalize":
                result = processor.normalize(*args, **kwargs)
            elif task_type == "neutralize":
                result = processor.neutralize(*args, **kwargs)
            elif task_type == "compute_factor":
                result = processor.compute_factor(*args, **kwargs)
            else:
                raise ValueError(f"不支持的任务类型: {task_type}")
                
        except Exception as e:
            error = str(e)
            # 更新统计信息
            self.stats["errors"] += 1
            raise
            
        finally:
            # 记录任务执行情况
            execution_time = time.time() - start_time
            task_record = {
                "task_type": task_type,
                "processor": processor_type,
                "time": execution_time,
                "error": error,
                "timestamp": time.time()
            }
            self.task_history.append(task_record)
            
        return result
    
    def _handle_task_with_fallback(self, task_type: str, *args, **kwargs) -> Any:
        """处理任务，失败时尝试降级
        
        Args:
            task_type: 任务类型
            *args, **kwargs: 任务参数
            
        Returns:
            Any: 任务执行结果
        """
        # 确定数据规模
        data_arg = kwargs.get("data", args[0] if len(args) > 0 else None)
        symbols_arg = kwargs.get("symbols", args[0] if task_type == "get_data" and len(args) > 0 else None)
        start_date = kwargs.get("start_date", args[1] if task_type == "get_data" and len(args) > 1 else None)
        end_date = kwargs.get("end_date", args[2] if task_type == "get_data" and len(args) > 2 else None)
        
        data_size = self._get_data_size(data_arg, symbols_arg, start_date, end_date)
        
        # 决定使用哪个处理器
        processor_type = self._decide_processor(task_type, data_size)
        logger.debug(f"任务 {task_type} 决定使用 {processor_type} 处理器，数据规模: {data_size}")
        
        try:
            # 尝试使用选定的处理器
            return self._execute_task(processor_type, task_type, *args, **kwargs)
        except Exception as e:
            logger.warning(f"{processor_type}处理器执行{task_type}失败: {e}")
            self.stats["errors"] += 1  # 手动记录错误
            # 如果使用DB处理器失败，尝试降级到本地处理
            if processor_type == "db":
                logger.info(f"降级到本地处理器执行{task_type}")
                self.stats["fallbacks"] += 1
                try:
                    return self._execute_task("local", task_type, *args, **kwargs)
                except Exception as e_local:
                    self.stats["errors"] += 1  # 记录本地失败
                    raise
    
    def get_data(self, symbols: Union[str, List[str]], 
                start_date: str, 
                end_date: str,
                fields: Optional[List[str]] = None,
                freq: str = "day",
                adjust: bool = True) -> pd.DataFrame:
        """获取数据，自动选择本地或DB处理
        
        Args:
            symbols: 股票代码或代码列表
            start_date: 开始日期，格式为 "YYYY-MM-DD"
            end_date: 结束日期，格式为 "YYYY-MM-DD"
            fields: 需要获取的字段列表，默认为所有字段
            freq: 数据频率，可选值为 "day", "minute"
            adjust: 是否进行复权处理
            
        Returns:
            pd.DataFrame: 股票数据
        """
        # 生成缓存键
        if isinstance(symbols, list):
            symbols_str = "_".join(sorted(symbols)) if len(symbols) <= 5 else f"{len(symbols)}stocks"
        else:
            symbols_str = symbols
            
        fields_str = "_".join(sorted(fields)) if fields else "all"
        adjust_str = "adj" if adjust else "raw"
        cache_key = f"hybrid_get_data_{symbols_str}_{start_date}_{end_date}_{fields_str}_{freq}_{adjust_str}"
        
        # 尝试从缓存获取
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logger.debug(f"从缓存获取数据: {cache_key}")
            self.stats["cache_hits"] += 1
            return cached_data
        
        # 执行任务
        result = self._handle_task_with_fallback(
            "get_data", 
            symbols, start_date, end_date, 
            fields=fields, freq=freq, adjust=adjust
        )
        
        # 缓存结果
        self.cache.set(cache_key, result)
        
        return result
    
    def resample(self, data: pd.DataFrame, 
                target_freq: str = "day", 
                method: str = "ohlc") -> pd.DataFrame:
        """将数据重采样到指定频率
        
        Args:
            data: 输入数据
            target_freq: 目标频率
            method: 重采样方法
            
        Returns:
            pd.DataFrame: 重采样后的数据
        """
        return self._handle_task_with_fallback("resample", data, target_freq, method=method)
    
    def handle_missing_values(self, data: pd.DataFrame, 
                             method: str = "ffill",
                             limit: Optional[int] = None) -> pd.DataFrame:
        """处理缺失值
        
        Args:
            data: 输入数据
            method: 处理方法
            limit: 最大填充长度
            
        Returns:
            pd.DataFrame: 处理缺失值后的数据
        """
        return self._handle_task_with_fallback("handle_missing_values", data, method=method, limit=limit)
    
    def handle_outliers(self, data: pd.DataFrame,
                       method: str = "winsorize") -> pd.DataFrame:
        """处理异常值
        
        Args:
            data: 输入数据
            method: 处理方法
            
        Returns:
            pd.DataFrame: 处理异常值后的数据
        """
        return self._handle_task_with_fallback("handle_outliers", data, method=method)
    
    def normalize(self, data: pd.DataFrame, 
                 method: str = "zscore",
                 by_cross_section: bool = True,
                 date_col: str = "date",
                 symbol_col: str = "symbol") -> pd.DataFrame:
        """标准化处理
        
        Args:
            data: 输入数据
            method: 标准化方法
            by_cross_section: 是否按横截面处理
            date_col: 日期列名
            symbol_col: 股票代码列名
            
        Returns:
            pd.DataFrame: 标准化后的数据
        """
        return self._handle_task_with_fallback(
            "normalize", 
            data, 
            method=method, 
            by_cross_section=by_cross_section,
            date_col=date_col,
            symbol_col=symbol_col
        )
    
    def neutralize(self, factor_data: pd.DataFrame,
                  date_col: str = "date",
                  symbol_col: str = "symbol",
                  factor_col: str = "factor",
                  industry_col: str = "industry") -> pd.DataFrame:
        """因子中性化处理
        
        Args:
            factor_data: 因子数据
            date_col: 日期列名
            symbol_col: 股票代码列名
            factor_col: 因子值列名
            industry_col: 行业列名
            
        Returns:
            pd.DataFrame: 中性化后的因子数据
        """
        return self._handle_task_with_fallback(
            "neutralize", 
            factor_data, 
            date_col=date_col, 
            symbol_col=symbol_col,
            factor_col=factor_col,
            industry_col=industry_col
        )
    
    def compute_factor(self, factor_name: str,
                      symbols: Union[str, List[str]],
                      start_date: str,
                      end_date: str,
                      params: Optional[Dict[str, Any]] = None,
                      freq: str = "day") -> pd.DataFrame:
        """计算指定因子
        
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
        # 生成缓存键
        if isinstance(symbols, list):
            symbols_str = "_".join(sorted(symbols)) if len(symbols) <= 5 else f"{len(symbols)}stocks"
        else:
            symbols_str = symbols
            
        params_str = ""
        if params:
            params_items = sorted(params.items())
            params_str = "_".join(f"{k}_{v}" for k, v in params_items)
            
        cache_key = f"hybrid_factor_{factor_name}_{symbols_str}_{start_date}_{end_date}_{params_str}_{freq}"
        
        # 尝试从缓存获取
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logger.debug(f"从缓存获取因子: {cache_key}")
            self.stats["cache_hits"] += 1
            return cached_data
        
        # 执行任务，因为DBProcessor暂不支持因子计算，此处强制使用本地处理
        try:
            result = self.local_processor.compute_factor(
                factor_name, symbols, start_date, end_date, params, freq
            )
            self.stats["local_tasks"] += 1
        except Exception as e:
            logger.error(f"计算因子失败: {e}")
            self.stats["errors"] += 1
            raise
            
        # 缓存结果
        self.cache.set(cache_key, result)
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            "task_stats": self.stats,
            "task_history": self.task_history[-10:],  # 只返回最近10条记录
            "processor_preference": {
                task: self._decide_processor(task, {"rows": 10000, "columns": 10, "symbols": 20, "days": 30})
                for task in ["get_data", "resample", "handle_missing_values", "normalize", "neutralize", "compute_factor"]
            }
        }