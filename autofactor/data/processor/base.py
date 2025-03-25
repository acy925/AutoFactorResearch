"""
数据处理基础抽象类 - 定义数据处理的通用接口
"""
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import DATA_PROCESS


class DataProcessor(ABC):
    """数据处理抽象基类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化数据处理器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self._init_logger()
        
    def _init_logger(self):
        """初始化日志系统"""
        log_level = self.config.get("log_level", "INFO")
        logger.remove()
        logger.add(sys.stderr, level=log_level)
        
    @abstractmethod
    def get_data(self, symbols: Union[str, List[str]], 
                 start_date: str, 
                 end_date: str,
                 fields: Optional[List[str]] = None,
                 freq: str = "day",
                 adjust: bool = True) -> pd.DataFrame:
        """获取指定股票在指定时间范围内的数据"""
        pass
    
    @abstractmethod
    def resample(self, data: pd.DataFrame, 
                 target_freq: str = "day", 
                 method: str = "ohlc") -> pd.DataFrame:
        """将数据重采样到指定频率"""
        pass
    
    @abstractmethod
    def handle_missing_values(self, data: pd.DataFrame, 
                             method: str = "ffill",
                             limit: Optional[int] = None) -> pd.DataFrame:
        """处理缺失值"""
        pass
    
    @abstractmethod
    def handle_outliers(self, data: pd.DataFrame,
                       method: str = "winsorize") -> pd.DataFrame:
        """处理异常值"""
        pass
    
    @abstractmethod
    def normalize(self, data: pd.DataFrame, 
                 method: str = "zscore",
                 by_cross_section: bool = True,
                 date_col: str = "date",
                 symbol_col: str = "symbol") -> pd.DataFrame:  
        """标准化处理"""
        pass
    
    @abstractmethod
    def neutralize(self, factor_data: pd.DataFrame,
                  date_col: str = "date",
                  symbol_col: str = "symbol",
                  factor_col: str = "factor",
                  industry_col: str = "industry") -> pd.DataFrame:
        """因子中性化处理"""
        pass
    
    @abstractmethod
    def compute_factor(self, factor_name: str,
                      symbols: Union[str, List[str]],
                      start_date: str,
                      end_date: str,
                      params: Optional[Dict[str, Any]] = None,
                      freq: str = "day") -> pd.DataFrame:
        """计算指定因子"""
        pass