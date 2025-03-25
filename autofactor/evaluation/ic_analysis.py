"""
IC分析模块 - 信息系数(Information Coefficient)相关分析
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union

from autofactor.evaluation.metrics import calc_ic_series, calc_ic_stats


class ICAnalyzer:
    """IC分析器，用于分析因子与未来收益的相关性"""
    
    def __init__(self):
        """初始化IC分析器"""
        self.ic_series = None
        self.ic_stats = None
        self.factor_name = None
        self.rank_ic = True  # 默认使用Rank IC (Spearman相关系数)
        
    def calculate_ic(self, factor_data: pd.DataFrame,
                   factor_col: str,
                   forward_periods: Union[int, List[int]] = 1,
                   return_col: str = 'close',
                   date_col: str = 'date',
                   symbol_col: str = 'symbol',
                   group_col: Optional[str] = None) -> Dict[int, pd.Series]:
        """计算指定周期的IC序列
        
        Args:
            factor_data: 包含因子值的DataFrame
            factor_col: 因子列名
            forward_periods: 未来收益周期(天)，可以是单个整数或列表
            return_col: 用于计算收益的价格列名
            date_col: 日期列名
            symbol_col: 股票代码列名
            group_col: 分组列名(可选)
            
        Returns:
            Dict: 映射周期到IC序列的字典
        """
        if isinstance(forward_periods, int):
            forward_periods = [forward_periods]
            
        # 验证数据
        required_cols = [factor_col, return_col, date_col, symbol_col]
        missing_cols = [col for col in required_cols if col not in factor_data.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要的列: {missing_cols}")
            
        # 确保日期列是日期类型
        if not pd.api.types.is_datetime64_any_dtype(factor_data[date_col]):
            factor_data[date_col] = pd.to_datetime(factor_data[date_col])
            
        # 记录因子名称
        self.factor_name = factor_col
        
        # 按symbol和date排序
        factor_data = factor_data.sort_values([symbol_col, date_col])
        
        # 计算各周期的未来收益
        data = factor_data.copy()
        
        # 初始化结果字典
        ic_results = {}
        
        for period in forward_periods:
            # 计算未来收益
            return_name = f'forward_return_{period}d'
            data[return_name] = data.groupby(symbol_col)[return_col].pct_change(period).shift(-period)
            
            # 计算IC序列
            ic_series = calc_ic_series(
                data, 
                factor_col=factor_col,
                return_col=return_name,
                date_col=date_col,
                group_col=group_col
            )
            
            # 计算IC统计特性
            ic_stats = calc_ic_stats(ic_series)
            
            # 存储结果
            ic_results[period] = {
                'ic_series': ic_series,
                'ic_stats': ic_stats
            }
            
        # 默认使用第一个周期
        default_period = forward_periods[0]
        self.ic_series = ic_results[default_period]['ic_series']
        self.ic_stats = ic_results[default_period]['ic_stats']
        
        return ic_results
        
    def get_ir(self) -> float:
        """获取信息比率(Information Ratio)
        
        Returns:
            float: 信息比率
        """
        if self.ic_stats is None:
            raise ValueError("请先调用calculate_ic方法计算IC")
            
        return self.ic_stats.get('IC IR', np.nan)
        
    def get_ic_decay(self, factor_data: pd.DataFrame,
                   factor_col: str,
                   max_periods: int = 10,
                   return_col: str = 'close',
                   date_col: str = 'date',
                   symbol_col: str = 'symbol') -> pd.Series:
        """计算IC衰减特性
        
        Args:
            factor_data: 包含因子值的DataFrame
            factor_col: 因子列名
            max_periods: 最大周期数
            return_col: 用于计算收益的价格列名
            date_col: 日期列名
            symbol_col: 股票代码列名
            
        Returns:
            pd.Series: 不同周期的平均IC值
        """
        # 计算各周期的IC
        periods = list(range(1, max_periods + 1))
        ic_results = self.calculate_ic(
            factor_data,
            factor_col=factor_col,
            forward_periods=periods,
            return_col=return_col,
            date_col=date_col,
            symbol_col=symbol_col
        )
        
        # 提取各周期的平均IC
        ic_means = {}
        for period, result in ic_results.items():
            ic_means[period] = result['ic_stats']['IC Mean']
            
        return pd.Series(ic_means)
        
    def plot_ic_series(self, figsize: Tuple[int, int] = (12, 6), title: Optional[str] = None) -> plt.Figure:
        """绘制IC时间序列图
        
        Args:
            figsize: 图表大小
            title: 图表标题
            
        Returns:
            plt.Figure: 图表对象
        """
        if self.ic_series is None:
            raise ValueError("请先调用calculate_ic方法计算IC")
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制IC序列
        ax.plot(self.ic_series.index, self.ic_series.values, marker='o', linestyle='-', markersize=4)
        
        # 添加零线
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        # 添加平均IC线
        mean_ic = self.ic_stats['IC Mean']
        ax.axhline(y=mean_ic, color='g', linestyle='--', alpha=0.7, 
                  label=f'Mean IC = {mean_ic:.4f}')
        
        # 设置标题和标签
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"{self.factor_name or 'Factor'} IC Time Series")
            
        ax.set_xlabel('Date')
        ax.set_ylabel('IC Value')
        
        # 添加网格和图例
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 旋转x轴标签
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
        
    def plot_ic_decay(self, ic_decay: pd.Series, figsize: Tuple[int, int] = (10, 6), 
                     title: Optional[str] = None) -> plt.Figure:
        """绘制IC衰减曲线
        
        Args:
            ic_decay: IC衰减序列
            figsize: 图表大小
            title: 图表标题
            
        Returns:
            plt.Figure: 图表对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制IC衰减曲线
        ax.plot(ic_decay.index, ic_decay.values, marker='o', linestyle='-', 
               markersize=6, color='blue')
        
        # 添加零线
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        # 设置标题和标签
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"{self.factor_name or 'Factor'} IC Decay")
            
        ax.set_xlabel('Forward Period (Days)')
        ax.set_ylabel('Mean IC')
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 设置x轴为整数
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        plt.tight_layout()
        return fig
        
    def plot_ic_distribution(self, figsize: Tuple[int, int] = (10, 6), 
                            bins: int = 20, title: Optional[str] = None) -> plt.Figure:
        """绘制IC分布直方图
        
        Args:
            figsize: 图表大小
            bins: 直方图的bin数量
            title: 图表标题
            
        Returns:
            plt.Figure: 图表对象
        """
        if self.ic_series is None:
            raise ValueError("请先调用calculate_ic方法计算IC")
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制IC直方图
        ax.hist(self.ic_series.dropna(), bins=bins, alpha=0.7, color='blue', edgecolor='black')
        
        # 添加均值线
        mean_ic = self.ic_stats['IC Mean']
        ax.axvline(x=mean_ic, color='r', linestyle='--', 
                  label=f'Mean IC = {mean_ic:.4f}')
        
        # 设置标题和标签
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"{self.factor_name or 'Factor'} IC Distribution")
            
        ax.set_xlabel('IC Value')
        ax.set_ylabel('Frequency')
        
        # 添加图例
        ax.legend()
        
        plt.tight_layout()
        return fig
        
    def summary(self) -> Dict[str, float]:
        """返回IC分析摘要
        
        Returns:
            Dict: IC分析统计结果
        """
        if self.ic_stats is None:
            raise ValueError("请先调用calculate_ic方法计算IC")
            
        return self.ic_stats