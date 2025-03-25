"""
分层回测分析模块 - 评估因子的预测能力
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union

from autofactor.evaluation.metrics import calc_quantile_returns


class QuantileAnalyzer:
    """分位数分析器，用于分析因子分层回测结果"""
    
    def __init__(self):
        """初始化分位数分析器"""
        self.quantile_returns = None
        self.cumulative_returns = None
        self.long_short_returns = None
        self.factor_name = None
        
    def calculate_quantile_returns(self, factor_data: pd.DataFrame,
                                 factor_col: str,
                                 n_quantiles: int = 5,
                                 forward_periods: Union[int, List[int]] = 1,
                                 price_col: str = 'close',
                                 date_col: str = 'date',
                                 symbol_col: str = 'symbol',
                                 group_col: Optional[str] = None) -> Dict[int, pd.DataFrame]:
        """计算因子分位数收益
        
        Args:
            factor_data: 包含因子值的DataFrame
            factor_col: 因子列名
            n_quantiles: 分位数数量
            forward_periods: 未来收益周期(天)，可以是单个整数或列表
            price_col: 价格列名
            date_col: 日期列名
            symbol_col: 股票代码列名
            group_col: 分组列名(可选)
            
        Returns:
            Dict: 映射周期到分位数收益的字典
        """
        if isinstance(forward_periods, int):
            forward_periods = [forward_periods]
            
        # 验证数据
        required_cols = [factor_col, price_col, date_col, symbol_col]
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
        
        # 初始化结果字典
        results = {}
        
        for period in forward_periods:
            # 计算未来收益
            return_name = f'forward_return_{period}d'
            data = factor_data.copy()
            data[return_name] = data.groupby(symbol_col)[price_col].pct_change(period).shift(-period)
            
            # 按日期计算分位数
            if group_col is not None:
                # 按组别计算分位数
                quantile_returns = []
                for date, date_group in data.groupby(date_col):
                    date_returns = []
                    for group_name, group_data in date_group.groupby(group_col):
                        # 确保每组有足够的样本
                        if len(group_data) >= n_quantiles:
                            try:
                                # 计算分位数
                                group_data['quantile'] = pd.qcut(
                                    group_data[factor_col], 
                                    n_quantiles, 
                                    labels=False, 
                                    duplicates='drop'
                                )
                                
                                # 计算各分位数平均收益
                                quant_rets = group_data.groupby('quantile')[return_name].mean()
                                quant_rets.name = f"{date}_{group_name}"
                                date_returns.append(quant_rets)
                            except:
                                # 如果分位数计算失败，跳过
                                continue
                            
                    if date_returns:
                        # 平均各组的分位数收益
                        avg_returns = pd.concat(date_returns, axis=1).mean(axis=1)
                        avg_returns.name = date
                        quantile_returns.append(avg_returns)
            else:
                # 直接计算分位数收益
                quantile_returns = []
                for date, date_group in data.groupby(date_col):
                    try:
                        # 计算分位数
                        date_group['quantile'] = pd.qcut(
                            date_group[factor_col], 
                            n_quantiles, 
                            labels=False, 
                            duplicates='drop'
                        )
                        
                        # 计算各分位数平均收益
                        quant_rets = date_group.groupby('quantile')[return_name].mean()
                        quant_rets.name = date
                        quantile_returns.append(quant_rets)
                    except:
                        # 如果分位数计算失败，跳过
                        continue
            
            # 合并结果
            if quantile_returns:
                quantile_df = pd.DataFrame(quantile_returns)
                quantile_df.index = pd.to_datetime(quantile_df.index)
                quantile_df = quantile_df.sort_index()
                
                # 计算多空收益
                long_short = quantile_df.iloc[:, -1] - quantile_df.iloc[:, 0]
                
                # 计算累积收益
                cumulative = (1 + quantile_df).cumprod() - 1
                
                # 计算多空累积收益
                long_short_cum = (1 + long_short).cumprod() - 1
                
                # 存储结果
                results[period] = {
                    'quantile_returns': quantile_df,
                    'cumulative_returns': cumulative,
                    'long_short_returns': long_short,
                    'long_short_cumulative': long_short_cum
                }
            else:
                results[period] = None
                
        # 默认使用第一个周期
        default_period = forward_periods[0]
        if results[default_period]:
            self.quantile_returns = results[default_period]['quantile_returns']
            self.cumulative_returns = results[default_period]['cumulative_returns']
            self.long_short_returns = results[default_period]['long_short_returns']
            self.long_short_cumulative = results[default_period]['long_short_cumulative']
        
        return results
        
    def plot_cumulative_returns(self, figsize: Tuple[int, int] = (12, 8), 
                               title: Optional[str] = None) -> plt.Figure:
        """绘制累积收益曲线
        
        Args:
            figsize: 图表大小
            title: 图表标题
            
        Returns:
            plt.Figure: 图表对象
        """
        if self.cumulative_returns is None:
            raise ValueError("请先调用calculate_quantile_returns方法计算分位数收益")
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制各分位数累积收益
        for col in self.cumulative_returns.columns:
            ax.plot(self.cumulative_returns.index, self.cumulative_returns[col], 
                   label=f'Q{col+1}')
            
        # 设置标题和标签
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"{self.factor_name or 'Factor'} Cumulative Returns by Quantile")
            
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        
        # 添加网格和图例
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 旋转x轴标签
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
        
    def plot_long_short_returns(self, figsize: Tuple[int, int] = (12, 6), 
                               title: Optional[str] = None) -> plt.Figure:
        """绘制多空收益曲线
        
        Args:
            figsize: 图表大小
            title: 图表标题
            
        Returns:
            plt.Figure: 图表对象
        """
        if self.long_short_cumulative is None:
            raise ValueError("请先调用calculate_quantile_returns方法计算分位数收益")
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制多空累积收益
        ax.plot(self.long_short_cumulative.index, self.long_short_cumulative.values, 
               color='blue', linewidth=2)
            
        # 设置标题和标签
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"{self.factor_name or 'Factor'} Long-Short Cumulative Returns")
            
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 添加零线
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # 旋转x轴标签
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
        
    def plot_quantile_returns_distribution(self, figsize: Tuple[int, int] = (10, 6), 
                                          title: Optional[str] = None) -> plt.Figure:
        """绘制分位数收益分布
        
        Args:
            figsize: 图表大小
            title: 图表标题
            
        Returns:
            plt.Figure: 图表对象
        """
        if self.quantile_returns is None:
            raise ValueError("请先调用calculate_quantile_returns方法计算分位数收益")
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # 计算每个分位数的平均收益
        mean_returns = self.quantile_returns.mean()
        
        # 绘制分位数平均收益条形图
        bar_positions = np.arange(len(mean_returns))
        ax.bar(bar_positions, mean_returns, alpha=0.7)
        
        # 添加数值标签
        for i, value in enumerate(mean_returns):
            ax.text(i, value + (0.0005 if value >= 0 else -0.002), 
                   f'{value:.4f}', ha='center')
        
        # 设置标题和标签
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"{self.factor_name or 'Factor'} Average Returns by Quantile")
            
        ax.set_xlabel('Quantile')
        ax.set_ylabel('Average Return')
        
        # 设置x轴刻度
        ax.set_xticks(bar_positions)
        ax.set_xticklabels([f'Q{i+1}' for i in range(len(mean_returns))])
        
        # 添加网格
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加零线
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
        
    def plot_quantile_returns_heatmap(self, figsize: Tuple[int, int] = (12, 8), 
                                     title: Optional[str] = None) -> plt.Figure:
        """绘制分位数收益热力图
        
        Args:
            figsize: 图表大小
            title: 图表标题
            
        Returns:
            plt.Figure: 图表对象
        """
        if self.quantile_returns is None:
            raise ValueError("请先调用calculate_quantile_returns方法计算分位数收益")
            
        # 获取最近的数据
        recent_returns = self.quantile_returns.tail(20)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制热力图
        im = ax.imshow(recent_returns.T, cmap='RdYlGn', aspect='auto')
        
        # 设置标题和标签
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"{self.factor_name or 'Factor'} Quantile Returns Heatmap (Recent 20 Periods)")
            
        # 设置x轴刻度
        dates = [d.strftime('%Y-%m-%d') for d in recent_returns.index]
        ax.set_xticks(np.arange(len(dates)))
        ax.set_xticklabels(dates, rotation=90)
        
        # 设置y轴刻度
        ax.set_yticks(np.arange(len(recent_returns.columns)))
        ax.set_yticklabels([f'Q{i+1}' for i in range(len(recent_returns.columns))])
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, label='Return')
        
        plt.tight_layout()
        return fig
        
    def summary(self) -> Dict[str, float]:
        """返回分层回测分析摘要
        
        Returns:
            Dict: 分析统计结果
        """
        if self.quantile_returns is None:
            raise ValueError("请先调用calculate_quantile_returns方法计算分位数收益")
            
        # 计算各分位数平均收益
        mean_returns = self.quantile_returns.mean()
        
        # 计算最高分位与最低分位收益差
        spread = mean_returns.iloc[-1] - mean_returns.iloc[0]
        
        # 计算多空收益统计
        long_short_mean = self.long_short_returns.mean()
        long_short_std = self.long_short_returns.std()
        long_short_sharpe = long_short_mean / long_short_std if long_short_std != 0 else np.nan
        long_short_win_rate = (self.long_short_returns > 0).mean()
        
        # 计算各分位数累积收益
        final_cumulative = self.cumulative_returns.iloc[-1]
        
        # 计算多空累积收益
        long_short_cum = self.long_short_cumulative.iloc[-1]
        
        return {
            'Quantile Mean Returns': mean_returns.to_dict(),
            'Spread (Q5-Q1)': spread,
            'Long-Short Mean Return': long_short_mean,
            'Long-Short Sharpe Ratio': long_short_sharpe,
            'Long-Short Win Rate': long_short_win_rate,
            'Quantile Cumulative Returns': final_cumulative.to_dict(),
            'Long-Short Cumulative Return': long_short_cum
        }