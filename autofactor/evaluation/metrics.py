"""
因子评价指标模块 - 提供计算因子质量的各种指标
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union

def calc_ic(factor: pd.Series, forward_return: pd.Series) -> float:
    """计算IC值(信息系数)

    Args:
        factor: 因子值Series
        forward_return: 未来收益率Series

    Returns:
        float: IC值
    """
    # 处理缺失值
    valid_data = pd.DataFrame({'factor': factor, 'return': forward_return}).dropna()
    
    # 计算相关系数
    if len(valid_data) < 5:  # 样本太少，返回NaN
        return np.nan
        
    # 使用spearman相关系数
    return stats.spearmanr(valid_data['factor'], valid_data['return'])[0]

def calc_ic_series(factor_data: pd.DataFrame, 
                  factor_col: str,
                  return_col: str,
                  date_col: str = 'date',
                  group_col: Optional[str] = None) -> pd.Series:
    """计算IC时间序列

    Args:
        factor_data: 包含因子值和未来收益的DataFrame
        factor_col: 因子列名
        return_col: 收益率列名
        date_col: 日期列名
        group_col: 分组列名(可选，用于计算分组IC)

    Returns:
        pd.Series: 按日期索引的IC时间序列
    """
    ic_series = []
    dates = []
    
    # 按日期分组计算IC
    for date, group in factor_data.groupby(date_col):
        if group_col is not None:
            # 分组IC
            group_ics = []
            for _, subgroup in group.groupby(group_col):
                ic = calc_ic(subgroup[factor_col], subgroup[return_col])
                if not np.isnan(ic):
                    group_ics.append(ic)
            
            # 计算分组IC的平均值
            if group_ics:
                ic = np.mean(group_ics)
            else:
                ic = np.nan
        else:
            # 直接计算IC
            ic = calc_ic(group[factor_col], group[return_col])
            
        ic_series.append(ic)
        dates.append(date)
        
    return pd.Series(ic_series, index=dates)

def calc_ic_stats(ic_series: pd.Series) -> Dict[str, float]:
    """计算IC统计特性

    Args:
        ic_series: IC时间序列

    Returns:
        Dict: 包含IC统计指标的字典
    """
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ic_ir = ic_mean / ic_std if ic_std != 0 else np.nan
    ic_pos_ratio = (ic_series > 0).mean()
    ic_neg_ratio = (ic_series < 0).mean()
    ic_t_value, ic_p_value = stats.ttest_1samp(ic_series.dropna(), 0)
    
    return {
        'IC Mean': ic_mean,
        'IC Std': ic_std,
        'IC IR': ic_ir,
        'IC > 0 %': ic_pos_ratio,
        'IC < 0 %': ic_neg_ratio,
        'IC t-value': ic_t_value,
        'IC p-value': ic_p_value,
        'IC Skew': ic_series.skew(),
        'IC Kurtosis': ic_series.kurtosis()
    }

def calc_quantile_returns(factor_data: pd.DataFrame,
                         factor_col: str,
                         return_col: str,
                         date_col: str = 'date',
                         n_quantiles: int = 5) -> pd.DataFrame:
    """计算因子分位数收益

    Args:
        factor_data: 包含因子值和未来收益的DataFrame
        factor_col: 因子列名
        return_col: 收益率列名
        date_col: 日期列名
        n_quantiles: 分位数数量，默认5

    Returns:
        pd.DataFrame: 各分位数的平均收益率
    """
    quantile_returns = []
    
    # 按日期分组计算分位数收益
    for date, group in factor_data.groupby(date_col):
        # 按因子值分组
        group['quantile'] = pd.qcut(group[factor_col], n_quantiles, labels=False, duplicates='drop')
        
        # 计算各分位数平均收益
        quant_rets = group.groupby('quantile')[return_col].mean()
        quant_rets.name = date
        
        quantile_returns.append(quant_rets)
        
    # 合并结果
    return pd.DataFrame(quantile_returns)

def calc_factor_turnover(factor_data: pd.DataFrame,
                        factor_col: str,
                        date_col: str = 'date',
                        id_col: str = 'symbol',
                        n_quantiles: int = 5) -> pd.Series:
    """计算因子换手率

    Args:
        factor_data: 因子数据
        factor_col: 因子列名
        date_col: 日期列名
        id_col: ID列名(如股票代码)
        n_quantiles: 分位数数量

    Returns:
        pd.Series: 因子换手率时间序列
    """
    turnover_series = []
    dates = []
    
    # 按日期排序
    factor_data = factor_data.sort_values(date_col)
    
    # 计算每个日期的分位数
    factor_data['quantile'] = factor_data.groupby(date_col)[factor_col].transform(
        lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates='drop')
    )
    
    # 获取唯一日期列表
    unique_dates = factor_data[date_col].unique()
    
    # 计算相邻日期间的换手率
    for i in range(1, len(unique_dates)):
        curr_date = unique_dates[i]
        prev_date = unique_dates[i-1]
        
        # 获取当前和上一日期的数据
        curr_df = factor_data[factor_data[date_col] == curr_date]
        prev_df = factor_data[factor_data[date_col] == prev_date]
        
        # 找到两个日期都存在的资产
        common_assets = set(curr_df[id_col]).intersection(set(prev_df[id_col]))
        
        if not common_assets:
            continue
            
        # 筛选共同资产
        curr_df = curr_df[curr_df[id_col].isin(common_assets)]
        prev_df = prev_df[prev_df[id_col].isin(common_assets)]
        
        # 设置索引便于比较
        curr_df = curr_df.set_index(id_col)
        prev_df = prev_df.set_index(id_col)
        
        # 计算分位数变化的资产比例
        changed = (curr_df['quantile'] != prev_df['quantile']).mean()
        
        turnover_series.append(changed)
        dates.append(curr_date)
        
    return pd.Series(turnover_series, index=dates)