"""
回测引擎模块 - 用于评估量化因子和策略的历史表现
"""
import os
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from autofactor.data.processor.frequency.daily_processor import DailyDataProcessor
from autofactor.data.processor.frequency.minute_processor import MinuteDataProcessor
from autofactor.evaluation.metrics import calc_ic_stats
from autofactor.evaluation.ic_analysis import ICAnalyzer
from autofactor.evaluation.return_analysis import QuantileAnalyzer
from config.settings import BACKTEST


class BacktestEngine:
    """回测引擎，用于评估量化因子和策略的历史表现"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化回测引擎
        
        Args:
            config: 配置参数，包括回测设置、佣金率等
        """
        # 初始化配置，合并传入的config和默认配置
        self.config = BACKTEST.copy()
        if config:
            self.config.update(config)
            
        # 初始化数据处理器
        self.daily_processor = DailyDataProcessor()
        self.minute_processor = MinuteDataProcessor()
        
        # 回测结果存储
        self.results = {}
        self.positions = {}
        self.trades = []
        self.performance_metrics = {}
        
        # 因子分析工具
        self.ic_analyzer = ICAnalyzer()
        self.quantile_analyzer = QuantileAnalyzer()
        
        # 回测状态
        self.current_date = None
        self.current_positions = None
        self.portfolio_value = None
        self.cash = None
        
    def set_db_client(self, db_client):
        """设置数据库客户端
        
        Args:
            db_client: DolphinDB客户端实例
        """
        self.daily_processor.set_db_client(db_client)
        self.minute_processor.set_db_client(db_client)
        
    def backtest_factor(self, 
                       factor_data: pd.DataFrame,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       universe: Optional[List[str]] = None,
                       quantiles: int = 5,
                       holding_period: int = 1,
                       long_short: bool = True,
                       commission_rate: Optional[float] = None,
                       slippage: Optional[float] = None,
                       capital: Optional[float] = None,
                       benchmark: Optional[str] = None,
                       rebalance_freq: str = "daily",
                       factor_col: str = None,
                       date_col: str = "date",
                       symbol_col: str = "symbol") -> Dict[str, Any]:
        """回测单因子策略
        
        Args:
            factor_data: 因子数据
            start_date: 回测开始日期
            end_date: 回测结束日期
            universe: 股票池
            quantiles: 分位数数量
            holding_period: 持仓周期（天）
            long_short: 是否做多做空
            commission_rate: 佣金率
            slippage: 滑点率
            capital: 初始资金
            benchmark: 基准指数
            rebalance_freq: 再平衡频率，'daily', 'weekly', 'monthly'
            factor_col: 因子列名
            date_col: 日期列名
            symbol_col: 股票代码列名
            
        Returns:
            Dict: 回测结果
        """
        # 清除之前的回测结果
        self._reset_backtest()
        
        # 参数处理
        commission_rate = commission_rate if commission_rate is not None else self.config["commission_rate"]
        slippage = slippage if slippage is not None else self.config["slippage"]
        capital = capital if capital is not None else self.config["capital"]
        benchmark = benchmark if benchmark is not None else self.config["benchmark"]
        
        # 如果没有提供因子列名，自动检测
        if factor_col is None:
            # 尝试查找因子列
            numeric_cols = factor_data.select_dtypes(include=[np.number]).columns
            potential_cols = [col for col in numeric_cols 
                              if col not in [date_col, symbol_col, "open", "high", "low", "close", 
                                          "volume", "amount", "market_cap", "adj_factor"]]
            if len(potential_cols) >= 1:
                factor_col = potential_cols[0]
                logger.info(f"自动选择因子列: {factor_col}")
            else:
                raise ValueError("找不到因子列，请明确指定factor_col参数")
        
        # 确保日期列是日期类型
        if not pd.api.types.is_datetime64_any_dtype(factor_data[date_col]):
            factor_data[date_col] = pd.to_datetime(factor_data[date_col])
            
        # 处理日期范围
        if start_date is None:
            start_date = factor_data[date_col].min().strftime("%Y-%m-%d")
        if end_date is None:
            end_date = factor_data[date_col].max().strftime("%Y-%m-%d")
            
        # 筛选回测区间的数据
        mask = (factor_data[date_col] >= pd.Timestamp(start_date)) & \
               (factor_data[date_col] <= pd.Timestamp(end_date))
        factor_data = factor_data[mask].copy()
        
        # 筛选股票池
        if universe is not None:
            factor_data = factor_data[factor_data[symbol_col].isin(universe)]
            
        # 获取交易日历
        trade_dates = self.daily_processor.db_client.get_trade_dates(start_date, end_date)
        trade_dates = [pd.Timestamp(d) for d in trade_dates]
        
        # 获取每日股票价格数据
        price_data = self.daily_processor.get_data(
            factor_data[symbol_col].unique().tolist(),
            start_date,
            end_date,
            fields=["symbol", "date", "open", "high", "low", "close", "volume", "amount", "adj_factor"]
        )
        
        # 确保价格数据的日期列是日期类型
        if not pd.api.types.is_datetime64_any_dtype(price_data["date"]):
            price_data["date"] = pd.to_datetime(price_data["date"])
            
        # 如果有复权因子，计算复权价格
        if "adj_factor" in price_data.columns:
            for col in ["open", "high", "low", "close"]:
                if col in price_data.columns:
                    price_data[f"{col}_adj"] = price_data[col] * price_data["adj_factor"]
                    
        # 计算每日收益率
        price_data["return_1d"] = price_data.groupby("symbol")["close_adj"].pct_change()
        
        # 初始化投资组合
        self.portfolio_value = capital
        self.cash = capital
        self.current_positions = {}
        
        # 确定再平衡日期
        rebalance_dates = self._get_rebalance_dates(trade_dates, rebalance_freq)
        
        # 回测主循环
        portfolio_values = []
        positions_history = []
        
        for i, date in enumerate(trade_dates):
            self.current_date = date
            
            # 更新持仓价值
            if i > 0:
                self._update_portfolio_value(price_data, date)
                
            # 记录当日组合价值
            portfolio_values.append({
                "date": date,
                "portfolio_value": self.portfolio_value,
                "cash": self.cash
            })
            
            # 记录持仓
            if self.current_positions:
                for symbol, position in self.current_positions.items():
                    positions_history.append({
                        "date": date,
                        "symbol": symbol,
                        "shares": position["shares"],
                        "cost": position["cost"],
                        "value": position["value"]
                    })
            
            # 如果是再平衡日，调整仓位
            if date in rebalance_dates:
                # 获取当天因子数据
                today_factor = factor_data[factor_data[date_col] == date].copy()
                
                if not today_factor.empty:
                    # 计算分位数
                    try:
                        today_factor["quantile"] = pd.qcut(today_factor[factor_col], 
                                                        quantiles, 
                                                        labels=False, 
                                                        duplicates="drop")
                    except:
                        logger.warning(f"计算分位数失败: {date}, 使用等间距划分")
                        today_factor["quantile"] = pd.cut(today_factor[factor_col], 
                                                       quantiles, 
                                                       labels=False)
                    
                    # 生成目标投资组合
                    target_portfolio = self._generate_target_portfolio(
                        today_factor, price_data, date, quantiles, long_short)
                    
                    # 执行交易
                    self._execute_trades(target_portfolio, price_data, date, 
                                       commission_rate, slippage)
        
        # 计算回测绩效指标
        returns = pd.DataFrame(portfolio_values)
        returns["daily_return"] = returns["portfolio_value"].pct_change()
        
        # 获取基准收益
        if benchmark:
            benchmark_data = self.daily_processor.get_data(
                benchmark, start_date, end_date, 
                fields=["symbol", "date", "close"]
            )
            benchmark_data["return"] = benchmark_data["close"].pct_change()
            benchmark_returns = benchmark_data.set_index("date")["return"]
            
            # 合并组合和基准收益
            returns = returns.set_index("date")
            returns["benchmark_return"] = benchmark_returns
            returns = returns.reset_index()
        
        # 计算性能指标
        self.performance_metrics = self._calculate_performance_metrics(returns)
        
        # 构建结果
        self.results = {
            "returns": returns,
            "positions": pd.DataFrame(positions_history),
            "trades": pd.DataFrame(self.trades),
            "performance": self.performance_metrics,
            "factor_analysis": self._analyze_factor(factor_data, price_data, factor_col, date_col, symbol_col),
            "config": {
                "start_date": start_date,
                "end_date": end_date,
                "quantiles": quantiles,
                "holding_period": holding_period,
                "long_short": long_short,
                "commission_rate": commission_rate,
                "slippage": slippage,
                "capital": capital,
                "benchmark": benchmark,
                "rebalance_freq": rebalance_freq
            }
        }
        
        return self.results
        
    def _reset_backtest(self):
        """重置回测状态"""
        self.results = {}
        self.positions = {}
        self.trades = []
        self.performance_metrics = {}
        self.current_date = None
        self.current_positions = {}
        self.portfolio_value = None
        self.cash = None
        
    def _get_rebalance_dates(self, trade_dates: List[pd.Timestamp], 
                           rebalance_freq: str) -> List[pd.Timestamp]:
        """根据再平衡频率获取再平衡日期
        
        Args:
            trade_dates: 交易日列表
            rebalance_freq: 再平衡频率
            
        Returns:
            List[pd.Timestamp]: 再平衡日期列表
        """
        if rebalance_freq == "daily":
            return trade_dates
        
        rebalance_dates = []
        
        if rebalance_freq == "weekly":
            # 每周最后一个交易日再平衡
            for i, date in enumerate(trade_dates):
                if i == len(trade_dates) - 1 or date.week != trade_dates[i+1].week:
                    rebalance_dates.append(date)
                    
        elif rebalance_freq == "monthly":
            # 每月最后一个交易日再平衡
            for i, date in enumerate(trade_dates):
                if i == len(trade_dates) - 1 or date.month != trade_dates[i+1].month:
                    rebalance_dates.append(date)
        
        elif rebalance_freq == "quarterly":
            # 每季度最后一个交易日再平衡
            for i, date in enumerate(trade_dates):
                if i == len(trade_dates) - 1 or date.quarter != trade_dates[i+1].quarter:
                    rebalance_dates.append(date)
                    
        else:
            raise ValueError(f"不支持的再平衡频率: {rebalance_freq}")
            
        return rebalance_dates
    
    def _update_portfolio_value(self, price_data: pd.DataFrame, date: pd.Timestamp):
        """更新投资组合价值
        
        Args:
            price_data: 价格数据
            date: 当前日期
        """
        if not self.current_positions:
            return
        
        # 获取当日价格
        today_prices = price_data[price_data["date"] == date]
        
        # 更新每只股票的价值
        portfolio_value = self.cash
        
        for symbol, position in self.current_positions.items():
            # 查找今日价格
            symbol_price = today_prices[today_prices["symbol"] == symbol]
            
            if not symbol_price.empty:
                # 更新股票价值
                close_price = symbol_price["close_adj"].values[0]
                position["price"] = close_price
                position["value"] = position["shares"] * close_price
                
                # 更新组合价值
                portfolio_value += position["value"]
            
        # 更新组合总价值
        self.portfolio_value = portfolio_value
    
    def _generate_target_portfolio(self, factor_data: pd.DataFrame, 
                                 price_data: pd.DataFrame,
                                 date: pd.Timestamp,
                                 quantiles: int,
                                 long_short: bool) -> Dict[str, float]:
        """生成目标投资组合
        
        Args:
            factor_data: 因子数据
            price_data: 价格数据
            date: 当前日期
            quantiles: 分位数数量
            long_short: 是否做多做空
            
        Returns:
            Dict[str, float]: 股票代码到目标权重的映射
        """
        # 获取当日价格数据
        today_prices = price_data[price_data["date"] == date]
        
        # 初始化目标组合
        target_portfolio = {}
        
        if long_short:
            # 多空组合：做多最高分位，做空最低分位
            long_stocks = factor_data[factor_data["quantile"] == quantiles - 1]["symbol"].tolist()
            short_stocks = factor_data[factor_data["quantile"] == 0]["symbol"].tolist()
            
            # 计算做多做空权重
            long_weight = 1.0 / len(long_stocks) if long_stocks else 0
            short_weight = -1.0 / len(short_stocks) if short_stocks else 0
            
            # 设置目标权重
            for symbol in long_stocks:
                target_portfolio[symbol] = long_weight
                
            for symbol in short_stocks:
                target_portfolio[symbol] = short_weight
        else:
            # 只做多：做多最高分位
            long_stocks = factor_data[factor_data["quantile"] == quantiles - 1]["symbol"].tolist()
            
            # 计算做多权重
            long_weight = 1.0 / len(long_stocks) if long_stocks else 0
            
            # 设置目标权重
            for symbol in long_stocks:
                target_portfolio[symbol] = long_weight
                
        return target_portfolio
    
    def _execute_trades(self, target_portfolio: Dict[str, float],
                       price_data: pd.DataFrame,
                       date: pd.Timestamp,
                       commission_rate: float,
                       slippage: float):
        """执行交易
        
        Args:
            target_portfolio: 目标投资组合
            price_data: 价格数据
            date: 当前日期
            commission_rate: 佣金率
            slippage: 滑点率
        """
        # 获取当日价格
        today_prices = price_data[price_data["date"] == date]
        
        # 计算目标持仓数量
        target_positions = {}
        
        for symbol, weight in target_portfolio.items():
            # 查找当日价格
            symbol_price = today_prices[today_prices["symbol"] == symbol]
            
            if not symbol_price.empty:
                # 计算目标持仓金额和数量
                price = symbol_price["close_adj"].values[0]
                target_value = self.portfolio_value * weight
                target_shares = int(target_value / price)  # 简化为整数股
                
                if target_shares != 0:
                    target_positions[symbol] = {
                        "shares": target_shares,
                        "price": price,
                        "value": target_shares * price,
                        "weight": weight,
                        "cost": target_shares * price * (1 + np.sign(target_shares) * (commission_rate + slippage))
                    }
        
        # 执行交易
        for symbol, target in target_positions.items():
            if symbol in self.current_positions:
                # 已有持仓，调整数量
                current = self.current_positions[symbol]
                shares_diff = target["shares"] - current["shares"]
                
                if shares_diff != 0:
                    # 交易成本
                    trade_cost = abs(shares_diff) * target["price"] * (commission_rate + slippage)
                    
                    # 记录交易
                    self.trades.append({
                        "date": date,
                        "symbol": symbol,
                        "shares": shares_diff,
                        "price": target["price"],
                        "cost": trade_cost,
                        "type": "buy" if shares_diff > 0 else "sell"
                    })
                    
                    # 更新持仓
                    if target["shares"] == 0:
                        # 清仓
                        self.cash += current["shares"] * target["price"] - trade_cost
                        del self.current_positions[symbol]
                    else:
                        # 调整持仓
                        trade_value = shares_diff * target["price"]
                        self.cash -= (trade_value + trade_cost)
                        current["shares"] = target["shares"]
                        current["price"] = target["price"]
                        current["value"] = target["value"]
                        current["cost"] += trade_cost
            else:
                # 新建持仓
                if target["shares"] != 0:
                    # 交易成本
                    trade_cost = abs(target["shares"]) * target["price"] * (commission_rate + slippage)
                    
                    # 记录交易
                    self.trades.append({
                        "date": date,
                        "symbol": symbol,
                        "shares": target["shares"],
                        "price": target["price"],
                        "cost": trade_cost,
                        "type": "buy" if target["shares"] > 0 else "sell"
                    })
                    
                    # 更新持仓和现金
                    self.current_positions[symbol] = target
                    self.cash -= (target["value"] + trade_cost)
                    self.current_positions[symbol]["cost"] = trade_cost
        
        # 处理需要清仓的股票
        for symbol in list(self.current_positions.keys()):
            if symbol not in target_positions:
                # 获取价格
                symbol_price = today_prices[today_prices["symbol"] == symbol]
                
                if not symbol_price.empty:
                    price = symbol_price["close_adj"].values[0]
                    position = self.current_positions[symbol]
                    
                    # 交易成本
                    trade_cost = abs(position["shares"]) * price * (commission_rate + slippage)
                    
                    # 记录交易
                    self.trades.append({
                        "date": date,
                        "symbol": symbol,
                        "shares": -position["shares"],
                        "price": price,
                        "cost": trade_cost,
                        "type": "sell"
                    })
                    
                    # 更新现金
                    self.cash += position["shares"] * price - trade_cost
                    
                    # 移除持仓
                    del self.current_positions[symbol]
    
    def _calculate_performance_metrics(self, returns: pd.DataFrame) -> Dict[str, float]:
        """计算绩效指标
        
        Args:
            returns: 收益率数据
            
        Returns:
            Dict[str, float]: 绩效指标
        """
        # 计算累积收益率
        returns["cumulative_return"] = (1 + returns["daily_return"]).cumprod() - 1
        
        # 计算年化收益率
        days = (returns["date"].max() - returns["date"].min()).days
        annualized_return = (1 + returns["cumulative_return"].iloc[-1]) ** (365 / days) - 1
        
        # 计算波动率
        volatility = returns["daily_return"].std() * np.sqrt(252)
        
        # 计算夏普比率
        risk_free_rate = 0.03  # 假设年化无风险利率3%
        daily_risk_free = risk_free_rate / 252
        sharpe_ratio = (returns["daily_return"].mean() - daily_risk_free) / returns["daily_return"].std() * np.sqrt(252)
        
        # 计算最大回撤
        cumulative = (1 + returns["daily_return"]).cumprod()
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()
        
        # 计算Alpha和Beta
        if "benchmark_return" in returns.columns:
            # 计算Beta
            covariance = np.cov(returns["daily_return"].dropna(), returns["benchmark_return"].dropna())[0, 1]
            benchmark_variance = returns["benchmark_return"].var()
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
            
            # 计算Alpha
            benchmark_annualized = (1 + returns["benchmark_return"]).cumprod().iloc[-1] ** (365 / days) - 1
            alpha = annualized_return - risk_free_rate - beta * (benchmark_annualized - risk_free_rate)
            
            # 计算信息比率
            tracking_error = (returns["daily_return"] - beta * returns["benchmark_return"]).std() * np.sqrt(252)
            information_ratio = (annualized_return - benchmark_annualized) / tracking_error if tracking_error != 0 else 0
        else:
            beta = 0
            alpha = 0
            information_ratio = 0
            
        # 计算胜率
        win_rate = (returns["daily_return"] > 0).mean()
        
        # 计算盈亏比
        profit_trades = returns[returns["daily_return"] > 0]["daily_return"]
        loss_trades = returns[returns["daily_return"] < 0]["daily_return"]
        profit_loss_ratio = abs(profit_trades.mean() / loss_trades.mean()) if len(loss_trades) > 0 and loss_trades.mean() != 0 else 0
        
        metrics = {
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "beta": beta,
            "alpha": alpha,
            "information_ratio": information_ratio,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "cumulative_return": returns["cumulative_return"].iloc[-1],
            "daily_return_mean": returns["daily_return"].mean(),
            "daily_return_std": returns["daily_return"].std()
        }
        
        return metrics
    
    def _analyze_factor(self, factor_data: pd.DataFrame, 
                      price_data: pd.DataFrame,
                      factor_col: str,
                      date_col: str,
                      symbol_col: str) -> Dict[str, Any]:
        """分析因子性能
        
        Args:
            factor_data: 因子数据
            price_data: 价格数据
            factor_col: 因子列名
            date_col: 日期列名
            symbol_col: 股票代码列名
            
        Returns:
            Dict[str, Any]: 因子分析结果
        """
        # 合并因子和收益率数据
        merged_data = pd.merge(
            factor_data,
            price_data[["symbol", "date", "return_1d"]],
            on=["symbol", "date"],
            how="inner"
        )
        
        # 计算IC
        ic_results = self.ic_analyzer.calculate_ic(
            merged_data,
            factor_col=factor_col,
            forward_periods=[1, 5, 10, 20],
            return_col="return_1d",
            date_col=date_col,
            symbol_col=symbol_col
        )
        
        # 计算分位数收益
        quantile_results = self.quantile_analyzer.calculate_quantile_returns(
            merged_data,
            factor_col=factor_col,
            n_quantiles=5,
            forward_periods=[1, 5, 10, 20],
            price_col="return_1d",
            date_col=date_col,
            symbol_col=symbol_col
        )
        
        # 提取关键指标
        ic_stats = {}
        for period, result in ic_results.items():
            ic_stats[f"{period}d"] = result["ic_stats"]
            
        quantile_stats = {}
        for period, result in quantile_results.items():
            if result:
                stats = self.quantile_analyzer.summary()
                quantile_stats[f"{period}d"] = stats
                
        return {
            "ic": ic_stats,
            "quantile": quantile_stats
        }
    
    def get_returns(self) -> pd.DataFrame:
        """获取回测收益率数据
        
        Returns:
            pd.DataFrame: 收益率数据
        """
        if not self.results:
            return pd.DataFrame()
            
        return self.results["returns"]
    
    def get_positions(self) -> pd.DataFrame:
        """获取回测持仓数据
        
        Returns:
            pd.DataFrame: 持仓数据
        """
        if not self.results:
            return pd.DataFrame()
            
        return self.results["positions"]
    
    def get_trades(self) -> pd.DataFrame:
        """获取回测交易数据
        
        Returns:
            pd.DataFrame: 交易数据
        """
        if not self.results:
            return pd.DataFrame()
            
        return self.results["trades"]
    
    def get_performance(self) -> Dict[str, float]:
        """获取回测绩效指标
        
        Returns:
            Dict[str, float]: 绩效指标
        """
        if not self.results:
            return {}
            
        return self.results["performance"]
    
    def get_factor_analysis(self) -> Dict[str, Any]:
        """获取因子分析结果
        
        Returns:
            Dict[str, Any]: 因子分析结果
        """
        if not self.results:
            return {}
            
        return self.results["factor_analysis"]
        
    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """生成回测报告
        
        Args:
            output_path: 报告输出路径
            
        Returns:
            Dict[str, Any]: 报告数据
        """
        if not self.results:
            raise ValueError("需要先运行回测才能生成报告")
            
        # 构建报告数据
        report = {
            "summary": {
                "start_date": self.results["config"]["start_date"],
                "end_date": self.results["config"]["end_date"],
                "initial_capital": self.results["config"]["capital"],
                "final_value": self.results["returns"]["portfolio_value"].iloc[-1],
                "total_return": self.results["performance"]["cumulative_return"],
                "annualized_return": self.results["performance"]["annualized_return"],
                "sharpe_ratio": self.results["performance"]["sharpe_ratio"],
                "max_drawdown": self.results["performance"]["max_drawdown"],
                "win_rate": self.results["performance"]["win_rate"]
            },
            "performance": self.results["performance"],
            "factor_analysis": self.results["factor_analysis"],
            "daily_returns": self.results["returns"][["date", "portfolio_value", "daily_return", "cumulative_return"]].to_dict("records"),
            "monthly_returns": self._calculate_monthly_returns(),
            "drawdowns": self._calculate_drawdowns(),
            "top_holdings": self._get_top_holdings(),
            "sector_exposure": self._get_sector_exposure(),
            "turnover": self._calculate_turnover()
        }
        
        # 如果提供了输出路径，保存报告
        if output_path:
            import json
            
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存JSON报告
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
                
            logger.info(f"回测报告已保存到: {output_path}")
            
        return report
    
    def _calculate_monthly_returns(self) -> List[Dict[str, Any]]:
        """计算月度收益率
        
        Returns:
            List[Dict[str, Any]]: 月度收益率数据
        """
        if not self.results:
            return []
            
        # 转换日期为月度
        returns = self.results["returns"].copy()
        returns["year_month"] = returns["date"].dt.strftime("%Y-%m")
        
        # 计算月度收益率
        monthly_returns = []
        
        for year_month, group in returns.groupby("year_month"):
            first_value = group["portfolio_value"].iloc[0]
            last_value = group["portfolio_value"].iloc[-1]
            monthly_return = (last_value / first_value) - 1
            
            monthly_returns.append({
                "year_month": year_month,
                "return": monthly_return,
                "start_value": first_value,
                "end_value": last_value
            })
            
        return monthly_returns
    
    def _calculate_drawdowns(self) -> List[Dict[str, Any]]:
        """计算回撤
        
        Returns:
            List[Dict[str, Any]]: 回撤数据
        """
        if not self.results:
            return []
            
        returns = self.results["returns"].copy()
        
        # 计算累积收益
        if "cumulative_return" not in returns.columns:
            returns["cumulative_return"] = (1 + returns["daily_return"]).cumprod() - 1
            
        # 计算回撤
        returns["underwater"] = 1 - (1 + returns["cumulative_return"]) / (1 + returns["cumulative_return"]).cummax()
        
        # 识别回撤区间
        drawdowns = []
        current_drawdown = None
        
        for _, row in returns.iterrows():
            if row["underwater"] == 0:
                if current_drawdown:
                    # 回撤结束
                    current_drawdown["end_date"] = row["date"]
                    current_drawdown["recovery_date"] = row["date"]
                    current_drawdown["duration"] = (current_drawdown["end_date"] - current_drawdown["start_date"]).days
                    current_drawdown["recovery_duration"] = (current_drawdown["recovery_date"] - current_drawdown["end_date"]).days
                    
                    drawdowns.append(current_drawdown)
                    current_drawdown = None
            else:
                if current_drawdown is None:
                    # 新回撤开始
                    current_drawdown = {
                        "start_date": row["date"],
                        "max_drawdown": row["underwater"],
                        "max_drawdown_date": row["date"]
                    }
                elif row["underwater"] > current_drawdown["max_drawdown"]:
                    # 更新最大回撤
                    current_drawdown["max_drawdown"] = row["underwater"]
                    current_drawdown["max_drawdown_date"] = row["date"]
        
        # 排序回撤，按最大回撤大小
        drawdowns.sort(key=lambda x: x["max_drawdown"], reverse=True)
        
        return drawdowns[:10]  # 返回前10个最大回撤
    
    def _get_top_holdings(self) -> List[Dict[str, Any]]:
        """获取top持仓
        
        Returns:
            List[Dict[str, Any]]: top持仓数据
        """
        if not self.results or "positions" not in self.results:
            return []
            
        # 获取最后一个日期的持仓
        positions = self.results["positions"]
        last_date = positions["date"].max()
        last_positions = positions[positions["date"] == last_date]
        
        # 按持仓价值排序
        last_positions = last_positions.sort_values("value", ascending=False)
        
        # 转换为列表
        top_holdings = []
        for _, row in last_positions.iterrows():
            top_holdings.append({
                "symbol": row["symbol"],
                "shares": row["shares"],
                "value": row["value"],
                "weight": row["value"] / last_positions["value"].sum()
            })
            
        return top_holdings
    
    def _get_sector_exposure(self) -> List[Dict[str, Any]]:
        """获取行业暴露
        
        Returns:
            List[Dict[str, Any]]: 行业暴露数据
        """
        # 此功能需要行业数据，暂不实现
        return []
    
    def _calculate_turnover(self) -> List[Dict[str, float]]:
        """计算换手率
        
        Returns:
            List[Dict[str, float]]: 换手率数据
        """
        if not self.results or "trades" not in self.results:
            return []
            
        # 获取交易数据
        trades = self.results["trades"]
        returns = self.results["returns"]
        
        # 按日期分组计算换手率
        turnover = []
        
        for date, date_returns in returns.groupby("date"):
            # 获取当日交易
            date_trades = trades[trades["date"] == date]
            
            if not date_trades.empty:
                # 计算交易金额
                trade_value = (date_trades["shares"].abs() * date_trades["price"]).sum()
                
                # 计算当日投资组合价值
                portfolio_value = date_returns["portfolio_value"].values[0]
                
                # 计算换手率
                turnover_rate = trade_value / (2 * portfolio_value) if portfolio_value > 0 else 0
                
                turnover.append({
                    "date": date,
                    "turnover_rate": turnover_rate,
                    "trade_value": trade_value,
                    "portfolio_value": portfolio_value
                })
                
        return turnover


def backtest_factor(factor_data, **kwargs):
    """回测因子的便捷函数
    
    Args:
        factor_data: 因子数据
        **kwargs: 回测参数
        
    Returns:
        Dict[str, Any]: 回测结果
    """
    engine = BacktestEngine()
    return engine.backtest_factor(factor_data, **kwargs)