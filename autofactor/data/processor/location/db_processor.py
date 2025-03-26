"""
DolphinDB数据处理器 - 在DolphinDB服务器上处理数据
"""
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from autofactor.data.dolphindb_client import DolphinDBClient
from autofactor.data.processor.base import DataProcessor
from config.settings import DOLPHINDB


class DBProcessor(DataProcessor):
    """DolphinDB数据处理器，将数据处理任务下发到DolphinDB服务器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化DolphinDB数据处理器"""
        super().__init__(config)
        # 初始化DolphinDB客户端
        self.db_client = DolphinDBClient()
        self.script_templates = self._init_script_templates()
        
    def _init_script_templates(self) -> Dict[str, str]:
        """初始化DolphinDB脚本模板"""
        templates = {
            "get_data": """
            def getStockData(symbols, startDate, endDate, fields=NULL, adjust=true) {{
                // 将日期转换为字符串
                start = temporalParse('{start_date}', 'yyyy-MM-dd')
                end = temporalParse('{end_date}', 'yyyy-MM-dd')
                
                // 构建SQL查询
                if(fields==NULL){{
                    fields = *
                }}
                
                // 构建股票列表
                symbolList = {symbols}
                
                // 加载表并过滤数据
                t = loadTable('{db_path}', '{table_name}')
                
                // 执行查询
                result = select {fields} from t 
                        where date between start:end, symbol in symbolList
                
                // 执行复权处理
                if(adjust && result.schema().colNames.contain("adj_factor")){{
                    cols = ["open", "high", "low", "close"]
                    for(col in cols){{
                        if(col in result.schema().colNames){{
                            result[col+"_adj"] = result[col] * result["adj_factor"]
                        }}
                    }}
                }}
                
                return result
            }}
            
            getStockData({symbols}, '{start_date}', '{end_date}', {fields}, {adjust})
            """,
            
            "resample": """
            def resampleData(t, freq, method) {{
                // 确保日期列已排序
                t = t.sortBy!(`date)
                
                // 按股票分组处理
                groups = t.groupBy(`symbol)
                
                if(method == "ohlc") {{
                    // 针对OHLC数据的特殊处理
                    result = select 
                        first(open) as open, 
                        max(high) as high, 
                        min(low) as low, 
                        last(close) as close, 
                        sum(volume) as volume,
                        sum(amount) as amount
                    from t
                    group by symbol, date.resample(freq) as date
                }}
                else if(method == "vwap") {{
                    // 成交量加权平均价格
                    result = select 
                        sum(close * volume) / sum(volume) as vwap,
                        sum(volume) as volume
                    from t
                    group by symbol, date.resample(freq) as date
                }}
                else if(method == "last") {{
                    // 使用最后一个值
                    agg_cols = dict()
                    for(col in t.schema().colNames) {{
                        if(col != `symbol && col != `date) {{
                            agg_cols[col] = <last>
                        }}
                    }}
                    
                    result = select {cols} from t
                    group by symbol, date.resample(freq) as date
                }}
                
                return result
            }}
            
            resampleData({table}, '{freq}', '{method}')
            """,
            
            "handle_missing_values": """
            def handleMissingValues(t, method, limit={limit}) {{
                // 按股票分组处理
                groups = t.groupBy(`symbol)
                
                if(method == "ffill") {{
                    result = groups.eachGroup(x -> {{
                        x = x.sortBy(`date)
                        for(col in x.schema().colNames) {{
                            if(col != `symbol && col != `date) {{
                                x[col] = x[col].nullFill(mode=`forward, limit=limit)
                            }}
                        }}
                        return x
                    }})
                }}
                else if(method == "bfill") {{
                    result = groups.eachGroup(x -> {{
                        x = x.sortBy(`date)
                        for(col in x.schema().colNames) {{
                            if(col != `symbol && col != `date) {{
                                x[col] = x[col].nullFill(mode=`backward, limit=limit)
                            }}
                        }}
                        return x
                    }})
                }}
                else if(method == "zero") {{
                    // 用0填充缺失值
                    result = t
                    for(col in t.schema().colNames) {{
                        if(col != `symbol && col != `date) {{
                            result[col] = result[col].nullFill(0)
                        }}
                    }}
                }}
                else if(method == "mean") {{
                    // 用均值填充缺失值
                    result = groups.eachGroup(x -> {{
                        for(col in x.schema().colNames) {{
                            if(col != `symbol && col != `date) {{
                                mean_val = mean(x[col])
                                x[col] = x[col].nullFill(mean_val)
                            }}
                        }}
                        return x
                    }})
                }}
                
                return result
            }}
            
            handleMissingValues({table}, '{method}', {limit})
            """,
            
            "normalize": """
            def normalizeData(t, method, byCS, date_col='date', symbol_col='symbol') {{
                if(byCS) {{
                    // 按横截面处理
                    result = t.copy()
                    dates = t[date_col].distinct()
                    
                    for(d in dates) {{
                        date_data = t[t[date_col] == d]
                        
                        // 选择需要标准化的列
                        cols = date_data.schema().colNames
                        numeric_cols = []
                        
                        for(col in cols) {{
                            if(col != date_col && col != symbol_col) {{
                                try {{
                                    // 检查列是否是数值类型
                                    if(date_data[col].typeString().find("DOUBLE") >= 0 || 
                                       date_data[col].typeString().find("FLOAT") >= 0 || 
                                       date_data[col].typeString().find("INT") >= 0) {{
                                        numeric_cols.append(col)
                                    }}
                                }} catch(ex) {{}}
                            }}
                        }}
                        
                        // 对每个数值列进行标准化
                        for(col in numeric_cols) {{
                            if(method == "zscore") {{
                                mean_val = mean(date_data[col])
                                std_val = std(date_data[col])
                                if(std_val != 0) {{
                                    result[col] = iif(result[date_col] == d, (date_data[col] - mean_val) / std_val, result[col])
                                }}
                            }}
                            else if(method == "rank") {{
                                ranks = date_data[col].rank() / count(date_data[col])
                                result[col] = iif(result[date_col] == d, ranks, result[col])
                            }}
                            else if(method == "min_max") {{
                                min_val = min(date_data[col])
                                max_val = max(date_data[col])
                                if(max_val > min_val) {{
                                    result[col] = iif(result[date_col] == d, (date_data[col] - min_val) / (max_val - min_val), result[col])
                                }}
                            }}
                        }}
                    }}
                }} else {{
                    // 按时间序列处理
                    result = t.copy()
                    
                    // 按股票分组处理
                    symbols = t[symbol_col].distinct()
                    
                    for(sym in symbols) {{
                        sym_data = t[t[symbol_col] == sym]
                        
                        // 选择需要标准化的列
                        cols = sym_data.schema().colNames
                        numeric_cols = []
                        
                        for(col in cols) {{
                            if(col != date_col && col != symbol_col) {{
                                try {{
                                    // 检查列是否是数值类型
                                    if(sym_data[col].typeString().find("DOUBLE") >= 0 || 
                                       sym_data[col].typeString().find("FLOAT") >= 0 || 
                                       sym_data[col].typeString().find("INT") >= 0) {{
                                        numeric_cols.append(col)
                                    }}
                                }} catch(ex) {{}}
                            }}
                        }}
                        
                        // 对每个数值列进行标准化
                        for(col in numeric_cols) {{
                            if(method == "zscore") {{
                                mean_val = mean(sym_data[col])
                                std_val = std(sym_data[col])
                                if(std_val != 0) {{
                                    result[col] = iif(result[symbol_col] == sym, (sym_data[col] - mean_val) / std_val, result[col])
                                }}
                            }}
                            else if(method == "rank") {{
                                ranks = sym_data[col].rank() / count(sym_data[col])
                                result[col] = iif(result[symbol_col] == sym, ranks, result[col])
                            }}
                            else if(method == "min_max") {{
                                min_val = min(sym_data[col])
                                max_val = max(sym_data[col])
                                if(max_val > min_val) {{
                                    result[col] = iif(result[symbol_col] == sym, (sym_data[col] - min_val) / (max_val - min_val), result[col])
                                }}
                            }}
                        }}
                    }}
                }}
                
                return result
            }}
            
            normalizeData({table}, '{method}', {by_cross_section}, '{date_col}', '{symbol_col}')
            """,
            
            "industry_neutralize": """
            def industryNeutralize(t, factor_col, industry_col, date_col='date') {{
                // 按日期和行业分组处理
                dates = t[date_col].distinct()
                result = t.copy()
                
                for(d in dates) {{
                    date_data = t[t[date_col] == d]
                    industries = date_data[industry_col].distinct()
                    
                    for(ind in industries) {{
                        ind_data = date_data[date_data[industry_col] == ind]
                        
                        // 计算行业均值
                        ind_mean = mean(ind_data[factor_col])
                        
                        // 更新结果表中对应行业的因子值
                        result[factor_col + "_neutral"] = iif(
                            result[date_col] == d && result[industry_col] == ind,
                            ind_data[factor_col] - ind_mean,
                            result[factor_col + "_neutral"]
                        )
                    }}
                }}
                
                return result
            }}
            
            industryNeutralize({table}, '{factor_col}', '{industry_col}', '{date_col}')
            """
        }
        
        return templates
    
    def get_data(self, symbols: Union[str, List[str]], 
                start_date: str, 
                end_date: str,
                fields: Optional[List[str]] = None,
                freq: str = "day",
                adjust: bool = True) -> pd.DataFrame:
        """从DolphinDB获取数据并在服务器端处理
        
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
        # 处理symbols参数
        if isinstance(symbols, str):
            symbols_str = f"['{symbols}']"
        else:
            symbols_str = str(symbols).replace("[", "[").replace("]", "]").replace("'", "'")
            
        # 处理fields参数
        fields_str = "*" if fields is None else ", ".join(fields)
            
        # 确定表名
        table_name = "daily_quote" if freq == "day" else "minute_quote"
            
        # 构建脚本
        script = self.script_templates["get_data"].format(
            symbols=symbols_str,
            start_date=start_date,
            end_date=end_date,
            fields=fields_str,
            adjust=str(adjust).lower(),
            db_path=DOLPHINDB["db_path"],
            table_name=table_name
        )
            
        # 执行脚本
        try:
            logger.debug(f"执行DolphinDB脚本获取数据:\n{script}")
            result = self.db_client.execute(script)
            
            # 转换为pandas DataFrame
            if isinstance(result, pd.DataFrame):
                return result
            else:
                logger.warning(f"DolphinDB返回值类型不是DataFrame: {type(result)}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"从DolphinDB获取数据失败: {e}")
            raise
    
    def resample(self, data: pd.DataFrame, 
                target_freq: str = "day", 
                method: str = "ohlc") -> pd.DataFrame:
        """将数据重采样到指定频率
        
        在DolphinDB服务器上执行重采样操作
        
        Args:
            data: 输入数据，必须包含date和symbol列
            target_freq: 目标频率，例如'D'(日),'W'(周),'M'(月)
            method: 重采样方法，'ohlc'(开高低收),'vwap'(成交量加权),'last'(最后值)
            
        Returns:
            pd.DataFrame: 重采样后的数据
        """
        # 将数据上传到DolphinDB临时表
        temp_table_name = f"tmp_resample_{id(data)}"
        
        # 将DataFrame上传到DolphinDB
        self.db_client.conn.upload({"data": data})
        self.db_client.execute(f"t = table(data); share t as {temp_table_name}")
        
        # 确定重采样频率映射
        freq_map = {
            "day": "1d",
            "week": "1w",
            "month": "1m",
            "quarter": "1q",
            "year": "1y",
            "hour": "1h",
            "minute": "1m",
            "D": "1d",
            "W": "1w",
            "M": "1m",
            "Q": "1q",
            "Y": "1y",
            "H": "1h",
            "min": "1m"
        }
        
        dolphin_freq = freq_map.get(target_freq, target_freq)
        
        # 构建重采样脚本
        cols = ", ".join([f"last({col}) as {col}" if col not in ["symbol", "date"] else col for col in data.columns])
        script = self.script_templates["resample"].format(
            table=temp_table_name,
            freq=dolphin_freq,
            method=method,
            cols=cols
        )
        
        # 执行脚本
        try:
            logger.debug(f"执行DolphinDB脚本重采样数据:\n{script}")
            result = self.db_client.execute(script)
            
            # 删除临时表
            self.db_client.execute(f"undef('{temp_table_name}', SHARED)")
            
            # 转换为pandas DataFrame
            if isinstance(result, pd.DataFrame):
                return result
            else:
                logger.warning(f"DolphinDB返回值类型不是DataFrame: {type(result)}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"DolphinDB重采样操作失败: {e}")
            
            # 确保临时表被删除
            try:
                self.db_client.execute(f"undef('{temp_table_name}', SHARED)")
            except:
                pass
                
            raise
    
    def handle_missing_values(self, data: pd.DataFrame, 
                             method: str = "ffill",
                             limit: Optional[int] = None) -> pd.DataFrame:
        """处理缺失值
        
        在DolphinDB服务器上执行缺失值处理
        
        Args:
            data: 输入数据
            method: 处理方法，可选值为 "ffill", "bfill", "zero", "mean"
            limit: 最大填充长度
            
        Returns:
            pd.DataFrame: 处理缺失值后的数据
        """
        # 将数据上传到DolphinDB临时表
        temp_table_name = f"tmp_missval_{id(data)}"
        
        # 将DataFrame上传到DolphinDB
        self.db_client.conn.upload({"data": data})
        self.db_client.execute(f"t = table(data); share t as {temp_table_name}")
        
        # 构建处理脚本
        limit_val = "NULL" if limit is None else str(limit)
        script = self.script_templates["handle_missing_values"].format(
            table=temp_table_name,
            method=method,
            limit=limit_val
        )
        
        # 执行脚本
        try:
            logger.debug(f"执行DolphinDB脚本处理缺失值:\n{script}")
            result = self.db_client.execute(script)
            
            # 删除临时表
            self.db_client.execute(f"undef('{temp_table_name}', SHARED)")
            
            # 转换为pandas DataFrame
            if isinstance(result, pd.DataFrame):
                return result
            else:
                logger.warning(f"DolphinDB返回值类型不是DataFrame: {type(result)}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"DolphinDB处理缺失值操作失败: {e}")
            
            # 确保临时表被删除
            try:
                self.db_client.execute(f"undef('{temp_table_name}', SHARED)")
            except:
                pass
                
            raise
    
    def handle_outliers(self, data: pd.DataFrame,
                       method: str = "winsorize") -> pd.DataFrame:
        """处理异常值
        
        Args:
            data: 输入数据
            method: 处理方法
            
        Returns:
            pd.DataFrame: 处理异常值后的数据
        """
        # 此处暂不实现DolphinDB版本的异常值处理
        # 因为复杂度较高，后续可根据需要添加
        logger.warning("DolphinDB处理器暂不支持异常值处理，返回原始数据")
        return data
    
    def normalize(self, data: pd.DataFrame, 
                 method: str = "zscore",
                 by_cross_section: bool = True,
                 date_col: str = "date",
                 symbol_col: str = "symbol") -> pd.DataFrame:
        """标准化处理
        
        在DolphinDB服务器上执行标准化处理
        
        Args:
            data: 输入数据
            method: 标准化方法，可选值为 "zscore", "rank", "min_max"
            by_cross_section: 是否按横截面处理
            date_col: 日期列名
            symbol_col: 股票代码列名
            
        Returns:
            pd.DataFrame: 标准化后的数据
        """
        # 将数据上传到DolphinDB临时表
        temp_table_name = f"tmp_norm_{id(data)}"
        
        # 将DataFrame上传到DolphinDB
        self.db_client.conn.upload({"data": data})
        self.db_client.execute(f"t = table(data); share t as {temp_table_name}")
        
        # 构建标准化脚本
        script = self.script_templates["normalize"].format(
            table=temp_table_name,
            method=method,
            by_cross_section=str(by_cross_section).lower(),
            date_col=date_col,
            symbol_col=symbol_col
        )
        
        # 执行脚本
        try:
            logger.debug(f"执行DolphinDB脚本标准化数据:\n{script}")
            result = self.db_client.execute(script)
            
            # 删除临时表
            self.db_client.execute(f"undef('{temp_table_name}', SHARED)")
            
            # 转换为pandas DataFrame
            if isinstance(result, pd.DataFrame):
                return result
            else:
                logger.warning(f"DolphinDB返回值类型不是DataFrame: {type(result)}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"DolphinDB标准化操作失败: {e}")
            
            # 确保临时表被删除
            try:
                self.db_client.execute(f"undef('{temp_table_name}', SHARED)")
            except:
                pass
                
            raise
    
    def neutralize(self, factor_data: pd.DataFrame,
                  date_col: str = "date",
                  symbol_col: str = "symbol",
                  factor_col: str = "factor",
                  industry_col: str = "industry") -> pd.DataFrame:
        """因子中性化处理
        
        在DolphinDB服务器上执行因子中性化处理
        
        Args:
            factor_data: 因子数据
            date_col: 日期列名
            symbol_col: 股票代码列名
            factor_col: 因子值列名
            industry_col: 行业列名
            
        Returns:
            pd.DataFrame: 中性化后的因子数据
        """
        # 将数据上传到DolphinDB临时表
        temp_table_name = f"tmp_neutral_{id(factor_data)}"
        
        # 将DataFrame上传到DolphinDB
        self.db_client.conn.upload({"factor_data": factor_data})
        self.db_client.execute(f"t = table(factor_data); share t as {temp_table_name}")
        
        # 构建中性化脚本
        script = self.script_templates["industry_neutralize"].format(
            table=temp_table_name,
            factor_col=factor_col,
            industry_col=industry_col,
            date_col=date_col
        )
        
        # 执行脚本
        try:
            logger.debug(f"执行DolphinDB脚本中性化因子:\n{script}")
            result = self.db_client.execute(script)
            
            # 删除临时表
            self.db_client.execute(f"undef('{temp_table_name}', SHARED)")
            
            # 转换为pandas DataFrame
            if isinstance(result, pd.DataFrame):
                return result
            else:
                logger.warning(f"DolphinDB返回值类型不是DataFrame: {type(result)}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"DolphinDB因子中性化操作失败: {e}")
            
            # 确保临时表被删除
            try:
                self.db_client.execute(f"undef('{temp_table_name}', SHARED)")
            except:
                pass
                
            raise
    
    def compute_factor(self, factor_name: str,
                      symbols: Union[str, List[str]],
                      start_date: str,
                      end_date: str,
                      params: Optional[Dict[str, Any]] = None,
                      freq: str = "day") -> pd.DataFrame:
        """计算指定因子
        
        根据因子名称和参数，在DolphinDB服务器上计算并返回因子值
        
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
        # 此处需要考虑因子是否在DolphinDB脚本库中定义
        # 如果未定义，则降级到Python环境计算
        
        # 暂不实现，返回未实现错误
        logger.warning("DBProcessor暂不支持直接因子计算，请使用HybridProcessor")
        raise NotImplementedError("DBProcessor暂不支持直接因子计算")
        
    def generate_script(self, operation: str, **kwargs) -> str:
        """生成DolphinDB脚本
        
        根据操作类型和参数生成相应的DolphinDB脚本
        
        Args:
            operation: 操作类型
            **kwargs: 操作参数
            
        Returns:
            str: 生成的DolphinDB脚本
        """
        if operation not in self.script_templates:
            raise ValueError(f"不支持的操作类型: {operation}")
            
        template = self.script_templates[operation]
        script = template.format(**kwargs)
        
        return script