"""
DolphinDB客户端模块 - 提供与DolphinDB数据库的交互功能
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import dolphindb as ddb
import numpy as np
import pandas as pd
from loguru import logger

from config.settings import DOLPHINDB


class DolphinDBClient:
    """DolphinDB数据库客户端"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super(DolphinDBClient, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """初始化DolphinDB客户端

        Args:
            host: DolphinDB服务器地址
            port: DolphinDB服务器端口
            username: 用户名
            password: 密码
        """
        self.host = host or DOLPHINDB["host"]
        self.port = port or DOLPHINDB["port"]
        self.username = username or DOLPHINDB["username"]
        self.password = password or DOLPHINDB["password"]
        self.conn = None
        self._connect()

    def _connect(self) -> None:
        """连接DolphinDB服务器"""
        try:
            self.conn = ddb.session()
            self.conn.connect(self.host, self.port, self.username, self.password)
            logger.info(f"成功连接到DolphinDB服务器: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"连接DolphinDB服务器失败: {e}")
            raise ConnectionError(f"连接DolphinDB服务器失败: {e}")

    def reconnect(self) -> None:
        """重新连接DolphinDB服务器"""
        if self.conn:
            try:
                self.conn.close()
            except:
                pass
        self._connect()

    def execute(self, script: str) -> Any:
        """执行DolphinDB脚本

        Args:
            script: DolphinDB脚本

        Returns:
            脚本执行结果
        """
        try:
            return self.conn.run(script)
        except ddb.OperationException as e:
            logger.error(f"DolphinDB脚本执行异常: {e}")
            if "connection" in str(e).lower():
                logger.info("尝试重新连接...")
                self.reconnect()
                return self.conn.run(script)
            raise
        except Exception as e:
            logger.error(f"执行DolphinDB脚本出错: {e}")
            raise

    def upload_dataframe(
        self, df: pd.DataFrame, table_name: str, db_path: Optional[str] = None
    ) -> bool:
        """上传Pandas DataFrame到DolphinDB表

        Args:
            df: Pandas DataFrame
            table_name: 目标表名
            db_path: 数据库路径，默认使用配置中的路径

        Returns:
            bool: 上传是否成功
        """
        if db_path is None:
            db_path = DOLPHINDB["db_path"]

        try:
            # 检查数据库是否存在，不存在则创建
            self.execute(f"""
            if(!existsDatabase("{db_path}")){{
                db = database("{db_path}", VALUE, 2018.01.01..2030.12.31)
            }}
            """)

            # 检查表是否存在，不存在则创建
            cols = ", ".join([f"{col} {self._map_dtype(df[col].dtype)}" for col in df.columns])
            self.execute(f"""
            if(!existsTable("{db_path}", "{table_name}")){{
                db = database("{db_path}")
                tb = db.createTable(table=`{table_name}, `{cols})
            }}
            """)

            # 上传数据
            batch_size = DOLPHINDB["batch_size"]
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i + batch_size]
                self.conn.upload({"batch_df": batch_df})
                self.execute(f"""
                loadTable("{db_path}", "{table_name}").append!(batch_df)
                """)
            
            logger.info(f"成功上传 {len(df)} 行数据到 {db_path}/{table_name}")
            return True
        except Exception as e:
            logger.error(f"上传数据到DolphinDB失败: {e}")
            return False

    def query_table(
        self, 
        table_name: str, 
        columns: Optional[List[str]] = None,
        condition: Optional[str] = None,
        db_path: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """查询DolphinDB表数据

        Args:
            table_name: 表名
            columns: 需要查询的列，默认查询所有列
            condition: 查询条件，例如 "date between 2020.01.01 and 2020.12.31"
            db_path: 数据库路径，默认使用配置中的路径
            limit: 限制返回行数

        Returns:
            pd.DataFrame: 查询结果
        """
        if db_path is None:
            db_path = DOLPHINDB["db_path"]

        try:
            # 构建SQL查询
            cols_str = "*" if columns is None else ", ".join(columns)
            query = f"select {cols_str} from loadTable('{db_path}', '{table_name}')"
            
            if condition:
                query += f" where {condition}"
                
            if limit:
                query += f" limit {limit}"
                
            # 执行查询
            result = self.execute(query)
            
            # 将结果转换为Pandas DataFrame
            if isinstance(result, (pd.DataFrame, pd.Series)):
                return result
            elif hasattr(result, "toDF"):
                return result.toDF()
            else:
                logger.warning(f"查询结果类型未知: {type(result)}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"查询DolphinDB表 {table_name} 失败: {e}")
            raise

    def get_stock_data(
        self,
        stock_codes: Union[str, List[str]],
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
        adj: bool = True,
        table_name: str = "daily_quote"
    ) -> pd.DataFrame:
        """获取股票行情数据

        Args:
            stock_codes: 股票代码或代码列表
            start_date: 开始日期，格式为 "YYYY-MM-DD"
            end_date: 结束日期，格式为 "YYYY-MM-DD"
            fields: 需要的字段列表，默认为所有字段
            adj: 是否进行复权处理
            table_name: 数据表名

        Returns:
            pd.DataFrame: 股票行情数据
        """
        if isinstance(stock_codes, str):
            stock_codes = [stock_codes]
            
        # 转换日期格式为DolphinDB格式
        start_date = start_date.replace("-", ".")
        end_date = end_date.replace("-", ".")
        
        # 构建股票代码列表字符串
        codes_str = ", ".join([f"'{code}'" for code in stock_codes])
        
        # 构建查询条件
        condition = f"date between {start_date} and {end_date} and stock_code in ({codes_str})"
        
        # 查询数据
        data = self.query_table(table_name, fields, condition)
        
        # 复权处理
        if adj and "close" in data.columns and "adj_factor" in data.columns:
            for col in ["open", "high", "low", "close"]:
                if col in data.columns:
                    data[f"{col}_adj"] = data[col] * data["adj_factor"]
                    
        return data

    def get_trade_dates(
        self, start_date: str, end_date: str, calendar_table: str = "trade_calendar"
    ) -> List[str]:
        """获取交易日历

        Args:
            start_date: 开始日期，格式为 "YYYY-MM-DD"
            end_date: 结束日期，格式为 "YYYY-MM-DD"
            calendar_table: 交易日历表名

        Returns:
            List[str]: 交易日列表
        """
        start_date = start_date.replace("-", ".")
        end_date = end_date.replace("-", ".")
        
        query = f"""
        select date from loadTable('{DOLPHINDB["db_path"]}', '{calendar_table}')
        where date between {start_date} and {end_date} and is_trading_day=1
        order by date
        """
        
        result = self.execute(query)
        
        if isinstance(result, pd.DataFrame):
            # 将日期转换为字符串格式 "YYYY-MM-DD"
            return [d.strftime("%Y-%m-%d") for d in result["date"]]
        else:
            logger.warning("获取交易日历结果类型未知")
            return []

    def _map_dtype(self, dtype: np.dtype) -> str:
        """将Pandas数据类型映射到DolphinDB数据类型

        Args:
            dtype: Pandas数据类型

        Returns:
            str: DolphinDB数据类型
        """
        if pd.api.types.is_integer_dtype(dtype):
            return "INT"
        elif pd.api.types.is_float_dtype(dtype):
            return "DOUBLE"
        elif pd.api.types.is_datetime64_dtype(dtype):
            return "TIMESTAMP"
        elif pd.api.types.is_bool_dtype(dtype):
            return "BOOL"
        else:
            return "STRING"
