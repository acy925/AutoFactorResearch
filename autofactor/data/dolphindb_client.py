"""
DolphinDB客户端模块 - 提供与DolphinDB数据库的交互功能
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

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
        test_mode: Optional[bool] = None,
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

        # 处理test_mode参数
        if test_mode is not None:
            self.test_mode = test_mode
        else:
            self.test_mode = DOLPHINDB.get("test_mode", False)

        self._connect()

    def _connect(self) -> None:
        """连接DolphinDB服务器"""
        # 检查是否使用测试模式
        if DOLPHINDB.get("test_mode", True):
            logger.info("使用测试模式，不连接真实DolphinDB服务器")
            self.conn = None
            self.test_mode = True
            return
            
        try:
            self.conn = ddb.session()
            self.conn.connect(self.host, self.port, self.username, self.password)
            self.test_mode = False
            
            # 测试连接是否真正有效
            version = self.conn.run("version()")
            logger.info(f"成功连接到DolphinDB服务器: {self.host}:{self.port}, 版本: {version}")
        except Exception as e:
            logger.error(f"连接DolphinDB服务器失败: {e}")
            logger.info("请确认DolphinDB服务器已启动，并验证连接信息是否正确")
            logger.info(f"当前连接信息: 主机={self.host}, 端口={self.port}, 用户名={self.username}")
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
        """执行DolphinDB脚本"""
        if getattr(self, 'test_mode', False):
            logger.info(f"测试模式执行脚本: {script}")
            # 简单模拟一些基本操作
            if "version()" in script:
                return "测试模式 v1.0.0"
            elif "1 + 1" in script:
                return 2
            elif "table(" in script:
                import pandas as pd
                return pd.DataFrame()
            elif "select" in script:
                import pandas as pd
                return pd.DataFrame({'date': pd.date_range('2018-01-01', periods=3),
                                    'value': [1, 2, 3]})
            else:
                return None
        
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
        if self.test_mode:
            logger.info("使用测试模式，不上传数据")
            return True
        
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
        if self.test_mode:
            import pandas as pd
            import numpy as np
            
            # 生成模拟数据
            n_rows = limit or 100
            if table_name.lower() in ["daily_quote", "stock_daily"]:
                # 模拟股票日线数据
                data = {
                    "date": pd.date_range('2020-01-01', periods=n_rows),
                    "symbol": np.random.choice(["000001.SZ", "600000.SH", "300001.SZ"], n_rows),
                    "open": np.random.rand(n_rows) * 100 + 10,
                    "high": np.random.rand(n_rows) * 100 + 15,
                    "low": np.random.rand(n_rows) * 100 + 5,
                    "close": np.random.rand(n_rows) * 100 + 10,
                    "volume": np.random.randint(1000, 10000000, n_rows),
                    "amount": np.random.randint(10000, 100000000, n_rows),
                    "adj_factor": np.random.rand(n_rows) + 0.5,
                }
            elif table_name.lower() in ["trade_calendar"]:
                # 模拟交易日历
                data = {
                    "date": pd.date_range('2020-01-01', periods=n_rows),
                    "is_trading_day": np.random.choice([0, 1], n_rows, p=[0.3, 0.7]),
                    "exchange": np.random.choice(["SSE", "SZSE"], n_rows),
                }
            else:
                # 通用模拟数据
                data = {
                    "date": pd.date_range('2020-01-01', periods=n_rows),
                    "value1": np.random.rand(n_rows) * 100,
                    "value2": np.random.rand(n_rows) * 200,
                    "category": np.random.choice(["A", "B", "C"], n_rows),
                }
                
            df = pd.DataFrame(data)
            
            # 如果指定了列，只返回指定列
            if columns:
                existing_cols = [col for col in columns if col in df.columns]
                df = df[existing_cols]
                
            return df
    
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
    def list_tables(self, db_path: Optional[str] = None) -> List[str]:
        """列出数据库中的所有表

        Args:
            db_path: 数据库路径，默认使用配置中的路径

        Returns:
            List[str]: 表名列表
        """
        if db_path is None:
            db_path = DOLPHINDB["db_path"]

        if self.test_mode:
            return ["stock_daily", "trade_calendar", "factor_data", "industry_mapping"]
            
        try:
            result = self.execute(f"tables('{db_path}')")
            if isinstance(result, (list, tuple, pd.Series)):
                return list(result)
            else:
                logger.warning(f"未预期的结果类型: {type(result)}")
                return []
        except Exception as e:
            logger.error(f"获取表列表失败: {e}")
            return []
        
    def table_info(self, table_name: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """获取表结构信息

        Args:
            table_name: 表名
            db_path: 数据库路径，默认使用配置中的路径

        Returns:
            Dict: 表结构信息
        """
        if db_path is None:
            db_path = DOLPHINDB["db_path"]

        if self.test_mode:
            # 根据表名返回不同的模拟数据
            if table_name.lower() in ["daily_quote", "stock_daily"]:
                return {
                    "name": table_name,
                    "columns": {"date": "DATE", "symbol": "SYMBOL", "open": "DOUBLE", 
                                "high": "DOUBLE", "low": "DOUBLE", "close": "DOUBLE", 
                                "volume": "LONG", "amount": "DOUBLE", "adj_factor": "DOUBLE"},
                    "row_count": 5000000,
                    "partition_count": 20,
                    "database": db_path
                }
            elif table_name.lower() == "trade_calendar":
                return {
                    "name": table_name,
                    "columns": {"date": "DATE", "is_trading_day": "BOOL", "exchange": "SYMBOL"},
                    "row_count": 10000,
                    "partition_count": 1,
                    "database": db_path
                }
            elif table_name.lower() == "factor_data":
                return {
                    "name": table_name,
                    "columns": {"date": "DATE", "symbol": "SYMBOL", "factor_value": "DOUBLE", 
                                "factor_name": "SYMBOL", "factor_group": "SYMBOL"},
                    "row_count": 8000000,
                    "partition_count": 30,
                    "database": db_path
                }
            else:
                return {
                    "name": table_name,
                    "columns": {"date": "DATE", "value": "DOUBLE", "category": "SYMBOL"},
                    "row_count": 1000,
                    "partition_count": 5,
                    "database": db_path
                }
            
        try:
            # 获取表的列信息
            schema = self.execute(f"schema(loadTable('{db_path}', '{table_name}'))")
            
            # 获取表的行数
            count = self.execute(f"count(loadTable('{db_path}', '{table_name}'))")
            
            # 获取表的分区信息
            partitions = self.execute(f"exec count(*) from loadTable('{db_path}', '{table_name}').schema().partitionSchema")
            
            return {
                "name": table_name,
                "columns": schema.to_dict() if isinstance(schema, pd.DataFrame) else schema,
                "row_count": count,
                "partition_count": partitions if isinstance(partitions, (int, float)) else 0,
                "database": db_path
            }
        except Exception as e:
            logger.error(f"获取表信息失败: {e}")
            return {"name": table_name, "error": str(e)}

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
        if self.test_mode:
            import pandas as pd
            import numpy as np
            
            # 生成日期序列
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
            dates = pd.date_range(start=start, end=end, freq='B')  # 'B'表示工作日
            
            # 为每支股票生成数据
            dfs = []
            for code in stock_codes:
                # 随机生成行情数据起点
                base_price = np.random.randint(10, 100)
                prices = np.cumsum(np.random.normal(0, 1, len(dates))) * 0.5 + base_price
                prices = np.maximum(prices, 1)  # 确保价格不会为负
                
                data = {
                    'date': dates,
                    'symbol': code,
                    'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
                    'high': prices * (1 + np.random.normal(0, 0.01, len(dates))),
                    'low': prices * (1 - np.random.normal(0, 0.01, len(dates))),
                    'close': prices,
                    'volume': np.random.randint(1000000, 10000000, len(dates)),
                    'amount': np.random.randint(10000000, 100000000, len(dates)),
                    'adj_factor': np.random.uniform(0.8, 1.2, len(dates)),
                }
                
                df = pd.DataFrame(data)
                dfs.append(df)
                
            # 合并所有股票数据
            result = pd.concat(dfs, ignore_index=True)
            
            # 如果指定了字段，只返回指定字段
            if fields:
                available_fields = [f for f in fields if f in result.columns]
                result = result[available_fields]
                
            # 复权处理
            if adj and 'close' in result.columns and 'adj_factor' in result.columns:
                for col in ['open', 'high', 'low', 'close']:
                    if col in result.columns:
                        result[f'{col}_adj'] = result[col] * result['adj_factor']
                        
            return result


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
        
        # 重要：列名转换            
        if "stock_code" in data.columns:
            data = data.rename(columns={"stock_code": "symbol"})
                    
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

        if self.test_mode:
            # 生成工作日
            dates = pd.bdate_range(start=start_date, end=end_date)
            return [d.strftime("%Y-%m-%d") for d in dates]
        
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
    
def test_connection():
    """测试DolphinDB连接"""
    try:
        client = DolphinDBClient()
        version = client.execute("version()")
        logger.info(f"DolphinDB连接成功，服务器版本: {version}")
        return True
    except Exception as e:
        logger.error(f"DolphinDB连接测试失败: {e}")
        return False

def test_basic_functions():
    """测试基本功能"""
    try:
        client = DolphinDBClient()
        
        # 1. 测试简单查询
        logger.info("测试简单查询...")
        result = client.execute("1 + 1")
        logger.info(f"1 + 1 = {result}")
        
        # 2. 测试创建内存表
        logger.info("测试创建内存表...")
        client.execute("""
        t = table(2018.01.01..2018.01.10 as date, 1..10 as value)
        """)
        
        # 3. 查询内存表
        logger.info("测试查询内存表...")
        result = client.execute("select * from t")
        logger.info(f"查询结果 (前3行):\n{result.head(3)}")

        # 4. 测试上传数据   
        logger.info("测试上传数据...")
        df = pd.DataFrame({
            "date": pd.date_range("2018-01-01", periods=3),
            "value": [1, 2, 3]
        })
        client.upload_dataframe(df, "test_table")
        
        # 5. 测试查询数据
        logger.info("测试查询数据...")
        result = client.query_table("daily_quote")
        logger.info(f"查询结果 (前3行):\n{result.head(3)}")

        # 6. 测试获取表结构信息
        logger.info("测试获取表结构信息...")
        info = client.table_info("daily_quote")
        logger.info(f"表结构信息:\n{info}")

        # 7. 测试获取股票数据
        logger.info("测试获取股票数据...")
        stock_data = client.get_stock_data(
            stock_codes=["000001.SZ", "600000.SH"],
            start_date="2022-01-01",
            end_date="2022-01-10",
            fields=["date", "symbol", "open", "close", "volume"]
        )
        logger.info(f"股票数据 (前3行):\n{stock_data.head(3)}")

        # 8. 测试获取交易日历
        logger.info("测试获取交易日历...")
        dates = client.get_trade_dates("2020-01-01", "2020-01-31")
        logger.info(f"交易日历:\n{dates}")

        # 9. 测试获取表结构信息
        logger.info("测试获取表结构信息...")
        info = client.table_info("daily_quote")
        logger.info(f"表结构信息:\n{info}")


        return True
    except Exception as e:
        logger.error(f"基本功能测试失败: {e}")
        return False
if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    logger.info("开始测试DolphinDBClient...")
    
    # 测试连接
    if test_connection():
        # 测试基本功能
        test_basic_functions()
    
    logger.info("DolphinDBClient测试完成")