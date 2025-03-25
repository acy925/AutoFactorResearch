"""
全局配置设置模块 - 用于配置项目中的各种参数和设置
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 基础路径
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"

# 创建必要的目录
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 应用设置
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# DolphinDB 设置
DOLPHINDB = {
    "host": os.getenv("DOLPHINDB_HOST", "localhost"),
    "port": int(os.getenv("DOLPHINDB_PORT", "8848")),
    "username": os.getenv("DOLPHINDB_USERNAME", "admin"),
    "password": os.getenv("DOLPHINDB_PASSWORD", "123456"),
    "db_path": os.getenv("DOLPHINDB_DB_PATH", "dfs://quantdb"),
    "batch_size": int(os.getenv("DOLPHINDB_BATCH_SIZE", "10000")),
    "timeout": int(os.getenv("DOLPHINDB_TIMEOUT", "60")),
    "test_mode": os.getenv("DOLPHINDB_TEST_MODE", "True").lower() in ("true", "1", "t"),  # 添加测试模式
}

# LLM 设置
LLM = {
    "provider": os.getenv("LLM_PROVIDER", "openai"),  # 'openai', 'anthropic', 'local', etc.
    "api_key": os.getenv("LLM_API_KEY", ""),
    "model": os.getenv("LLM_MODEL", "gpt-4"),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "4000")),
    "timeout": int(os.getenv("LLM_TIMEOUT", "120")),
    "local_model_path": os.getenv("LOCAL_MODEL_PATH", ""),
    "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
}

# 回测设置
BACKTEST = {
    "default_start_date": os.getenv("DEFAULT_START_DATE", "2018-01-01"),
    "default_end_date": os.getenv("DEFAULT_END_DATE", "2023-12-31"),
    "benchmark": os.getenv("BENCHMARK", "000300.SH"),  # 沪深300指数
    "commission_rate": float(os.getenv("COMMISSION_RATE", "0.0003")),  # 手续费率
    "slippage": float(os.getenv("SLIPPAGE", "0.0001")),  # 滑点
    "capital": float(os.getenv("CAPITAL", "10000000")),  # 初始资金
    "rebalance_frequency": os.getenv("REBALANCE_FREQUENCY", "monthly"),  # 再平衡频率
}

# 因子设置
FACTOR = {
    "cache_dir": os.getenv("FACTOR_CACHE_DIR", str(DATA_DIR / "factor_cache")),
    "neutralize": os.getenv("FACTOR_NEUTRALIZE", "False").lower() in ("true", "1", "t"),
    "universe": os.getenv("FACTOR_UNIVERSE", "A股全市场"),  # 股票池
    "preprocess": os.getenv("FACTOR_PREPROCESS", "standardize"),  # 预处理方法
}

# API设置
API = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4")),
    "reload": os.getenv("API_RELOAD", "False").lower() in ("true", "1", "t"),
    "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
}

# 可视化设置
VISUALIZATION = {
    "theme": os.getenv("VIZ_THEME", "plotly"),
    "color_palette": os.getenv("VIZ_COLOR_PALETTE", "blues"),
    "default_figsize": (
        int(os.getenv("VIZ_DEFAULT_WIDTH", "12")),
        int(os.getenv("VIZ_DEFAULT_HEIGHT", "8")),
    ),
}

# 系统设置
SYSTEM = {
    "n_workers": int(os.getenv("N_WORKERS", "4")),  # 工作进程数
    "chunk_size": int(os.getenv("CHUNK_SIZE", "100000")),  # 数据处理分块大小
    "cache_enabled": os.getenv("CACHE_ENABLED", "True").lower() in ("true", "1", "t"),
    "cache_ttl": int(os.getenv("CACHE_TTL", "86400")),  # 缓存过期时间（秒）
}

# 数据字段映射
FIELD_MAPPING = {
    "date": "trade_date",
    "open": "open",
    "high": "high", 
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
    "adj_factor": "adj_factor",
    "pe": "pe_ttm",
    "pb": "pb",
    "ps": "ps_ttm",
    "market_cap": "total_mv",
    "float_market_cap": "float_mv",
    "industry": "industry",
}

# 因子评价指标
FACTOR_METRICS = [
    "ic",
    "ic_ir",
    "rank_ic",
    "annualized_return", 
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "turnover_rate",
]

# 股票交易日历（简化版）
TRADING_DAYS_SETTINGS = {
    "exchange": "SSE",  # 上海证券交易所
    "start_time": "09:30:00",
    "end_time": "15:00:00",
    "lunch_break_start": "11:30:00", 
    "lunch_break_end": "13:00:00",
    "timezone": "Asia/Shanghai",
}
