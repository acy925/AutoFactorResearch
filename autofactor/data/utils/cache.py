"""
缓存管理模块 - 提供数据缓存功能
"""
import os
import pickle
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger


class CacheManager:
    """缓存管理器，提供内存和磁盘两级缓存"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化缓存管理器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.memory_cache = {}
        self.enabled = config.get("enabled", True)
        self.cache_dir = config.get("cache_dir", "./cache")
        # self.cache_dir = self.config.get("cache_dir", "cache")
        self.memory_limit = self.config.get("memory_cache_limit", 10)  # 默认最多缓存10个对象
        self.disk_enabled = self.config.get("disk_cache_enabled", True)
        
        # 创建缓存目录
        if self.disk_enabled and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
    def get(self, key: str, default: Any = None) -> Any:
        """从缓存获取数据
        
        先尝试从内存缓存获取，如果不存在再尝试从磁盘缓存获取
        
        Args:
            key: 缓存键
            default: 默认值，如果缓存不存在则返回此值
            
        Returns:
            缓存的数据，或默认值
        """
        # 尝试从内存缓存获取
        if key in self.memory_cache:
            logger.debug(f"从内存缓存获取: {key}")
            return self.memory_cache[key]
            
        # 尝试从磁盘缓存获取
        if self.disk_enabled:
            disk_path = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(disk_path):
                try:
                    with open(disk_path, 'rb') as f:
                        data = pickle.load(f)
                    # 加载到内存缓存
                    self._update_memory_cache(key, data)
                    logger.debug(f"从磁盘缓存获取: {key}")
                    return data
                except Exception as e:
                    logger.warning(f"从磁盘缓存加载失败: {key}, 错误: {e}")
                    
        return default
        
    def set(self, key: str, value: Any) -> None:
        """设置缓存数据
        
        同时更新内存缓存和磁盘缓存
        
        Args:
            key: 缓存键
            value: 要缓存的数据
        """
        # 更新内存缓存
        self._update_memory_cache(key, value)
        
        # 更新磁盘缓存
        if self.disk_enabled:
            disk_path = os.path.join(self.cache_dir, f"{key}.pkl")
            try:
                with open(disk_path, 'wb') as f:
                    pickle.dump(value, f)
                logger.debug(f"写入磁盘缓存: {key}")
            except Exception as e:
                logger.warning(f"写入磁盘缓存失败: {key}, 错误: {e}")
                
    def _update_memory_cache(self, key: str, value: Any) -> None:
        """更新内存缓存，并控制缓存大小"""
        # 如果达到内存缓存上限，删除最早的缓存
        if len(self.memory_cache) >= self.memory_limit:
            # 简单策略：删除第一个键
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
            logger.debug(f"内存缓存达到上限，删除: {oldest_key}")
            
        # 添加到内存缓存
        self.memory_cache[key] = value
        logger.debug(f"更新内存缓存: {key}")
        
    def clear(self, key: Optional[str] = None) -> None:
        """清除缓存
        
        Args:
            key: 要清除的缓存键，如果为None则清除所有缓存
        """
        if key is None:
            # 清除所有缓存
            self.memory_cache.clear()
            logger.debug("清除所有内存缓存")
            
            if self.disk_enabled:
                for file_name in os.listdir(self.cache_dir):
                    if file_name.endswith('.pkl'):
                        os.remove(os.path.join(self.cache_dir, file_name))
                logger.debug("清除所有磁盘缓存")
        else:
            # 清除指定缓存
            if key in self.memory_cache:
                del self.memory_cache[key]
                logger.debug(f"清除内存缓存: {key}")
                
            if self.disk_enabled:
                disk_path = os.path.join(self.cache_dir, f"{key}.pkl")
                if os.path.exists(disk_path):
                    os.remove(disk_path)
                    logger.debug(f"清除磁盘缓存: {key}")