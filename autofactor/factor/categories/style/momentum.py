class MomentumFactor(Factor):
    """动量因子"""
    
    def __init__(self, window=20):
        """初始化动量因子
        
        Args:
            window: 窗口期(天数)
        """
        name = f"momentum_{window}d"
        super().__init__(
            name=name,
            description=f"{window}天价格动量因子",
            category="Technical",
            subcategory="Momentum",
            parameters={"window": window}
        )
        self.window = window
        
    def compute(self, data, processor=None, **kwargs):
        """计算动量因子
        
        Args:
            data: 输入数据
            processor: 数据处理器(可选)
            **kwargs: 其他参数
            
        Returns:
            pandas DataFrame: 包含动量因子的数据
        """
        symbol_col = kwargs.get("symbol_col", "symbol")
        close_col = kwargs.get("close_col", "close")
        
        # 检查必要列
        required = self.get_requirements()
        missing = [col for col in required if col not in data.columns]
        if missing:
            if processor is not None and symbol_col in data.columns:
                # 尝试通过处理器获取缺失字段
                symbols = data[symbol_col].unique().tolist()
                start_date = data["date"].min()
                end_date = data["date"].max()
                missing_data = processor.get_data(symbols, start_date, end_date, fields=missing)
                data = pd.merge(data, missing_data, on=["date", symbol_col])
            else:
                raise ValueError(f"缺少必要列: {missing}")
                
        # 计算动量因子
        data[self.name] = data.groupby(symbol_col)[close_col].pct_change(self.window)
        
        # 如果有处理器，可以选择进行标准化
        if processor is not None and kwargs.get("normalize", False):
            norm_data = processor.normalize(
                data,
                method=kwargs.get("normalize_method", "zscore"),
                by_cross_section=kwargs.get("by_cross_section", True)
            )
            # 仅复制标准化后的因子列
            data[f"{self.name}_norm"] = norm_data[self.name]
            
        return data
        
    def get_requirements(self):
        """获取计算因子所需字段"""
        return ["symbol", "date", "close"]