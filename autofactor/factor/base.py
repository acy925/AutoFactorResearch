class Factor:
    """因子基类"""
    
    def __init__(self, name, description, category, subcategory, parameters=None):
        """初始化因子
        
        Args:
            name: 因子名称
            description: 因子描述
            category: 因子类别 (如"Technical", "Fundamental")
            subcategory: 子类别 (如"Momentum", "Value") 
            parameters: 因子参数字典
        """
        self.name = name
        self.description = description
        self.category = category
        self.subcategory = subcategory
        self.parameters = parameters or {}
        
    def compute(self, data, processor=None, **kwargs):
        """计算因子值
        
        Args:
            data: 输入数据 (pandas DataFrame)
            processor: 数据处理器实例(可选)
            **kwargs: 其他参数
            
        Returns:
            pandas DataFrame: 包含计算因子值的数据
        """
        raise NotImplementedError("子类必须实现此方法")
        
    def get_requirements(self):
        """获取计算因子所需的数据字段
        
        Returns:
            List[str]: 所需字段列表
        """
        raise NotImplementedError("子类必须实现此方法")
        
    def estimate_complexity(self):
        """估计计算复杂度
        
        Returns:
            str: 复杂度级别 ("low", "medium", "high")
        """
        return "medium"  # 默认复杂度