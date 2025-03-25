class FactorRegistry:
    """因子注册表，管理所有可用因子"""
    
    def __init__(self):
        """初始化因子注册表"""
        self.factors = {}  # 名称到因子映射
        self.categories = {}  # 类别索引
        
    def register(self, factor):
        """注册因子
        
        Args:
            factor: Factor实例
        """
        self.factors[factor.name] = factor
        
        # 添加到类别索引
        if factor.category not in self.categories:
            self.categories[factor.category] = {}
        
        if factor.subcategory not in self.categories[factor.category]:
            self.categories[factor.category][factor.subcategory] = []
            
        self.categories[factor.category][factor.subcategory].append(factor.name)
        
    def register_all(self, factors):
        """注册多个因子
        
        Args:
            factors: Factor实例列表
        """
        for factor in factors:
            self.register(factor)
            
    def get_factor(self, name):
        """通过名称获取因子
        
        Args:
            name: 因子名称
            
        Returns:
            Factor: 对应因子实例，未找到返回None
        """
        return self.factors.get(name)
        
    def list_categories(self):
        """列出所有类别
        
        Returns:
            List[str]: 类别列表
        """
        return list(self.categories.keys())
        
    def list_subcategories(self, category):
        """列出类别下的所有子类别
        
        Args:
            category: 类别名称
            
        Returns:
            List[str]: 子类别列表
        """
        if category not in self.categories:
            return []
        return list(self.categories[category].keys())
        
    def list_factors(self, category=None, subcategory=None):
        """列出符合条件的因子
        
        Args:
            category: 类别名称(可选)
            subcategory: 子类别名称(可选)
            
        Returns:
            List[str]: 因子名称列表
        """
        if category is None:
            return list(self.factors.keys())
            
        if category not in self.categories:
            return []
            
        if subcategory is None:
            # 返回所有子类别下的因子
            result = []
            for subcat in self.categories[category]:
                result.extend(self.categories[category][subcat])
            return result
            
        if subcategory not in self.categories[category]:
            return []
            
        return self.categories[category][subcategory]