"""
LLM提示词模板模块 - 为与大语言模型交互提供标准化的提示词模板
"""
from typing import Dict, List, Optional, Union

# 因子提取提示词模板
FACTOR_EXTRACTION_PROMPT = """
你是一个专业的量化金融分析助手，擅长从研究报告中提取量化因子信息。请仔细分析以下研究报告内容，识别并提取其中描述的量化因子的详细信息，包括：

1. 因子名称
2. 因子类别（基本面/技术面/市场情绪/另类数据等）
3. 因子计算公式
4. 因子参数设置
5. 因子实现步骤
6. 数据要求（所需数据字段）
7. 因子使用建议（如中性化处理、行业分组等）

请以JSON格式返回，确保公式和计算逻辑尽可能详细准确。如果报告中包含多个因子，请分别提取每个因子的信息。

研究报告内容：
{report_content}
"""

# 代码生成提示词模板
CODE_GENERATION_PROMPT = """
你是一个专业的量化金融编程助手，擅长将量化因子逻辑转化为Python代码。请根据以下因子信息，生成计算该因子的Python函数。

因子信息:
{factor_info}

要求：
1. 使用pandas和numpy进行数据处理
2. 代码需包含详细注释
3. 处理可能的异常情况（如缺失值、极端值等）
4. 确保代码结构清晰，函数接口规范
5. 提供使用示例

请生成完整可执行的Python代码。
"""

# 因子评估提示词模板
FACTOR_EVALUATION_PROMPT = """
你是一个专业的量化金融分析助手，擅长评估量化因子性能。请根据以下因子回测结果，对该因子进行全面评估。

因子信息:
{factor_info}

回测结果:
{backtest_results}

要求：
1. 评估因子的预测能力（信息比率、IC值等）
2. 分析因子的稳定性和持续性
3. 评估因子收益的风险调整表现（夏普比率、最大回撤等）
4. 分析因子在不同市场环境下的表现
5. 给出因子优化或组合建议

请提供专业、客观的分析评估。
"""

# 研究报告解析提示词模板
REPORT_PARSING_PROMPT = """
你是一个专业的量化研究助手，擅长解析金融研究报告。请仔细阅读以下研究报告内容，提取其中的核心信息。

研究报告内容：
{report_content}

请提取以下信息：
1. 报告主题和研究目标
2. 研究方法和数据来源
3. 主要发现和结论
4. 量化策略或因子的详细描述
5. 实证结果和统计数据
6. 研究局限性和未来改进方向

请以结构化的方式返回上述信息。
"""

# DolphinDB脚本生成提示词模板
DOLPHINDB_SCRIPT_PROMPT = """
你是一个DolphinDB专家，擅长将量化因子逻辑转换为高效的DolphinDB脚本。请根据以下因子信息和计算逻辑，生成相应的DolphinDB脚本代码。

因子信息:
{factor_info}

Python实现代码:
{python_code}

要求：
1. 充分利用DolphinDB的向量化计算特性
2. 考虑大数据集的处理效率
3. 包含必要的异常处理
4. 提供详细注释
5. 考虑分布式计算优化（如适用）

请生成高效的DolphinDB脚本代码。
"""

# 量化问题解答提示词模板
QUANT_QA_PROMPT = """
你是一个专业的量化金融专家，擅长解答量化投资和因子研究相关问题。请根据你的专业知识，回答以下问题：

问题：
{question}

请提供详细、准确、专业的回答，必要时包含公式、例子或参考文献。
"""

# 提示词模板映射
PROMPT_TEMPLATES = {
    "factor_extraction": FACTOR_EXTRACTION_PROMPT,
    "code_generation": CODE_GENERATION_PROMPT,
    "factor_evaluation": FACTOR_EVALUATION_PROMPT,
    "report_parsing": REPORT_PARSING_PROMPT,
    "dolphindb_script": DOLPHINDB_SCRIPT_PROMPT,
    "quant_qa": QUANT_QA_PROMPT,
}


def get_prompt(template_name: str, **kwargs) -> str:
    """获取提示词模板并填充参数

    Args:
        template_name: 模板名称
        **kwargs: 模板参数

    Returns:
        str: 填充参数后的提示词
    """
    if template_name not in PROMPT_TEMPLATES:
        raise ValueError(f"提示词模板不存在: {template_name}")
        
    prompt_template = PROMPT_TEMPLATES[template_name]
    return prompt_template.format(**kwargs)


def get_all_templates() -> Dict[str, str]:
    """获取所有提示词模板

    Returns:
        Dict[str, str]: 提示词模板字典
    """
    return PROMPT_TEMPLATES.copy()
