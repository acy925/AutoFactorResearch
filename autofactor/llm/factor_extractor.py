"""
因子提取模块 - 使用LLM从研究报告中提取量化因子逻辑
"""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import openai
from loguru import logger

from config.llm_prompts import get_prompt
from config.settings import DATA_DIR, LLM
from llm.report_parser import ReportParser, parse_report


class FactorExtractor:
    """因子提取器"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """初始化因子提取器

        Args:
            api_key: LLM API密钥，默认使用配置中的API密钥
            model: LLM模型名称，默认使用配置中的模型
            cache_dir: 缓存目录，默认使用DATA_DIR/factor_cache
        """
        self.api_key = api_key or LLM["api_key"] or os.getenv("OPENAI_API_KEY")
        self.model = model or LLM["model"]
        self.temperature = LLM["temperature"]
        self.max_tokens = LLM["max_tokens"]

        if self.api_key is None:
            raise ValueError("未设置API密钥，请在环境变量或配置文件中设置")

        if cache_dir is None:
            cache_dir = Path(DATA_DIR) / "factor_cache"
        else:
            cache_dir = Path(cache_dir)

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)

    def extract_factor_from_report(
        self, report_path: Union[str, Path], use_cache: bool = True
    ) -> Dict[str, Any]:
        """从研究报告中提取因子信息

        Args:
            report_path: 研究报告文件路径
            use_cache: 是否使用缓存

        Returns:
            Dict: 因子信息字典
        """
        report_path = Path(report_path)
        cache_file = self.cache_dir / f"{report_path.stem}_factor.json"

        # 如果启用缓存且缓存文件存在，直接返回缓存结果
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_result = json.load(f)
                logger.info(f"使用缓存的因子提取结果: {cache_file}")
                return cached_result
            except Exception as e:
                logger.warning(f"读取缓存文件失败: {e}")

        # 解析研究报告
        logger.info(f"解析研究报告: {report_path}")
        report_data = parse_report(report_path)
        report_content = report_data["content"]

        # 使用LLM提取因子信息
        logger.info("使用LLM提取因子信息...")
        prompt = get_prompt("factor_extraction", report_content=report_content[:20000])  # 限制内容长度
        
        try:
            # 调用OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的量化金融分析助手，擅长从研究报告中提取量化因子信息。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # 解析响应
            extraction_result = response.choices[0].message.content
            
            try:
                # 尝试解析JSON
                factor_info = self._parse_factor_json(extraction_result)
            except json.JSONDecodeError:
                logger.warning("LLM返回的结果不是有效的JSON格式，尝试结构化提取")
                factor_info = self._extract_structured_info(extraction_result)
            
            # 添加元信息
            result = {
                "factor_info": factor_info,
                "report_meta": report_data["meta_info"],
                "extraction_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_used": self.model,
                "report_file": str(report_path),
            }
            
            # 缓存结果
            if use_cache:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                logger.info(f"因子提取结果已缓存: {cache_file}")
                
            return result
            
        except Exception as e:
            logger.error(f"调用LLM API提取因子信息失败: {e}")
            raise

    def _parse_factor_json(self, text: str) -> Dict[str, Any]:
        """解析LLM返回的JSON格式因子信息

        Args:
            text: LLM返回的文本

        Returns:
            Dict: 解析后的因子信息
        """
        # 尝试提取JSON部分
        json_start = text.find("{")
        json_end = text.rfind("}")
        
        if json_start >= 0 and json_end >= 0:
            json_text = text[json_start:json_end + 1]
            return json.loads(json_text)
        else:
            # 如果没有找到JSON格式，抛出异常
            raise json.JSONDecodeError("未找到有效的JSON格式", text, 0)

    def _extract_structured_info(self, text: str) -> Dict[str, Any]:
        """从非JSON格式的LLM响应中提取结构化信息

        Args:
            text: LLM返回的文本

        Returns:
            Dict: 结构化的因子信息
        """
        lines = text.split("\n")
        factor_info = {}
        current_factor = None
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检查是否是因子名称行
            if line.startswith("因子名称") or "因子名称" in line:
                parts = line.split(":", 1) if ":" in line else line.split("：", 1)
                if len(parts) > 1:
                    factor_name = parts[1].strip()
                    current_factor = factor_name
                    factor_info[current_factor] = {"名称": factor_name}
                    current_section = None
                    
            # 检查是否是因子属性行
            elif current_factor and ":" in line:
                parts = line.split(":", 1)
                key = parts[0].strip()
                value = parts[1].strip()
                factor_info[current_factor][key] = value
                current_section = key
                
            # 检查是否是因子属性行（中文冒号）
            elif current_factor and "：" in line:
                parts = line.split("：", 1)
                key = parts[0].strip()
                value = parts[1].strip()
                factor_info[current_factor][key] = value
                current_section = key
                
            # 接续上一个属性的内容
            elif current_factor and current_section:
                factor_info[current_factor][current_section] += "\n" + line
                
        # 如果只有一个因子，则简化结构
        if len(factor_info) == 1:
            return next(iter(factor_info.values()))
            
        return {"factors": factor_info}

    def generate_factor_code(self, factor_info: Dict[str, Any]) -> str:
        """生成因子计算代码

        Args:
            factor_info: 因子信息字典

        Returns:
            str: 生成的Python代码
        """
        # 将因子信息转换为格式化字符串
        if isinstance(factor_info, dict):
            factor_info_str = json.dumps(factor_info, ensure_ascii=False, indent=2)
        else:
            factor_info_str = str(factor_info)
            
        # 使用代码生成提示词
        prompt = get_prompt("code_generation", factor_info=factor_info_str)
        
        try:
            # 调用OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的量化金融编程助手，擅长将量化因子逻辑转化为Python代码。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # 提取代码部分
            code_text = response.choices[0].message.content
            return self._extract_code_block(code_text)
            
        except Exception as e:
            logger.error(f"生成因子代码失败: {e}")
            raise

    def _extract_code_block(self, text: str) -> str:
        """从LLM响应中提取代码块

        Args:
            text: LLM返回的文本

        Returns:
            str: 提取的代码
        """
        # 查找Python代码块
        python_pattern = r"```python\n(.*?)```"
        import re
        matches = re.findall(python_pattern, text, re.DOTALL)
        
        if matches:
            return matches[0]
        
        # 查找普通代码块
        code_pattern = r"```(.*?)```"
        matches = re.findall(code_pattern, text, re.DOTALL)
        
        if matches:
            return matches[0]
            
        # 如果没有代码块标记，返回整个文本
        return text


def extract_factor(report_path: Union[str, Path]) -> Dict[str, Any]:
    """从研究报告中提取因子的便捷函数

    Args:
        report_path: 研究报告文件路径

    Returns:
        Dict: 因子信息字典
    """
    extractor = FactorExtractor()
    return extractor.extract_factor_from_report(report_path)


def generate_code(factor_info: Dict[str, Any]) -> str:
    """根据因子信息生成代码的便捷函数

    Args:
        factor_info: 因子信息字典

    Returns:
        str: 生成的Python代码
    """
    extractor = FactorExtractor()
    return extractor.generate_factor_code(factor_info)


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) > 1:
        report_file = sys.argv[1]
        result = extract_factor(report_file)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        print("\n生成因子代码:")
        code = generate_code(result["factor_info"])
        print(code)
    else:
        print("请提供研究报告文件路径")
