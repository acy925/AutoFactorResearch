"""
研究报告解析模块 - 负责处理研究报告文件并提取文本内容
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import docx
import PyPDF2
from loguru import logger

from config.settings import REPORT_DIR


class ReportParser:
    """研究报告解析器"""

    def __init__(self, report_path: Union[str, Path]):
        """初始化解析器

        Args:
            report_path: 研究报告文件路径
        """
        self.report_path = Path(report_path)
        if not self.report_path.exists():
            raise FileNotFoundError(f"研究报告文件不存在: {self.report_path}")

        self.file_extension = self.report_path.suffix.lower()
        self.content = ""
        self.meta_info = {}

    def parse(self) -> Dict[str, Union[str, Dict]]:
        """解析研究报告

        Returns:
            Dict: 包含报告内容和元信息的字典
        """
        try:
            if self.file_extension == ".pdf":
                self._parse_pdf()
            elif self.file_extension in (".docx", ".doc"):
                self._parse_docx()
            elif self.file_extension == ".txt":
                self._parse_txt()
            else:
                raise ValueError(f"不支持的文件格式: {self.file_extension}")

            # 提取元信息
            self._extract_meta_info()

            return {
                "content": self.content,
                "meta_info": self.meta_info,
                "file_name": self.report_path.name,
                "file_path": str(self.report_path),
                "file_extension": self.file_extension,
            }

        except Exception as e:
            logger.error(f"解析研究报告 {self.report_path} 失败: {e}")
            raise

    def _parse_pdf(self) -> None:
        """解析PDF文件"""
        try:
            with open(self.report_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                # 提取元信息
                if pdf_reader.metadata:
                    self.meta_info = {
                        "title": pdf_reader.metadata.get("/Title", ""),
                        "author": pdf_reader.metadata.get("/Author", ""),
                        "subject": pdf_reader.metadata.get("/Subject", ""),
                        "keywords": pdf_reader.metadata.get("/Keywords", ""),
                        "creator": pdf_reader.metadata.get("/Creator", ""),
                        "producer": pdf_reader.metadata.get("/Producer", ""),
                        "creation_date": pdf_reader.metadata.get("/CreationDate", ""),
                    }
                
                # 提取文本内容
                num_pages = len(pdf_reader.pages)
                content = []
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        content.append(text)
                        
                self.content = "\n\n".join(content)
                
        except Exception as e:
            logger.error(f"解析PDF文件 {self.report_path} 失败: {e}")
            raise

    def _parse_docx(self) -> None:
        """解析DOCX文件"""
        try:
            doc = docx.Document(self.report_path)
            
            # 提取元信息
            self.meta_info = {
                "title": doc.core_properties.title or "",
                "author": doc.core_properties.author or "",
                "subject": doc.core_properties.subject or "",
                "keywords": doc.core_properties.keywords or "",
                "created": str(doc.core_properties.created) if doc.core_properties.created else "",
                "modified": str(doc.core_properties.modified) if doc.core_properties.modified else "",
                "last_modified_by": doc.core_properties.last_modified_by or "",
            }
            
            # 提取文本内容
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            self.content = "\n\n".join(paragraphs)
            
        except Exception as e:
            logger.error(f"解析DOCX文件 {self.report_path} 失败: {e}")
            raise

    def _parse_txt(self) -> None:
        """解析TXT文件"""
        try:
            with open(self.report_path, "r", encoding="utf-8") as f:
                self.content = f.read()
                
            # TXT文件通常没有元信息，使用文件名作为标题
            self.meta_info = {
                "title": self.report_path.stem,
                "author": "",
                "created": str(self.report_path.stat().st_ctime),
                "modified": str(self.report_path.stat().st_mtime),
            }
            
        except UnicodeDecodeError:
            # 尝试其他编码
            with open(self.report_path, "r", encoding="gbk") as f:
                self.content = f.read()
        except Exception as e:
            logger.error(f"解析TXT文件 {self.report_path} 失败: {e}")
            raise

    def _extract_meta_info(self) -> None:
        """从内容中提取更多元信息"""
        # 尝试从内容中提取发布机构
        institution_patterns = [
            r"([\u4e00-\u9fa5]{2,}证券(?:股份)?有限责任?公司)",
            r"([\u4e00-\u9fa5]{2,}基金(?:管理)?(?:股份)?有限责任?公司)",
            r"([\u4e00-\u9fa5]{2,}研究院)",
            r"([\u4e00-\u9fa5]{2,}(?:投资)?研究所)",
        ]
        
        for pattern in institution_patterns:
            matches = re.findall(pattern, self.content)
            if matches:
                self.meta_info["institution"] = matches[0]
                break
                
        # 尝试提取日期
        date_patterns = [
            r"(\d{4})[-年/](\d{1,2})[-月/](\d{1,2})日?",
            r"报告日期[：:]?\s*(\d{4})[-年/](\d{1,2})[-月/](\d{1,2})日?",
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, self.content[:1000])  # 只在开头部分查找
            if matches:
                year, month, day = matches[0]
                self.meta_info["date"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                break
                
        # 尝试提取报告类型
        report_type_patterns = [
            r"(量化(?:投资|策略|研究)报告)",
            r"(因子(?:研究|分析)报告)",
            r"((?:选股|多因子)模型(?:研究|报告))",
        ]
        
        for pattern in report_type_patterns:
            matches = re.findall(pattern, self.content[:2000])
            if matches:
                self.meta_info["report_type"] = matches[0]
                break


def parse_report(report_path: Union[str, Path]) -> Dict[str, Union[str, Dict]]:
    """解析研究报告的便捷函数

    Args:
        report_path: 研究报告文件路径

    Returns:
        Dict: 包含报告内容和元信息的字典
    """
    parser = ReportParser(report_path)
    return parser.parse()


def list_reports(
    directory: Optional[Union[str, Path]] = None,
    extensions: Optional[Set[str]] = None,
    recursive: bool = True,
) -> List[Path]:
    """列出目录中的研究报告文件

    Args:
        directory: 目录路径，默认为配置中的报告目录
        extensions: 文件扩展名集合，默认为 {'.pdf', '.docx', '.doc', '.txt'}
        recursive: 是否递归搜索子目录

    Returns:
        List[Path]: 报告文件路径列表
    """
    if directory is None:
        directory = REPORT_DIR
    else:
        directory = Path(directory)

    if extensions is None:
        extensions = {".pdf", ".docx", ".doc", ".txt"}

    if not directory.exists():
        logger.warning(f"目录不存在: {directory}")
        return []

    result = []
    
    if recursive:
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                result.append(path)
    else:
        for path in directory.glob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                result.append(path)

    return result


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) > 1:
        report_file = sys.argv[1]
        result = parse_report(report_file)
        print(f"文件名: {result['file_name']}")
        print(f"元信息: {result['meta_info']}")
        print(f"内容前500字符: {result['content'][:500]}...")
    else:
        print("请提供研究报告文件路径")
