# AutoFactorResearch

自动化量化因子研究框架 - 从研究报告到因子回测的全流程自动化解决方案

## 项目简介

AutoFactorResearch 是一个自动复现量化研究报告的框架。

本项目利用大语言模型(LLM)分析研究报告，提取量化因子逻辑，自动生成代码并在历史数据上回测，最终提供全面的因子表现评估。

### 主要功能

- 📄 **研究报告解析**：自动处理PDF/Word格式的量化研究报告
- 🧠 **因子逻辑提取**：使用LLM从文本中提取因子公式和实现逻辑
- 💻 **代码自动生成**：将提取的因子逻辑转换为可执行代码
- 📊 **因子回测分析**：在历史数据上回测因子表现
- 📈 **评价指标生成**：计算因子评价指标
- 🖥️ **可视化与报告**：自动生成分析报告和可视化结果

## 系统架构

本系统由以下主要模块组成:

1. **数据处理模块**：基于DolphinDB的高性能数据管理系统
2. **LLM处理模块**：处理和解析研究报告，提取因子信息
3. **因子生成模块**：将因子描述转换为可执行代码
4. **回测分析模块**：评估因子在历史数据上的表现
5. **可视化模块**：展示回测结果和关键指标

## 快速开始

### 环境要求

- Python 3.8+
- DolphinDB 服务器
- 其他依赖见 `requirements.txt`

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/yourusername/AutoFactorResearch.git
cd AutoFactorResearch
```

2. 创建虚拟环境
```bash
conda create -n autofactor python=3.9
conda activate autofactor
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 配置DolphinDB连接
编辑 `config/settings.py` 文件，配置DolphinDB服务器地址和凭据。

5. 配置LLM API (如果使用外部API)
编辑 `config/settings.py` 文件，添加LLM API密钥和相关配置。

## 使用指南

1. **数据准备**
```python
from autofactor.data import data_loader
data_loader.import_stock_data('path/to/your/data')
```

2. **研究报告处理**
```python
from autofactor.llm import report_parser
factor_info = report_parser.parse('path/to/research_report.pdf')
```

3. **因子生成与回测**
```python
from autofactor.factor import factor_generator
from autofactor.backtest import engine

# 生成因子
factor = factor_generator.generate_from_info(factor_info)

# 回测因子
results = engine.backtest(factor, start_date='2018-01-01', end_date='2022-12-31')
```

4. **结果可视化**
```python
from autofactor.visualization import dashboard
dashboard.plot_performance(results)
```

## 项目路线图

- [x] 项目结构设计与初始化
- [ ] DolphinDB数据模块实现
- [ ] LLM研究报告解析模块
- [ ] 因子代码生成模块
- [ ] 回测引擎开发
- [ ] 可视化与报告系统
- [ ] Web界面开发
- [ ] 系统整合与优化

## 贡献指南

欢迎对本项目提出问题和改进建议。请遵循以下步骤：

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详情请见 [LICENSE](LICENSE) 文件

## 联系方式

项目维护者: acy925 - aichengyuan925@gmail.com

项目链接: [https://github.com/acy925/AutoFactorResearch](https://github.com/yourusername/AutoFactorResearch)
