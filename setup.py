"""
安装脚本
"""
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="autofactor",
    version="0.1.0",
    author="acy",
    author_email="aichengyuan925@gmail.com",
    description="自动化量化因子研究框架",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/acy925/AutoFactorResearch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "polars>=0.8.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "dolphindb>=1.30.17.1",
        "openai>=1.0.0",
        "langchain>=0.0.267",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "pypdf2>=3.0.0",
        "python-docx>=0.8.11",
        "pyfolio>=0.9.2",
        "empyrical>=0.5.5",
        "pyarrow>=8.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.10.0",
        "dash>=2.6.0",
        "dash-bootstrap-components>=1.2.0",
        "fastapi>=0.89.0",
        "uvicorn>=0.20.0",
        "pydantic>=1.10.0",
        "jinja2>=3.0.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "python-dotenv>=0.20.0",
        "loguru>=0.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.10.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
    },
    entry_points={
        "console_scripts": [
            "autofactor=autofactor.cli:main",
        ],
    },
)
