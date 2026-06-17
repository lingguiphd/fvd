# 安装
from setuptools import setup, find_packages

setup(
    name="fvd",  # 包名
    version="0.1.0",  # 版本
    package_dir={"": "src"},
    packages=find_packages(where="src"),  # 自动发现所有 Python 包
    python_requires="==3.11.11",  # Python 版本要求
)