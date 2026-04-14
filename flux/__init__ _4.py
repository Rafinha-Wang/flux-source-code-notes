"""
本文件核心：
Python 包的初始化文件 (Package Entry)。
告诉 Python 解释器 flux 文件夹是一个完整的工程包。它会尝试读取版本号，并获取项目的绝对根目录路径 (PACKAGE_ROOT)
有了它定义的绝对路径，无论你在哪个文件夹下敲击运行命令，
系统都能准确无误地找到动辄几十 GB 的模型权重文件 (Checkpoints) 而不报错
"""
try:
    from ._version import (
        version as __version__,  # type: ignore
        version_tuple,
    )
except ImportError:
    # 容错机制：如果用户是直接 clone 代码而不是通过 pip 安装的，找不到版本文件时，给一个兜底信息
    __version__ = "unknown (no version information available)"
    version_tuple = (0, 0, "unknown", "noinfo")

from pathlib import Path

# 获取包的名称和根目录的绝对路径，方便整个项目在后续寻找权重文件 (checkpoints) 时不会因为当前运行路径的不同而报错
PACKAGE = __package__.replace("_", "-")
PACKAGE_ROOT = Path(__file__).parent