"""
本文件核心：
把复杂的 Python 函数调用, 变成像 t2i、control、fill 极其简短、语义明确的终端命令行指令
系统的“大门”和“路由器”。
当你使用 python -m flux <指令> 时，它会截获这个指令，
并通过 fire 库将任务分发给不同的子脚本(cli.py、cli_control.py、cli_fill.py、cli_kontext.py、cli_redux.py)。
每个子脚本负责一个功能模块，解析特定的命令行参数，并调用 flux 包内的核心功能来完成任务
"""
from fire import Fire

from .cli import main as cli_main             # 负责标准文生图 (t2i)
from .cli_control import main as control_main # 负责控制网 (Canny/Depth) 与 LoRA 微调
from .cli_fill import main as fill_main       # 负责局部重绘 (Inpainting)
from .cli_kontext import main as kontext_main # 负责上下文设计 (特定宽高比的生成)
from .cli_redux import main as redux_main     # 负责垫图/风格迁移 (Image Prompt)

if __name__ == "__main__":
    # Fire 接收一个字典，将字典的 key 作为终端命令，value 作为对应的执行函数
    Fire(
        {
            "t2i": cli_main,
            "control": control_main,
            "fill": fill_main,
            "kontext": kontext_main,
            "redux": redux_main,
        }
    )