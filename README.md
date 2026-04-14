# flux-source-code-notes
Flux 核心代码的中文注释与原理解析 

这是一个针对 [flux.1官方仓库](https://github.com/black-forest-labs/flux.git) 的深入解析与中文注释项目。旨在帮助 AI 开发者、研究人员以及对底层生成逻辑感兴趣的同学，更低门槛地啃透这套先进的图像生成架构。

在注释的过程中, 我保留了原来的代码, 并将原来的少部分英文注释翻译成中文, 并补入了自己对代码的理解, 以及.py之间关系的理解

## 声明:
着重注释了原库中的 src/flux 文件, 我的库中没有翻译:

demo_xx.py一系列的文件 -- 官方提供的推理演示脚本

pyproject.toml & setup.py -- Python 的环境和依赖配置文件

## 为了获得最佳的阅读体验，建议按照 _n 的顺序阅读源码:
**说明**: n从1开始到6, 从小到大, 同时这也是我的阅读顺序

xx_1.py => 底层基石(独立工具与数学)

xx_2.py => 核心积木(神经网络层)

xx_3.py => 主架构与大脑(模型与调度)

xx_4.py, xx_5.py => 顶层应用与交互(命令行入口)

xx_6.py => TensorRT的加速相关代码

## 🤝 致谢与交流

感谢 FLUX.1 原始开发团队的开源贡献。如果这份注释对你的学习或科研有帮助，欢迎点个 **Star** ⭐！
如果在阅读中发现我对某段逻辑的理解有误，非常欢迎提交 Issue 或 Pull Request 一起探讨。

## 引用 (Citation)

本项目基于 Black Forest Labs 的 FLUX.1 模型结构与官方代码。如果你在研究中发现官方的这些模型或代码对你有帮助，请考虑引用原作者的工作：

```bibtex
@misc{labs2025flux1kontextflowmatching,
      title={FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space}, 
      author={Black Forest Labs and Stephen Batifol and Andreas Blattmann and Frederic Boesel and Saksham Consul and Cyril Diagne},
      year={2025},
      eprint={2506.15742},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={[https://arxiv.org/abs/2506.15742](https://arxiv.org/abs/2506.15742)}, 
}

@misc{flux2024,
      author={Black Forest Labs},
      title={FLUX},
      year={2024},
      howpublished={\url{[https://github.com/black-forest-labs/flux](https://github.com/black-forest-labs/flux)}},
}
