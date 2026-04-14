# flux-source-code-notes
Chinese Annotations and Conceptual Analysis of the FLUX Core Source Code

This project provides a detailed, Chinese-annotated walkthrough of the [official FLUX.1 repository](https://github.com/black-forest-labs/flux.git). It is designed to help AI developers, researchers, and anyone interested in the underlying mechanics of generative models better understand this advanced image generation architecture with a lower barrier to entry.

While annotating the codebase, I preserved the original implementation, translated part of the original English comments into Chinese, and added my own explanations of the code logic, as well as the relationships between different `.py` files.

## Disclaimer

This project focuses primarily on the `src/flux` directory from the original repository. The following parts are **not** annotated in this version:

- `demo_xx.py` files — inference demo scripts provided by the official repository
- `pyproject.toml` and `setup.py` — Python environment and dependency configuration files

## Recommended Reading Order

For the best reading experience, I recommend going through the source files in the order of `_n`.

**Note:** `n` ranges from 1 to 6 in ascending order, which also reflects the order in which I studied the codebase.

- `xx_1.py` → Foundational building blocks (standalone utilities and mathematical components)
- `xx_2.py` → Core modules (neural network layers)
- `xx_3.py` → Main architecture and model logic (models and schedulers)
- `xx_4.py`, `xx_5.py` → High-level applications and interfaces (command-line entry points)
- `xx_6.py` → TensorRT acceleration-related code

## 🤝 Acknowledgments & Contributions

My sincere thanks to the original FLUX.1 team for open-sourcing their work.

If this annotated project helps with your learning or research, a **Star** ⭐ would be greatly appreciated.

If you find any misunderstandings, inaccuracies, or areas that could be explained better, feel free to open an Issue or submit a Pull Request. Discussion and corrections are always welcome.

## Citation

This project is based on the FLUX.1 model architecture and official code released by Black Forest Labs. If the official models or code have been helpful to your research, please consider citing the original authors' work:

```bibtex
@misc{labs2025flux1kontextflowmatching,
      title={FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space}, 
      author={Black Forest Labs and Stephen Batifol and Andreas Blattmann and Frederic Boesel and Saksham Consul and Cyril Diagne},
      year={2025},
      eprint={2506.15742},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2506.15742}, 
}

@misc{flux2024,
      author={Black Forest Labs},
      title={FLUX},
      year={2024},
      howpublished={\url{https://github.com/black-forest-labs/flux}},
}
