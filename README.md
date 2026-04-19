<div align="center">

# 🧩 RankSEG

#### Boost Segmentation Performance Instantly via Direct Dice/IoU Post-Optimization

[![PyPI](https://badge.fury.io/py/rankseg.svg)](https://pypi.org/project/rankseg/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![GitHub Stars](https://img.shields.io/github/stars/rankseg/rankseg?style=social)](https://github.com/rankseg/rankseg)
[![Documentation](https://img.shields.io/badge/docs-rankseg-brightgreen.svg)](https://rankseg.readthedocs.io/en/latest/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/statmlben/rankseg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c2znXP7_yt_9MrE75p-Ag82LHz-WfKq-?usp=sharing)
[![中文文档](https://img.shields.io/badge/中文文档-CN-red)](https://github.com/rankseg/rankseg/blob/main/README_zh.md)

[![JMLR](https://img.shields.io/badge/JMLR-v24|22.0712-black.svg)](https://www.jmlr.org/papers/v24/22-0712.html)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-black.svg)](https://openreview.net/pdf?id=4tRMm1JJhw)


[**Quick Start**](#-quick-start) | [**Official Integrations**](#-official-integrations) | [**Key Features**](#-key-features) | [**Benchmarks**](#-benchmarks) | [**Citation**](#-citation)
</div>

---

**RankSEG** is a **plug-and-play** post-processing module that boosts segmentation performance (Dice/IoU) during inference. It works with **ANY pre-trained probabilistic segmentation model** (SAM, DeepLab, SegFormer, etc.) without any retraining or fine-tuning.

Explore RankSEG by reading our [documentation](https://rankseg.readthedocs.io/en/latest/).

> If RankSEG improves your segmentation workflow, please consider starring the repo:
> https://github.com/rankseg/rankseg

### 🌟 Why RankSEG?
Conventional methods use `argmax` or fixed `thresholding`, which are **not theoretically optimized** for non-decomposable metrics like Dice or IoU. RankSEG bridges this gap by directly optimizing the target metric, yielding "free" performance gains.

<div align="center">
  <p align="center"><b>Demo: RankSEG vs. Argmax on <i>Segformer ADE20k</i></b></p>
  <img src="./fig/tmpclkalz9y.gif" alt="RankSEG vs Argmax Comparison" width="85%">
</div>

## ⚡ Quick Start

RankSEG is designed to drop into an existing PyTorch segmentation pipeline with just a few lines of code.

### 1. Installation
```bash
pip install -U rankseg
```

### 2. Basic Usage
![](https://raw.githubusercontent.com/rankseg/rankseg/main/fig/rankseg_workflow.svg)

### PyTorch Native Flow

```python
from rankseg import RankSEG
import torch.nn.functional as F

# 1. Initialize RankSEG with the official default configuration
rankseg = RankSEG(metric="dice", solver="RMA", output_mode="multiclass")

# 2. Get probability output from YOUR model
# probs: (batch_size, num_classes, *image_shape)
probs = F.softmax(model_logits, dim=1)

# 3. Get optimized predictions
preds = rankseg.predict(probs)
```

> 💡 **Try it now:**
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c2znXP7_yt_9MrE75p-Ag82LHz-WfKq-?usp=sharing)
>
> Official PyTorch integration:
> [Docs](https://rankseg.readthedocs.io/en/latest/integrations_pytorch.html) · [Example](https://github.com/rankseg/rankseg/blob/main/examples/pytorch_native_rankseg.py)
>
> Hugging Face Transformers integration:
> [Notebook](./notebooks/rankseg_with_transformers.ipynb) · [Colab](https://colab.research.google.com/github/Leev1s/rankseg/blob/feat/transformers-adapter/notebooks/rankseg_with_transformers.ipynb)


## 🔌 Official Integrations

These are the maintained integration entry points documented by this repository.

| Path | Status | Entry |
| :--- | :---: | :--- |
| **PyTorch Native** | **Ready** | [Docs](https://rankseg.readthedocs.io/en/latest/integrations_pytorch.html) · [Example](./examples/pytorch_native_rankseg.py) |
| **Transformers** | **Ready** | [Docs](https://rankseg.readthedocs.io/en/latest/integrations_transformers.html) · [Example](./examples/transformers_rankseg.py) |
<!-- | **MMSegmentation** | Planned | Official integration guide in progress | -->

## 🌐 External Integrations

These integrations already exist, but are currently maintained outside the main
repository.

| Integration | Status | Entry |
| :--- | :---: | :--- |
| **PaddleSeg** | External | [Docs](https://rankseg.readthedocs.io/en/latest/integrations_paddleseg.html) · [Branch](https://github.com/Leev1s/rankseg/tree/paddleseg/rankseg/paddleseg) · [Docker](https://ghcr.io/leev1s/rankseg) |


## ✨ Key Features

- **🚀 Performance Boost**: Consistently improves mIoU/mDice scores over standard `argmax`.
- **🔌 Zero Effort**: Compatible with **any** PyTorch model. No retraining, no fine-tuning.
- **🆓 Training-Free**: Purely post-processing. Works with frozen weights.
- **⚡ Real-time Inference**: Efficient `RMA` (Reciprocal Moment Approximation) solver.
- **🧩 Versatile**: Supports semantic (multi-class) and binary (multi-label) tasks.


## 📊 Benchmarks

RankSEG delivers consistent gains across various architectures and datasets **without touching a single weight**.

| Model | Dataset | mIoU (Argmax) | mIoU (**RankSEG**) | Gain |
| :--- | :--- | :---: | :---: | :---: |
| **DeepLabV3+** | PASCAL VOC | 77.25% | **78.14%** | +0.89% |
| **SegFormer** | PASCAL VOC | 77.57% | **78.59%** | +1.02% |
| **UPerNet** | PASCAL VOC | 79.52% | **80.31%** | +0.79% |
| **SegFormer** | ADE20K | 40.00% | **40.82%** | +0.82% |
| **UPerNet** | ADE20K | 42.86% | **43.84%** | +0.98% |

*Detailed results available in our [NeurIPS 2025 paper](https://openreview.net/forum?id=4tRMm1JJhw).*


## 🧪 Additional Demos

| Framework | Task | Quick Start |
| :--- | :--- | :---: |
| **Segment Anything (SAM)** | Zero-shot Segmentation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Gj-rG3ZnFN5OYTcgdJHfUuiSJtWVpgfu?usp=sharing) |
| **Hugging Face** | Interactive Demo | [![Spaces](https://img.shields.io/badge/%F0%9F%A4%97-Spaces-blue)](https://huggingface.co/spaces/statmlben/rankseg) |


## 🔗 Citation

If you use RankSEG in your research, please cite our papers:

> - Dai, B., & Li, C. (2023). RankSEG: A Consistent Ranking-based Framework for Segmentation. *Journal of Machine Learning Research*, **24**(224), 1-50. [[link]](https://www.jmlr.org/papers/v24/22-0712.html)
> - Wang, Z., & Dai, B. (2025). RankSEG-RMA: An Efficient Segmentation Algorithm via Reciprocal Moment Approximation. *Advances in Neural Information Processing Systems (NeurIPS 2025)*. [[link]](https://openreview.net/pdf?id=4tRMm1JJhw)


```bibtex
@article{dai2023rankseg,
  title={RankSEG: A Consistent Ranking-based Framework for Segmentation},
  author={Dai, Ben and Li, Chunlin},
  journal={Journal of Machine Learning Research},
  volume={24},
  number={224},
  pages={1--50},
  url={https://www.jmlr.org/papers/v24/22-0712.html},
  year={2023}
}

@inproceedings{wang2025rankseg,
  title={RankSEG-RMA: An Efficient Segmentation Algorithm via Reciprocal Moment Approximation},
  author={Wang, Zixun and Dai, Ben},
  booktitle={Advances in Neural Information Processing Systems},
  url={https://arxiv.org/abs/2510.15362},
  year={2025}
}
```

---

<div align="center">
  <p>Star us on GitHub if RankSEG helps your project! ⭐</p>
</div>
