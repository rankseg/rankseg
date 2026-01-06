<div align="center">

# 🧩 RankSEG

### 🚀 RankSEG: Boost Segmentation Metrics without Retraining

[![PyPI](https://badge.fury.io/py/rankseg.svg)](https://pypi.org/project/rankseg/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![GitHub Stars](https://img.shields.io/github/stars/rankseg/rankseg?style=social)](https://github.com/rankseg/rankseg)
[![Documentation](https://img.shields.io/badge/docs-rankseg-brightgreen.svg)](https://rankseg.readthedocs.io/en/latest/)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/statmlben/rankseg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c2znXP7_yt_9MrE75p-Ag82LHz-WfKq-?usp=sharing)

[![JMLR](https://img.shields.io/badge/JMLR-v24|22.0712-black.svg)](https://www.jmlr.org/papers/v24/22-0712.html)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-black.svg)](https://openreview.net/pdf?id=4tRMm1JJhw)
[![中文文档](https://img.shields.io/badge/中文文档-CN-red)](./README_zh.md)

<br>

**Improve your existing segmentation models performance in Dice/IoU metrics instantly.**

</div>

**RankSEG** is a **plug-and-play** post-processing module that improves segmentation results during inference. It works with **ANY pre-trained logit/prob-outcome segmentation model** (SAM, DeepLab, SegFormer, UPerNet, etc.) without any retraining or fine-tuning.

Instead of using simple `thresholding` or `argmax` (which don't care about Dice/IoU scores), RankSEG directly optimizes for these metrics - giving you better results without any extra training.

Explore RankSEG by reading our [documentation](https://rankseg.readthedocs.io/en/latest/).

![image](./fig/demo.png)

---

## ⚡ Quick Start

Get started in seconds. RankSEG is designed to be dropped into your existing inference pipeline.

### 1. Installation

```bash
pip install -U rankseg
```

### 2. Optimize Predictions

Add 3 lines of code to your inference loop:

```python
import torch
import torch.nn.functional as F
from rankseg import RankSEG

# 1. Initialize RankSEG (optimizing for Dice score)
rankseg = RankSEG(metric='dice', solver='RMA')

# 2. Get your model's probability outputs (Batch, Class, Height, Width)
# Example: probs = model(images).softmax(dim=1)
probs = F.softmax(torch.randn(4, 21, 256, 256), dim=1)

# 3. Get optimized predictions (No retraining needed!)
preds = rankseg.predict(probs)
```

> 💡 **Try it now:**
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c2znXP7_yt_9MrE75p-Ag82LHz-WfKq-?usp=sharing)

---

## ✨ Key Features

- **🚀 Instant Metric Boost**: Consistently improves mIoU and mDice scores over standard `argmax`.
- **🔌 Plug-and-Play**: Compatible with **any** PyTorch segmentation model.
- **🆓 Training-Free**: Purely post-processing. No gradients, no backprop, no dataset needed.
- **⚡ Efficient**: Optimized solvers (RMA) for real-time inference.
- **🧩 Flexible**: Supports multi-class and multi-label segmentation.

---

## 📊 Why RankSEG?

Standard segmentation methods use `argmax` or thresholding, which are **not optimized** for evaluation metrics like Dice or IoU. RankSEG solves this by directly optimizing the target metric during inference.

**Performance Comparison (No Retraining):**

| Model | Dataset | mIoU (Argmax) | mIoU (RankSEG) | mDice (Argmax) | mDice (RankSEG) |
|-------|---------|---------------|----------------|----------------|-----------------|
| DeepLabV3+ (ResNet101) | PASCAL VOC | 77.25% | **78.14%** ↑0.89% | 82.08% | **83.14%** ↑1.06% |
| SegFormer (MiT-B4) | PASCAL VOC | 77.57% | **78.59%** ↑1.02% | 82.15% | **83.22%** ↑1.07% |
| UPerNet (ConvNeXt) | PASCAL VOC | 79.52% | **80.31%** ↑0.79% | 84.11% | **84.98%** ↑0.87% |
| PSPNet (ResNet101) | Cityscapes | 65.89% | **66.53%** ↑0.64% | 73.55% | **74.28%** ↑0.73% |
| DeepLabV3+ (ResNet101) | Cityscapes | 66.17% | **66.68%** ↑0.51% | 73.71% | **74.33%** ↑0.62% |
| UPerNet (ConvNeXt) | Cityscapes | 68.83% | **69.57%** ↑0.74% | 76.08% | **76.97%** ↑0.89% |
| SegFormer (MiT-B4) | ADE20K | 40.00% | **40.82%** ↑0.82% | 46.50% | **47.57%** ↑1.07% |
| UPerNet (ConvNeXt) | ADE20K | 42.86% | **43.84%** ↑0.98% | 49.61% | **50.85%** ↑1.24% |
| CPT (Swin-Large) | ADE20K | 44.59% | **45.56%** ↑0.97% | 51.27% | **52.58%** ↑1.31% |

*Results from our [NeurIPS 2025 paper](https://openreview.net/forum?id=4tRMm1JJhw).*

---

## 🛠️ Integrations

RankSEG works out-of-the-box with any PyTorch-based segmentation framework.

| Framework | Task | Integration Guide |
| :--- | :--- | :--- |
| **PyTorch (Native)** | Semantic Seg. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c2znXP7_yt_9MrE75p-Ag82LHz-WfKq-?usp=sharing) |
| **SegmentAnything** | Semantic Seg. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Gj-rG3ZnFN5OYTcgdJHfUuiSJtWVpgfu?usp=sharing) |
| **MMSegmentation** | Semantic Seg. | *Coming Soon* |
| **PaddleSeg** | Semantic Seg. | *Coming Soon* |

> **Note**: Don't see your favorite framework? Open an [issue](https://github.com/rankseg/rankseg/issues) or submit a PR!

---

## 📚 Documentation & Resources

- **Full Documentation**: [rankseg.readthedocs.io](https://rankseg.readthedocs.io/en/latest/)
- **JMLR Paper**: [RankSEG: A Consistent Ranking-based Framework for Segmentation](https://www.jmlr.org/papers/v24/22-0712.html)
- **NeurIPS Paper**: [RankSEG-RMA: An Efficient Segmentation Algorithm via Reciprocal Moment Approximation](https://openreview.net/pdf?id=4tRMm1JJhw)

## 🔗 Citation

If you use RankSEG in your research, please cite our papers:

```bibtex
@article{dai2023rankseg,
  title={RankSEG: A Consistent Ranking-based Framework for Segmentation},
  author={Dai, Ben and Li, Chunlin},
  journal={Journal of Machine Learning Research},
  volume={24},
  number={224},
  pages={1--50},
  year={2023}
}

@inproceedings{wang2025rankseg,
  title={RankSEG-RMA: An Efficient Segmentation Algorithm via Reciprocal Moment Approximation},
  author={Wang, Zixun and Dai, Ben},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues or pull requests on our [GitHub repository](https://github.com/rankseg/rankseg).

<div align="center">
  <br>
  <p>If you find RankSEG useful, please give it a star! ⭐</p>
</div>
