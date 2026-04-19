<div align="center">

# 🧩 RankSEG: 无需重新训练即可瞬间提升分割模型的 Dice/IoU 指标

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
[![English Documentation](https://img.shields.io/badge/English-EN-blue)](https://github.com/rankseg/rankseg/blob/main/README.md)

</div>

---

**RankSEG** 是一个**即插即用**的后处理模块，可在推理过程中改善分割结果。它适用于**任何预训练的概率输出分割模型**（SAM, DeepLab, SegFormer, UPerNet 等），无需任何重新训练或微调。

不像使用简单的`阈值化`或`argmax`（这些方法不关心 Dice/IoU 分数），RankSEG 直接针对这些指标进行优化，从而为您提供更好的结果，而无需任何额外的训练。

了解 RankSEG 的更多信息，请查看[我们的文档](https://rankseg.readthedocs.io/en/latest/)。

> 如果 RankSEG 对您的分割工作流有帮助，欢迎给项目点一个 star：
> https://github.com/rankseg/rankseg

<!--![image](./fig/rankseg.png)-->

<div align="center">
  <img src="./fig/rankseg.png" alt="RankSEG Overview">
</div>

## 🌟 为什么选择 RankSEG?

传统分割通常使用 `argmax` 或固定阈值，但这些方法并没有直接针对 Dice / IoU 等非可分解指标进行优化。RankSEG 在推理阶段直接优化目标指标，因此在不重训模型的情况下，往往可以获得“免费”的性能提升。

## ⚡ 快速开始

RankSEG 可以直接插入现有的 PyTorch 分割推理流程中。

### 1. 安装

```bash
pip install -U rankseg
```

### 2. 基本用法

![](https://raw.githubusercontent.com/rankseg/rankseg/main/fig/rankseg_workflow.svg)

```python
from rankseg import RankSEG
import torch.nn.functional as F

# 1. 使用官方默认配置初始化 RankSEG
rankseg = RankSEG(metric="dice", solver="RMA", output_mode="multiclass")

# 2. 获取模型的概率输出
# probs: (batch_size, num_classes, *image_shape)
probs = F.softmax(model_logits, dim=1)

# 3. 获取优化后的预测结果
preds = rankseg.predict(probs)
```

> 💡 **立即尝试:**
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c2znXP7_yt_9MrE75p-Ag82LHz-WfKq-?usp=sharing)
>
> 官方 PyTorch 集成路径：
> [Docs](https://rankseg.readthedocs.io/en/latest/integrations_pytorch.html) · [Example](./examples/pytorch_native_rankseg.py)
>
> Transformers 集成路径：
> [Notebook](./notebooks/rankseg_with_transformers.ipynb) · [Colab](https://colab.research.google.com/github/Leev1s/rankseg/blob/feat/transformers-adapter/notebooks/rankseg_with_transformers.ipynb)

## 🔌 官方集成路径

这些是当前由本仓库维护的官方集成入口。

| 路径 | 状态 | 入口 |
| :--- | :---: | :--- |
| **PyTorch Native** | **Ready** | [Docs](https://rankseg.readthedocs.io/en/latest/integrations_pytorch.html) · [Example](./examples/pytorch_native_rankseg.py) |
| **Transformers** | **Ready** | [Docs](https://rankseg.readthedocs.io/en/latest/integrations_transformers.html) · [Example](./examples/transformers_rankseg.py) |

## 🌐 外部集成路径

以下集成已经存在，但当前主要由主仓库之外的实现维护。

| 集成 | 状态 | 入口 |
| :--- | :---: | :--- |
| **PaddleSeg** | External | [Docs](https://rankseg.readthedocs.io/en/latest/integrations_paddleseg.html) · [Branch](https://github.com/Leev1s/rankseg/tree/paddleseg/rankseg/paddleseg) · [Docker](https://ghcr.io/leev1s/rankseg) |

## ✨ 主要特性

- **🚀 指标瞬间提升**：相比标准的 `argmax`，持续提升 mIoU 和 mDice 分数。
- **🔌 即插即用**：兼容**任何** PyTorch 分割模型。无需重训。
- **🆓 无需训练**：纯后处理。无需梯度、无需反向传播、无需数据集。
- **⚡ 高效默认路径**：推荐使用 `RMA` 作为默认推理求解器。
- **🧩 灵活**：支持多类和多标签分割任务。

## 📊 Benchmarks

RankSEG 在不改动模型权重的情况下，能在多个模型和数据集上持续带来稳定增益。

| 模型 | 数据集 | mIoU (Argmax) | mIoU (RankSEG) | mDice (Argmax) | mDice (RankSEG) |
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

*结果来自我们的 [NeurIPS 2025 论文](https://openreview.net/forum?id=4tRMm1JJhw)。*

## 🧪 更多演示

以下内容作为额外演示和扩展生态入口保留：

| 框架 | 任务 | 快速入口 |
| :--- | :--- | :--- |
| **SegmentAnything** | 语义分割 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Gj-rG3ZnFN5OYTcgdJHfUuiSJtWVpgfu?usp=sharing) |
| **Hugging Face** | 互动演示 | [![Spaces](https://img.shields.io/badge/%F0%9F%A4%97-Spaces-blue)](https://huggingface.co/spaces/statmlben/rankseg) |

## 🔗 引用

如果您在研究中使用了 RankSEG，请引用我们的论文：

- Dai, B., & Li, C. (2023). RankSEG: A Consistent Ranking-based Framework for Segmentation. *Journal of Machine Learning Research*, **24**(224), 1-50. [[link]](https://www.jmlr.org/papers/v24/22-0712.html)
- Wang, Z., & Dai, B. (2025). RankSEG-RMA: An Efficient Segmentation Algorithm via Reciprocal Moment Approximation. *Advances in Neural Information Processing Systems (NeurIPS 2025)*. [[link]](https://openreview.net/pdf?id=4tRMm1JJhw)


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

## 🤝 贡献

我们欢迎贡献！请随时在我们的 [GitHub 仓库](https://github.com/rankseg/rankseg)上提交 issue 或 pull request。

<div align="center">
  <br>
  <p>如果您觉得 RankSEG 有用，请给我们一颗星！ ⭐</p>
</div>
