<div align="center">

# 🧩 RankSEG

#### 无需重新训练的 Dice/IoU 指标感知后处理

[![PyPI](https://badge.fury.io/py/rankseg.svg)](https://pypi.org/project/rankseg/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![GitHub Stars](https://img.shields.io/github/stars/rankseg/rankseg?style=social)](https://github.com/rankseg/rankseg)
[![Documentation](https://img.shields.io/badge/docs-rankseg-brightgreen.svg)](https://rankseg.readthedocs.io/en/latest/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/statmlben/rankseg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c2znXP7_yt_9MrE75p-Ag82LHz-WfKq-?usp=sharing)
[![English Documentation](https://img.shields.io/badge/English-EN-blue)](https://github.com/rankseg/rankseg/blob/main/README.md)

[![JMLR](https://img.shields.io/badge/JMLR-v24|22.0712-black.svg)](https://www.jmlr.org/papers/v24/22-0712.html)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-black.svg)](https://openreview.net/pdf?id=4tRMm1JJhw)


[**新闻**](#-新闻) | [**快速开始**](#-快速开始) | [**基准测试**](#-基准测试) | [**生态集成**](#-生态集成) | [**文档**](https://rankseg.readthedocs.io/en/latest/) | [**引用**](#-引用)
</div>

---

**RankSEG** 以指标感知的后处理替代 `argmax` 或固定阈值，旨在改善 Dice 或 IoU，
**无需重新训练或微调**。可直接用于冻结参数的 PyTorch 分割模型，
支持多类、二分类和多标签任务，覆盖自然图像到 3D 医学影像。

<div align="center">
  <p><b>RankSEG 应用于 3D CT</b> · 相同模型，无需重训。</p>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./fig/monai_pancreas_rankseg_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="./fig/monai_pancreas_rankseg.png">
    <img src="./fig/monai_pancreas_rankseg.png" alt="胰腺 CT 切片：真实标注、argmax（Dice 35.0）与 RankSEG（Dice 49.3）。白色轮廓表示真实标注。" width="100%">
  </picture>
  <p><b>20 例三维 CT 的平均 Dice：50.30 → 54.87（+4.57 个百分点）</b><br>
  冻结的 MONAI BTCV Swin UNETR · MSD Pancreas 数据集。</p>
  <p>上图为示例切片；白色轮廓表示真实标注。仅供研究使用。<br>
  <a href="https://github.com/rankseg/rankseg-benchmark#monai-datasets-and-checkpoints">评测协议与示例选择方式</a>。</p>
</div>

## 📰 新闻

- **2026 年 8 月——RankSEG 已正式收录于 MONAI 官方 Tutorials！** 体验面向 3D 医学分割的指标感知后处理。[查看教程](https://github.com/Project-MONAI/tutorials/blob/main/modules/rankseg_integration.ipynb) · [Colab](https://colab.research.google.com/github/Project-MONAI/tutorials/blob/main/modules/rankseg_integration.ipynb)

## ⚡ 快速开始

```bash
pip install -U rankseg
```

对于形状为 `(batch, classes, *spatial)`、至少包含两个类别的多类模型 logits：

```python
from rankseg import RankSEG

probs = model_logits.softmax(dim=1)
preds = RankSEG(metric="dice")(probs)  # 替代 argmax；输出形状为 (batch, *spatial)
```

二分类／多标签示例、函数式 API 及求解器选项，请参阅
[入门指南](https://rankseg.readthedocs.io/en/latest/getting_started.html)。

**在线体验：** [Colab](https://colab.research.google.com/drive/1c2znXP7_yt_9MrE75p-Ag82LHz-WfKq-?usp=sharing) · [交互演示](https://huggingface.co/spaces/statmlben/rankseg)

## 📊 基准测试

**相同的概率输出，无需重训。** 以下精选结果比较了 `argmax` 与采用 Dice 目标的
RankSEG-RMA。分数以百分比表示，增益以百分点（pp）表示。

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./fig/benchmark_results_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="./fig/benchmark_results.png">
    <img src="./fig/benchmark_results.png" alt="RankSEG 在五个精选数据集上与 argmax 的 Dice 和 IoU 对比结果" width="100%">
  </picture>
</div>

不同数据集上的增益有所差异。完整结果、指标定义、运行时间及复现命令见
[rankseg-benchmark](https://github.com/rankseg/rankseg-benchmark)。
算法细节及更多实验见我们的 [NeurIPS 2025 论文](https://openreview.net/forum?id=4tRMm1JJhw)。

## 🔌 生态集成

选择适合你的工作流：

| 生态 | 入门 | 在线体验 |
| :--- | :--- | :--- |
| **PyTorch** | [文档](https://rankseg.readthedocs.io/en/latest/integrations_pytorch.html) · [示例](./examples/pytorch_native_rankseg.py) | [Colab](https://colab.research.google.com/drive/1c2znXP7_yt_9MrE75p-Ag82LHz-WfKq-?usp=sharing) |
| **Hugging Face** | [文档](https://rankseg.readthedocs.io/en/latest/integrations_transformers.html) · [Notebook](./notebooks/rankseg_with_transformers.ipynb) | [Colab](https://colab.research.google.com/github/rankseg/rankseg/blob/main/notebooks/rankseg_with_transformers.ipynb) |
| **SAM 系列** | [文档](https://rankseg.readthedocs.io/en/latest/integrations_sam.html) · [Notebook](./notebooks/rankseg_with_sam_family.ipynb) | [Colab](https://colab.research.google.com/github/rankseg/rankseg/blob/main/notebooks/rankseg_with_sam_family.ipynb) |
| **MONAI** | [文档](https://rankseg.readthedocs.io/en/latest/integrations_monai.html) · [官方教程](https://github.com/Project-MONAI/tutorials/blob/main/modules/rankseg_integration.ipynb) | [Colab](https://colab.research.google.com/github/Project-MONAI/tutorials/blob/main/modules/rankseg_integration.ipynb) |
| **PaddleSeg**（外部维护） | [文档](https://rankseg.readthedocs.io/en/latest/integrations_paddleseg.html) · [社区分支](https://github.com/Leev1s/rankseg/tree/paddleseg/rankseg/paddleseg) | — |

## 🔗 引用

如果您在研究中使用了 RankSEG，请引用我们的论文：

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
  <p>如果 RankSEG 对你的项目有帮助，欢迎在 GitHub 上点一颗星！⭐</p>
</div>
