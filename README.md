<div align="center">

# 🧩 RankSEG

### RankSEG: A Statistically Consistent Segmentation Prediction Module <br> for Dice and IoU Metrics Optimization

[![PyPI](https://badge.fury.io/py/rankseg.svg)](https://pypi.org/project/rankseg/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![GitHub Stars](https://img.shields.io/github/stars/rankseg/rankseg?style=social)](https://github.com/rankseg/rankseg)
[![Documentation](https://img.shields.io/badge/docs-rankseg-brightgreen.svg)](https://rankseg.readthedocs.io/en/latest/)

[![JMLR](https://img.shields.io/badge/JMLR-v24|22.0712-black.svg)](https://www.jmlr.org/papers/v24/22-0712.html)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-black.svg)](https://openreview.net/pdf?id=4tRMm1JJhw)

</div>

**RankSEG** is a statistically consistent framework for semantic segmentation that provides **plug-and-play** modules to improve segmentation results during inference. It works with **ANY pre-trained segmentation model** without retraining.

RankSEG-based methods are theoretically-grounded segmentation approaches that are **statistically consistent** with respect to popular segmentation metrics like **Dice** and **IoU**. They provide *almost guaranteed* improved performance over traditional thresholding or argmax segmentation methods.


Explore RankSEG by reading our [documentation](https://rankseg.readthedocs.io/en/latest/).

## Why RankSEG?

Traditional segmentation methods use **argmax** or **thresholding** to convert model outputs to predictions. However, these methods are **not optimized** for the actual evaluation metrics (Dice, IoU).

**RankSEG consistently outperforms standard argmax prediction without any model retraining:**

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

*Results from our [NeurIPS 2025 paper](https://openreview.net/forum?id=4tRMm1JJhw). RankSEG uses Dice metric with RMA solver.*

## Key Features

- **🎯 Metric-Optimized**: Directly optimizes for Dice or IoU metrics instead of using generic ad-hoc `argmax` during inference
- **🔌 Plug-and-Play**: Works with ANY pre-trained segmentation model without retraining
- **⚡ Efficient Solvers**: Multiple solver options (BA, TRNA, RMA) for different speed-accuracy trade-offs
- **🧩 Flexible Tasks**: Supports both multi-class and multi-label segmentation tasks, whether objects overlap or not

# Get Started with RankSEG

## Installing RankSEG

`rankseg` is available on PyPI. Run the following command to get the latest version of the package:

```bash
pip install -U rankseg
```

## First Steps with RankSEG

Once you installed `rankseg`, you can run the following code snippet to optimize predictions from your pre-trained segmentation model:

```python
import torch
import torch.nn.functional as F
from rankseg import RankSEG

# Your pre-trained model's probability output
probs = F.softmax(torch.randn(4, 21, 256, 256), dim=1)  # (batch, classes, height, width)

# Create RankSEG predictor optimized for Dice metric
rankseg = RankSEG(metric='dice', solver='RMA')

# Get optimized predictions
preds = rankseg.predict(probs)
```

You can refer to the [documentation](https://rankseg.readthedocs.io/en/latest/) to explore more options and detailed API reference.

## Cite RankSEG

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

## Useful Links

- **Documentation**: https://rankseg.readthedocs.io/en/latest/
- **GitHub Repository**: https://github.com/rankseg/rankseg
- **PyPI Package**: https://pypi.org/project/rankseg/
- **JMLR Paper**: https://www.jmlr.org/papers/v24/22-0712.html
- **NeurIPS Paper**: https://openreview.net/pdf?id=4tRMm1JJhw
- **Issue Tracker**: https://github.com/rankseg/rankseg/issues

## Contributing

We welcome contributions! Please feel free to submit issues or pull requests on our [GitHub repository](https://github.com/rankseg/rankseg).
