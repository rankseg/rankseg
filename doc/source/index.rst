.. RankSEG documentation master file, created by
   sphinx-quickstart on Nov 6, 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

🧩 RankSEG
==========

.. .. image:: _static/logo.png
..    :width: 28%
..    :align: right

.. -*- mode: rst -*-

|PyPi|_ |BSD|_ |Python3|_ |PyTorch|_ |GitHub|_  |Docs|_

|JMLR|_ |NeurIPS|_

.. |PyPi| image:: https://badge.fury.io/py/rankseg.svg
.. _PyPi: https://pypi.org/project/rankseg/

.. |BSD| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
.. _BSD: https://opensource.org/licenses/BSD-3-Clause

.. |Python3| image:: https://img.shields.io/badge/python-3.10+-blue.svg
.. _Python3: https://www.python.org

.. |PyTorch| image:: https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white
.. _PyTorch: https://pytorch.org

.. |Downloads| image:: https://static.pepy.tech/badge/rankseg
.. _Downloads: https://pepy.tech/project/rankseg

.. |GitHub| image:: https://img.shields.io/github/stars/rankseg/rankseg?style=social
.. _GitHub: https://github.com/rankseg/rankseg

.. |JMLR| image:: https://img.shields.io/badge/JMLR-v24|22.0712-black.svg
.. _JMLR: https://www.jmlr.org/papers/v24/22-0712.html

.. |NeurIPS| image:: https://img.shields.io/badge/NeurIPS-2025-black.svg
.. _NeurIPS: https://openreview.net/pdf?id=4tRMm1JJhw

.. |Docs| image:: https://img.shields.io/badge/docs-rankseg-brightgreen.svg
.. _Docs: https://rankseg.readthedocs.io/en/latest/


**RankSEG** is a statistically consistent framework for semantic segmentation that provides *plug-and-play* modules to improve segmentation results during inference.

RankSEG-based methods are theoretically-grounded segmentation approaches that are **statistically consistent** with respect to popular segmentation metrics like **Dice** and **IoU**. They provide *almost guaranteed* improved performance over traditional thresholding or argmax segmentation methods.

.. note::
    RankSEG optimizes metrics using a *samplewise* aggregation: the score is computed per sample and then averaged across the dataset (akin to ``aggregation_level='samplewise'`` in `TorchMetrics DiceScore <https://lightning.ai/docs/torchmetrics/stable/segmentation/dice.html>`_). See :ref:`metrics` for details.

Key Properties
--------------

.. list-table::
    :widths: 25 75

    * - **🎯 Metric-Optimized**
      - Directly optimizes for Dice or IoU metrics instead of using generic ad-hoc `argmax` during inference.
    * - **🔌 Plug-and-Play**
      - Works with ANY pre-trained logit/prob-outcome segmentation model without retraining
    * - **⚡ Efficient Solvers**
      - Multiple solver options (BA, TRNA, RMA) for different speed-accuracy trade-offs
    * - **🧩 Flexible Tasks**
      - Supports both multi-class and multi-label segmentation tasks, whether objects overlap or not.

✨ Quick Start
--------------

Install ``rankseg`` using ``pip``:

.. code-block:: bash

   pip install rankseg

Basic usage example:

.. code-block:: python

   import torch
   import torch.nn.functional as F
   from rankseg import RankSEG

   # Your pre-trained model's probability output
   probs = F.softmax(torch.randn(4, 21, 256, 256), dim=1) # (batch, classes, height, width)

   # Create RankSEG predictor optimized for Dice metric
   rankseg = RankSEG(metric='dice')
   
   # Get optimized predictions
   preds = rankseg.predict(probs)

Why RankSEG?
------------

Traditional segmentation methods use **argmax** or **thresholding** to convert model outputs to predictions. However, these methods are not optimized for the actual evaluation metrics (Dice, IoU).

**Performance Improvements Across Models and Datasets:**

RankSEG consistently outperforms standard argmax prediction without any model retraining:

.. list-table::
   :widths: 25 20 12 12 12 12
   :header-rows: 1
   :align: left

   * - Model
     - Dataset
     - mIoU (Argmax)
     - mIoU (RankSEG)
     - mDice (Argmax)
     - mDice (RankSEG)
   * - DeepLabV3+ (ResNet101)
     - PASCAL VOC
     - 77.25%
     - **78.14%** ↑0.89%
     - 82.08%
     - **83.14%** ↑1.06%
   * - SegFormer (MiT-B4)
     - PASCAL VOC
     - 77.57%
     - **78.59%** ↑1.02%
     - 82.15%
     - **83.22%** ↑1.07%
   * - UPerNet (ConvNeXt)
     - PASCAL VOC
     - 79.52%
     - **80.31%** ↑0.79%
     - 84.11%
     - **84.98%** ↑0.87%
   * - PSPNet (ResNet101)
     - Cityscapes
     - 65.89%
     - **66.53%** ↑0.64%
     - 73.55%
     - **74.28%** ↑0.73%
   * - DeepLabV3+ (ResNet101)
     - Cityscapes
     - 66.17%
     - **66.68%** ↑0.51%
     - 73.71%
     - **74.33%** ↑0.62%
   * - UPerNet (ConvNeXt)
     - Cityscapes
     - 68.83%
     - **69.57%** ↑0.74%
     - 76.08%
     - **76.97%** ↑0.89%
   * - SegFormer (MiT-B4)
     - ADE20K
     - 40.00%
     - **40.82%** ↑0.82%
     - 46.50%
     - **47.57%** ↑1.07%
   * - UPerNet (ConvNeXt)
     - ADE20K
     - 42.86%
     - **43.84%** ↑0.98%
     - 49.61%
     - **50.85%** ↑1.24%
   * - CPT (Swin-Large)
     - ADE20K
     - 44.59%
     - **45.56%** ↑0.97%
     - 51.27%
     - **52.58%** ↑1.31%


.. note::
    Results from our `NeurIPS 2025 paper <https://openreview.net/forum?id=4tRMm1JJhw>`_. RankSEG uses Dice metric with RMA solver.


Learn More
----------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 📖 Getting Started
      :link: getting_started
      :link-type: doc

      Learn how to use RankSEG with your segmentation models

   .. grid-item-card:: 📚 API Reference
      :link: API
      :link-type: doc

      Detailed documentation of all classes and functions

   .. grid-item-card:: 📝 Citation
      :link: citation
      :link-type: doc

      How to cite RankSEG in your research

   .. grid-item-card:: 💻 GitHub
      :link: https://github.com/rankseg/rankseg
      
      Source code, issues, and contributions

References
----------
.. bibliography::

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   metrics
   citation
   API

