.. RankSEG documentation master file, created by
   sphinx-quickstart on Nov 6, 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

🧩 RankSEG
==========

.. image:: figs/logo.png
   :width: 18%
   :align: right

.. -*- mode: rst -*-

|PyPi|_ |BSD|_ |Python3|_ |PyTorch|_ |downloads|_

.. |PyPi| image:: https://badge.fury.io/py/rankseg.svg
.. _PyPi: https://pypi.org/project/rankseg/

.. |BSD| image:: https://img.shields.io/pypi/l/rankseg.svg
.. _BSD: https://opensource.org/licenses/BSD-3-Clause

.. |Python3| image:: https://img.shields.io/badge/python-3-blue.svg
.. _Python3: https://www.python.org

.. |PyTorch| image:: https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white
.. _PyTorch: https://pytorch.org

.. |downloads| image:: https://static.pepy.tech/badge/rankseg
.. _downloads: https://pepy.tech/project/rankseg


**RankSEG** is a statistically consistent framework for semantic segmentation that provides *plug-and-play* modules to improve segmentation results during inference.

- GitHub repo: `https://github.com/statmlben/rankseg <https://github.com/statmlben/rankseg>`_
- Documentation: `https://rankseg.readthedocs.io <https://rankseg.readthedocs.io/en/latest/>`_
- PyPi: `https://pypi.org/project/rankseg <https://pypi.org/project/rankseg>`_
- Paper: `JMLR | 2024 <https://www.jmlr.org/papers/v24/22-0712.html>`_

RankSEG-based methods are theoretically-grounded segmentation approaches that are **statistically consistent** with respect to popular segmentation metrics like **Dice**, **IoU**, and **AP**. They provide *almost guaranteed* improved performance over traditional thresholding or argmax segmentation methods.

Key Properties
--------------

.. list-table::
    :widths: 25 75

    * - **🎯 Metric-Optimized**
      - Directly optimizes for Dice, IoU, or AP metrics instead of using proxy losses during inference
    * - **🔌 Plug-and-Play**
      - Works with ANY pre-trained segmentation model without retraining
    * - **📊 Statistically Consistent**
      - Theoretically guaranteed to converge to the optimal segmentation under the target metric
    * - **⚡ Efficient Solvers**
      - Multiple solver options (BA, TRNA, RMA) for different speed-accuracy trade-offs
    * - **🧩 Flexible Tasks**
      - Supports both multi-class and multi-label segmentation tasks

✨ Quick Start
--------------

Install ``rankseg`` using ``pip``:

.. code-block:: bash

   pip install rankseg

Basic usage example:

.. code-block:: python

   import torch
   from rankseg import RankSEG

   # Your pre-trained model's probability output
   probs = torch.rand(4, 21, 256, 256)  # (batch, classes, height, width)

   # Create RankSEG predictor optimized for Dice metric
   rankseg = RankSEG(metric='dice', solver='RMA')
   
   # Get optimized predictions
   preds = rankseg.predict(probs)

🔧 Supported Metrics & Solvers
-------------------------------

RankSEG supports multiple segmentation metrics and solver algorithms:

**Metrics:**
   - ``'dice'``: Dice coefficient (F1 score)
   - ``'IoU'``: Intersection over Union
   - ``'AP'``: Average Precision

**Solvers:**
   - ``'BA'``: Blind Approximation (fast, good for Dice)
   - ``'TRNA'``: Truncated Refined Normal Approximation (accurate)
   - ``'BA+TRNA'``: Automatic selection based on data
   - ``'RMA'``: Reciprocal Moment Approximation (versatile, supports all metrics)

📚 Citation
-----------

If you use this code, please star 🌟 the repository and cite the following paper:

.. code-block:: bibtex

   @article{dai2024rankseg,
      title={RankSEG: A Statistically Consistent Framework for Segmentation},
      author={Dai, Ben and Wang, Zixun},
      journal={Journal of Machine Learning Research},
      volume={24},
      year={2024}
   }


.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
