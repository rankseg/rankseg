.. _metrics:

Metrics
=======

This page provides comprehensive definitions of segmentation metrics used in RankSEG, including Dice, IoU, and Accuracy. Understanding these metrics is essential for choosing the right optimization strategy for your segmentation task.

Definitions
-----------

In brief, the definitions of Dice and IoU optimized in ``rankseg`` are consistent with **per-image** level Dice and IoU in [2]_ or ``aggregation_level='samplewise'`` in TorchMetrics [1]_.

Dice Coefficient
****************

The **Dice coefficient**, also known as the F1 score or Sørensen-Dice coefficient, measures the overlap between predicted and ground-truth segmentations. It is the harmonic mean of precision and recall.

The **per-image** level averaged Dice score for class :math:`c` across all samples :math:`i` is computed as:

.. math::
   \boxed{\text{Dice}_c = \frac{1}{n} \sum_{i=1}^{n} \frac{2 \cdot \text{TP}_{ic} + \gamma}{2\text{TP}_{ic} + \text{FP}_{ic} + \text{FN}_{ic} + \gamma} = \frac{1}{n} \sum_{i=1}^{n} \frac{2 \cdot |\hat{\mathbf{y}}_{ic} \cap \mathbf{y}_{ic}| + \gamma}{|\hat{\mathbf{y}}_{ic}| + |\mathbf{y}_{ic}| + \gamma}}

Where :math:`\text{TP}_{ic}`, :math:`\text{FP}_{ic}` and :math:`\text{FN}_{ic}` denote the number of true positive, false positive and false negative pixels for class :math:`c` in sample :math:`i`, respectively.

IoU
***

The **IoU** (Intersection over Union), also known as the Jaccard Index, measures the ratio of intersection to union between predicted and ground-truth segmentations.

The **per-image** level averaged IoU score for class :math:`c` across all samples :math:`i` is computed as:

.. math::
   \boxed{\text{IoU}_c = \frac{1}{n} \sum_{i=1}^{n} \frac{ \text{TP}_{ic} + \gamma}{\text{TP}_{ic} + \text{FP}_{ic} + \gamma} = \frac{1}{n} \sum_{i=1}^{n} \frac{|\hat{\mathbf{y}}_{ic} \cap \mathbf{y}_{ic}| + \gamma}{|\hat{\mathbf{y}}_{ic} \cup \mathbf{y}_{ic}| + \gamma}}

Where :math:`\text{TP}_{ic}` and :math:`\text{FP}_{ic}` denote the number of true positive and false positive pixels for class :math:`c` in sample :math:`i`, respectively.

Properties
**********

- **Range**: Both Dice and IoU have range :math:`[0, 1]`, where 1 indicates perfect overlap and 0 indicates no overlap.
- **Smoothing**: :math:`\gamma` (controlled via ``smooth`` parameter) prevents division by zero. A small value of ``smooth`` is added to the numerator and denominator of both metrics. This helps to avoid division by zero and mitigates the effect of classes with very few positive pixels.

.. note::

   Dice and IoU metrics optimized in ``rankseg`` follows a **samplewise aggregation** strategy, which matches the aggregation level used in TorchMetrics [1]_ (``aggregation_level='samplewise'``). The per-image level Dice and IoU are denoted as :math:`\text{Dice}_c^C` and :math:`\text{IoU}_c^C` in [2]_. 
   
   Moreover, for aggregation over multiple classes, ``rankseg`` supports optimization of both :math:`(\text{mDice}^I, \text{mIoU}^I)` and  :math:`(\text{mDice}^C, \text{mIoU}^C)` in [2]_, see definitions (4)-(8) in [2]_.

References
----------

.. [1] `torchmetrics.segmentation.DiceScore <https://lightning.ai/docs/torchmetrics/stable/segmentation/dice.html>`_

.. [2] Wang, Zifu, etal. "Revisiting evaluation metrics for semantic segmentation: Optimization and evaluation of fine-grained intersection over union." *Advances in Neural Information Processing Systems* 36 (2023): 60144-60225. `paper link <https://arxiv.org/pdf/2310.19252>`__
