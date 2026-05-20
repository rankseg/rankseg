.. image:: _static/Pytorch_logo.png
   :width: 150px
   :align: right

PyTorch Native
==============

This page documents the official RankSEG integration path for standard PyTorch
semantic-segmentation inference.

If you already have a PyTorch semantic-segmentation model and want to replace
``argmax`` with a better inference-time post-processing step, this is the
recommended starting point.

Where RankSEG fits
------------------

Most PyTorch segmentation code has three stages:

.. code-block:: text

   images -> model(images) -> logits -> argmax/threshold -> masks

RankSEG keeps the model and logits unchanged. The only change is the final
prediction step:

.. code-block:: text

   images -> model(images) -> logits -> probabilities -> RankSEG -> masks

The default official configuration is:

.. code-block:: python

   rankseg = RankSEG(metric="dice", solver="RMA", output_mode="multiclass")

This is the most practical setup for standard semantic segmentation and should
be the first configuration users try unless they already know they need a
different metric or output mode.

Minimal integration
-------------------

If your model already returns a logits tensor, the integration is only a few
lines:

.. code-block:: python

   import torch
   from rankseg import RankSEG

   model.eval()
   with torch.inference_mode():
       logits = model(images)              # (batch_size, num_classes, *image_shape)
       probs = torch.softmax(logits, dim=1)

       rankseg = RankSEG(metric="dice", solver="RMA", output_mode="multiclass")
       preds = rankseg.predict(probs)      # (batch_size, *image_shape)

Usual PyTorch inference vs RankSEG inference
--------------------------------------------

This example is self-contained and uses a tiny convolutional model only to show
the tensor contract. Replace ``model`` with your trained network.

.. code-block:: python

   import torch
   from rankseg import RankSEG

   model = torch.nn.Conv2d(in_channels=3, out_channels=21, kernel_size=1)
   images = torch.randn(2, 3, 256, 256)

   model.eval()
   with torch.inference_mode():
       logits = model(images)
       probs = torch.softmax(logits, dim=1)

       baseline_preds = probs.argmax(dim=1)

       rankseg = RankSEG(metric="dice", solver="RMA", output_mode="multiclass")
       rankseg_preds = rankseg.predict(probs)

   print("logits:  ", tuple(logits.shape))          # (2, 21, 256, 256)
   print("baseline:", tuple(baseline_preds.shape))  # (2, 256, 256)
   print("RankSEG: ", tuple(rankseg_preds.shape))   # (2, 256, 256)

Choosing options
----------------

.. list-table::
   :widths: 22 34 44
   :header-rows: 1

   * - Option
     - Recommended starting value
     - When to change it
   * - ``metric``
     - ``"dice"``
     - Use ``"iou"`` when your benchmark or report is IoU-centered.
   * - ``solver``
     - ``"RMA"``
     - Start here for Dice/IoU. Use other solvers only when you need their specific trade-offs.
   * - ``output_mode``
     - ``"multiclass"``
     - Use ``"multilabel"`` when the final masks may overlap.
   * - Activation before RankSEG
     - ``torch.softmax(logits, dim=1)``
     - Use ``torch.sigmoid(logits)`` for independent multilabel probabilities.

Notebook and script
-------------------

For a notebook-style walkthrough with a pretrained model, see:

- `notebooks/quickstart.ipynb <https://github.com/rankseg/rankseg/blob/main/notebooks/quickstart.ipynb>`_
- `Open in Colab <https://colab.research.google.com/github/rankseg/rankseg/blob/main/notebooks/quickstart.ipynb>`_

The maintained script version is:

- `examples/pytorch_native_rankseg.py <https://github.com/rankseg/rankseg/blob/main/examples/pytorch_native_rankseg.py>`_

For a more complete explanation of:

- probability conventions
- ``softmax`` vs ``sigmoid``
- ``multiclass`` vs ``multilabel``
- input / output tensor shapes
- common integration mistakes

see :doc:`getting_started`.
