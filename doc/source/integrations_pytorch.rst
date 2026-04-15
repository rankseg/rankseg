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

The default official configuration is:

.. code-block:: python

   rankseg = RankSEG(metric="dice", solver="RMA", output_mode="multiclass")

This is the most practical setup for standard semantic segmentation and should
be the first configuration users try unless they already know they need a
different metric or output mode.

Minimal integration
-------------------

.. code-block:: python

   import torch
   from rankseg import RankSEG

   model.eval()
   with torch.inference_mode():
       logits = model(images)              # (batch_size, num_classes, *image_shape)
       probs = torch.softmax(logits, dim=1)

       rankseg = RankSEG(metric="dice", solver="RMA", output_mode="multiclass")
       preds = rankseg.predict(probs)      # (batch_size, *image_shape)

For a more complete explanation of:

- probability conventions
- ``softmax`` vs ``sigmoid``
- ``multiclass`` vs ``multilabel``
- input / output tensor shapes
- common integration mistakes

see :doc:`getting_started`.

Official example
----------------

See the maintained example script:

- `examples/pytorch_native_rankseg.py <https://github.com/rankseg/rankseg/blob/main/examples/pytorch_native_rankseg.py>`_
