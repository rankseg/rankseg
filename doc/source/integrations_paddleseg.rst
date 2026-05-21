PaddleSeg
=========

This page documents the current PaddleSeg integration status for RankSEG.

Current status
--------------

The PaddleSeg integration already exists, but it is currently maintained
outside the main ``rankseg`` branch.

At this stage, the main repository provides an entry point to that work rather
than duplicating or re-implementing the full PaddleSeg integration locally.

How it fits the RankSEG integration pattern
-------------------------------------------

The insertion point is the same as the first-party integrations:

.. code-block:: text

   PaddleSeg model -> probability map -> convert to PyTorch tensor
   -> RankSEG -> prediction mask

The difference is maintenance scope. RankSEG's public predictor currently
expects a PyTorch tensor, so Paddle outputs need an explicit conversion step
before calling ``RankSEG.predict``.

.. code-block:: python

   import torch
   from rankseg import RankSEG

   # probs_paddle has shape (batch_size, num_classes, height, width)
   probs = torch.from_numpy(probs_paddle.numpy())

   rankseg = RankSEG(metric="dice", solver="RMA", output_mode="multiclass")
   preds = rankseg.predict(probs)

Who should use this path
------------------------

This path is useful if:

- you already deploy segmentation pipelines in PaddleSeg;
- you want to evaluate RankSEG as an inference-time post-processing module in
  the Paddle ecosystem;
- you are comfortable using an externally maintained integration branch.

Available entry points
----------------------

- External integration branch:
  `Leev1s/rankseg (paddleseg branch) <https://github.com/Leev1s/rankseg/tree/paddleseg/rankseg/paddleseg>`_
- Docker image:
  `ghcr.io/leev1s/rankseg <https://ghcr.io/leev1s/rankseg>`_
- Notebook:
  `notebooks/rankseg_with_paddleseg.ipynb <https://github.com/rankseg/rankseg/blob/main/notebooks/rankseg_with_paddleseg.ipynb>`_

Scope and maintenance
---------------------

This integration is currently treated as an external/community-maintained path.

That means:

- the main RankSEG repository links to it;
- the main RankSEG repository does not yet treat it as a first-party official
  integration path;
- once the integration assets become stable enough, it can be promoted into a
  first-party maintained path later.

Relationship to official integrations
-------------------------------------

If you are new to RankSEG, the recommended first entry point remains:

- :doc:`integrations_pytorch`

That path is maintained directly in this repository and is the current default
official integration route.
