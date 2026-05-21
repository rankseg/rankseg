Official Integrations
=====================

RankSEG is designed to fit into existing inference pipelines with minimal code
changes. Every integration page in this section answers the same practical
question:

   Given a trained segmentation model, where do I replace the usual
   ``argmax`` or ``0.5`` threshold with ``RankSEG``?

The common insertion point is:

.. code-block:: text

   image -> model -> logits/probabilities -> RankSEG -> prediction mask

For multiclass semantic segmentation, the model usually returns logits with
shape ``(B, C, H, W)``. Convert them to probabilities with ``softmax`` before
calling RankSEG. RankSEG then produces the final mask for the metric you care
about, such as Dice or IoU.

For multilabel segmentation, replace ``softmax`` with ``sigmoid`` and use
``output_mode="multilabel"``.

Inputs and outputs
------------------

.. list-table::
   :widths: 24 34 42
   :header-rows: 1

   * - Item
     - Code object
     - What to check
   * - Model scores
     - ``logits``
     - Tensor of shape ``(B, C, *image_shape)`` before activation.
   * - Probability map
     - ``probs``
     - Tensor in ``[0, 1]`` with shape ``(B, C, *image_shape)``.
   * - Target metric
     - ``RankSEG(metric=...)``
     - ``"dice"``, ``"iou"``, or ``"acc"``.
   * - Prediction family
     - ``output_mode``
     - ``"multiclass"`` returns one class index per pixel; ``"multilabel"``
       returns one binary mask per class.
   * - Solver
     - ``solver``
     - ``"RMA"`` is the recommended default for Dice/IoU inference.
   * - Final prediction
     - ``preds = rankseg.predict(probs)``
     - Tensor consumed by your evaluator or visualization code.

Available integrations
----------------------

.. list-table::
   :widths: 18 27 27 28
   :header-rows: 1

   * - Backend
     - Best starting point
     - Where RankSEG is called
     - Executable tutorial
   * - PyTorch Native
     - You already own the model forward pass.
     - ``RankSEG(...).predict(probs)``
     - `quickstart.ipynb <https://github.com/rankseg/rankseg/blob/main/notebooks/quickstart.ipynb>`__
   * - Transformers
     - You use Hugging Face semantic segmentation through
       ``processor -> model -> outputs``.
     - ``rankseg.integration.transformers.postprocess(...)``
     - `rankseg_with_transformers.ipynb <https://github.com/rankseg/rankseg/blob/main/notebooks/rankseg_with_transformers.ipynb>`__
   * - SAM Family
     - You use SAM1, SAM2, or SAM3 outputs from Hugging Face Transformers.
     - ``sam.Sam1`` / ``sam.Sam2`` / ``sam.Sam3`` adapters
     - `rankseg_with_sam_family.ipynb <https://github.com/rankseg/rankseg/blob/main/notebooks/rankseg_with_sam_family.ipynb>`__
   * - PaddleSeg
     - You use PaddleSeg and can work from the external integration branch.
     - Convert Paddle probabilities to a PyTorch tensor, then call RankSEG.
     - `rankseg_with_paddleseg.ipynb <https://github.com/rankseg/rankseg/blob/main/notebooks/rankseg_with_paddleseg.ipynb>`__

Choose your path
----------------

- Start with :doc:`integrations_pytorch` if your inference code already
  exposes logits or probabilities as PyTorch tensors.
- Start with :doc:`integrations_transformers` if you want to keep the standard
  Hugging Face ``processor -> model -> outputs`` workflow unchanged.
- Start with :doc:`integrations_sam` for SAM-family models, because SAM
  outputs include family-specific geometry restoration before the final mask
  step.
- Start with :doc:`integrations_paddleseg` only when your deployment is already
  in PaddleSeg. It is currently linked as an external/community-maintained
  integration path.

Recommended defaults
--------------------

For ordinary semantic segmentation, the first configuration to try is:

.. code-block:: python

   from rankseg import RankSEG

   rankseg = RankSEG(metric="dice", solver="RMA", output_mode="multiclass")
   preds = rankseg.predict(probs)

This setup targets the Dice metric, uses the efficient RMA solver, and returns
one non-overlapping class label per pixel. Change the metric only when your
evaluation protocol uses a different metric, and change ``output_mode`` only
when your task allows overlapping class masks.

Tutorial pages
--------------

.. toctree::
   :maxdepth: 1

   integrations_pytorch
   integrations_transformers
   integrations_sam
   integrations_paddleseg

Common mistakes
---------------

- Passing raw logits directly to ``RankSEG``. Use ``softmax`` for multiclass
  probabilities or ``sigmoid`` for multilabel probabilities first.
- Mixing output semantics. ``output_mode="multiclass"`` returns
  ``(B, *image_shape)`` class-index masks, while ``output_mode="multilabel"``
  returns ``(B, C, *image_shape)`` binary masks.
- Using SAM outputs with the generic Transformers helper. SAM-family models
  require the explicit adapters documented in :doc:`integrations_sam`.
- Comparing against a dataset-level Dice definition without checking
  aggregation. RankSEG targets the samplewise metric convention explained in
  :doc:`metrics`.

The API reference remains available at :doc:`API`, but these integration pages
are the intended tutorial entry points for applying RankSEG in real inference
code.
