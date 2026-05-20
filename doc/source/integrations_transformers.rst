Transformers
============

This page documents the RankSEG integration path for standard Hugging Face
``transformers`` semantic-segmentation outputs.

Use this path when you already run inference through a standard
``processor -> model -> outputs`` workflow and want RankSEG to replace the
final ``argmax``-style prediction step.

Where RankSEG fits
------------------

Hugging Face segmentation models do not all expose the same raw tensor field.
Some return ``outputs.logits``; query-based models return class-query and mask
logits; some processors also own model-specific resizing logic. The RankSEG
Transformers helper keeps the official inference flow intact and handles the
last post-processing step:

.. code-block:: text

   processor(images) -> model(**inputs) -> restore semantic probabilities
   -> RankSEG -> prediction masks

.. code-block:: python

   from rankseg.integration import transformers

The main helper is:

.. code-block:: python

   transformers.postprocess(
       outputs,
       *,
       model=None,
       target_sizes=None,
       rankseg_kwargs=None,
   )

Its role is intentionally narrow:

- restore probabilities from supported Hugging Face output families;
- resize them to the original image size when needed;
- apply ``RankSEG`` as the final post-processing step.

Helper arguments
----------------

.. list-table::
   :widths: 22 34 44
   :header-rows: 1

   * - Argument or return
     - Shape or type
     - Meaning
   * - ``outputs``
     - Structured Transformers output
     - The object returned by ``model(**inputs)``. Tuple-style
       ``return_dict=False`` outputs are intentionally unsupported.
   * - ``model``
     - Optional Transformers model
     - Required for output families whose semantic reconstruction depends on
       the model configuration.
   * - ``target_sizes``
     - List or tensor of ``(height, width)`` pairs
     - One original output size per image. For a PIL image, use
       ``[image.size[::-1]]``.
   * - ``rankseg_kwargs``
     - ``dict`` forwarded to ``RankSEG``
     - Example: ``{"metric": "dice", "solver": "RMA"}``.
   * - Return value
     - ``list[torch.Tensor]``
     - One predicted mask per input image.

Minimal integration
-------------------

The standard Hugging Face inference structure stays the same. The only
integration change happens after ``outputs = model(**inputs)``.

.. code-block:: python

   from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
   from rankseg.integration import transformers
   from PIL import Image
   import requests

   processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
   model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

   image = Image.open(requests.get("https://plus.unsplash.com/premium_photo-1673210886161-bfcc40f54d1f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8cGVyc29uJTIwc3RhbmRpbmd8ZW58MHx8MHx8&w=1000&q=80", stream=True).raw)
   inputs = processor(images=image, return_tensors="pt")
   outputs = model(**inputs)

   preds = transformers.postprocess(
       outputs,
       target_sizes=[image.size[::-1]],
       rankseg_kwargs={"metric": "dice"},
   )

For supported output families, ``transformers.postprocess(...)`` preserves the
surrounding Hugging Face inference code and replaces only the final prediction
step. SAM-family outputs are intentionally handled by ``sam.Sam1``,
``sam.Sam2``, or ``sam.Sam3`` after ``from rankseg.integration import sam``
instead of this helper.

Compare with the usual argmax step
----------------------------------

The usual SegFormer-style baseline is:

.. code-block:: python

   import torch.nn.functional as F

   upsampled_logits = F.interpolate(
       outputs.logits,
       size=image.size[::-1],
       mode="bilinear",
       align_corners=False,
   )
   baseline_pred = upsampled_logits.argmax(dim=1)[0]

The RankSEG version asks the helper to restore probabilities and then produce
the final prediction:

.. code-block:: python

   rankseg_pred = transformers.postprocess(
       outputs,
       target_sizes=[image.size[::-1]],
       rankseg_kwargs={"metric": "dice", "solver": "RMA"},
   )[0]

This is the intended replacement point: the processor, model, checkpoint, and
input preparation remain unchanged.

Advanced probability helper
---------------------------

The namespace also exposes:

.. code-block:: python

   from rankseg.integration import transformers

``transformers.restore_semantic_probs(...)`` returns restored semantic
probability maps directly as a per-image list of ``(C, H, W)`` tensors. Use it
when you need probability tensors instead of final RankSEG predictions.

Pass ``target_sizes`` as one ``(height, width)`` entry per batch item, for
example ``target_sizes=[image.size[::-1]]`` for a single PIL image.
``transformers.postprocess(...)`` follows the same per-image list convention
for prediction outputs.

Explicit helper imports are also supported when you prefer shorter local names:

.. code-block:: python

   from rankseg.integration.transformers import postprocess, restore_semantic_probs

Supported output families
-------------------------

The standard Transformers helper supports the main semantic-segmentation
output families used by ``transformers``:

- ``outputs.logits``
- ``outputs.class_queries_logits`` + ``outputs.masks_queries_logits``
- ``outputs.logits`` + ``outputs.pred_masks``
- ``outputs.semantic_seg``

When a branch requires model-specific handling, pass ``model=...`` so the
helper can follow the corresponding official post-processing behavior.

Output defaults
---------------

If ``rankseg_kwargs`` omits ``output_mode``, the helper chooses:

- ``"multiclass"`` when the restored semantic probability map has more than
  one class channel;
- ``"multilabel"`` when the restored probability map has one channel.

You can override this explicitly:

.. code-block:: python

   preds = transformers.postprocess(
       outputs,
       target_sizes=[image.size[::-1]],
       rankseg_kwargs={
           "metric": "dice",
           "solver": "RMA",
           "output_mode": "multiclass",
       },
   )

Current exclusions
------------------

The simplified API does not currently support:

- SAM-family outputs, which use the explicit adapters in
  ``rankseg.integration.sam``;
- outputs with ``patch_offsets`` that require official patch-merge logic;
- tuple-style outputs such as ``return_dict=False`` returns;
- custom unstructured outputs from ``trust_remote_code=True`` models;
- SegGPT-style ``pred_masks`` semantic reconstruction.

These cases should fail explicitly rather than silently using an incorrect
semantic restoration path.

Executable tutorial
-------------------

The notebook below is written as a user-facing tutorial: it first runs the
official Hugging Face baseline, then repeats the same inference flow with only
the final post-processing step replaced by RankSEG.

- `notebooks/rankseg_with_transformers.ipynb <https://github.com/rankseg/rankseg/blob/main/notebooks/rankseg_with_transformers.ipynb>`_
- `Open in Colab <https://colab.research.google.com/github/rankseg/rankseg/blob/main/notebooks/rankseg_with_transformers.ipynb>`_

The maintained script version is:

- `examples/transformers_rankseg.py <https://github.com/rankseg/rankseg/blob/main/examples/transformers_rankseg.py>`_
