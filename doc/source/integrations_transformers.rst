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
     - One positive-integer output size per image. For a PIL image, use
       ``[image.size[::-1]]``. A tensor must have integer dtype and shape
       ``(B, 2)``.
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

For ordinary dense ``outputs.logits``, the helper first resizes logits to the
requested target size, then applies ``sigmoid`` to a single-channel binary
output or class-wise ``softmax`` to a multi-channel output. In particular,
softmax is never applied to a single channel, where it would otherwise produce
an all-one probability map.

For query-based outputs, multi-channel semantic scores are normalized across
classes for every positive total, including very small ``float64`` totals. A
pixel whose class scores are all exactly zero remains all-zero rather than
producing an undefined ``0 / 0`` result. A single foreground channel instead
retains its spatial semantic score and clips only overlap sums outside
``[0, 1]``; dividing that channel by itself would incorrectly turn every
nonzero pixel into probability one.

Pass ``target_sizes`` as one ``(height, width)`` entry per batch item, for
example ``target_sizes=[image.size[::-1]]`` for a single PIL image.
``transformers.postprocess(...)`` follows the same per-image list convention
for prediction outputs.

The helper validates model-output geometry before reconstruction. Dense logits
must have shape ``(B, C, H, W)`` with at least one class and non-empty spatial
dimensions. Query-based class logits must have shape ``(B, Q, C + 1)`` (the
last entry is the no-object class), mask logits must have shape
``(B, Q, H, W)``, and their batch/query dimensions and devices must match.
An empty batch is valid and produces an empty list.

Model outputs must use a real floating-point dtype and contain only finite
values; NaN and positive or negative infinity are rejected before probability
restoration. ``float16`` and ``bfloat16`` values are promoted to ``float32``
for stable interpolation and probability calculations; ``float32`` and
``float64`` retain their supplied precision. Query class and mask logits with
different floating dtypes are promoted to their common working dtype.

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

DETR and Conditional DETR ``logits`` + ``pred_masks`` outputs can be restored
without retaining the model object. Pass ``model=...`` only when model config
is needed to identify family-specific behavior; for example, an untyped
``class_queries_logits`` + ``masks_queries_logits`` container from
Mask2Former needs its model config to select Mask2Former's pre-resize step.
The native ``Mask2FormerForUniversalSegmentationOutput`` type is recognized
without a model. Output/config subclasses are recognized through their class
inheritance chain, and lightweight config wrappers can expose the standard
``model_type="mask2former"`` identifier instead of preserving the exact
Transformers config class.

``outputs.semantic_seg`` is a SAM3-specific field and is intentionally routed
away from this generic helper. Native SAM output classes, their subclasses,
and mappings or wrappers with characteristic SAM named fields fail with an
explicit message directing callers to ``sam.Sam1``, ``sam.Sam2``, or
``sam.Sam3``. This keeps cached or wrapped SAM outputs on the same
family-specific geometry and shape contract as native outputs.

Output defaults
---------------

If ``rankseg_kwargs`` omits ``output_mode``, the helper chooses:

- ``"multiclass"`` when the restored semantic probability map has more than
  one class channel;
- ``"multilabel"`` when the restored probability map has one channel.

Accordingly, the default multi-channel prediction is a ``torch.int64`` class
index map, while the default single-channel prediction is a ``torch.bool``
mask. This dtype contract is independent of the model probability dtype.

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
- EOMT outputs, including runs without split-image patch offsets. EOMT requires
  processor-specific working-size restoration, aspect-ratio unpadding, and
  optional patch merging that cannot be reconstructed from its logits alone;
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
