Transformers
============

This page documents the RankSEG integration path for standard Hugging Face
``transformers`` semantic-segmentation outputs.

Use this path when you already run inference through a standard
``processor -> model -> outputs`` workflow and want RankSEG to replace the
final ``argmax``-style decision step.

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
decision. SAM-family outputs are intentionally handled by ``sam.Sam1``,
``sam.Sam2``, or ``sam.Sam3`` after ``from rankseg.integration import sam``
instead of this helper.

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
