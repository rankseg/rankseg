Transformers
============

This page documents the official RankSEG integration path for Hugging Face
``transformers`` semantic-segmentation outputs.

If you already run inference manually through a ``processor -> model ->
outputs`` workflow, this is the recommended entry point.

Recommended entry point
-----------------------

.. code-block:: python

   from rankseg.transformers import postprocess

The main helper is:

.. code-block:: python

   postprocess(outputs, *, model=None, target_sizes, rankseg_kwargs=None)

Its role is intentionally narrow:

- restore semantic probabilities from supported Hugging Face output families;
- resize them to the original image size when needed;
- apply ``RankSEG`` as the final post-processing step.

Minimal integration
-------------------

The standard Hugging Face inference structure stays the same. The only
integration change happens after ``outputs = model(**inputs)``.

.. code-block:: python

   from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
   from rankseg.transformers import postprocess
   from PIL import Image
   import requests

   processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
   model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

   image = Image.open(requests.get("https://plus.unsplash.com/premium_photo-1673210886161-bfcc40f54d1f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8cGVyc29uJTIwc3RhbmRpbmd8ZW58MHx8MHx8&w=1000&q=80", stream=True).raw)
   inputs = processor(images=image, return_tensors="pt")
   outputs = model(**inputs)

   preds = postprocess(
       outputs,
       target_sizes=image.size[::-1],
       rankseg_kwargs={"metric": "dice"},
   )

For supported output families, ``postprocess(...)`` replaces the final
``argmax``-style decision step while preserving the surrounding Hugging Face
inference code.

Low-level helper
----------------

The module also exposes:

.. code-block:: python

   from rankseg.transformers import restore_semantic_probs

``restore_semantic_probs(...)`` is a lower-level helper for users who want the
restored semantic probability map directly. It may be imported and used
directly, but it is not the primary recommended API.

Supported output families
-------------------------

The current helper supports the main semantic-segmentation output families used
by ``transformers``:

- ``outputs.logits``
- ``outputs.class_queries_logits`` + ``outputs.masks_queries_logits``
- ``outputs.logits`` + ``outputs.pred_masks``
- ``outputs.semantic_seg``

When a branch requires model-specific handling, pass ``model=...`` so the
helper can follow the corresponding official post-processing behavior.

Current exclusions
------------------

The simplified API does not currently support:

- outputs with ``patch_offsets`` that require official patch-merge logic;
- tuple-style outputs such as ``return_dict=False`` returns;
- custom unstructured outputs from ``trust_remote_code=True`` models;
- SegGPT-style ``pred_masks`` semantic reconstruction.

These cases should fail explicitly rather than silently using an incorrect
semantic restoration path.

Notebook and Colab demo
-----------------------

- Example script: `examples/transformers_rankseg.py <https://github.com/rankseg/rankseg/blob/main/examples/transformers_rankseg.py>`_
- Notebook: `notebooks/rankseg_with_transformers.ipynb <https://github.com/rankseg/rankseg/blob/main/notebooks/rankseg_with_transformers.ipynb>`_
- Colab: `Open the notebook in Colab <https://colab.research.google.com/github/Leev1s/rankseg/blob/feat/transformers-adapter/notebooks/rankseg_with_transformers.ipynb>`_
