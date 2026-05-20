SAM Family
==========

This page documents the RankSEG integration path for SAM-family outputs from
Hugging Face ``transformers``.

SAM-family outputs use explicit adapter classes instead of the standard
Transformers semantic-segmentation helper.

Why SAM has its own adapters
----------------------------

SAM outputs are not plain semantic-segmentation logits. Before RankSEG is
called, SAM masks must be restored from model space back to image space using
family-specific geometry:

.. code-block:: text

   SAM processor -> SAM model outputs -> restore mask probabilities
   -> RankSEG -> prompt, instance, or semantic masks

The adapters in ``rankseg.integration.sam`` keep this restoration step
explicit. RankSEG only replaces the final binary mask step after masks have
been resized and converted to probabilities.

.. code-block:: python

   from rankseg.integration import sam

Adapter map
-----------

.. list-table::
   :widths: 18 28 28 26
   :header-rows: 1

   * - Adapter
     - Input family
     - Main prediction method
     - Probability-only method
   * - ``sam.Sam1``
     - SAM1 and SAM-HQ prompt masks
     - ``postprocess(...)``
     - ``restore_mask_probs(...)``
   * - ``sam.Sam2``
     - SAM2 prompt masks
     - ``postprocess(...)``
     - ``restore_mask_probs(...)``
   * - ``sam.Sam3``
     - SAM3 instance masks
     - ``postprocess_instance(...)``
     - ``restore_instance_mask_probs(...)``
   * - ``sam.Sam3``
     - SAM3 semantic masks
     - ``postprocess_semantic(...)``
     - ``restore_semantic_mask_probs(...)``

Recommended RankSEG options
---------------------------

SAM prompt and instance masks are naturally represented as per-mask binary
probability maps, so the adapters default to ``output_mode="multilabel"`` when
``rankseg_kwargs`` does not specify an output mode.

.. code-block:: python

   adapter = sam.Sam1(
       rankseg_kwargs={"metric": "dice", "solver": "RMA"}
   )

SAM1 prompt masks
-----------------

.. code-block:: python

   adapter = sam.Sam1(rankseg_kwargs={"metric": "dice"})
   preds = adapter.postprocess(
       outputs,
       original_sizes=inputs["original_sizes"],
       reshaped_input_sizes=inputs["reshaped_input_sizes"],
   )

``original_sizes`` and ``reshaped_input_sizes`` should come from the SAM
processor inputs. The adapter removes padding, resizes masks back to the
original image size, applies ``sigmoid``, and then calls RankSEG.

SAM2 prompt masks
-----------------

.. code-block:: python

   adapter = sam.Sam2(
       rankseg_kwargs={"metric": "dice"},
       apply_non_overlapping_constraints=False,
   )
   preds = adapter.postprocess(
       outputs,
       original_sizes=inputs["original_sizes"],
   )

Set ``apply_non_overlapping_constraints=True`` when you want lower-scoring
overlapping SAM2 masks to be suppressed before converting logits to
probabilities.

SAM3 instance masks
-------------------

.. code-block:: python

   adapter = sam.Sam3(rankseg_kwargs={"metric": "dice"}, threshold=0.3)
   results = adapter.postprocess_instance(
       outputs,
       target_sizes=target_sizes,
   )

The instance method returns one dictionary per image with ``scores``, ``boxes``,
and ``masks``. ``threshold`` filters low-confidence instances before RankSEG is
applied to the remaining mask probabilities.

SAM3 semantic masks
-------------------

.. code-block:: python

   adapter = sam.Sam3(rankseg_kwargs={"metric": "dice"})
   preds = adapter.postprocess_semantic(
       outputs,
       target_sizes=target_sizes,
   )

The SAM adapters follow the official Transformers post-processing order
through geometry and score restoration. RankSEG replaces the final binary mask
step. For SAM3, callers choose ``postprocess_instance(...)`` or
``postprocess_semantic(...)`` explicitly.

Shape conventions
-----------------

.. list-table::
   :widths: 28 34 38
   :header-rows: 1

   * - Stage
     - Typical shape
     - Meaning
   * - Restored SAM prompt probabilities
     - ``(num_masks, 1, H, W)`` or compatible per-image tensors
     - One probability map per proposed prompt mask.
   * - Restored SAM3 instance probabilities
     - ``(num_instances, H, W)``
     - One probability map per retained instance.
   * - RankSEG prompt predictions
     - Binary mask tensors matching restored mask geometry
     - Final prompt masks after RankSEG post-processing.
   * - RankSEG semantic predictions
     - One tensor per image
     - Final semantic masks from SAM3 semantic probabilities.

Restored probabilities
----------------------

Use the ``restore_*`` methods when you need the restored mask probabilities
instead of final RankSEG predictions:

.. code-block:: python

   sam1_probs = sam.Sam1().restore_mask_probs(
       outputs,
       original_sizes=inputs["original_sizes"],
       reshaped_input_sizes=inputs["reshaped_input_sizes"],
   )

   sam2_probs = sam.Sam2().restore_mask_probs(
       outputs,
       original_sizes=inputs["original_sizes"],
   )

   sam3_instances = sam.Sam3(threshold=0.3).restore_instance_mask_probs(
       outputs,
       target_sizes=target_sizes,
   )

   sam3_semantic_probs = sam.Sam3().restore_semantic_mask_probs(
       outputs,
       target_sizes=target_sizes,
   )

Explicit adapter imports are also supported when you prefer shorter local names:

.. code-block:: python

   from rankseg.integration.sam import Sam1, Sam2, Sam3

Current exclusions
------------------

The SAM integration does not currently support SAM video tracker state.

Executable tutorial
-------------------

The SAM-family notebook is the recommended way to learn this integration. It
runs SAM1, SAM2, and SAM3 examples in separate sections, compares the official
post-processing path with the RankSEG path, and keeps the model outputs shared
between the two paths so the replacement point is visible.

- `notebooks/rankseg_with_sam_family.ipynb <https://github.com/rankseg/rankseg/blob/main/notebooks/rankseg_with_sam_family.ipynb>`_
- `Open in Colab <https://colab.research.google.com/github/rankseg/rankseg/blob/main/notebooks/rankseg_with_sam_family.ipynb>`_
