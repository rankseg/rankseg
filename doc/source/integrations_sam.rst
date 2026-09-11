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
``rankseg_kwargs`` does not specify an output mode. Prediction masks have dtype
``torch.bool`` and retain their per-mask geometry. ``output_mode="multiclass"``
is rejected: SAM's multiple prompt masks or instances are independent binary
masks, not semantic class channels that can be collapsed into one class-index
map.

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
Each size must contain two positive integers, and each reshaped input size must
fit inside ``pad_size``.

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
probabilities. This option must be a boolean; string values such as ``"false"``
are rejected instead of being interpreted as truthy.

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
applied to the remaining mask probabilities. It must be finite and in
``[0, 1]``. Filtering is strict: an instance is retained only when
``score > threshold``.

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

All ``original_sizes``, ``reshaped_input_sizes``, and ``target_sizes`` values
use ``(height, width)`` order, contain positive integers, and provide one entry
per batch item. Tensor size inputs use shape ``(2,)`` for a single image or
``(B, 2)`` for a batch.

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

SAM1/SAM2 prompt outputs require ``pred_masks`` with shape ``(B, N, H, W)`` or
``(B, P, N, H, W)``. Their ``iou_scores`` field is not used to reconstruct
mask probabilities and may be omitted from cached or reduced outputs. SAM3
instance outputs use ``pred_logits: (B, Q)``, ``pred_boxes: (B, Q, 4)``, and
``pred_masks: (B, Q, H, W)``; optional ``presence_logits`` uses ``(B, 1)``.
SAM3 semantic logits use ``(B, 1, H, W)``. Related tensors must agree on batch,
query, and device dimensions.

Each adapter accepts the native Transformers ``ModelOutput`` as well as a
mapping or attribute-based structured object containing the required fields.
This allows cached and wrapped outputs without depending on an exact Python
class name. When a native output or subclass can be identified as belonging to
a different SAM generation, the adapter rejects it rather than applying the
wrong family's resize geometry. Tuple-style ``return_dict=False`` outputs
remain unsupported because their positional field layout is ambiguous across
model families.

SAM model outputs must use a real floating-point dtype and contain only finite
values; NaN and positive or negative infinity are rejected rather than being
propagated into masks or silently filtered as low-confidence instances.
Probability restoration promotes ``float16`` and ``bfloat16`` to ``float32``
for stable computation, while preserving ``float32`` and ``float64``
precision. Scores, boxes, and mask probabilities therefore retain the
appropriate working precision instead of being unconditionally converted to
``float32``.

When SAM3 instance filtering retains no masks, the returned empty mask tensor
keeps its low-resolution spatial shape, matching the Transformers processor
contract; resizing is performed only when at least one mask remains. Empty
input batches are supported and return empty lists.

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
