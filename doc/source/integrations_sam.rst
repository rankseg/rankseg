SAM Family
==========

This page documents the RankSEG integration path for SAM-family outputs from
Hugging Face ``transformers``.

SAM-family outputs use explicit adapter classes instead of the standard
Transformers semantic-segmentation helper.

.. code-block:: python

   from rankseg.integration.sam import Sam1, Sam2, Sam3

SAM1 prompt masks
-----------------

.. code-block:: python

   adapter = Sam1(rankseg_kwargs={"metric": "dice"})
   preds = adapter.postprocess(
       outputs,
       original_sizes=inputs["original_sizes"],
       reshaped_input_sizes=inputs["reshaped_input_sizes"],
   )

SAM2 prompt masks
-----------------

.. code-block:: python

   adapter = Sam2(
       rankseg_kwargs={"metric": "dice"},
       apply_non_overlapping_constraints=False,
   )
   preds = adapter.postprocess(
       outputs,
       original_sizes=inputs["original_sizes"],
   )

SAM3 instance masks
-------------------

.. code-block:: python

   adapter = Sam3(rankseg_kwargs={"metric": "dice"}, threshold=0.3)
   results = adapter.postprocess_instance(
       outputs,
       target_sizes=target_sizes,
   )

SAM3 semantic masks
-------------------

.. code-block:: python

   adapter = Sam3(rankseg_kwargs={"metric": "dice"})
   preds = adapter.postprocess_semantic(
       outputs,
       target_sizes=target_sizes,
   )

The SAM adapters follow the official Transformers post-processing order
through geometry and score restoration. RankSEG replaces the final binary mask
decision. For SAM3, callers choose ``postprocess_instance(...)`` or
``postprocess_semantic(...)`` explicitly.

Restored probabilities
----------------------

Use the ``restore_*`` methods when you need the restored mask probabilities
instead of final RankSEG predictions:

.. code-block:: python

   sam1_probs = Sam1().restore_mask_probs(
       outputs,
       original_sizes=inputs["original_sizes"],
       reshaped_input_sizes=inputs["reshaped_input_sizes"],
   )

   sam2_probs = Sam2().restore_mask_probs(
       outputs,
       original_sizes=inputs["original_sizes"],
   )

   sam3_instances = Sam3(threshold=0.3).restore_instance_mask_probs(
       outputs,
       target_sizes=target_sizes,
   )

   sam3_semantic_probs = Sam3().restore_semantic_mask_probs(
       outputs,
       target_sizes=target_sizes,
   )

Current exclusions
------------------

The SAM integration does not currently support SAM video tracker state.
