.. _api:

.. meta::
   :description: Browse the rankseg API documentation.

=================
API
=================

Module Interface
================

.. autoapisummary::

   rankseg.RankSEG
   rankseg.functional.rankseg


Integration Interface
=====================

.. autoapisummary::

   rankseg.integration.transformers.postprocess
   rankseg.integration.transformers.restore_semantic_probs
   rankseg.integration.sam.Sam1
   rankseg.integration.sam.Sam2
   rankseg.integration.sam.Sam3


Algorithms
==========

.. autoapisummary::

   rankseg.rankdice_ba
   rankseg.rankseg_rma


Experimental RMA safe screening
-------------------------------

``safe_screening`` controls a sorting optimization for RMA Dice
with ``smooth=0``. The default is ``False`` (opt-in). It is accepted by
``rankseg_rma``, the functional interface, and ``RankSEG``:

.. code-block:: python

    from rankseg import RankSEG

    decoder = RankSEG(
        metric="dice", solver="RMA", smooth=0,
        output_mode="multilabel", safe_screening="auto",
    )
    masks = decoder(probs)

The three modes are:

* ``False`` (default): retain the original reference path.
* ``True``: force screening for eligible, nonempty inputs, regardless of size
  or candidate density. CUDA uses optional Triton kernels when available,
  otherwise the portable PyTorch implementation. Fully pruned inputs can still
  skip the candidate search.
* ``"auto"``: screen CPU inputs without a size cutoff. On CUDA, screen only
  when Triton is available and ``probs.numel() >= 1_280_000``; otherwise use
  optimized full sort. Other devices retain the original path.

The CUDA threshold is **1,280,000 probability values**, including batch and
class dimensions (B*C*D), not 1280*1024, bytes, spatial pixels alone, or the
number of foreground pixels. Equality selects screening. Dispatch reads only
tensor metadata and cached backend availability; it adds no probability scan,
device synchronization or runtime timing. Triton is loaded lazily on CUDA and
is never installed automatically. The threshold reflects a memory-conscious
policy targeting at least 0.8x full-sort speed (up to 25% more latency) on real
calibration inputs; it is not a per-input or cross-hardware guarantee. The
calibration used RTX 3090, batch size 1 and float32; other workloads should be
benchmarked independently. Incremental postprocessing memory savings do not
represent savings for the entire model/inference process.

Unlike the previous experimental boolean option, ``True`` no longer bypasses
small CUDA inputs. Use ``"auto"`` for automatic routing. Its optimized
full-sort branch is not the same implementation as ``False`` and does not
promise bitwise equivalence to it.

**Compatibility:** screening remains opt-in in this release. Omitting the
option is equivalent to ``safe_screening=False`` and retains the original
computation path. Explicitly select ``True`` or ``"auto"`` to opt in.
Neither enabling screening nor selecting optimized
full sort promises identical masks: floating-point reductions can alter close
binary decisions and, consequently, final multiclass assignments. IoU,
positive-smoothing Dice and non-RMA solvers keep their existing paths.

Retaining the original path is not a cross-run or cross-device bitwise
guarantee. CUDA floating-point cumulative sums can round differently between
executions, including on the original ``False`` path, and near-tied volume
scores can therefore produce slightly different masks.

For each sample and class, the unsmoothed Dice RMA objective is
:math:`2S/(k+\mu+1)`, where :math:`\mu` is the probability sum over the
**entire** image, :math:`S` is the selected probability sum, and :math:`k` is
the selected pixel count. Screening certifies some pixels as selected or
excluded, then sorts only the unresolved pixels. Different per-class lengths
are grouped into bounded padded workspaces; invalid padding never competes in
the volume search or writes into the output mask. Inactive classes contribute
no candidates. A single-row sort needs no padding.

Among the searched candidate volumes, use **direct argmax** of the computed
scores. Only exactly equal computed maxima prefer the smallest searched
volume. There is no epsilon comparison, near-tie pass or close-score full-sort
retry at inference time. Float16/bfloat16 inputs use float32 arithmetic.
Certified selected/excluded pixels remain fixed during the candidate search.

This does not change ``pruning_prob`` or multiclass assignment. In particular,
a pixel excluded from a class's independent binary mask can still be assigned
to that class by ``unassigned_policy="max_score"`` if no class selected the
pixel. Original probabilities and full-image sums are retained for that step.

The first implementation screens only Dice with ``smooth=0``. IoU and Dice
with positive smoothing use the original full-sort path even when screening
is requested. The positive-smoothing Dice RMA objective is a sum of two
fractions, so its screening bounds cannot be substituted from the unsmoothed
case.

.. warning::

   This option is experimental. Mathematical safe screening preserves the
   independent binary optimum, but floating-point reductions can change close
   score comparisons. This option accepts epsilon-scale objective differences
   and does **not** promise bitwise equality with the default masks. Epsilon
   tolerances are used in validation, not in inference decisions, and are not
   rigorous bounds on all floating-point reduction errors. Tests check the independent
   binary objectives against high-precision full-prefix oracles as well as
   pruning, scatter, padding and multiclass assignment behavior. Close binary
   objectives do not imply an epsilon bound on final multiclass Dice/IoU:
   changed binary masks can affect eligibility in the unchanged assignment
   rule. ``safe_screening=False`` retains the original full-sort reference.

Performance depends on image size, class count, precision, and the unresolved
set sizes. Once screening starts, all unresolved candidates are solved using
the direct-argmax volume rule, without full-sort retries based on candidate
count or fraction. Multi-row padded groups are limited to 1,048,576 elements;
longer individual rows run alone without padding. These grouping limits bound
padding overhead, not the size of a real candidate row or the underlying
sorting library's total allocation. Screening can still be slower on dense
candidate sets; allocation failures are not silently retried with full sort.
If validation has already established that every class is pruned, reuse its
global maximum and skip candidate construction. Multiclass ``max_score`` still
evaluates all classes using the original probabilities; it is not raw argmax.
Auto routing can bypass screening before candidate construction according to
the metadata rule above. There is no candidate-count or candidate-fraction
retry after screening starts. Benchmark representative data before enabling
this option; it is not a universal speedup. When Triton is available, CUDA screening fuses the full-image
sum, maximum and forced-positive statistics. For float32/float64 CUDA inputs
that enter this path and can be viewed as probability rows without copying,
finite-value and range validation share that statistics scan. Nonfinite errors
still take precedence over range errors, including in pruned classes. Validation
finishes before candidate construction, and the full binary mask allocation is
deferred until it is needed. The validation host transfer remains; this removes
a separate full-input read, not all synchronization. CPU, half-precision,
copy-requiring layouts and small-input bypasses keep ordinary validation.
Globally pruned inputs still skip candidate construction and sorting, but fused
validation has already calculated statistics and can cost more than validation
alone. Their multiclass fallback retains the original sum reduction.
The CUDA decoder stores the binary
mask, lower bounds and integer block candidate counts, without allocating a
second full-size candidate mask. GPU block offsets then pack indices in stable
row-major order: only nonempty blocks of retained rows reread probabilities and
replay the original candidate comparisons against the stored bounds. Bounds and
statistics are not recomputed, and neither candidate membership nor order
changes. Pruned and empty rows are skipped during packing.
The source probabilities remain unchanged throughout. This avoids dynamic
``nonzero`` allocation on the prepared screening path. A bounded host metadata
transfer still determines group sizes
and workspace allocation; this is not a synchronization-free implementation.
Within each CUDA candidate group, a fused gather reads the original probability
strides and copies statistics into the existing bounded sort workspace. Group
membership, candidate order, padding values and sort shapes are unchanged;
singleton groups retain one-dimensional sorting. The sort permutation then
writes decisions directly to the unique packed candidate indices. Padding never
writes to the image, and forced-positive and pruned positions are
untouched. This avoids per-candidate owner/position arrays, concatenated group
index copies and inverse-permutation masks; it does not change sorting or
scoring. CPU and no-Triton paths retain the portable implementation.
The candidate prefix sum remains a PyTorch operation, followed by fused scoring
and direct argmax. Multiclass output optionally fuses overlap classification and
uniquely selected pixel statistics, followed by incremental scoring, eligibility
and class argmax. It stores one byte per pixel for zero/unique/overlapping status,
with bounded partial reductions instead of full class-by-pixel integer copies,
unique masks or floating-point products. Unique counts use integer accumulation;
probability sums retain the working dtype. Unsupported kernel shapes use portable
assignment. A pixel selected by exactly one binary class must retain that class,
not the probability argmax. All-unique tiles therefore emit the selected class
IDs without probability loads or incremental scoring. For up to 254 classes,
the statistics pass also caches that unique class ID in the same one-byte
status buffer: zero means unassigned, one means overlapping, and two plus the
class index means unique. Exact integer packing combines the count and class
identity in one reduction; the unique probability-sum expression is unchanged.
All-unique assignment tiles decode these IDs without rereading the binary
masks or reducing across classes again. With 255 or 256 classes, the original
status encoding and mask-based identity lookup are retained. Mixed tiles keep
the original vectorized scoring path. High-class tiles isolate this scoring in a
device function to limit register pressure, without an extra kernel launch.
Overlapping and unassigned pixels keep the same score expression, eligibility,
tie order and all-pruned/void handling. This shortcut preserves labels exactly
for identical binary masks and statistics; it does not treat screened-negative
pixels as ineligible for the unassigned-pixel fallback.
Full-image sums always include screened-negative pixels, but both
full-image and unique-pixel fused reductions can round differently from
``torch.sum``. This can change near-tied final labels; bitwise identity is not
guaranteed. The existing floating-point CUDA prefix scan can also vary across
repeated calls on long candidate rows. Dense binary masks and the final class-index output still require
memory even when screening removes almost all sorting candidates, so screening
proportions are not percentages of total memory saved.
Without Triton, screening uses pure PyTorch;
Triton is not a base dependency. The optional ``rankseg[cuda]`` extra declares
it on Linux x86-64; see :ref:`cuda-installation` for installation and version
compatibility. CPU execution does not load the CUDA backend. First-use kernel compilation is excluded
from warmed benchmark timings. No iterative threshold solver is used.

From a repository checkout, run the paired synthetic benchmark with:

.. code-block:: bash

    python -m scripts.benchmark_rma_screening --device cuda --force-screening --output /tmp/rma-screening.json

``--force-screening`` disables the small-input performance bypasses for the
benchmark process only, so screening is actually measured at every requested
size. Workspace-budget fallbacks still apply. Add ``--torch-screening`` to
measure the portable implementation without fused kernels.

The benchmark records pixel differences and checks independent binary objective
regret against a float64 full-prefix oracle before timing. Its measured check
allows four float32 epsilons (16 for float64, where the oracle also rounds),
for arithmetic roundoff;
this is a regression threshold, not a universal analytical error guarantee.
It alternates method order and reports median synchronized wall-clock latency
(including host overhead). CUDA memory is incremental peak allocated memory with the input
already resident. Probability generation and warm-up are excluded. Synthetic
``all_pruned`` cases measure early class-pruning savings, not the two-sided
pixel certificates in isolation. This is not a clinical/model-accuracy study.


Distribution Interface
======================

.. autoapisummary::

   rankseg.RefinedNormalPB
   rankseg.RefinedNormal


.. toctree::
   :hidden:

   autoapi/rankseg/rankseg/index
