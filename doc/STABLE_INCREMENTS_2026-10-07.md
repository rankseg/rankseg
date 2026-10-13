# Stable RMA multiclass increments: validation and smoke benchmark

## Scope

Replace subtraction of nearby overall objective values with algebraically
equivalent factored increments. Dice with `smooth=0` uses
`(p - S/d) * (2/(d+1))`, where `d = t + M + 1`. The ordinary IoU branch uses
`(p - (a/b)*(1-p)) / (b + (1-p))`, with the existing zero-denominator convention.
The optional Triton Dice assignment uses the same factorization.

There is no new epsilon tie band, probability-argmax fallback, float64 retry,
or device synchronization. Exactly equal computed scores retain the smallest
eligible class index. Positive-smoothing Dice and very-large-smoothing scaled
formulas are unchanged. Binary search, screening certificates, pruning and
eligibility are unchanged. AST comparisons also verified that `rankdice_ba`,
`compute_opt_tau` and `full_sort_masks` were not modified.

Final multiclass labels can change: eliminating cancellation is not a bitwise
compatibility change. It is also not a guarantee of better ground-truth Dice.

## Correctness

The independent regression uses `2**20` copies of each foreground probability
`[7/8, 1/8, 15/32]`, plus its complementary background channel. Both classes
select the third block. Exact rational before/after objectives select class 1
at every overlap; the published 0.0.7 assigned the entire 1,048,576-pixel block
to class 0 in float32. The new formulas select class 1 on CPU and CUDA, for
Dice and IoU, with screening both off and on. This is a constructed numerical
regression, not a medical accuracy measurement.

The new test file first failed all four CPU large-volume cases against the
unchanged implementation. Its small-input exact-rational oracle also covers
zero classes, all-pruned fallback, positive smoothing, eligibility and voids.
Two existing floating-point parity helpers now use the stable expressions;
the independent objective oracles were not replaced with implementation math.

| Check | Passed | Skipped |
| --- | ---: | ---: |
| Full CPU suite, Torch 2.11 | 1,292 | 1,739 |
| Full CUDA suite, Torch 2.11 / Triton 3.6 | 3,014 | 17 |
| Assignment/stability suites, Torch 2.8 / Triton 3.4 | 649 | 0 |
| Large-index CUDA suite with `RANKSEG_TEST_LARGE_CUDA=1` | 44 | 0 |

CPU portable coverage: 92.67%; CUDA full coverage: 85.92% (JIT kernel execution
is not measured as Python line coverage). Sphinx `-W --keep-going`, focused
lint and `git diff --check` passed. The large-index run includes the existing
liver-sized public assignment tests. Counts are per run, not disjoint totals.

## Real-input smoke benchmark

Baseline: published RankSEG 0.0.7 wheel, SHA256
`a421c6e198bcd5b45d1972f8e867d3e2d75b9b5227cbc435a0412359cdd313bc`.
Environment: Python 3.10, Torch 2.11, Triton 3.6, NVIDIA RTX 3090, one CPU
thread. No inference, training, downloading or resampling was performed.

One existing full-resolution cache sample per dataset:

- VOC: `2007_000033`, shape `(1,21,512,512)`.
- Cityscapes: `lindau_000036_000019`, shape `(1,19,1024,2048)`.
- ADE20K: `ADE_val_00001969`, shape `(1,150,512,683)`.
- nnU-Net Hippocampus: `hippocampus_017`, shape `(1,3,32,48,35)`.
- nnU-Net Pancreas: `pancreas_041`, shape `(1,3,137,512,512)`, CUDA only.

Timing uses both versions in one process, three warm-ups per mode/version,
then alternating execution order for six CPU or twenty CUDA repetitions.
Times below are medians, in milliseconds, including dispatch and host
overhead; CUDA is synchronized. Inputs are already resident. The frozen
baseline namespace was byte-verified against the published wheel, apart from
renaming absolute imports to isolate it from the modified package.

An initial separate-process run showed a CPU VOC slowdown; the interleaved
measurement below did not reproduce it. Small timing differences are not
hardware-independent speed guarantees.

| Input | CPU Dice, off | CPU Dice, on | CPU IoU | CUDA Dice, off | CUDA Dice, on | CUDA IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VOC | 185.80 → 182.18 | 72.31 → 67.52 | 185.36 → 176.13 | 3.209 → 3.100 | 0.548 → 0.542 | 3.047 → 3.018 |
| Cityscapes | 2022.56 → 1885.76 | 955.01 → 826.85 | 1928.07 → 1863.60 | 19.209 → 18.461 | 1.518 → 1.507 | 18.613 → 18.427 |
| ADE20K | 2336.25 → 2181.87 | 853.07 → 698.98 | 2280.56 → 2190.68 | 25.375 → 24.385 | 1.616 → 1.563 | 24.840 → 24.590 |
| Hippocampus | 7.796 → 7.762 | 4.849 → 4.808 | 7.756 → 7.745 | 0.377 → 0.376 | 0.532 → 0.533 | 0.378 → 0.375 |
| Pancreas | — | — | — | 98.718 → 96.676 | 3.370 → 3.377 | 97.127 → 96.604 |

`off/on` refers to `safe_screening`; IoU uses full sort. CUDA incremental peak
allocated memory was unchanged in every paired case. Examples: Cityscapes
Dice off/on 1536/56 MiB; Pancreas Dice off/on 4391/411 MiB. These are decoder
peaks above resident inputs, not total inference-process memory. Sorting or
other workspaces can dominate the peak even when scoring uses fewer temporaries.

The separate correctness pass found identical binary-mask hashes for old/new
on all sampled modes and devices. Final multiclass differences were 0–1 pixels
on Hippocampus and 146–152 voxels on Pancreas. VOC/Cityscapes/ADE20K changed
39–1,433 pixels, depending on input, device and mode. Their per-image Dice/IoU
changes were between -0.02058 and +0.00473 percentage points. Those small changes
include both improvements and declines. No medical ground-truth comparison
was performed in this smoke run, and this is not a full-cohort benchmark.

Local diagnostic scripts, timing samples and reports are retained in
`/tmp/rankseg-stable-increments.aGQh4k/`; this temporary directory is not part
of the published package.
