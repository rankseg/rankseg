# Release acceptance: screening remains opt-in

Local acceptance completed on 2026-09-25. **`safe_screening=False` is the
default**. Explicit `True` and `"auto"` remain available; auto's decimal
1,280,000-entry CUDA threshold is unchanged. There is no release, commit or
push associated with this check.

## Changes and compatibility

- Reverted only the RMA option default from `"auto"` to `False`; synchronized
  the module/functional docstrings, English/Chinese READMEs and Sphinx docs.
- Added default-versus-explicit-False tests across all three public interfaces,
  CPU/CUDA, four floating dtypes, binary/multiclass output, empty/all-pruned
  inputs, ties, non-contiguous layouts, void assignment and both internal CUDA
  mask reconstruction routes. Omitted-default calls must not initialize the
  screening selector, fused statistics or Triton backend.
- Updated installed-artifact smoke checks to assert the default signature and
  exercise omission / False / True / auto through all four calling forms
  (module call, `.predict`, functional API and `rankseg_rma`).
- Updated the mini acceptance harness to compare omission against **False**,
  regardless of which opt-in mode is being benchmarked. Previous auto-default
  reports are marked as historical experiments; raw results remain untouched.
- AST comparison with the preserved pre-change snapshot verifies that package
  implementation changes consist only of the default and docstrings. All
  other package source bytes match. No score, bound, pruning, scatter,
  multiclass-assignment or CUDA kernel math was changed by this revision.

## Test results

Hardware: NVIDIA RTX 3090. Local Python: **3.10.12**.

| Check | Passed | Skipped | Notes |
| --- | ---: | ---: | --- |
| Full core suite, CPU | 1,190 | 1,639 | Torch 2.11; portable coverage 92.65% |
| Full core suite, CUDA | 2,821 | 8 | Torch 2.11 / Triton 3.6; coverage 85.99%; large-index tests enabled |
| Algorithm + screening suites, CUDA | 2,281 | 6 | Torch 2.8 / Triton 3.4; large-index tests enabled |
| Full suite from extracted sdist, CPU | 1,190 | 1,639 | Outside checkout; portable coverage 92.65% |
| Mini benchmark harness, CUDA available | 242 | 0 | Includes explicit auto/default-False regressions |
| nnU-Net screening/acceptance/audit harness | 207 | 0 | Includes subset-selection and independent-summary checks |
| Installed wheel smoke, CPU / CUDA | 33 / 33 | 0 | Imported outside checkout; module bytes verified |
| Installed sdist smoke, CPU / CUDA | 33 / 33 | 0 | Independently installed; module bytes verified |

Counts are per run, not a sum of unique tests. Torch 2.8's algorithm-only run is
deliberately not described as a full-suite run: that environment lacks the
optional `torchmetrics` test dependency. CPU skips include CUDA-only tests.

Both installed CUDA artifacts executed actual Triton statistics, packing,
gather, scoring, scatter and multiclass-assignment kernels via explicit opt-in
modes, including an input exactly at the auto threshold. CPU execution does
not require activation of those kernels. A genuinely Triton-free installation
was not created locally; missing-backend behavior is covered by regression
tests, and the distribution CI has a separate Triton-free CPU environment.

## Real-input frozen-reference audit

Reused all **6,012 cached mini inputs**: 100 VOC, 100 Cityscapes, 100 ADE20K and
5,712 KiTS slices (210 cases, five folds). The natural-image caches are subsets,
not complete published datasets. No checkpoints or inference were needed.

Every input passed six exact mask comparisons against the frozen candidate
from the prior acceptance:

1. Omitted `rankseg_rma` option versus previous explicit False.
2. Omitted functional option versus previous explicit False.
3. Omitted module option versus previous explicit False.
4. Current explicit False versus previous explicit False.
5. Current explicit True versus previous explicit True.
6. Current explicit auto versus previous explicit auto.

All **36,072 comparisons passed bitwise**. Dataset/input/label hashes and
identities matched the prior evidence. Default predictions' TP/FP/FN matched
the previously recorded full-sort reference for **every input**, preserving
the corresponding Dice/IoU aggregates. Probability tensors were not mutated.

This was a compatibility check, **not a new timing benchmark**. The nnU-Net
778-volume cohort and 41-volume subset were not rerun in this revision; their
earlier explicit-mode results remain historical evidence. In particular,
restoring False does not give its full-sort path screening's memory savings:
the earlier original-False control had six OOMs in the 11-volume Liver subset.

## CUDA numerical finding

A large random-input route test exposed existing CUDA floating-point `cumsum`
non-determinism. Repeated decoding of identical probabilities with the
**original False path** also produced volume differences of 15 boundary
pixels in the diagnostic fixture. This is not a change caused by the default.

The exact threshold/route test now uses integer-valued probabilities whose
prefixes are exactly representable in float32. A separate randomized test
repeats the original and auto-full-sort paths on three large fixtures, four
times each, and checks **every run** against an independent float64 exhaustive
prefix oracle. The existing **four-float32-eps objective budget is unchanged**.
No runtime retry, epsilon-based tie handling or extra inference work was added.

The real-cache bitwise matches above are observations on those inputs, not a
universal cross-run or cross-device mask guarantee. Near-tied binary objectives
also do not provide a bound on final multiclass ground-truth metrics.

## Packaging and release status

- Sphinx HTML build with `-W --keep-going`: passed.
- Wheel and sdist build, package-source verification, packaged test/script
  checks and `twine check --strict`: passed.
- Main environment `pip check`: no broken requirements.
- Focused lint and changed-tree whitespace checks: passed.
- CI configuration retains Python 3.10–3.14, CPU coverage, documentation,
  distribution smoke and extracted-sdist tests. **Remote CI was not triggered
  or verified in this revision.** Only Python 3.10 was executed locally.

**Version confirmed: 0.0.7.** `setup.py` has been updated and fresh release
artifacts built. The algorithm/package Python sources still match the full
acceptance hashes above; only release metadata and version-specific packaging
test coverage changed in this step. New post-bump checks passed:

- Packaging/release-smoke regression tests: **48 passed**.
- Full CPU suite from the **0.0.7 sdist**: **1,192 passed, 1,639 skipped**,
  coverage **92.65%** (two additional version-parameter tests).
- Independently installed 0.0.7 wheel and sdist: **33 prediction checks each
  on CPU and CUDA**, including the default-False signature. All imported and
  installed versions are verified as **0.0.7**.
- Sphinx, distribution source/README verification and strict Twine checks:
  passed. No upload, tag, commit or push was performed. Check the resulting
  commit's remote CI before publishing with release tag `v0.0.7`.

Final local artifacts are under `dist/0.0.7/`; the earlier temporary 0.0.6
builds are obsolete test artifacts and should not be uploaded:

- `rankseg-0.0.7-py3-none-any.whl` SHA-256:
  `a421c6e198bcd5b45d1972f8e867d3e2d75b9b5227cbc435a0412359cdd313bc`.
- `rankseg-0.0.7.tar.gz` SHA-256:
  `75f64a8230f3253fcd9ae8a2fd1dfb7f7bfb0a93ae00112598aa4ce54df8da38`.

Logs, the real-input audit script, source hashes and per-input compatibility
records are preserved locally in the Git-ignored directory
`.cache/release-default-false-2026-09-25/`. The original temporary directory
also retains isolated installs, coverage XML and extracted-sdist tests.
Post-version-bump logs are in `.cache/release-0.0.7-2026-09-25/`, with isolated
installs and extracted source under `/tmp/rankseg-007-release.Dd8H5g/`.
