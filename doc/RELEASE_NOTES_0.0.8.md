# RankSEG 0.0.8

This release improves the numerical stability of RMA multiclass assignment.
When multiple classes select the same pixel, RankSEG compares their expected
Dice or IoU increments. The Dice path with `smooth=0` and the ordinary IoU path
now evaluate those increments in factored form, avoiding cancellation from
subtracting nearby objective values. The optional CUDA Dice assignment uses
the same stable expression.

## Compatibility

- The public API, dependencies, and `safe_screening=False` default are unchanged.
- Binary and multilabel mask selection, pruning, and screening certificates are
  unchanged by this update.
- Multiclass output **can change** where the previous floating-point calculation
  misordered close increments, with screening either on or off. This numerical
  fix does not guarantee a higher ground-truth score on every image. Keep the
  RankSEG version attached to benchmark results.

Regression tests cover a large-volume case that was misassigned in 0.0.7 and
small cases checked against an exact-rational objective. Validation details and
exploratory runtime measurements are in
[Stable RMA multiclass increments](STABLE_INCREMENTS_2026-10-07.md).
