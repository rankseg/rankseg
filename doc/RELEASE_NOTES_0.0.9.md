# RankSEG 0.0.9

RankSEG 0.0.9 adds **DEP**, an optional Dice decoder that models spatially
localized dependence between pixel labels. It operates on existing model
probabilities without retraining.

```python
from rankseg import RankSEG

probs = model_logits.float().softmax(dim=1)  # [B, C, H, W], C >= 2
preds = RankSEG(metric="dice", solver="DEP", theta=300.0)(probs)
```

## New functionality

- Select DEP through `RankSEG(solver="DEP")`,
  `rankseg.functional.rankseg(..., solver="DEP")`, or the exported
  `rankseg_dep()` function.
- Run on CPU or CUDA with multiclass label maps or independent multilabel masks.
- Control the Gaussian spatial bandwidth with `theta` (default: 300 pixels).
  `theta=0` removes dependence between distinct pixels; it does not guarantee
  identical predictions to RMA.
- DEP uses fixed-point ranking and volume search, with at most 10 iterations.
  Multiclass assignment uses factored objective increments to avoid cancellation
  from subtracting nearly equal overall scores. Gradient recording is disabled
  in all public decoding entry points.

## Compatibility and scope

- RMA remains the default. Existing RMA behavior and its `safe_screening=False`
  default are unchanged. No external dependencies have been added.
- DEP supports only `metric="dice"` and `smooth=0`. Do not pass
  `safe_screening`, which is specific to RMA.
- Start with 2D images and validate accuracy and runtime against RMA. Although
  DEP accepts 3D tensors, `theta` does not account for anisotropic voxel spacing,
  and improved 3D accuracy is not established.

## Validation

Tests cover conditional volume estimates, multiclass objective increments,
zero bandwidth, disabled autograd, and large-input float32 assignment on CPU
and CUDA. Installed-wheel and source-distribution smoke checks now exercise
DEP through the class, functional, and direct interfaces in both output modes.

See [DEP usage and parameters](https://rankseg.readthedocs.io/en/latest/API.html#dep-spatial-dependence)
and [PR #28](https://github.com/rankseg/rankseg/pull/28).
