"""Multiclass assignment must preserve differences between small increments."""

from fractions import Fraction

import pytest
import torch

from rankseg import rankseg_rma

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
]


def _objective(metric, mass, count, mean, smooth):
    """Exact rational before/after objective, independent of the implementation."""
    if metric == "dice":
        value = 2 * mass / (count + mean + smooth + 1)
        if smooth:
            value += smooth / (count + mean + smooth)
        return value
    denominator = count + mean - mass + smooth
    return (mass + smooth) / denominator if denominator else Fraction(0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("metric", ["dice", "iou"])
@pytest.mark.parametrize("screening", [False, True])
def test_large_volume_increment_order_matches_exact_rational_objective(device, metric, screening):
    # Exactly representable probabilities and sums. Both classes select the
    # third block, but class 1 has the larger increment at EVERY overlap.
    # The old float32 subtraction misassigned the entire 1,048,576-pixel block
    # on both CPU and CUDA, despite neither exact increment being tied.
    size = 1 << 20
    p = [Fraction(15, 32), Fraction(17, 32)]
    mean = [Fraction(47 * size, 32), Fraction(49 * size, 32)]
    mass = Fraction(7 * size, 8)
    increments = [
        _objective(metric, mass + p[c], size + 1, mean[c], 0)
        - _objective(metric, mass, size, mean[c], 0)
        for c in range(2)
    ]
    assert increments[1] > increments[0] > 0

    foreground = torch.tensor([7 / 8, 1 / 8, 15 / 32], device=device).repeat_interleave(size)
    probs = torch.stack((foreground, 1 - foreground)).reshape(1, 2, 192, 128, 128)
    binary = rankseg_rma(probs, metric=metric, output_mode="multilabel", safe_screening=screening).flatten(2)
    # Check the binary stage separately, rather than blaming different binary
    # optima for a multiclass assignment error.
    assert binary[0, 0, :size].all() and not binary[0, 1, :size].any()
    assert binary[0, 1, size:2 * size].all() and not binary[0, 0, size:2 * size].any()
    assert binary[..., 2 * size:].all()
    del binary

    actual = rankseg_rma(probs, metric=metric, safe_screening=screening).flatten()
    assert (actual[:size] == 0).all()
    assert (actual[size:] == 1).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("metric", ["dice", "iou"])
@pytest.mark.parametrize("smooth", [0.0, 0.25, 100.0])
@pytest.mark.parametrize("screening", [False, True])
@pytest.mark.parametrize("policy", ["max_score", "void"])
@pytest.mark.parametrize("pruning", [0.5, 1.0])
def test_assignment_matches_fraction_oracle(device, dtype, metric, smooth, screening, policy, pruning):
    # Include an all-zero class and a fully pruned sample: IoU's 0/0=0 rule
    # must survive the algebraic rewrite, including in max-score fallback.
    probs = torch.tensor([
        [[15, 4, 6, 0, 9, 1, 3], [1, 12, 7, 16, 3, 14, 3], [0, 0, 3, 0, 4, 1, 10]],
        [[4, 3, 2, 1, 3, 1, 0], [1, 2, 3, 4, 2, 3, 0], [0, 0, 0, 0, 0, 0, 0]],
    ], device=device, dtype=dtype) / 16
    before = probs.clone()
    options = dict(metric=metric, smooth=smooth, pruning_prob=pruning, safe_screening=screening)
    masks = rankseg_rma(probs, output_mode="multilabel", **options).cpu().tolist()
    actual = rankseg_rma(probs, unassigned_policy=policy,
                         **({"void_index": -7} if policy == "void" else {}), **options).cpu().tolist()
    assert torch.equal(probs, before)
    for values, selected, labels in zip(probs.cpu().tolist(), masks, actual):
        values = [[Fraction(v) for v in row] for row in values]
        active = [max(row) > pruning for row in values]
        counts = [sum(row[i] for row in selected) for i in range(len(labels))]
        means = [sum(row) for row in values]
        unique = [[flag and counts[i] == 1 for i, flag in enumerate(row)] for row in selected]
        volumes = [sum(row) for row in unique]
        masses = [sum(v for v, flag in zip(row, flags) if flag) for row, flags in zip(values, unique)]
        for i, predicted in enumerate(labels):
            if counts[i] == 0 and policy == "void":
                assert predicted == -7
                continue
            eligible = [c for c in range(len(values)) if (
                selected[c][i] if counts[i] else active[c] or not any(active)
            )]
            scores = {
                c: (_objective(metric, masses[c] + values[c][i], volumes[c] + 1, means[c], Fraction(smooth))
                    - _objective(metric, masses[c], volumes[c], means[c], Fraction(smooth)))
                for c in eligible
            }
            assert predicted in scores
            # Allow either exact mathematical maximizer, not an absolute-eps
            # budget that would mask misordered small increments.
            assert scores[predicted] == max(scores.values())
