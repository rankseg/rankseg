"""Objective accuracy, small-volume ties and regressions for RMA screening."""

from fractions import Fraction
from itertools import product

import pytest
import torch

from rankseg import RankSEG
from rankseg import _rankseg_algo as algo
from rankseg import _screening as screening
from rankseg.functional import rankseg

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]
DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def _assert_epsilon_optimal(probs, mask, pruning_prob=0.5):
    """Independently evaluate every full prefix in float64 on the CPU."""
    values = probs.detach().cpu().double().flatten(2)
    selected = mask.detach().cpu().flatten(2)
    means = values.sum(-1)
    sorted_values = values.sort(descending=True).values
    counts = torch.arange(1, values.shape[-1] + 1, dtype=torch.float64)
    best = (2 * sorted_values.cumsum(-1) / (means[..., None] + counts + 1)).amax(-1)
    actual = 2 * (values * selected).sum(-1) / (means + selected.sum(-1) + 1)
    active = values.amax(-1) > pruning_prob
    assert not selected[~active].any()
    dtype = torch.float64 if probs.dtype == torch.float64 else torch.float32
    # Arithmetic regression budget only; inference has no epsilon comparison.
    tolerance = (16 if dtype == torch.float64 else 4) * torch.finfo(dtype).eps
    assert torch.all((best - actual)[active] <= tolerance), (best - actual)[active]


def _multiclass_reference(probs, masks, pruning_prob, policy, void_index, means=None, unique_stats=None):
    """Apply the existing unsmoothed assignment rule to screened binary masks."""
    probs = probs if probs.dtype == torch.float64 else probs.float()
    values = probs.flatten(2)
    masks = masks.flatten(2)
    counts = masks.sum(1)
    single = masks & (counts == 1)[:, None]
    mass = (values * single).sum(-1, keepdim=True)
    volume = single.sum(-1, keepdim=True).to(probs.dtype)
    if unique_stats is not None:
        # Separately validated fused sums can round differently. Supplying
        # them isolates exact assignment/eligibility from the reduction tree.
        volume, mass = unique_stats
        volume = volume.to(probs.dtype).reshape(*values.shape[:2], 1)
        mass = mass.reshape(*values.shape[:2], 1)
    # Optional full-image means isolate the assignment rule from differences
    # in the CUDA statistics reduction tree (tested independently).
    means = values.sum(-1, keepdim=True) if means is None else means[..., None]
    denom = volume + means + 1
    increments = 2 * ((mass + values) / (denom + 1) - mass / denom)
    active = values.amax(-1) > pruning_prob
    if policy == "max_score":
        active = active | ~active.any(1, keepdim=True)
    eligible = torch.where((counts > 0)[:, None], masks, active[..., None])
    result = increments.masked_fill(~eligible, -torch.inf).argmax(1)
    if policy == "void":
        result[counts == 0] = void_index
    return result.reshape(probs.shape[0], *probs.shape[2:])


def _compare(probs, **kwargs):
    before = probs.clone()
    expected = algo.rankseg_rma(probs, safe_screening=False, **kwargs)
    actual = algo.rankseg_rma(probs, safe_screening=True, **kwargs)
    if kwargs.get("metric", "dice") != "dice" or kwargs.get("smooth", 0) != 0 or not probs.shape[0]:
        assert torch.equal(actual, expected)
    elif kwargs.get("output_mode", "multiclass") == "multilabel":
        _assert_epsilon_optimal(probs, actual, kwargs.get("pruning_prob", 0.5))
    else:
        binary = algo.rankseg_rma(probs, output_mode="multilabel", safe_screening=True,
                                  pruning_prob=kwargs.get("pruning_prob", 0.5))
        _assert_epsilon_optimal(probs, binary, kwargs.get("pruning_prob", 0.5))
        reference = _multiclass_reference(probs, binary, kwargs.get("pruning_prob", 0.5),
                                          kwargs.get("unassigned_policy", "max_score"),
                                          kwargs.get("void_index", 255))
        assert torch.equal(actual, reference)
    assert torch.equal(probs, before)
    assert actual.device == probs.device
    assert actual.dtype == expected.dtype
    assert not actual.requires_grad
    return actual


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("output_mode", ["multiclass", "multilabel"])
@pytest.mark.parametrize("pruning_prob", [0.0, 0.5, 1.0])
def test_random_and_ragged_masks_preserve_objective(device, dtype, output_mode, pruning_prob):
    generator = torch.Generator().manual_seed(972)
    for dim in (1, 7, 129, 2049):
        probs = torch.rand(2, 4, dim, generator=generator)
        probs[:, 0] = probs[:, 0].pow(16)
        probs[:, 1] = 1 - probs[:, 1].pow(16)
        probs[:, 2] = probs[:, 2].round()
        # The last row often has many more unresolved values than the others.
        probs = probs.to(device=device, dtype=dtype)
        _compare(probs, output_mode=output_mode, pruning_prob=pruning_prob)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ragged_padded_scatter_matches_individual_rows_and_full_sort(device, dtype, monkeypatch):
    probs = torch.tensor(
        [[
            [0.9, 0.4, 0.2, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0.9, 0.45, 0.35, 0.34, 0.2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]], dtype=dtype, device=device,
    )
    # Include differently sized M sets in one padded workspace and exercise
    # fake padding positions whose apparent index could otherwise corrupt H.
    monkeypatch.setattr(screening, "_MAX_PADDING_RATIO", 100)
    expected = algo.rankseg_rma(probs, output_mode="multilabel", pruning_prob=0)
    batched = screening._rma_dice_screened_masks(probs, probs.sum(-1), probs.amax(-1) > 0)
    assert torch.equal(batched, expected)
    monkeypatch.setattr(screening, "_MAX_BATCHED_ROWS", 1)
    individual = screening._rma_dice_screened_masks(probs, probs.sum(-1), probs.amax(-1) > 0)
    assert torch.equal(individual, batched)


@pytest.mark.parametrize("device", DEVICES)
def test_screening_reduces_sorted_dimension_and_skips_resolved_rows(device, monkeypatch):
    probs = torch.zeros(1, 3, 64, device=device)
    probs[0, 0, :3] = torch.tensor([0.9, 0.4, 0.2], device=device)
    probs[0, 1, :2] = 1
    expected = algo.rankseg_rma(probs, output_mode="multilabel", pruning_prob=0)
    dimensions = []
    tensor_sort = torch.Tensor.sort

    def tracked_sort(self, *args, **kwargs):
        dimensions.append(self.shape[-1])
        return tensor_sort(self, *args, **kwargs)

    def no_full_sort(*args, **kwargs):
        pytest.fail("well-separated screened example should not use full sort")

    monkeypatch.setattr(torch.Tensor, "sort", tracked_sort)
    monkeypatch.setattr(torch, "sort", no_full_sort)
    actual = algo.rankseg_rma(probs, output_mode="multilabel", pruning_prob=0, safe_screening=True)
    assert torch.equal(actual, expected)
    assert dimensions == [1]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ties_and_adjacent_probabilities_prefer_smaller_volume_without_retry(device, dtype, monkeypatch):
    # [3/4, 1/4] has mu=1 and exactly tied Dice scores: 1.5/3 == 2/4.
    quarter = torch.tensor(0.25, dtype=dtype, device=device)
    near = [quarter, torch.nextafter(quarter, quarter.new_tensor(0)), torch.nextafter(quarter, quarter.new_tensor(1))]
    for value in near:
        probs = torch.stack((value.new_tensor(0.75), value)).view(1, 1, -1)
        direct = screening._rma_dice_screened_masks(probs, probs.sum(-1), probs.amax(-1) > 0)
        _assert_epsilon_optimal(probs, direct, 0)
        with monkeypatch.context() as patch:
            patch.setattr(torch, "sort", lambda *args, **kwargs: pytest.fail("near tie must not retry full sort"))
            actual = algo.rankseg_rma(probs, output_mode="multilabel", pruning_prob=0, safe_screening=True)
        assert actual.flatten().tolist() == [True, False]
        _assert_epsilon_optimal(probs, actual, 0)


@pytest.mark.parametrize("device", DEVICES)
def test_fully_certified_row_does_not_retry_for_close_omitted_prefix(device):
    # No undecided entries: H minus one pixel lies within the guard of H,
    # despite there being only one entry in the reduced score array.
    probs = torch.ones(1, 1, 262_144, device=device)
    direct = screening._rma_dice_screened_masks(probs, probs.sum(-1), probs.amax(-1) > 0)
    assert direct.all()
    _compare(probs, output_mode="multilabel")


@pytest.mark.parametrize("device", DEVICES)
def test_candidate_rows_larger_than_group_budget_are_solved(device, monkeypatch):
    monkeypatch.setattr(screening, "_MAX_PADDED_ELEMENTS", 1)
    probs = torch.tensor([[[0.9, 0.4, 0.35, 0], [1, 1, 0, 0]]], device=device)
    direct = screening._rma_dice_screened_masks(probs, probs.sum(-1), probs.amax(-1) > 0)
    _assert_epsilon_optimal(probs, direct, 0)
    _compare(probs, output_mode="multilabel")
    _compare(probs, output_mode="multiclass")


def test_candidate_groups_respect_padding_workspace_and_row_limits(monkeypatch):
    monkeypatch.setattr(screening, "_MAX_PADDED_ELEMENTS", 64)
    monkeypatch.setattr(screening, "_MAX_BATCHED_ROWS", 2)
    lengths = [0, 1, 2, 5, 6, 31, 32, 64]
    groups = list(screening._candidate_groups(lengths))
    assert sorted(row for group in groups for row in group) == list(range(1, len(lengths)))
    for group in groups:
        assert len(group) <= 2
        assert max(lengths[row] for row in group) * len(group) <= 64
        assert max(lengths[row] for row in group) <= 2 * min(lengths[row] for row in group)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("policy", ["max_score", "void"])
def test_screened_negatives_remain_eligible_for_unassigned_fallback(device, policy):
    probs = torch.tensor([[
        [1, 1, 0, 0, 0, 0, 0.34],
        [0, 0, 1, 1, 0, 0, 0.33],
        [0, 0, 0, 0, 1, 1, 0.33],
    ]], dtype=torch.float64, device=device)
    masks = _compare(probs, output_mode="multilabel")
    assert not masks[..., -1].any()
    labels = _compare(probs, output_mode="multiclass", unassigned_policy=policy)
    assert labels[0, -1].item() == (0 if policy == "max_score" else 255)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("policy", ["max_score", "void"])
def test_all_pruned_fallback_overlap_and_nonselecting_classes(device, policy):
    probs = torch.tensor([
        [[0.4, 0.2, 0.0], [0.2, 0.3, 0.0], [0.1, 0.5, 0.0]],
        [[0.9, 0.8, 0.0], [0.8, 0.9, 0.0], [0.0, 0.0, 0.9]],
    ], device=device)
    _compare(probs, output_mode="multiclass", unassigned_policy=policy)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("metric,smooth", [("dice", 0.2), ("dice", 100), ("dice", 1e308), ("iou", 0), ("iou", 0.2)])
def test_other_objectives_use_unchanged_full_sort(device, metric, smooth, monkeypatch):
    def fail(*args):
        pytest.fail("unsupported objective must not enter screening")

    monkeypatch.setattr(algo, "_rma_dice_screened_masks", fail)
    probs = torch.tensor([[[0.501, 0.2], [0.499, 0.8]]], device=device)
    for mode in ("multilabel", "multiclass"):
        _compare(probs, metric=metric, smooth=smooth, output_mode=mode, pruning_prob=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("output_mode", ["multilabel", "multiclass"])
def test_empty_batch_noncontiguous_and_repeated_outputs(device, output_mode):
    _compare(torch.empty(0, 3, 2, 4, device=device), output_mode=output_mode)
    generator = torch.Generator().manual_seed(18)
    probs = torch.rand(2, 3, 2, 7, generator=generator).to(device).transpose(-1, -2).requires_grad_()
    first = _compare(probs, output_mode=output_mode)
    saved = first.clone()
    _compare(torch.ones_like(probs), output_mode=output_mode)
    assert torch.equal(first, saved)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("output_mode", ["multilabel", "multiclass"])
def test_channel_strided_candidate_scatter_updates_actual_output(device, dtype, output_mode):
    probs = torch.tensor([[
        [0.9, 0.4, 0.2, 0], [1, 1, 0, 0], [0.9, 0.45, 0.35, 0.2],
    ]], device=device, dtype=dtype)
    probs = probs.transpose(1, 2).contiguous().transpose(1, 2)
    assert not probs.is_contiguous()
    direct = screening._rma_dice_screened_masks(probs, probs.sum(-1), probs.amax(-1) > 0)
    _assert_epsilon_optimal(probs, direct, 0)
    _compare(probs, output_mode=output_mode, pruning_prob=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("output_mode", ["multilabel", "multiclass"])
def test_original_mask_vectorization_boundary(device, output_mode, monkeypatch):
    monkeypatch.setattr(algo, "_RMA_CUDA_VECTORIZED_MASK_MAX_DIM", 4)
    for dim in (3, 4, 5):
        probs = torch.full((2, 3, dim), 0.5, device=device)
        # The reference exercises both original mask branches; screening also
        # accepts a dense candidate set without a full-sort retry.
        _compare(probs, output_mode=output_mode, pruning_prob=0)


def test_independent_exact_rational_brute_force_oracle():
    generator = torch.Generator().manual_seed(981)
    for n in range(1, 9):
        for _ in range(12):
            values = [Fraction(int(v), 16) for v in torch.randint(0, 17, (n,), generator=generator)]
            mu = sum(values)
            best = max(
                2 * sum(p for p, keep in zip(values, mask) if keep) / (sum(mask) + mu + 1)
                for mask in product((False, True), repeat=n)
            )
            probs = torch.tensor([[[float(v) for v in values]]], dtype=torch.float64)
            mask = _compare(probs, output_mode="multilabel", pruning_prob=0).flatten().tolist()
            score = 2 * sum(p for p, keep in zip(values, mask) if keep) / (sum(mask) + mu + 1)
            assert score == best


@pytest.mark.parametrize("output_mode", ["multilabel", "multiclass"])
def test_public_interfaces_pass_screening_option(output_mode):
    probs = torch.tensor([[[0.9, 0.4, 0.0], [0.1, 0.6, 1.0]]])
    expected = algo.rankseg_rma(probs, output_mode=output_mode, safe_screening=True)
    assert torch.equal(rankseg(probs, output_mode=output_mode, safe_screening=True), expected)
    assert torch.equal(RankSEG(output_mode=output_mode, safe_screening=True)(probs), expected)


@pytest.mark.parametrize("value", [0, 1, "AUTO", " auto ", "true", None, torch.tensor(True)])
def test_screening_requires_boolean_or_auto(value):
    probs = torch.ones(1, 2, 3)
    with pytest.raises(TypeError, match="safe_screening must be a bool"):
        rankseg(probs, safe_screening=value)


def test_other_solvers_reject_screening_option():
    with pytest.raises(TypeError, match="does not accept solver parameters: safe_screening"):
        rankseg(torch.ones(1, 1, 3), solver="BA", output_mode="multilabel", safe_screening=True)


def test_explicit_false_retains_original_path(monkeypatch):
    def fail(*args):
        pytest.fail("explicit False must not enter screening")

    monkeypatch.setattr(algo, "_rma_dice_screened_masks", fail)
    rankseg(torch.ones(1, 2, 3), safe_screening=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scenario", ["sparse_resolved", "softmax_sparse", "all_pruned", "uniform", "ragged"])
def test_benchmark_inputs_preserve_dtype_and_seed(dtype, scenario):
    from scripts.benchmark_rma_screening import make_probabilities

    first = make_probabilities(scenario, 64, dtype, 38)
    assert first.dtype == dtype
    assert first.shape == (2, 3, 64)
    assert torch.equal(first, make_probabilities(scenario, 64, dtype, 38))
    if scenario == "softmax_sparse":
        torch.testing.assert_close(first.sum(dim=1), torch.ones(2, 64, dtype=dtype))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_ties_and_extreme_probabilities(device, dtype):
    generator = torch.Generator().manual_seed(98172)
    for classes in (2, 3, 11):
        logits = torch.randn(2, classes, 257, generator=generator)
        for temperature in (0.05, 1.0, 20.0):
            probs = (logits / temperature).softmax(dim=1).to(device=device, dtype=dtype)
            for mode in ("multiclass", "multilabel"):
                for pruning in (0.0, 0.5):
                    _compare(probs, output_mode=mode, pruning_prob=pruning)
        # Quantized inputs stress large blocks of identical probabilities.
        quantized = (logits.sigmoid() * 4).round().div(4).to(device=device, dtype=dtype)
        _compare(quantized, output_mode="multiclass", pruning_prob=0)
        _compare(quantized, output_mode="multilabel", pruning_prob=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_direct_argmax_does_not_merge_near_scores(device, dtype):
    eps = torch.finfo(dtype).eps
    values = torch.tensor([[0.25], [0.25 + eps]], device=device, dtype=dtype)
    means = values.new_ones(2)
    counts = torch.ones(2, device=device, dtype=torch.int64)
    mass = values.new_full((2,), 0.75)
    # Row 0 is an exact tie. Row 1 has a strictly better nonempty score,
    # despite improving by less than one epsilon.
    result = screening._score_argmax(values, means, counts, mass, None, None)
    assert result.tolist() == [0, 1]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unpadded_candidate_uses_same_small_volume_tie_rule(device, dtype, monkeypatch):
    probs = torch.tensor([[[0.75, 0.25]]], device=device, dtype=dtype)
    monkeypatch.setattr(screening, "_MAX_PADDED_ELEMENTS", 1)
    actual = _compare(probs, output_mode="multilabel")
    assert actual.flatten().tolist() == [True, False]


@pytest.mark.parametrize("device", DEVICES)
def test_compacted_single_active_row_preserves_inactive_classes(device):
    probs = torch.zeros(2, 3, 32, device=device)
    probs[1, 1, :3] = torch.tensor([0.75, 0.25, 0.0], device=device)
    actual = _compare(probs, output_mode="multilabel")
    assert actual.sum().item() == 1
    assert actual[1, 1, 0]


@pytest.mark.parametrize("device", DEVICES)
def test_tiny_positive_objective_is_not_rounded_to_empty(device):
    probs = torch.tensor([[[1e-9, 0]]], device=device)
    actual = _compare(probs, output_mode="multilabel", pruning_prob=0)
    assert actual.flatten().tolist() == [True, False]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dim", [8, 262_144, 262_145])
def test_true_forces_single_channel_cuda_dispatch(dim, monkeypatch):
    calls = []
    original = algo._rma_dice_screened_masks

    def tracked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(algo, "_rma_dice_screened_masks", tracked)
    probs = torch.zeros(2, 1, dim, device="cuda")
    probs[:, :, :2] = 1
    actual = algo.rankseg_rma(probs, output_mode="multilabel", safe_screening=True)
    assert actual.sum().item() == 4
    assert bool(calls)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("batch,classes,dim", [
    (2, 3, 65_536), (2, 3, 65_537), (1, 17, 64),
])
def test_true_forces_few_rows_cuda_dispatch(batch, classes, dim, monkeypatch):
    calls = []
    original = algo._rma_dice_screened_masks

    def tracked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(algo, "_rma_dice_screened_masks", tracked)
    probs = torch.zeros(batch, classes, dim, device="cuda")
    probs[:, :, :2] = 1
    actual = algo.rankseg_rma(probs, output_mode="multilabel", safe_screening=True)
    assert actual.sum().item() == 2 * batch * classes
    assert bool(calls)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch,classes", [(1, 1), (4, 1), (1, 4), (2, 3)])
def test_large_close_optima_against_float64_oracle(device, batch, classes):
    generator = torch.Generator().manual_seed(461)
    probs = torch.rand(batch, classes, 32_769, generator=generator).pow(12).to(device)
    # Quantized probabilities resemble cached low-precision model outputs.
    probs = probs.half().float()
    actual = algo.rankseg_rma(probs, output_mode="multilabel", safe_screening=True)
    _assert_epsilon_optimal(probs, actual)


def test_benchmark_objective_oracle_rejects_incorrect_mask():
    from scripts.benchmark_rma_screening import objective_diagnostics

    probs = torch.tensor([[[0.9, 0.8, 0.0]]])
    with pytest.raises(AssertionError, match="objective regret"):
        objective_diagnostics(probs, torch.zeros_like(probs, dtype=torch.bool))
    result = objective_diagnostics(probs, torch.tensor([[[True, True, False]]]))
    assert result["max_regret"] == 0
