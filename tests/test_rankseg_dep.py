"""RankSEG-DEP: spatially localized dependence solver."""

import itertools
import math

import pytest
import torch
import torch.nn.functional as F

import rankseg._rankseg_algo as algo_module
from rankseg import RankSEG, rankseg_dep, rankseg_rma
from rankseg._rankseg_algo import _dep_fixed_point_masks, _dep_volume_means
from rankseg.functional import rankseg

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
]


def _smooth_probs(batch, classes, shape, seed=0, scale=3.0, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.randn(batch, classes, *(max(2, n // 8) for n in shape), generator=generator) * scale
    mode = {1: "linear", 2: "bilinear", 3: "trilinear"}[len(shape)]
    return F.interpolate(coarse, size=shape, mode=mode, align_corners=False).softmax(dim=1).to(dtype)


def _oracle_volume_means(probs, theta):
    """Direct O(d^2) evaluation of q + (nu / p) * (nu * kappa) in float64."""
    rows, *shape = probs.shape
    probs = probs.double()
    coords = torch.tensor(list(itertools.product(*(range(n) for n in shape))), dtype=torch.float64)
    delta = coords[:, None, :] - coords[None, :, :]
    kernel = torch.exp(-delta.square().sum(-1) / (2 * theta**2))
    flat = probs.reshape(rows, -1)
    nu = torch.sqrt(flat * (1 - flat) + 1e-10)
    return flat.sum(-1, keepdim=True) + nu / (flat + 1e-8) * (nu @ kernel.T)


def _oracle_fixed_point_mask(prob, mu, max_iters=10, tol=1e-6):
    """Row-by-row transcription of the published fixed-point search."""
    dim = prob.numel()
    taus = torch.arange(1, dim + 1, dtype=prob.dtype)
    score, last = prob, -1.0
    for _ in range(max_iters):
        index = torch.sort(score, descending=True)[1]
        p, m = prob[index], mu[index]
        mean = torch.cumsum(m, 0) / taus
        z0, z1, z2 = torch.cumsum(p, 0), torch.cumsum(p * m, 0), torch.cumsum(p * m**2, 0)
        disc = 1 / (taus + mean)
        objective = z0 * disc - (z1 - mean * z0) * disc**2 + (z2 - 2 * mean * z1 + mean**2 * z0) * disc**3
        opt = int(torch.argmax(objective))
        best = float(objective[opt])
        score = prob / (mu + opt)
        if abs(best - last) < tol:
            break
        last = best
    mask = torch.zeros(dim, dtype=torch.bool)
    mask[index[: opt + 1]] = True
    return mask


@pytest.mark.parametrize("shape", [(11,), (7, 9), (4, 5, 6)])
@pytest.mark.parametrize("theta", [1.5, 300.0])
def test_volume_means_match_direct_kernel_sum(shape, theta):
    probs = _smooth_probs(1, 3, shape, dtype=torch.float64)[0]
    actual = _dep_volume_means(probs, theta)
    expected = _oracle_volume_means(probs, theta)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=0)


def test_volume_means_chunking_matches_unchunked(monkeypatch):
    probs = _smooth_probs(1, 5, (12, 10), dtype=torch.float64)[0]
    expected = _dep_volume_means(probs, 4.0)
    monkeypatch.setattr(algo_module, "_DEP_FFT_CHUNK_NUMEL", 1)
    torch.testing.assert_close(_dep_volume_means(probs, 4.0), expected, rtol=0, atol=0)


def test_volume_means_with_zero_theta_uses_identity_kernel():
    probs = _smooth_probs(1, 3, (7, 9), dtype=torch.float64)[0]
    flat = probs.flatten(1)
    nu = torch.sqrt(flat * (1 - flat) + 1e-10)
    expected = flat.sum(-1, keepdim=True) + (nu / (flat + 1e-8)) * nu
    torch.testing.assert_close(_dep_volume_means(probs, 0.0), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fixed_point_masks_match_row_by_row_reference(dtype):
    probs = _smooth_probs(2, 4, (24, 20), seed=3).flatten(2).reshape(8, -1).to(dtype)
    mu = _dep_volume_means(probs.reshape(8, 24, 20), 300.0)
    sorted_prob, sorted_index = torch.sort(probs, dim=-1, descending=True)
    actual, unconverged = _dep_fixed_point_masks(probs, mu, sorted_prob, sorted_index)
    assert unconverged == 0
    for row in range(probs.shape[0]):
        assert torch.equal(actual[row], _oracle_fixed_point_mask(probs[row], mu[row]))


def _objective(probs, mu, mask):
    t = mask.sum()
    return (2 * probs[mask] / (t + mu[mask])).sum()


def _block_probs(third_probability, size=1 << 20, device="cpu", dtype=torch.float32):
    """Three equal-sized blocks per class, arranged along the flattened image."""
    foreground = torch.tensor([7 / 8, 1 / 8, third_probability], device=device, dtype=dtype).repeat_interleave(size)
    return torch.stack((foreground, 1 - foreground)).reshape(1, 2, 192, 128, 128)


@pytest.mark.parametrize("device", DEVICES)
def test_large_volume_multiclass_increment_does_not_collapse_to_class_zero(device):
    # Constructed numerical regression, not a real probability map. Class 0
    # selects blocks 0 and 2, class 1 selects blocks 1 and 2, and the exact
    # increments differ by only about 4e-8 while both are about 1e-7. In
    # float32, subtracting the two almost equal overall objectives rounds the
    # increments together (or reverses them) and assigns the whole 2**20-pixel
    # overlap block to class 0. The factored increment keeps the ordering.
    size = 1 << 20
    probs = torch.zeros(1, 2, 192, 128, 128, device=device)
    flat_probs = probs.flatten(2)
    flat_probs[:, 0, :size] = 9 / 16
    flat_probs[:, 0, 2 * size :] = 13 / 32
    flat_probs[:, 1, size : 2 * size] = 63 / 64
    flat_probs[:, 1, 2 * size :] = 19 / 32

    # Float64 reference for the DEP theta=0 increment at an overlap pixel.
    flat = flat_probs[0].double()
    q = flat.sum(-1, keepdim=True)
    mu = q + (flat * (1 - flat) + 1e-10) / (flat + 1e-8)
    safe = torch.zeros_like(flat, dtype=torch.bool)
    safe[0, :size] = True
    safe[1, size : 2 * size] = True
    denom = size + mu
    increments = 2 * flat[:, 2 * size] / (denom[:, 2 * size] + 1)
    increments -= (2 * flat / (denom * (denom + 1)) * safe).sum(dim=1)
    assert increments[1] > increments[0] > 0

    masks = rankseg_dep(probs, theta=0.0, output_mode="multilabel")[0].flatten(1)
    overlap = masks[0] & masks[1]
    assert not overlap[: 2 * size].any()
    # Both classes select essentially all of the third block. The fixed-point
    # search can lose a few boundary pixels in float32.
    assert overlap[2 * size :].sum() > size * 0.99

    actual = rankseg_dep(probs, theta=0.0).flatten()
    assert (actual[:size] == 0).all()
    assert (actual[size : 2 * size] == 1).all()
    assert (actual[overlap] == 1).all()


def test_theta_zero_matches_rma_on_controlled_large_input():
    # theta=0 is the delta-kernel limit: distinct pixels are independent.
    # The two algorithms can still differ on a handful of fixed-point ties.
    probs = _block_probs(13 / 32)
    assert (rankseg_dep(probs, theta=0.0) != rankseg_rma(probs)).sum() <= 16
    assert (
        rankseg_dep(probs, theta=0.0, output_mode="multilabel") != rankseg_rma(probs, output_mode="multilabel")
    ).sum() <= 16


def _assert_multiclass_assignment_matches_objective(probs, theta, policy):
    kwargs = {"unassigned_policy": policy, "void_index": -1} if policy == "void" else {}
    actual = rankseg_dep(probs, theta=theta, **kwargs).flatten()
    masks = rankseg_dep(probs, theta=theta, output_mode="multilabel")[0].flatten(1)
    flat = probs[0].flatten(1)
    active = flat.amax(-1) > 0.5
    mu = _dep_volume_means(probs[0], theta)
    counts = masks.sum(0)
    fixed = masks & (counts == 1)
    for j in range(flat.shape[1]):
        if counts[j] == 0 and policy == "void":
            assert actual[j] == -1
            continue
        eligible = masks[:, j] if counts[j] > 0 else active
        increments = []
        for c in range(flat.shape[0]):
            if not eligible[c]:
                increments.append(-math.inf)
                continue
            after = fixed[c].clone()
            after[j] = True
            increments.append(float(_objective(flat[c], mu[c], after) - _objective(flat[c], mu[c], fixed[c])))
        assert actual[j] == max(range(len(increments)), key=increments.__getitem__)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("policy", ["max_score", "void"])
def test_multiclass_assignment_matches_objective_increment_oracle(seed, policy):
    probs = _smooth_probs(1, 3, (10, 12), seed=seed, scale=1.2, dtype=torch.float64)
    _assert_multiclass_assignment_matches_objective(probs, 5.0, policy)


@pytest.mark.parametrize("policy", ["max_score", "void"])
def test_multiclass_increment_uses_the_appendix_denominator(policy):
    # One pixel flips when the increment denominator carries the former extra
    # +1, so this pins the Appendix F formula d_j = t + mu_j. The winning
    # margin is large enough to be stable across platforms.
    probs = _smooth_probs(1, 3, (6, 7), seed=4, scale=2.0, dtype=torch.float64)
    _assert_multiclass_assignment_matches_objective(probs, 0.5, policy)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(17,), (12, 16), (5, 6, 7)])
@pytest.mark.parametrize("output_mode", ["multiclass", "multilabel"])
def test_output_contract_for_any_spatial_rank(device, shape, output_mode):
    probs = _smooth_probs(2, 3, shape).to(device)
    preds = RankSEG(solver="DEP", output_mode=output_mode)(probs)
    if output_mode == "multilabel":
        assert preds.dtype == torch.bool and preds.shape == probs.shape
    else:
        assert preds.dtype == torch.int64 and preds.shape == (2, *shape)
        assert ((preds >= 0) & (preds < 3)).all()
    assert preds.device == probs.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("output_mode", ["multiclass", "multilabel"])
def test_cuda_matches_cpu_in_float64(output_mode):
    probs = _smooth_probs(2, 4, (32, 24), dtype=torch.float64)
    expected = rankseg_dep(probs, output_mode=output_mode)
    assert torch.equal(rankseg_dep(probs.cuda(), output_mode=output_mode).cpu(), expected)


def test_class_and_functional_interfaces_agree():
    probs = _smooth_probs(2, 4, (16, 16))
    expected = rankseg_dep(probs, theta=50.0)
    assert torch.equal(RankSEG(solver="dep", theta=50.0)(probs), expected)
    assert torch.equal(rankseg(probs, solver=" DEP ", theta=50.0), expected)


def test_zero_theta_is_accepted_by_all_interfaces():
    probs = _smooth_probs(1, 3, (16, 16))
    expected = rankseg_dep(probs, theta=0.0)
    assert torch.equal(RankSEG(solver="DEP", theta=0.0)(probs), expected)
    assert torch.equal(rankseg(probs, solver="DEP", theta=0.0), expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_inputs_are_computed_in_float32(dtype):
    probs = _smooth_probs(1, 3, (16, 16))
    assert torch.equal(rankseg_dep(probs.to(dtype)), rankseg_dep(probs.to(dtype).float()))


def test_empty_batch():
    probs = torch.rand(0, 3, 8, 8)
    assert rankseg_dep(probs).shape == (0, 8, 8)
    assert rankseg_dep(probs, output_mode="multilabel").shape == (0, 3, 8, 8)


def test_pruned_classes_and_all_pruned_fallback():
    probs = torch.full((2, 3, 6, 6), 1 / 3)
    probs[0] = _smooth_probs(1, 3, (6, 6), scale=6.0)[0]
    probs[0, 2] = 0.0
    probs[0, :2] /= probs[0, :2].sum(0, keepdim=True)
    multilabel = rankseg_dep(probs, output_mode="multilabel")
    assert not multilabel[0, 2].any() and not multilabel[1].any()
    multiclass = rankseg_dep(probs)
    assert (multiclass[0] != 2).all()
    assert ((multiclass[1] >= 0) & (multiclass[1] < 3)).all()
    voided = rankseg_dep(probs, unassigned_policy="void")
    assert (voided[1] == 255).all()


def test_non_convergence_warns(monkeypatch):
    monkeypatch.setattr(algo_module, "_DEP_MAX_ITERS", 1)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        rankseg_dep(_smooth_probs(1, 3, (16, 16)))


@pytest.mark.parametrize("metric", ["iou", "IoU", "acc"])
def test_dep_rejects_non_dice_metrics(metric):
    with pytest.raises(ValueError, match="DEP"):
        RankSEG(metric=metric, solver="DEP")(_smooth_probs(1, 3, (8, 8)))
    if metric.lower() == "iou":
        with pytest.raises(ValueError, match="IoU is not supported"):
            rankseg_dep(_smooth_probs(1, 3, (8, 8)), metric=metric)


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"smooth": 1.0}, ValueError, "smooth=0"),
        ({"theta": -1.0}, ValueError, "greater than or equal to 0"),
        ({"theta": float("nan")}, ValueError, "theta must be finite"),
        ({"theta": "300"}, TypeError, "theta must be a real number"),
        ({"kernel_size": 3}, TypeError, "does not accept solver parameters: kernel_size"),
        ({"amplitude": 1.0}, TypeError, "does not accept solver parameters: amplitude"),
        ({"safe_screening": True}, TypeError, "does not accept solver parameters: safe_screening"),
        ({"output_mode": "multilabel", "unassigned_policy": "void"}, TypeError, "does not accept"),
        ({"void_index": 7}, ValueError, "void_index requires unassigned_policy='void'"),
        ({"unassigned_policy": "void", "void_index": 1}, ValueError, "outside the valid class index range"),
    ],
)
def test_dep_parameter_validation(kwargs, error, match):
    with pytest.raises(error, match=match):
        rankseg(_smooth_probs(1, 3, (8, 8)), solver="DEP", **kwargs)


def test_dep_rejects_single_channel_multiclass():
    with pytest.raises(ValueError, match="Single-channel"):
        rankseg_dep(torch.rand(1, 1, 8, 8))
