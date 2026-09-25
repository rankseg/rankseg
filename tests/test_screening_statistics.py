"""Full-image fused reductions must preserve certificates and pruning."""

import pytest
import torch

from rankseg import _rankseg_algo as algo
from rankseg import _screening as screening
from scripts.benchmark_rma_screening import objective_diagnostics

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.fixture
def backend():
    backend = screening._cuda_backend()
    if backend is None:
        pytest.skip("Triton unavailable")
    return backend


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["contiguous", "channel_strided", "spatial_strided"])
@pytest.mark.parametrize("dim", [1, 4095, 4096, 4097, 65537])
def test_full_statistics_and_prepared_certificates(backend, dtype, layout, dim):
    rows = torch.rand(6, dim, generator=torch.Generator().manual_seed(623), dtype=dtype).cuda()
    if layout == "channel_strided":
        rows = rows.T.contiguous().T
    elif layout == "spatial_strided":
        rows = rows.repeat_interleave(2, -1)[:, ::2]
    guard = screening._ROUNDING_GUARD * torch.finfo(dtype).eps
    rows[0] = 0
    rows[1] = 1
    rows[2] = 0.5 + guard
    rows[3] *= 0.25
    rows[4] = rows[4].pow(16)
    original = rows.clone()
    means, maxima, state = backend.screening_statistics(rows, guard)
    torch.testing.assert_close(means, rows.sum(-1), rtol=4 * torch.finfo(dtype).eps, atol=0)
    assert torch.equal(maxima, rows.amax(-1))
    # The same reduced means and mass must be used in all lower-bound terms.
    for pruning in (0.0, 0.5, 1.0):
        copied_state = tuple(x.clone() for x in state)
        active = maxima > pruning
        masks, candidates, count, mass = backend.screening_candidates(
            rows, means, maxima, guard, active, statistics=copied_state,
        )
        full_h = rows > 0.5 + guard
        assert torch.equal(count, full_h.sum(-1))
        torch.testing.assert_close(mass, (rows * full_h).sum(-1))
        lower = torch.maximum(torch.maximum(maxima / (means + 2), mass / (count + means + 1)),
                              means / (dim + means + 1))
        lower = (lower - guard).clamp(0, 0.5)
        assert torch.equal(masks, full_h & active[:, None])
        assert torch.equal(candidates, active[:, None] & (rows >= lower[:, None]) & ~full_h)
    assert torch.equal(rows, original)


@pytest.mark.parametrize("pruning", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_prepared_pipeline_objective_and_pruning(backend, monkeypatch, pruning, dtype):
    probs = torch.rand(2, 4, 8193, generator=torch.Generator().manual_seed(927)).pow(12).to("cuda", dtype)
    probs[:, 0] = 0
    probs[:, 1] *= 0.4
    original = probs.clone()
    masks = algo.rankseg_rma(probs, output_mode="multilabel", pruning_prob=pruning, safe_screening=True)
    objective_diagnostics(probs, masks, pruning)
    assert torch.equal(probs, original)


def test_full_statistics_respects_large_grid_limit(backend, monkeypatch):
    monkeypatch.setattr(backend, "_MAX_GRID_Y", 2)
    rows = torch.linspace(0, 1, 32769, device="cuda").repeat(2, 1)
    means, maxima, _ = backend.screening_statistics(rows, 32 * torch.finfo(rows.dtype).eps)
    torch.testing.assert_close(means, rows.sum(-1))
    assert torch.equal(maxima, rows.amax(-1))


@pytest.mark.parametrize("dim", [1, 4095, 4096, 4097, 65537])
@pytest.mark.parametrize("drop_row", [False, True])
def test_stable_candidate_packing_matches_nonzero(backend, dim, drop_row):
    rows = torch.rand(5, dim, device="cuda", generator=torch.Generator(device="cuda").manual_seed(164))
    rows[0] = 0
    rows[1] = 1
    rows[2] *= 0.3
    means, maxima, state = backend.screening_statistics(rows, 32 * torch.finfo(rows.dtype).eps)
    _, candidates, _, _, counts = backend.screening_candidates(
        rows, means, maxima, 32 * torch.finfo(rows.dtype).eps, maxima > 0.5,
        statistics=state, return_counts=True,
    )
    offsets = counts.cumsum(-1)
    lengths = offsets[:, -1].cpu().tolist()
    assert lengths == candidates.sum(-1).cpu().tolist()
    if drop_row:
        lengths[3] = 0
    actual = backend.pack_candidates(candidates, offsets, lengths)
    if drop_row:
        candidates[3] = False
    assert torch.equal(actual, candidates.flatten().nonzero(as_tuple=True)[0])


def test_long_candidate_row_does_not_use_padding_budget(backend, monkeypatch):
    monkeypatch.setattr(screening, "_MAX_PADDED_ELEMENTS", 1)
    probs = torch.tensor([[[.9, .45, .4, .35, 0, 0, 0, 0],
                           [1., 0., 0., 0., 0., 0., 0., 0.]]], device="cuda")
    means, maxima, state = screening._rma_dice_screening_statistics(probs)
    masks = screening._rma_dice_screened_masks(
        probs, means, maxima > .5, maxima=maxima, statistics=state,
    )
    objective_diagnostics(probs, masks)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("channels", [2, 3, 8, 9, 19, 21, 150, 256])
@pytest.mark.parametrize("policy", ["max_score", "void"])
def test_fused_assignment_exact_reference(backend, dtype, channels, policy):
    from test_screening import _multiclass_reference
    generator = torch.Generator().manual_seed(314)
    # Spatially strided probabilities, ragged class selections and all three
    # cases (unassigned / unique / overlapping), including all-pruned samples.
    probs = torch.rand(2, channels, 1026, generator=generator, dtype=dtype).cuda()[..., ::2]
    masks = torch.rand(2, channels, 513, generator=generator).cuda() > .7
    masks[..., :10] = False
    masks[:, 1:, 10:20] = False
    probs[1] *= .4
    masks[1] = False
    means = probs.sum(-1)
    active = probs.amax(-1) > .5
    counts = masks.sum(1)
    unique = masks & (counts == 1)[:, None]
    h = (probs * unique).sum(-1, keepdim=True)
    n = algo._count_selected_pixels(unique, dtype)
    result = backend.dice_nonoverlap(masks, probs, counts, means, n, h, active, policy == "void", -5)
    assert torch.equal(result, _multiclass_reference(probs, masks, .5, policy, -5, means))


@pytest.mark.parametrize("void_index", [-(2**63), 2**63 - 1])
def test_fused_assignment_int64_void_labels(backend, void_index):
    probs = torch.zeros(1, 3, 7, device="cuda")
    masks = probs.bool()
    stats = probs.sum(-1)
    result = backend.dice_nonoverlap(masks, probs, masks.sum(1), stats, stats, stats, stats.bool(), True, void_index)
    assert result.dtype == torch.int64
    assert (result == void_index).all()


def test_assignment_large_class_count_uses_portable_path(backend):
    probs = torch.zeros(1, 257, 2, device="cuda")
    masks = probs.bool()
    stats = probs.sum(-1)
    assert backend.dice_nonoverlap(masks, probs, masks.sum(1), stats, stats, stats, stats.bool()) is None
    actual = algo.rankseg_rma(probs, safe_screening=True)
    assert torch.equal(actual, algo.rankseg_rma(probs))


def test_prepared_pipeline_does_not_use_dynamic_nonzero(backend, monkeypatch):
    probs = torch.tensor([[[.9, .4, .2, 0], [0., 0., 0., 0.], [1., 0., 0., 0.]]], device="cuda")
    expected = algo.rankseg_rma(probs, output_mode="multilabel", safe_screening=True)
    def reject(*args, **kwargs):
        pytest.fail("prepared candidate packing must not call dynamic nonzero")
    monkeypatch.setattr(torch.Tensor, "nonzero", reject)
    result = algo.rankseg_rma(probs, output_mode="multilabel", safe_screening=True)
    assert torch.equal(result, expected)


@pytest.mark.parametrize("mode", ["multilabel", "multiclass"])
@pytest.mark.parametrize("pruning", [0.0, 0.5, 1.0])
def test_all_pruned_skips_candidate_statistics(backend, monkeypatch, mode, pruning):
    probs = torch.full((2, 3, 8193), pruning, device="cuda")
    expected = algo.rankseg_rma(probs, output_mode=mode, pruning_prob=pruning)
    def reject(*args, **kwargs):
        pytest.fail("all-pruned input should reuse the validated maximum")
    monkeypatch.setattr(algo, "_rma_dice_screening_statistics", reject)
    actual = algo.rankseg_rma(probs, output_mode=mode, pruning_prob=pruning, safe_screening=True)
    assert torch.equal(actual, expected)


def test_validation_range_reuses_one_reduction(monkeypatch):
    from rankseg._validation import validate_probability_tensor
    calls = []
    original = torch.aminmax
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "aminmax", counted)
    probs = torch.tensor([[[0., .25, .5]]], device="cuda")
    assert validate_probability_tensor(probs, return_bounds=True) == (0., .5)
    assert calls == [1]
    assert validate_probability_tensor(probs) is None
    assert calls == [1, 1]
    assert validate_probability_tensor(probs, check_values=False, return_bounds=True) is None
    assert calls == [1, 1]
