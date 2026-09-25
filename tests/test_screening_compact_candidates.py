"""Compact candidate replay must preserve every index, bound and row count."""

import os

import pytest
import torch

from rankseg import _rankseg_algo as algo
from rankseg import _screening as screening

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.fixture
def backend():
    module = screening._cuda_backend()
    if module is None:
        pytest.skip("Triton unavailable")
    return module


def compare_forms(backend, rows, active, prepared, *, drop_rows=False):
    guard = screening._ROUNDING_GUARD * torch.finfo(rows.dtype).eps
    original = rows.clone()
    if prepared:
        means, maxima, state = backend.screening_statistics(rows, guard)
    else:
        means, maxima, state = rows.sum(-1), rows.amax(-1), None
    dense = backend.screening_candidates(rows, means, maxima, guard, active,
                                         statistics=state, return_counts=True)
    old_mask = dense[0].clone()
    compact = backend.screening_candidates(rows, means, maxima, guard, active,
                                           statistics=state, return_counts=True,
                                           materialize_candidates=False)
    masks, bounds, n, h, counts = compact
    assert isinstance(bounds, backend._CandidateBounds)
    assert bounds.probabilities is rows
    if state is not None:
        assert bounds.lower is state[-1]
    assert torch.equal(masks, old_mask)
    for a, b in zip(dense[2:], compact[2:]):
        assert torch.equal(a, b)
    # Independent candidate predicate using the exact stored, not re-reduced,
    # lower bound. Include all original probabilities, regardless of pruning.
    expected_mask = rows > .5 + guard
    expected = (rows >= bounds.lower[:, None]) & ~expected_mask
    if active is not None:
        expected &= active[:, None]
        expected_mask &= active[:, None]
    assert torch.equal(masks, expected_mask)
    assert torch.equal(dense[1], expected)
    offsets = counts.cumsum(-1)
    lengths = offsets[:, -1].cpu().tolist()
    assert lengths == expected.sum(-1).cpu().tolist()
    if drop_rows:
        lengths[::2] = [0] * len(lengths[::2])
        expected[::2] = False
    result = backend.pack_candidates(bounds, offsets, lengths)
    reference = expected.flatten().nonzero(as_tuple=True)[0]
    assert result.dtype == torch.int64
    assert torch.equal(result, reference)
    assert torch.equal(result, backend.pack_candidates(dense[1], offsets, lengths))
    assert torch.equal(rows, original)
    return compact, offsets, lengths, reference


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["contiguous", "channel_strided", "spatial_strided"])
@pytest.mark.parametrize("dim", [1, 31, 4095, 4096, 4097, 8193, 65537])
@pytest.mark.parametrize("prepared", [False, True])
def test_exact_compaction_strides_and_tails(backend, dtype, layout, dim, prepared):
    rows = torch.rand(6, dim, dtype=dtype, generator=torch.Generator().manual_seed(392)).cuda()
    if layout == "channel_strided":
        rows = rows.T.contiguous().T
    elif layout == "spatial_strided":
        rows = rows.repeat_interleave(2, -1)[:, ::2]
    rows[0] = 0
    rows[1] = 1
    rows[2] = .5 + screening._ROUNDING_GUARD * torch.finfo(dtype).eps
    rows[3] = rows[3].pow(12)
    rows[4] *= .3
    active = torch.tensor([False, True, True, True, True, False], device="cuda")
    compare_forms(backend, rows, active, prepared, drop_rows=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("pruning", [None, "none", "all", "partial"])
@pytest.mark.parametrize("n_rows", [1, 7])
def test_empty_full_and_ragged_candidates(backend, dtype, pruning, n_rows):
    rows = torch.zeros(n_rows, 16385, device="cuda", dtype=dtype)
    rows[:, :100] = .9
    for row in range(n_rows):
        rows[row, 8190:8190 + 3 * row + 1] = .4
    active = None if pruning is None else torch.full((n_rows,), pruning == "all", device="cuda")
    if pruning == "partial":
        active[::2] = True
    compare_forms(backend, rows, active, True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("lower", [0., .25, .5])
def test_exact_boundaries_and_adjacent_floats(backend, dtype, lower):
    guard = screening._ROUNDING_GUARD * torch.finfo(dtype).eps
    threshold = torch.tensor(.5 + guard, device="cuda", dtype=dtype)
    lo = threshold.new_tensor(lower)
    zero, one = lo.new_tensor(0.), lo.new_tensor(1.)
    values = torch.stack((zero, one, lo, torch.nextafter(lo, zero), torch.nextafter(lo, one),
                          threshold, torch.nextafter(threshold, zero), torch.nextafter(threshold, one)))
    rows = values.repeat(2, 1025)
    # A supplied state isolates exact comparisons from bound derivation.
    state = (torch.empty_like(rows, dtype=torch.bool), torch.zeros(2, dtype=torch.int64, device="cuda"),
             rows.new_zeros(2), rows.new_full((2,), lower))
    active = torch.tensor([True, False], device="cuda")
    masks, bounds, _, _, counts = backend.screening_candidates(
        rows, rows.sum(-1), rows.amax(-1), guard, active, statistics=state,
        return_counts=True, materialize_candidates=False,
    )
    offsets = counts.cumsum(-1)
    indices = backend.pack_candidates(bounds, offsets, offsets[:, -1].cpu().tolist())
    expected = (rows >= lower) & ~(rows > threshold) & active[:, None]
    assert torch.equal(indices, expected.flatten().nonzero(as_tuple=True)[0])
    assert torch.equal(masks, (rows > threshold) & active[:, None])


def test_compact_mode_requires_counts_before_any_allocation(backend, monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail("invalid compact request must fail before statistics")
    monkeypatch.setattr(backend, "_statistics", reject)
    with pytest.raises(ValueError, match="return_counts"):
        backend.screening_candidates(None, None, None, 0., materialize_candidates=False)


def test_no_dense_candidate_allocation(backend, monkeypatch):
    rows = torch.linspace(0, 1, 8193, device="cuda").repeat(3, 1)
    original = torch.empty_like
    def checked(tensor, *args, **kwargs):
        if tensor.dtype == torch.bool and tensor.shape == rows.shape:
            pytest.fail("compact mode must not allocate the dense candidate mask")
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch, "empty_like", checked)
    means, maxima, state = backend.screening_statistics(rows, 32 * torch.finfo(rows.dtype).eps)
    _, bounds, _, _, counts = backend.screening_candidates(
        rows, means, maxima, 32 * torch.finfo(rows.dtype).eps,
        statistics=state, return_counts=True, materialize_candidates=False,
    )
    offsets = counts.cumsum(-1)
    assert backend.pack_candidates(bounds, offsets, offsets[:, -1].cpu().tolist()).numel() > 0


def test_compact_descriptor_retains_adaptive_tile_size(backend, monkeypatch):
    rows = torch.linspace(0, 1, 16385, device="cuda").repeat(3, 1)
    with monkeypatch.context() as patch:
        patch.setattr(backend, "_MAX_GRID_Y", 2)
        result, offsets, lengths, expected = compare_forms(backend, rows, None, True)
    assert result[1].block > backend._BLOCK
    assert torch.equal(backend.pack_candidates(result[1], offsets, lengths), expected)


def test_empty_packing_does_not_launch_kernel(backend, monkeypatch):
    rows = torch.ones(1, 17, device="cuda")
    bounds = backend._CandidateBounds(rows, rows.new_zeros(1), 0., 4096)
    monkeypatch.setattr(backend, "_pack_candidates", None)
    result = backend.pack_candidates(bounds, torch.zeros(1, 1, dtype=torch.int64, device="cuda"), [0])
    assert result.numel() == 0 and result.dtype == torch.int64 and result.is_cuda


def test_compaction_on_nondefault_stream(backend):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        rows = torch.linspace(0, 1, 8193, device="cuda").repeat(3, 1)
        compare_forms(backend, rows, None, True)
    stream.synchronize()


def test_candidate_stage_peak_memory(backend):
    rows = torch.full((4, 1048576), .25, device="cuda")
    means, maxima = rows.sum(-1), rows.amax(-1)
    guard = 32 * torch.finfo(rows.dtype).eps
    for materialize in (True, False):
        result = backend.screening_candidates(rows, means, maxima, guard, return_counts=True,
                                               materialize_candidates=materialize)
        del result
    peaks = []
    for materialize in (True, False):
        torch.cuda.synchronize()
        resident = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = backend.screening_candidates(rows, means, maxima, guard, return_counts=True,
                                               materialize_candidates=materialize)
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated() - resident)
        del result
    assert peaks[1] < .55 * peaks[0]


def test_multiclass_peak_after_eliminating_candidate_mask(backend):
    dim = 1048576
    probs = torch.full((1, 21, dim), .0001, device="cuda")
    probs[:, 0] = .999
    result = algo.rankseg_rma(probs, safe_screening=True)
    del result
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = algo.rankseg_rma(probs, safe_screening=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - resident
    # Status+class-index output+binary masks: (21+9)*D, plus bounded metadata.
    # The previous dense M allocation alone raised the peak to ~42*D.
    assert peak < 32 * dim
    assert (result == 0).all()


@pytest.mark.skipif(os.environ.get("RANKSEG_TEST_LARGE_CUDA") != "1", reason="opt-in large CUDA test")
def test_liver_sized_compaction_has_candidates_in_last_block(backend):
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < 8 * 1024**3:
        pytest.skip("requires 8 GiB free CUDA memory")
    dim = 846 * 512 * 512
    rows = torch.zeros(3, dim, device="cuda")
    rows[:, :2] = .9
    rows[:, -2:] = .4
    guard = 32 * torch.finfo(rows.dtype).eps
    means, maxima, state = backend.screening_statistics(rows, guard)
    masks, bounds, _, _, counts = backend.screening_candidates(
        rows, means, maxima, guard, statistics=state,
        return_counts=True, materialize_candidates=False,
    )
    offsets = counts.cumsum(-1)
    lengths = offsets[:, -1].cpu().tolist()
    assert lengths == [2, 2, 2]
    expected = torch.tensor([dim - 2, dim - 1, 2 * dim - 2, 2 * dim - 1,
                              3 * dim - 2, 3 * dim - 1], device="cuda")
    assert torch.equal(backend.pack_candidates(bounds, offsets, lengths), expected)
    assert masks[:, :2].all() and not masks[:, 2:].any()
