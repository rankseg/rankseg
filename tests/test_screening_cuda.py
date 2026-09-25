"""Force both screening backends, including kernel/block/packing boundaries."""

import builtins
import os

import pytest
import torch

from rankseg import _rankseg_algo as algo
from rankseg import _screening as screening
from scripts.benchmark_rma_screening import objective_diagnostics


@pytest.fixture
def cuda_backend():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    backend = screening._cuda_backend()
    if backend is None:
        pytest.skip("optional Triton unavailable")
    return backend


@pytest.fixture
def force_screening(monkeypatch):
    monkeypatch.setattr(algo, "_rma_dice_use_screening", lambda probs, mode: True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("width", [1, 8, 4095, 4096, 4097, 20000])
def test_fused_score_exactly_matches_materialized_scores(cuda_backend, dtype, width):
    generator = torch.Generator().manual_seed(2126)
    values = torch.rand(4, width, generator=generator, dtype=dtype).cuda().sort(descending=True).values
    lengths = torch.tensor([0, width // 2, width - 1, width], device="cuda")
    values[torch.arange(width, device="cuda") >= lengths[:, None]] = -1
    means = torch.tensor([1, 2, 3, 9], device="cuda", dtype=dtype)
    counts = torch.tensor([0, 0, 1, 2], device="cuda")
    mass = torch.tensor([0, 0, 0.8, 1.7], device="cuda", dtype=dtype)
    reference = screening._score_argmax(values, means, counts, mass, lengths, None)
    actual = screening._score_argmax(values, means, counts, mass, lengths, cuda_backend)
    assert torch.equal(actual, reference)
    # Also exercise the scalar, unpadded return shape and first exact maximum.
    for value in (0, 0.25, 0.25 + torch.finfo(dtype).eps):
        one = values.new_tensor([value])
        mu, n, h = means.new_tensor(1), counts.new_tensor(1), mass.new_tensor(0.75)
        expected = screening._score_argmax(one, mu, n, h, None, None)
        result = screening._score_argmax(one, mu, n, h, None, cuda_backend)
        assert torch.equal(result, expected)
        assert result.item() == (1 if value > 0.25 else 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["contiguous", "channel_strided", "spatial_strided"])
@pytest.mark.parametrize("dim", [1, 129, 4095, 4096, 4097, 65537])
def test_fused_statistics_strides_and_boundaries(cuda_backend, dtype, layout, dim):
    generator = torch.Generator().manual_seed(31)
    rows = torch.rand((5, dim), generator=generator, dtype=dtype).cuda()
    if layout == "channel_strided":
        rows = rows.T.contiguous().T
    elif layout == "spatial_strided":
        rows = rows.repeat_interleave(2, dim=-1)[:, ::2]
    guard = screening._ROUNDING_GUARD * torch.finfo(dtype).eps
    rows[0] = 0
    rows[1] = 1
    rows[2] = 0.5 + guard
    rows[3, 0] = torch.nextafter(rows.new_tensor(0.5 + guard), rows.new_tensor(1))
    means, maxima = rows.sum(-1), rows.amax(-1)
    masks, candidates, count, mass = cuda_backend.screening_candidates(rows, means, maxima, guard)
    expected_masks = rows > 0.5 + guard
    assert masks.is_contiguous()
    assert torch.equal(masks, expected_masks)
    assert torch.equal(count, expected_masks.sum(-1))
    torch.testing.assert_close(mass, (rows * expected_masks).sum(-1))
    # Isolate bound construction from differences in reduction order.
    lower = torch.maximum(torch.maximum(maxima / (means + 2), mass / (count + means + 1)),
                          means / (dim + means + 1))
    lower = (lower - guard).clamp(0, 0.5)
    assert torch.equal(candidates, (rows >= lower[:, None]) & ~expected_masks)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1, 1, 32), (4, 1, 8193), (2, 5, 65537)])
def test_forced_fused_and_portable_objectives(cuda_backend, force_screening, monkeypatch, dtype, shape):
    generator = torch.Generator().manual_seed(9742)
    probs = torch.rand(shape, generator=generator).pow(12).to(device="cuda", dtype=dtype)
    probs[:, -1, -1] = 1
    original = probs.clone()
    fused = algo.rankseg_rma(probs, output_mode="multilabel", safe_screening=True)
    objective_diagnostics(probs, fused)
    # The final multiclass mapping still uses original probabilities and means.
    from test_screening import _multiclass_reference
    if shape[1] > 1:
        fused_classes = algo.rankseg_rma(probs, output_mode="multiclass", safe_screening=True)
        working = probs if dtype == torch.float64 else probs.float()
        means, _, _ = screening._rma_dice_screening_statistics(working)
        status, n, h = cuda_backend.unique_statistics(fused, working)
        counts = fused.sum(1)
        unique = fused & (counts == 1)[:, None]
        assert torch.equal(status, counts.clamp(max=2).byte())
        assert torch.equal(n, unique.sum(-1))
        torch.testing.assert_close(h, (working * unique).sum(-1),
                                   rtol=4 * torch.finfo(working.dtype).eps, atol=0)
        assert torch.equal(fused_classes, _multiclass_reference(
            probs, fused, 0.5, "max_score", 255, means, unique_stats=(n, h),
        ))
    monkeypatch.setattr(screening, "_cuda_backend", lambda: None)
    portable = algo.rankseg_rma(probs, output_mode="multilabel", safe_screening=True)
    objective_diagnostics(probs, portable)
    assert torch.equal(probs, original)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("budget", [1, 1048576])
def test_mask_only_result_preserves_pruning_with_small_groups(device, budget, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setattr(screening, "_MAX_PADDED_ELEMENTS", budget)
    for shape in ((1, 1, 32), (2, 3, 32)):
        values = torch.zeros(shape, device=device)
        values[..., :3] = values.new_tensor([0.9, 0.4, 0.35])
        for pruning in (0, 1):
            masks = screening._rma_dice_screened_masks(
                values, values.sum(-1), values.amax(-1) > pruning,
                maxima=values.amax(-1),
            )
            assert masks.shape == values.shape
            assert masks.dtype == torch.bool
            if pruning == 1:
                assert not masks.any()
            else:
                objective_diagnostics(values, masks)


def test_optional_triton_import_failure(monkeypatch):
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "triton":
            raise ImportError("Triton intentionally unavailable")
        return original(name, *args, **kwargs)

    screening._cuda_backend.cache_clear()
    try:
        monkeypatch.setattr(builtins, "__import__", missing)
        assert screening._cuda_backend() is None
    finally:
        screening._cuda_backend.cache_clear()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_maximum_plateau_spans_multiple_blocks(cuda_backend, dtype):
    # 1025 forced ones and 8196 undecided quarters have score 0.5 at
    # every candidate volume. The first block must win, not the final block.
    values = torch.full((8196,), 0.25, device="cuda", dtype=dtype)
    mu = values.new_tensor(3074)
    count = torch.tensor(1025, device="cuda")
    mass = values.new_tensor(1025)
    assert screening._score_argmax(values, mu, count, mass, None, cuda_backend).item() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rows", [1, 3])
def test_long_score_grid_and_hierarchical_reduction(cuda_backend, dtype, rows, monkeypatch):
    # Nine partial blocks exercise 9 -> 5 -> 3 -> 2 reduction stages without
    # allocating hundreds of millions of probabilities in normal CI.
    width = 9 * cuda_backend._BLOCK - 1
    values = torch.full((rows, width), .25, device="cuda", dtype=dtype)
    means = values.new_full((rows,), 3074)
    n = torch.full((rows,), 1025, device="cuda", dtype=torch.int64)
    mass = values.new_full((rows,), 1025)
    lengths = torch.full((rows,), width, device="cuda", dtype=torch.int64)
    if rows > 1:
        values[1, -1] = 1  # best volume in the final partial block
        lengths[2] = width // 2  # padded tail cannot win
    expected = screening._score_argmax(values, means, n, mass, lengths, None)
    monkeypatch.setattr(cuda_backend, "_SCORE_REDUCTION_BLOCK", 2)
    grids = []
    original = cuda_backend._score_partials

    class TrackedKernel:
        def __getitem__(self, grid):
            grids.append(grid)
            # Long candidate dimension must be grid.x, not limited grid.y.
            assert grid == (9, rows)
            return original[grid]

    monkeypatch.setattr(cuda_backend, "_score_partials", TrackedKernel())
    actual = screening._score_argmax(values, means, n, mass, lengths, cuda_backend)
    assert grids == [(9, rows)]
    assert torch.equal(actual, expected)
    assert actual[0].item() == 0  # exact plateau keeps its first maximum
    if rows > 1:
        assert actual[1].item() == width


def test_score_reduction_preserves_int64_indices(cuda_backend):
    # Original indices can exceed int32 even though this small reduction
    # fixture needs almost no storage. Equal maxima select the smaller index.
    scores = torch.tensor([[.1, .9, .9, .2]], device="cuda")
    indices = torch.tensor([[0, 2**31 + 9, 2**31 + 3, 2**31 + 11]], device="cuda")
    values_out = torch.empty((1, 2), device="cuda")
    indices_out = torch.empty((1, 2), device="cuda", dtype=torch.int64)
    cuda_backend._reduce_score_partials[(2, 1)](scores, indices, values_out, indices_out, 4, 2, 2)
    result = indices.new_empty(1)
    cuda_backend._score_final[(1,)](values_out, indices_out, result, 2, 2)
    assert result.item() == 2**31 + 3


@pytest.mark.skipif(os.environ.get("RANKSEG_TEST_LARGE_CUDA") != "1",
                    reason="opt-in 1 GiB score-grid regression")
def test_score_grid_beyond_cuda_y_limit(cuda_backend):
    width = cuda_backend._MAX_GRID_Y * cuda_backend._BLOCK
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < 2 * 1024**3:
        pytest.skip("requires at least 2 GiB free CUDA memory")
    prefix = torch.zeros(width, device="cuda")
    zero = prefix.new_zeros(())
    count = torch.zeros((), device="cuda", dtype=torch.int64)
    # The old (rows, parts) launch required grid.y=65536 and failed. This
    # also exercises the real multi-stage reduction, not just a small mock.
    assert cuda_backend.score_argmax(prefix, zero, count, zero, None).item() == 0
    prefix[-1] = 1
    assert cuda_backend.score_argmax(prefix, zero, count, zero, None).item() == width


def test_fused_screening_on_nondefault_stream(cuda_backend, force_screening):
    generator = torch.Generator().manual_seed(211)
    source = torch.rand(2, 3, 8193, generator=generator).pow(12)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        probs = source.cuda()
        result = algo.rankseg_rma(probs, output_mode="multilabel", safe_screening=True)
    stream.synchronize()
    objective_diagnostics(probs, result)


def test_statistics_respects_cuda_grid_limit(cuda_backend, monkeypatch):
    # Exercise the large-row tiling branch with a small allocation.
    monkeypatch.setattr(cuda_backend, "_MAX_GRID_Y", 2)
    rows = torch.linspace(0, 1, 8193, device="cuda").repeat(2, 1)
    guard = 32 * torch.finfo(rows.dtype).eps
    masks, candidates, count, mass = cuda_backend.screening_candidates(
        rows, rows.sum(-1), rows.amax(-1), guard,
    )
    assert torch.equal(masks, rows > 0.5 + guard)
    assert torch.equal(count, masks.sum(-1))
    torch.testing.assert_close(mass, (rows * masks).sum(-1))
    assert not (masks & candidates).any()
