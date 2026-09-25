"""Memory-bounded multiclass statistics, independent of binary screening."""

import pytest
import torch

from rankseg import _rankseg_algo as algo
from rankseg import _screening as screening

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.fixture
def backend():
    backend = screening._cuda_backend()
    if backend is None:
        pytest.skip("Triton unavailable")
    return backend


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("channels", [2, 3, 8, 9, 21, 150, 256])
@pytest.mark.parametrize("layout", ["contiguous", "channel_strided", "spatial_strided"])
def test_unique_statistics_independent_reference(backend, dtype, channels, layout):
    generator = torch.Generator().manual_seed(634)
    probs = torch.rand(2, channels, 2051, dtype=dtype, generator=generator).cuda()
    if layout == "channel_strided":
        probs = probs.transpose(1, 2).contiguous().transpose(1, 2)
    elif layout == "spatial_strided":
        probs = probs.repeat_interleave(2, -1)[..., ::2]
    masks = torch.rand(2, channels, 2051, generator=generator).cuda() > .7
    masks[..., :23] = False
    masks[:, 0, 23:100] = True
    masks[:, 1:, 23:100] = False
    # With C=256 all selected must be 2+ rather than wrapping to zero.
    masks[..., 100:200] = True
    masks[1] = False
    before = probs.clone()
    mask_before = masks.clone()
    counts = masks.sum(1)
    unique = masks & (counts == 1)[:, None]
    expected_n = unique.sum(-1)
    expected_h = (probs * unique).sum(-1)

    status, n, h = backend.unique_statistics(masks, probs)

    assert status.dtype == torch.uint8
    assert n.dtype == torch.int64
    assert h.dtype == dtype
    assert torch.equal(status, counts.clamp(max=2).to(torch.uint8))
    assert torch.equal(n, expected_n)
    torch.testing.assert_close(h, expected_h, rtol=4 * torch.finfo(dtype).eps, atol=0)
    assert torch.equal(probs, before)
    assert torch.equal(masks, mask_before)


@pytest.mark.parametrize("dim", [1, 63, 64, 65, 1023, 1024, 1025, 16385])
@pytest.mark.parametrize("channels", [3, 19])
def test_unique_tile_tails_and_hierarchical_reduction(backend, monkeypatch, dim, channels):
    # Force multiple reduction levels using small, exactly summable inputs.
    monkeypatch.setattr(backend, "_BLOCK", 2)
    probs = torch.full((2, channels, dim), .125, device="cuda")
    masks = torch.zeros_like(probs, dtype=torch.bool)
    masks[:, 1] = True
    status, n, h = backend.unique_statistics(masks, probs)
    assert (status == 1).all()
    expected = torch.zeros((2, channels), dtype=torch.int64, device="cuda")
    expected[:, 1] = dim
    assert torch.equal(n, expected)
    assert torch.equal(h, expected * .125)


@pytest.mark.parametrize("policy", ["max_score", "void"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_statistics_assignment_matches_exact_fixture(backend, policy, dtype):
    from test_screening import _multiclass_reference
    # Binary fractions make both reduction orders exact. Cover unique,
    # overlapping and unassigned pixels, partly pruned and all-pruned samples.
    probs = torch.tensor([
        [[.875, .5, .125, .25, 0], [.5, .875, .5, .25, 0], [.125, .25, .125, .25, 0]],
        [[.25, .125, .375, 0, 0], [.125, .25, .25, .125, 0], [0, .375, .125, 0, 0]],
    ], device="cuda", dtype=dtype)
    masks = torch.tensor([
        [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0]] * 3,
    ], device="cuda", dtype=torch.bool)
    # The public dispatcher must also normalize non-contiguous mask storage.
    masks = masks.transpose(1, 2).contiguous().transpose(1, 2)
    means, active = probs.sum(-1), probs.amax(-1) > .5
    actual = backend.dice_nonoverlap_from_masks(masks, probs, means, active, policy == "void", -9)
    expected = _multiclass_reference(probs, masks, .5, policy, -9, means)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("void_index", [-(2**63), 2**63 - 1])
def test_new_statistics_path_preserves_int64_void(backend, void_index):
    probs = torch.zeros(2, 3, 13, device="cuda")
    actual = backend.dice_nonoverlap_from_masks(
        probs.bool(), probs, probs.sum(-1), probs.amax(-1) > .5, True, void_index,
    )
    assert actual.dtype == torch.int64
    assert (actual == void_index).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("channels", [3, 19])
@pytest.mark.parametrize("policy", ["max_score", "void"])
def test_assignment_eligibility_and_float64_increment_oracle(backend, dtype, channels, policy):
    generator = torch.Generator(device="cuda").manual_seed(531)
    probs = torch.rand((3, channels, 8193), device="cuda", dtype=dtype, generator=generator)
    masks = torch.zeros_like(probs, dtype=torch.bool)
    labels = torch.randint(channels, (3, 1, 8193), device="cuda", generator=generator)
    masks.scatter_(1, labels, True)
    masks[..., ::3] = True  # overlapping
    masks[..., ::7] = False  # unassigned
    probs[:, -1] *= .25
    masks[:, -1] = False  # partly pruned
    probs[2] *= .25
    masks[2] = False  # all-pruned
    means = probs.sum(-1)
    active = probs.amax(-1) > .5
    actual = backend.dice_nonoverlap_from_masks(masks, probs, means, active, policy == "void", -7)

    # Independent dense float64 oracle; rounding may change only near-tied
    # incremental-score winners, never eligibility, unique labels or voids.
    counts = masks.sum(1)
    unique = masks & (counts == 1)[:, None]
    n = unique.sum(-1)
    h = (probs.double() * unique).sum(-1)
    status, new_n, new_h = backend.unique_statistics(masks, probs)
    assert torch.equal(status, counts.clamp(max=2).byte())
    assert torch.equal(new_n, n)
    torch.testing.assert_close(new_h.double(), h, rtol=4 * torch.finfo(dtype).eps, atol=0)
    eligible_active = active | ~active.any(1, keepdim=True) if policy == "max_score" else active
    eligible = torch.where((counts > 0)[:, None], masks, eligible_active[..., None])
    if policy == "void":
        assert torch.equal(actual == -7, counts == 0)
    valid = counts > 0 if policy == "void" else torch.ones_like(counts, dtype=torch.bool)
    index = actual.clamp(min=0)[:, None]
    assert eligible.gather(1, index)[:, 0][valid].all()
    assert torch.equal(actual[counts == 1], masks.long().argmax(1)[counts == 1])
    denom = (n.double() + means.double() + 1)[..., None]
    scores = 2 * ((h[..., None] + probs.double()) / (denom + 1) - h[..., None] / denom)
    scores.masked_fill_(~eligible, -torch.inf)
    regret = scores.amax(1) - scores.gather(1, index)[:, 0]
    assert (regret[valid] <= 16 * torch.finfo(dtype).eps).all()


def test_public_fused_path_skips_portable_unique_statistics(backend, monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail("fused path must not allocate portable unique statistics")
    monkeypatch.setattr(algo, "_count_selected_pixels", reject)
    probs = torch.full((1, 3, 65537), .875, device="cuda")
    actual = algo.rankseg_rma(probs, safe_screening=True)
    assert (actual == 0).all()


def test_unsupported_shapes_return_before_statistics(backend, monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail("unsupported shapes must fall back before new statistics")
    monkeypatch.setattr(backend, "unique_statistics", reject)
    for shape in ((1, 257, 1), (65536, 2, 1)):
        probs = torch.zeros(shape, device="cuda")
        assert backend.dice_nonoverlap_from_masks(
            probs.bool(), probs, probs.sum(-1), probs.amax(-1) > .5,
        ) is None


def test_empty_batch(backend):
    probs = torch.empty(0, 3, 17, device="cuda")
    actual = backend.dice_nonoverlap_from_masks(probs.bool(), probs, probs.sum(-1), probs.amax(-1) > .5)
    assert actual.shape == (0, 17)
    assert actual.dtype == torch.int64


def test_multiclass_incremental_peak_memory(backend):
    dim = 1048576
    probs = torch.full((1, 3, dim), .875, device="cuda")
    probs[:, 1] = .125
    warmup = algo.rankseg_rma(probs, safe_screening=True)
    del warmup
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    actual = algo.rankseg_rma(probs, safe_screening=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - resident
    # Old dense int64 reduction peaked at 35*D bytes. New status+output+mask
    # peaks near 12*D; allow allocator rounding and small reduction workspaces.
    assert peak < 16 * dim
    assert (actual == 0).all()


def test_unique_counts_above_float32_integer_precision(backend):
    # The count must not round early when the final score works in float32.
    dim = 2**24 + 33
    probs = torch.full((1, 2, dim), .125, device="cuda")
    masks = torch.zeros_like(probs, dtype=torch.bool)
    masks[:, 1] = True
    status, n, h = backend.unique_statistics(masks, probs)
    assert (status == 1).all()
    assert n.tolist() == [[0, dim]]
    expected = torch.tensor([[0, dim / 8]], dtype=torch.float32, device="cuda")
    assert torch.equal(h, expected)
