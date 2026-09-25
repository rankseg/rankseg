"""Cached unique class identities must not change statistics or assignment."""

import pytest
import torch

from rankseg import _screening as screening

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.fixture
def backend():
    module = screening._cuda_backend()
    if module is None:
        pytest.skip("Triton unavailable")
    return module


def check_cached(backend, probs, masks, pruning=.5, void_index=-7):
    """Exact pre-cache statistics plus the independent dense scoring oracle."""
    from test_screening_unique_fastpath import check_assignment

    original_probs, original_masks = probs.clone(), masks.clone()
    masks = masks.contiguous()
    status, n, h = backend.unique_statistics(masks, probs)
    cached, cached_n, cached_h = backend.unique_statistics(masks, probs, cache_unique=True)
    counts = masks.sum(1)
    assert cached.dtype == torch.uint8 and cached.numel() == status.numel()
    if probs.shape[1] <= 254:
        expected = torch.where(counts == 0, 0,
                               torch.where(counts == 1, masks.long().argmax(1) + 2, 1)).byte()
    else:
        expected = counts.clamp_max(2).byte()
    assert torch.equal(cached, expected)
    assert torch.equal(n, cached_n)
    assert torch.equal(h, cached_h)  # No floating tolerance for this optimization.
    means, active = probs.sum(-1), probs.amax(-1) > pruning
    for policy in ("max_score", "void"):
        expected = check_assignment(backend, probs, masks, pruning=pruning,
                                    policy=policy, void_index=void_index)
        actual = backend.dice_nonoverlap(
            masks, probs, cached, means, cached_n, cached_h, active,
            policy == "void", void_index, cache_unique=True,
        )
        assert torch.equal(actual, expected)
        dispatched = backend.dice_nonoverlap_from_masks(
            original_masks, probs, means, active, policy == "void", void_index,
        )
        assert torch.equal(dispatched, expected)
    assert torch.equal(probs, original_probs)
    assert torch.equal(masks, original_masks)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("channels", [1, 2, 3, 8, 9, 21, 64, 65, 128, 129, 150, 254, 255, 256])
@pytest.mark.parametrize("scenario", ["unique", "mixed", "overlapping", "unassigned"])
def test_exact_cached_statistics_and_labels(backend, dtype, channels, scenario):
    generator = torch.Generator().manual_seed(479)
    probs = torch.rand(2, channels, 513, dtype=dtype, generator=generator).cuda()
    labels = torch.arange(513, device="cuda")[None].expand(2, -1) % channels
    masks = torch.zeros_like(probs, dtype=torch.bool).scatter_(1, labels[:, None], True)
    if scenario == "mixed":
        masks[0, :, [0, 31, 32, 255, 256, 512]] = False
        masks[1, :, [1, 30, 33, 254, 257, 511]] = True
    elif scenario == "overlapping":
        masks[:] = True  # Includes all 254 selected: no packed count carry.
    elif scenario == "unassigned":
        masks[:] = False
    check_cached(backend, probs, masks)


@pytest.mark.parametrize("channels", [3, 21, 254, 255, 256])
@pytest.mark.parametrize("dim", [1, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025])
def test_cached_tail_last_class_and_class_zero(backend, channels, dim):
    probs = torch.zeros(2, channels, dim, device="cuda")
    probs[:, 0] = 1  # Class identity is NOT the raw probability argmax.
    masks = torch.zeros_like(probs, dtype=torch.bool)
    masks[0, -1] = True
    masks[1, 0] = True
    check_cached(backend, probs, masks)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["spatial", "channel", "batch", "mask", "expanded"])
def test_cached_strides(backend, dtype, layout):
    generator = torch.Generator().manual_seed(426)
    probs = torch.rand(3, 9, 257, generator=generator, dtype=dtype).cuda()
    masks = torch.zeros_like(probs, dtype=torch.bool)
    masks[0, 8] = True
    masks[1, :2] = True
    if layout == "mask":
        masks = masks.repeat_interleave(2, -1)[..., ::2]
    elif layout == "expanded":
        probs = probs[:1].expand(3, -1, -1)
    else:
        axis = {"batch": 0, "channel": 1, "spatial": 2}[layout]
        slices = [slice(None)] * 3
        slices[axis] = slice(None, None, 2)
        probs = probs.repeat_interleave(2, axis)[tuple(slices)]
    check_cached(backend, probs, masks)


@pytest.mark.parametrize("pruning", [0., .5, 1.])
@pytest.mark.parametrize("void_index", [-(2**63), 2**63 - 1])
def test_cached_pruning_and_void(backend, pruning, void_index):
    probs = torch.full((3, 9, 257), .25, device="cuda")
    probs[0, 8] = .75
    probs[1] = 0
    masks = torch.zeros_like(probs, dtype=torch.bool)
    masks[0, 8, :128] = True
    masks[2, 0, :32] = True
    masks[2, :2, 32:64] = True
    masks &= (probs.amax(-1) > pruning)[:, :, None]
    check_cached(backend, probs, masks, pruning, void_index)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all_three_class_three_pixel_masks_cached(backend, dtype):
    bits = torch.arange(9, device="cuda")
    masks = ((torch.arange(512, device="cuda")[:, None] >> bits) & 1).bool().reshape(512, 3, 3)
    probs = torch.rand(512, 3, 3, device="cuda", dtype=dtype)
    check_cached(backend, probs, masks)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cached_exact_and_adjacent_score_ties(backend, dtype):
    probs = torch.full((1, 3, 513), .5, dtype=dtype, device="cuda")
    masks = torch.zeros_like(probs, dtype=torch.bool)
    masks[:, 0, :64] = True
    masks[:, 1, 64:128] = True
    masks[:, 2, 128:192] = True
    masks[:, :, 192:384] = True
    check_cached(backend, probs, masks, pruning=0.)
    probs[:, 2, 200] = torch.nextafter(probs[:, 2, 200], probs.new_tensor(1.))
    probs[:, 1, 400] = torch.nextafter(probs[:, 1, 400], probs.new_tensor(0.))
    check_cached(backend, probs, masks, pruning=0.)


@pytest.mark.parametrize("channels", [3, 254, 255, 256])
def test_cached_hierarchical_statistics(backend, monkeypatch, channels):
    monkeypatch.setattr(backend, "_BLOCK", 2)
    probs = torch.full((2, channels, 16385), .125, device="cuda")
    masks = torch.zeros_like(probs, dtype=torch.bool)
    masks[0, -1] = True
    masks[1, :2] = True
    old = backend.unique_statistics(masks, probs)
    cached, n, h = backend.unique_statistics(masks, probs, cache_unique=True)
    assert torch.equal(n, old[1]) and torch.equal(h, old[2])
    assert n[0, -1] == 16385 and (n[1] == 0).all()
    assert (cached[0] == (channels + 1 if channels <= 254 else 1)).all()
    assert (cached[1] == (1 if channels <= 254 else 2)).all()


def test_cached_statistics_above_float32_integer_precision(backend):
    dim = 2**24 + 33
    probs = torch.full((1, 2, dim), .125, device="cuda")
    masks = torch.zeros_like(probs, dtype=torch.bool)
    masks[:, 1] = True
    old = backend.unique_statistics(masks, probs)
    cached, n, h = backend.unique_statistics(masks, probs, cache_unique=True)
    assert (cached == 3).all()
    assert n.tolist() == [[0, dim]]
    assert torch.equal(n, old[1]) and torch.equal(h, old[2])


def test_cached_nondefault_stream(backend):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        probs = torch.full((2, 3, 2051), .125, device="cuda")
        masks = torch.zeros_like(probs, dtype=torch.bool)
        masks[0, 2] = True
        masks[1, 0] = True
        status, n, h = backend.unique_statistics(masks, probs, cache_unique=True)
        actual = backend.dice_nonoverlap(masks, probs, status, probs.sum(-1), n, h,
                                         probs.amax(-1) > .5, cache_unique=True)
    stream.synchronize()
    assert (actual[0] == 2).all() and (actual[1] == 0).all()


@pytest.mark.parametrize("channels", [3, 254])
def test_all_unique_cached_assignment_uses_cached_identity(backend, channels):
    # Freeze stats/cached identity, then supply an all-zero mask to distinguish
    # the cached shortcut from re-deriving a winner from the mask. This is a
    # private path guard, not permission to mutate masks in the public decoder.
    probs = torch.full((1, channels, 513), .125, device="cuda")
    masks = torch.zeros_like(probs, dtype=torch.bool)
    masks[:, -1] = True
    status, n, h = backend.unique_statistics(masks, probs, cache_unique=True)
    actual = backend.dice_nonoverlap(
        torch.zeros_like(masks), probs, status,
        probs.sum(-1), n, h, probs.amax(-1) > .5, cache_unique=True,
    )
    assert (actual == channels - 1).all()
