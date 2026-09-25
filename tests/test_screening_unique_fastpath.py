"""Unique assignment is an exact eligibility shortcut, not an argmax heuristic."""

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


def check_assignment(backend, probs, masks, *, policy="max_score", pruning=.5,
                     void_index=-7, saturated=True):
    """Dense pre-shortcut formula, using identical supplied reduction results."""
    original_probs, original_masks = probs.clone(), masks.clone()
    counts = masks.sum(1)
    status, n, h = backend.unique_statistics(masks.contiguous(), probs)
    assert torch.equal(status, counts.clamp_max(2).to(torch.uint8))
    means = probs.sum(-1)
    active = probs.amax(-1) > pruning
    effective_active = active | ~active.any(1, keepdim=True) if policy == "max_score" else active
    eligible = torch.where((counts > 0)[:, None], masks, effective_active[:, :, None])
    denom = (n.to(probs.dtype) + means + 1)[:, :, None]
    mass = h[:, :, None]
    scores = 2 * ((mass + probs) / (denom + 1) - mass / denom)
    expected = scores.masked_fill(~eligible, -torch.inf).argmax(1)
    if policy == "void":
        expected[counts == 0] = void_index
    actual = backend.dice_nonoverlap(
        masks, probs, status if saturated else counts, means, n, h, active,
        policy == "void", void_index,
    )
    assert actual.dtype == torch.int64
    assert torch.equal(actual, expected)
    unique = counts == 1
    assert torch.equal(actual[unique], masks.long().argmax(1)[unique])
    assert torch.equal(probs, original_probs)
    assert torch.equal(masks, original_masks)
    return actual


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("channels", [1, 2, 3, 8, 9, 21, 64, 65, 128, 129, 150, 255, 256])
@pytest.mark.parametrize("scenario", ["unique", "mixed", "overlapping", "unassigned"])
def test_exact_all_assignment_paths(backend, dtype, channels, scenario):
    generator = torch.Generator().manual_seed(281)
    probs = torch.rand(2, channels, 513, generator=generator, dtype=dtype).cuda()
    labels = torch.arange(513, device="cuda")[None].expand(2, -1) % channels
    masks = torch.zeros_like(probs, dtype=torch.bool).scatter_(1, labels[:, None], True)
    if scenario == "mixed":
        # Tile boundaries and both samples; remaining tiles retain only uniques.
        masks[0, :, [0, 31, 32, 255, 256, 512]] = False
        masks[1, :, [1, 30, 33, 254, 257, 511]] = True
    elif scenario == "overlapping":
        masks[:] = True
    elif scenario == "unassigned":
        masks[:] = False
    for policy in ("max_score", "void"):
        check_assignment(backend, probs, masks, policy=policy)


@pytest.mark.parametrize("channels", [3, 9, 256])
@pytest.mark.parametrize("dim", [1, 31, 32, 33, 63, 64, 65, 255, 256, 257, 511, 512, 513])
def test_all_unique_tail_and_class_zero(backend, channels, dim):
    probs = torch.zeros(2, channels, dim, device="cuda")
    probs[:, 0] = 1  # Raw probability argmax must NOT override the binary mask.
    masks = torch.zeros_like(probs, dtype=torch.bool)
    masks[0, -1] = True
    masks[1, 0] = True
    result = check_assignment(backend, probs, masks, saturated=False)
    assert (result[0] == channels - 1).all()
    assert (result[1] == 0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["spatial", "channel", "batch", "mask"])
def test_strided_inputs_mixed_samples(backend, dtype, layout):
    generator = torch.Generator().manual_seed(956)
    probs = torch.rand(3, 9, 257, generator=generator, dtype=dtype).cuda()
    masks = torch.zeros_like(probs, dtype=torch.bool)
    masks[0, 8] = True
    masks[1, :2] = True
    if layout == "mask":
        masks = masks.repeat_interleave(2, -1)[..., ::2]
    else:
        axis = {"batch": 0, "channel": 1, "spatial": 2}[layout]
        slices = [slice(None)] * 3
        slices[axis] = slice(None, None, 2)
        probs = probs.repeat_interleave(2, axis)[tuple(slices)]
    check_assignment(backend, probs, masks)
    check_assignment(backend, probs, masks, policy="void")


@pytest.mark.parametrize("policy", ["max_score", "void"])
@pytest.mark.parametrize("pruning", [0., .5, 1.])
@pytest.mark.parametrize("void_index", [-(2**63), 2**63 - 1])
def test_pruning_with_mixed_tile_and_int64_void(backend, policy, pruning, void_index):
    probs = torch.full((3, 9, 257), .25, device="cuda")
    probs[0, 8] = .75
    probs[1] = 0  # All classes pruned, including at pruning=0.
    masks = torch.zeros_like(probs, dtype=torch.bool)
    masks[0, 8, :128] = True
    masks[2, 0, :32] = True
    masks[2, :2, 32:64] = True
    masks &= (probs.amax(-1) > pruning)[:, :, None]
    check_assignment(backend, probs, masks, pruning=pruning, policy=policy, void_index=void_index)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_overlap_ties_and_adjacent_float_scores(backend, dtype):
    # Equal per-class unique statistics; ambiguous ties favor the first class.
    probs = torch.full((1, 3, 513), .5, dtype=dtype, device="cuda")
    masks = torch.zeros_like(probs, dtype=torch.bool)
    masks[:, 0, :64] = True
    masks[:, 1, 64:128] = True
    masks[:, 2, 128:192] = True
    masks[:, :, 192:384] = True
    result = check_assignment(backend, probs, masks, pruning=0.)
    assert (result[:, 192:] == 0).all()
    probs[:, 2, 200] = torch.nextafter(probs[:, 2, 200], probs.new_tensor(1.))
    probs[:, 1, 400] = torch.nextafter(probs[:, 1, 400], probs.new_tensor(0.))
    check_assignment(backend, probs, masks, pruning=0.)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exhaustive_three_class_three_pixel_masks(backend, dtype):
    # Every possible mask: 512 samples spanning all class-selection patterns.
    bits = torch.arange(9, device="cuda")
    masks = ((torch.arange(512, device="cuda")[:, None] >> bits) & 1).bool().reshape(512, 3, 3)
    generator = torch.Generator().manual_seed(982)
    probs = torch.rand(512, 3, 3, dtype=dtype, generator=generator).cuda()
    for policy in ("max_score", "void"):
        check_assignment(backend, probs, masks, policy=policy)
