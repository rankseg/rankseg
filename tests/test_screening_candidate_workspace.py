"""Fused candidate movement must preserve sort inputs and every mask write."""

import itertools
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


def candidate_indices(lengths, dim, device="cuda"):
    starts = [0]
    indices = []
    for row, length in enumerate(lengths):
        indices.append(row * dim + torch.arange(length, device=device) * dim // max(1, length))
        starts.append(starts[-1] + length)
    return torch.cat(indices), starts


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["contiguous", "channel_strided", "spatial_strided", "expanded", "offset"])
@pytest.mark.parametrize("dim", [1, 31, 4095, 4096, 4097, 8193])
@pytest.mark.parametrize("group", [[2], [4, 0, 2], [3, 1, 0, 4]])
def test_exact_group_gather_and_scatter(backend, dtype, layout, dim, group):
    rows = torch.rand(5, dim, generator=torch.Generator().manual_seed(286), dtype=dtype).cuda()
    if layout == "channel_strided":
        rows = rows.T.contiguous().T
    elif layout == "spatial_strided":
        rows = rows.repeat_interleave(2, -1)[:, ::2]
    elif layout == "expanded":
        rows = rows[:1].expand(5, -1)
    elif layout == "offset":
        rows = torch.cat((rows, rows), -1)[:, dim:]
    original = rows.clone()
    lengths = [dim, 0, max(1, dim // 2), 0, 1]
    packed, starts = candidate_indices(lengths, dim)
    group_lengths = [lengths[r] for r in group]
    width = max(group_lengths)
    # Non-contiguous statistics; large counts must not round through float32.
    means = (torch.arange(10, dtype=dtype, device="cuda") + .5)[::2]
    mass = torch.arange(10, dtype=dtype, device="cuda")[::2]
    counts = (torch.arange(10, device="cuda") + 2**33)[::2]
    values, mu, n, h, metadata = backend.gather_candidate_group(
        rows, packed, group, [starts[r] for r in group], group_lengths, means, counts, mass,
    )
    expected = rows.new_full((len(group), width), -1)
    for i, row in enumerate(group):
        local_indices = packed[starts[row]:starts[row + 1]] - row * dim
        expected[i, :lengths[row]] = rows[row, local_indices]
    if len(group) == 1:
        expected = expected[0]
    assert torch.equal(values, expected)
    assert torch.equal(mu, means[group]) and torch.equal(n, counts[group]) and torch.equal(h, mass[group])
    assert metadata.tolist() == [group, [starts[r] for r in group], group_lengths]
    _, order = values.sort(dim=-1, descending=True)
    opt = torch.tensor([length // 2 for length in group_lengths], device="cuda")
    masks = torch.rand(rows.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(311)) > .5
    reference = masks.clone()
    for i, row in enumerate(group):
        permutation = order if len(group) == 1 else order[i]
        selected = torch.empty(width, dtype=torch.bool, device="cuda")
        selected.scatter_(0, permutation, torch.arange(width, device="cuda") < opt[i])
        reference.reshape(-1)[packed[starts[row]:starts[row + 1]]] = selected[:lengths[row]]
    backend.scatter_candidate_group(masks, packed, metadata, order, opt)
    assert torch.equal(masks, reference)
    assert torch.equal(rows, original)


def test_all_permutations_and_selection_boundaries(backend):
    # Exhaust all 4! permutations and tau=0..4, with real and padding entries.
    permutations = list(itertools.permutations(range(4)))
    for length in (0, 1, 2, 4):
        n_rows, dim = len(permutations) * 5, 9
        lengths = [length] * n_rows
        packed, starts = candidate_indices(lengths, dim)
        metadata = torch.tensor([list(range(n_rows)), starts[:-1], lengths], device="cuda")
        order = torch.tensor(permutations * 5, device="cuda")
        opt = torch.arange(5, device="cuda").repeat_interleave(len(permutations))
        masks = torch.ones(n_rows, dim, dtype=torch.bool, device="cuda")
        expected = masks.clone()
        selected = torch.empty(n_rows, 4, dtype=torch.bool, device="cuda")
        selected.scatter_(1, order, torch.arange(4, device="cuda") < opt[:, None])
        expected.reshape(-1)[packed] = selected[:, :length].reshape(-1)
        backend.scatter_candidate_group(masks, packed, metadata, order, opt)
        assert torch.equal(masks, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n_rows", [2, 17, 128])
def test_group_limits_and_zero_one_all_selected(backend, dtype, n_rows):
    dim = 16385
    rows = torch.linspace(0, 1, dim, device="cuda", dtype=dtype).repeat(n_rows, 1)
    lengths = [4097 + r % 3 for r in range(n_rows)]
    packed, starts = candidate_indices(lengths, dim)
    stats = rows.new_zeros(n_rows)
    counts = torch.zeros(n_rows, device="cuda", dtype=torch.int64)
    group = list(reversed(range(n_rows)))
    values, _, _, _, metadata = backend.gather_candidate_group(
        rows, packed, group, [starts[r] for r in group], [lengths[r] for r in group], stats, counts, stats,
    )
    _, order = values.sort(descending=True)
    for volume in (0, 1, values.shape[-1]):
        masks = torch.zeros_like(rows, dtype=torch.bool)
        opt = torch.full((n_rows,), volume, device="cuda", dtype=torch.int64)
        backend.scatter_candidate_group(masks, packed, metadata, order, opt)
        for r in range(n_rows):
            assert int(masks[r].count_nonzero()) == min(volume, lengths[r])
            chosen = packed[starts[r]:starts[r + 1]] - r * dim
            if volume:
                assert masks[r, chosen[-min(volume, lengths[r]):]].all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_groups_match_portable_prefix_and_writeback_oracle(backend, dtype):
    dim = 8193
    rows = torch.rand(7, dim, dtype=dtype, generator=torch.Generator().manual_seed(394)).cuda()
    lengths = [1, 10, 12, 3, 4097, 0, 4095]
    packed, starts = candidate_indices(lengths, dim)
    means = rows.sum(-1)
    n = torch.tensor([0, 3, 1, 2, 50, 0, 49], device="cuda")
    h = rows.new_tensor([0, 2., .9, 1.8, 43., 0, 42.])
    masks = torch.zeros_like(rows, dtype=torch.bool)
    expected = masks.clone()
    for group in screening._candidate_groups(lengths):
        width = max(lengths[r] for r in group)
        reference = rows.new_full((len(group), width), -1)
        for i, row in enumerate(group):
            reference[i, :lengths[row]] = rows[row, packed[starts[row]:starts[row + 1]] - row * dim]
        if len(group) == 1:
            reference = reference[0]
        sorted_values, order = reference.sort(descending=True)
        repeats = torch.tensor([lengths[r] for r in group], device="cuda") if len(group) > 1 else None
        # Freeze the exact same scan result to isolate data movement from CUDA
        # scan repeatability. The score kernel and sort themselves are unchanged.
        prefix = sorted_values.cumsum(-1)
        opt = backend.score_argmax(prefix, means[group], n[group], h[group], repeats)
        values, mu, gn, gh, metadata = backend.gather_candidate_group(
            rows, packed, group, [starts[r] for r in group], [lengths[r] for r in group], means, n, h,
        )
        assert torch.equal(values, reference)
        actual_values, actual_order = values.sort(descending=True)
        assert torch.equal(actual_values, sorted_values) and torch.equal(actual_order, order)
        actual_opt = backend.score_argmax(prefix, mu, gn, gh, metadata[2] if len(group) > 1 else None)
        assert torch.equal(actual_opt, opt)
        backend.scatter_candidate_group(masks, packed, metadata, actual_order, actual_opt)
        for i, row in enumerate(group):
            permutation = order if len(group) == 1 else order[i]
            ell = opt if len(group) == 1 else opt[i]
            selection = torch.empty(width, dtype=torch.bool, device="cuda")
            selection.scatter_(0, permutation, torch.arange(width, device="cuda") < ell)
            expected.reshape(-1)[packed[starts[row]:starts[row + 1]]] = selection[:lengths[row]]
    assert torch.equal(masks, expected)


def test_public_path_avoids_owner_arrays_and_dense_probability_copy(backend, monkeypatch):
    dim = 65537
    probs = torch.rand(3, dim, generator=torch.Generator().manual_seed(134)).pow(12).cuda().T.contiguous().T[None]
    expected = algo.rankseg_rma(probs, safe_screening=True)
    original = torch.Tensor.reshape
    def checked(tensor, *shape):
        if tensor.shape == (3, dim) and shape == (-1,):
            pytest.fail("CUDA candidate grouping must not flatten/copy full probabilities")
        return original(tensor, *shape)
    def reject(*args, **kwargs):
        pytest.fail("CUDA candidate grouping must not create owner/inverse-permutation arrays")
    monkeypatch.setattr(torch.Tensor, "reshape", checked)
    monkeypatch.setattr(torch, "repeat_interleave", reject)
    monkeypatch.setattr(torch, "cat", reject)
    monkeypatch.setattr(torch.Tensor, "scatter_", reject)
    assert torch.equal(expected, algo.rankseg_rma(probs, safe_screening=True))


def test_nondefault_stream(backend):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        rows = torch.arange(100, device="cuda", dtype=torch.float32).reshape(2, 50) / 100
        packed, starts = candidate_indices([3, 5], 50)
        stats = rows.sum(-1)
        values, _, _, _, metadata = backend.gather_candidate_group(
            rows, packed, [1, 0], [starts[1], starts[0]], [5, 3], stats,
            torch.zeros(2, dtype=torch.int64, device="cuda"), stats,
        )
        _, order = values.sort(descending=True)
        masks = torch.zeros_like(rows, dtype=torch.bool)
        backend.scatter_candidate_group(masks, packed, metadata, order, torch.tensor([2, 1], device="cuda"))
        assert int(masks.count_nonzero()) == 3
    stream.synchronize()


@pytest.mark.skipif(os.environ.get("RANKSEG_TEST_LARGE_CUDA") != "1", reason="opt-in large CUDA test")
def test_multigigabyte_probability_offsets_and_last_pixel(backend):
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < 8 * 1024**3:
        pytest.skip("requires 8 GiB free CUDA memory")
    dim = 846 * 512 * 512
    rows = torch.zeros(3, dim, device="cuda")
    rows[0, 0], rows[0, -1] = .1, .2
    rows[2, 0], rows[2, -1] = .3, .4
    packed = torch.tensor([0, dim - 1, 2 * dim, 3 * dim - 1], device="cuda")
    stats = rows.new_zeros(3)
    values, _, _, _, metadata = backend.gather_candidate_group(
        rows, packed, [2, 0], [2, 0], [2, 2], stats,
        torch.zeros(3, device="cuda", dtype=torch.int64), stats,
    )
    assert torch.equal(values, rows.new_tensor([[.3, .4], [.1, .2]]))
    _, order = values.sort(descending=True)
    masks = torch.zeros_like(rows, dtype=torch.bool)
    backend.scatter_candidate_group(masks, packed, metadata, order, torch.tensor([1, 2], device="cuda"))
    assert masks[0, 0] and masks[0, -1] and masks[2, -1]
    assert int(masks.count_nonzero()) == 3


@pytest.mark.skipif(os.environ.get("RANKSEG_TEST_LARGE_CUDA") != "1", reason="opt-in large CUDA test")
def test_packed_indices_above_signed_int32_limit(backend):
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < 14 * 1024**3:
        pytest.skip("requires 14 GiB free CUDA memory")
    # Cross 2**31 in element indices, not just byte addresses. Only candidate
    # probabilities are initialized: the gather must never read other entries.
    dim = 16777216
    rows = torch.empty(129, dim, device="cuda")
    rows[0, 0], rows[0, -1] = .1, .2
    rows[-1, 0], rows[-1, -1] = .3, .4
    packed = torch.tensor([0, dim - 1, 128 * dim, 129 * dim - 1], device="cuda")
    stats = rows.new_zeros(129)
    values, _, _, _, metadata = backend.gather_candidate_group(
        rows, packed, [128, 0], [2, 0], [2, 2], stats,
        torch.zeros(129, device="cuda", dtype=torch.int64), stats,
    )
    assert torch.equal(values, rows.new_tensor([[.3, .4], [.1, .2]]))
    _, order = values.sort(descending=True)
    masks = torch.zeros_like(rows, dtype=torch.bool)
    backend.scatter_candidate_group(masks, packed, metadata, order, torch.tensor([1, 2], device="cuda"))
    assert masks[0, 0] and masks[0, -1] and masks[-1, -1]
    # Boolean reductions avoid a full-size int64 mask copy in this large test.
    flat = masks.view(-1)
    assert not flat[1:dim - 1].any()
    assert not flat[dim:129 * dim - 1].any()
