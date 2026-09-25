"""Row compaction must also work for very wide CUDA probability tensors."""

import os
from contextlib import contextmanager

import pytest
import torch

from rankseg import _rankseg_algo as algo
from rankseg import _screening as screening


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]


@contextmanager
def _reject_advanced_row_copies(monkeypatch):
    """Catch the failing operation with small tensors, including on CPU CI."""
    original_get = torch.Tensor.__getitem__
    original_set = torch.Tensor.__setitem__

    def check(tensor, key):
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]
        if tensor.ndim == 2 and isinstance(key, torch.Tensor) and key.ndim == 1:
            pytest.fail("wide row copies must use index_select/index_copy_, not advanced indexing")

    def get(tensor, key):
        check(tensor, key)
        return original_get(tensor, key)

    def put(tensor, key, value):
        check(tensor, key)
        return original_set(tensor, key, value)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__getitem__", get)
        patch.setattr(torch.Tensor, "__setitem__", put)
        yield


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("active_rows", [(4,), (0, 3, 5), tuple(range(6))])
@pytest.mark.parametrize("strided", [False, True])
def test_active_row_compaction_preserves_masks(device, dtype, active_rows, strided, monkeypatch):
    storage = torch.zeros((2, 6, 64) if strided else (2, 3, 32), device=device, dtype=dtype)
    probs = storage[:, ::2, ::2] if strided else storage
    for row in active_rows:
        probs[row // 3, row % 3, :3] = probs.new_tensor([0.9, 0.4, 0.2])
    before = probs.clone()
    expected = algo.rankseg_rma(probs, output_mode="multilabel")
    maxima = probs.amax(-1)
    with _reject_advanced_row_copies(monkeypatch):
        actual = screening._rma_dice_screened_masks(
            probs, probs.sum(-1), maxima > 0.5, maxima=maxima,
        )
    assert torch.equal(actual, expected)
    assert torch.equal(probs, before)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["multilabel", "multiclass"])
@pytest.mark.parametrize("strided", [False, True])
def test_separate_candidate_groups_use_safe_row_copies(device, dtype, mode, strided, monkeypatch):
    storage = torch.zeros((2, 6, 64) if strided else (2, 3, 32), device=device, dtype=dtype)
    probs = storage[:, ::2, ::2] if strided else storage
    probs[0, 0, :3] = probs.new_tensor([0.9, 0.4, 0.35])
    probs[1, 1, :2] = probs.new_tensor([0.9, 0.4])
    before = probs.clone()
    expected = algo.rankseg_rma(probs, output_mode=mode)
    monkeypatch.setattr(screening, "_MAX_PADDED_ELEMENTS", 1)
    with _reject_advanced_row_copies(monkeypatch):
        actual = algo.rankseg_rma(probs, output_mode=mode, safe_screening=True)
    assert torch.equal(actual, expected)
    assert torch.equal(probs, before)


@pytest.mark.skipif(os.environ.get("RANKSEG_TEST_LARGE_CUDA") != "1",
                    reason="opt-in large-memory CUDA regression")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("active_rows", [(0,), (0, 2)])
def test_liver_sized_cuda_compaction(active_rows):
    # liver_115: D=846*512*512. PyTorch 2.8 advanced row reads fail here
    # with 'invalid configuration argument' even when enough memory is free.
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    if free < 10 * 1024**3:
        pytest.skip("requires at least 10 GiB free CUDA memory")
    probs = torch.empty((1, 3, 846 * 512 * 512), device="cuda")
    for row in range(3):
        probs[0, row].fill_(0.875 if row in active_rows else 0.125)
    maxima = probs.amax(-1)
    masks = screening._rma_dice_screened_masks(
        probs, probs.sum(-1), maxima > 0.5, maxima=maxima,
    )
    for row in range(3):
        # For a positive constant row the objective increases with volume;
        # pruned rows are empty. This oracle does not require sorting D pixels.
        assert (masks[0, row] == (row in active_rows)).all()
        assert (probs[0, row] == (0.875 if row in active_rows else 0.125)).all()
    torch.cuda.synchronize()


@pytest.mark.skipif(os.environ.get("RANKSEG_TEST_LARGE_CUDA") != "1",
                    reason="opt-in large-memory CUDA regression")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("unique", [False, True])
def test_liver_sized_public_fused_assignment(unique):
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        pytest.skip("requires at least 12 GiB free CUDA memory")
    if screening._cuda_backend() is None:
        pytest.skip("Triton unavailable")
    # Both row indexing and the small-tile assignment launch must support
    # D > 65535 * 256. Cover both scored overlaps and direct unique output.
    probs = torch.full((1, 3, 846 * 512 * 512), .875, device="cuda")
    probs[:, 1] = .125
    if unique:
        probs[:, 0] = .125
    actual = algo.rankseg_rma(probs, safe_screening=True)
    assert (actual == (2 if unique else 0)).all()
    torch.cuda.synchronize()
