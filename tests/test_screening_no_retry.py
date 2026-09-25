"""Dense/long real candidate rows must be solved, never retried as full sort."""
import pytest
import torch

from rankseg import _rankseg_algo as algo
from rankseg import _screening as screening
from test_screening import _assert_epsilon_optimal, _multiclass_reference


@pytest.fixture(params=["cpu", "cuda_torch", "cuda_triton"])
def backend(request, monkeypatch):
    name = request.param
    if name != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if name == "cuda_triton" and screening._cuda_backend() is None:
        pytest.skip("Triton unavailable")
    if name == "cuda_torch":
        monkeypatch.setattr(screening, "_cuda_backend", lambda: None)
    return "cpu" if name == "cpu" else "cuda"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1, 1, 129), (2, 3, 4097)])
@pytest.mark.parametrize("strided", [False, True])
def test_dense_candidates_no_retry_and_correct_assignment(backend, dtype, shape, strided, monkeypatch):
    p = torch.full(shape, .49, device=backend, dtype=dtype)
    p[..., 0] = .9
    if shape[1] > 1:
        p[0, 1] = 0  # inactive
        p[1, 2] = 1  # fully resolved, mixed with nearly 100% undecided rows
    if strided:
        p = p.transpose(1, 2).contiguous().transpose(1, 2)
    before = p.clone()
    # Deliberately make the group budget much smaller than one real row.
    monkeypatch.setattr(screening, "_MAX_PADDED_ELEMENTS", 32)
    seen = []
    original = torch.Tensor.sort

    def tracked(values, *args, **kwargs):
        seen.append(tuple(values.shape))
        return original(values, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "sort", lambda *a, **kw: pytest.fail("candidate rows must not retry full sort"))
        patch.setattr(torch.Tensor, "sort", tracked)
        mask = algo.rankseg_rma(p, safe_screening=True, output_mode="multilabel")
        if shape[1] > 1:
            classes = algo.rankseg_rma(p, safe_screening=True, output_mode="multiclass")
    assert seen and all(size == (shape[-1] - 1,) for size in seen)
    if dtype == torch.float64:
        # For these constant positive blocks the objective strictly increases
        # with volume. Check the exact optimal mask: accumulating .49 thousands
        # of times in a float64 oracle itself drifts by >16 eps versus sum().
        assert torch.equal(mask, p > 0)
    else:
        _assert_epsilon_optimal(p, mask)
    assert torch.equal(p, before)
    if shape[1] > 1:
        means = screening._rma_dice_screening_statistics(p)
        reference = _multiclass_reference(p, mask, .5, "max_score", 255,
                                          None if means is None else means[0])
        assert torch.equal(classes, reference)


@pytest.mark.parametrize("fraction", [1., .5])
def test_candidates_exceed_old_absolute_limit(backend, fraction, monkeypatch):
    # M exceeds the removed 2**22 limit. One case also exceeds the removed
    # 75% fraction limit; the other isolates the absolute count condition.
    candidates = 4_194_304 + 16
    dim = int((candidates + 1) / fraction)
    p = torch.zeros((1, 1, dim), device=backend)
    p[..., :candidates + 1] = .49
    p[..., 0] = .9
    seen = []
    original = torch.Tensor.sort

    def tracked(values, *args, **kwargs):
        seen.append(tuple(values.shape))
        return original(values, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "sort", lambda *a, **kw: pytest.fail("long candidates must not retry full sort"))
        patch.setattr(torch.Tensor, "sort", tracked)
        mask = algo.rankseg_rma(p, safe_screening=True, output_mode="multilabel")
    assert seen == [(candidates,)]
    assert not mask[..., candidates + 1:].any()
    # Exact ordering is known: one .9, then identical .49s, then zeros.
    # All positive entries maximize the binary objective (strictly increasing
    # over the equal .49 block), so no large oracle sort is required.
    high, low = float(p[0, 0, 0]), float(p[0, 0, 1])
    mu = high + candidates * low
    optimum = 2 * mu / (mu + candidates + 2)
    mass = high * int(mask[0, 0, 0]) + low * int(mask[..., 1:candidates + 1].sum())
    score = 2 * mass / (mu + int(mask.sum()) + 1)
    assert optimum - score <= 4 * torch.finfo(torch.float32).eps


def test_group_budget_splits_rows_without_losing_long_candidates(monkeypatch):
    monkeypatch.setattr(screening, "_MAX_PADDED_ELEMENTS", 32)
    lengths = [0, 1, 2, 17, 33, 4_194_305, 5_000_000]
    groups = list(screening._candidate_groups(lengths))
    assert sorted(row for group in groups for row in group) == list(range(1, len(lengths)))
    for group in groups:
        assert len(group) == 1 or len(group) * max(lengths[r] for r in group) <= 32
    assert [5] in groups and [6] in groups
