"""Fused input checks must preserve validation and screening statistics exactly."""

import os

import pytest
import torch

from rankseg import _rankseg_algo as algo
from rankseg import _screening as screening
from rankseg._validation import validate_probability_tensor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.fixture
def backend():
    module = screening._cuda_backend()
    if module is None:
        pytest.skip("Triton unavailable")
    return module


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["contiguous", "channel_strided", "spatial_strided", "offset"])
@pytest.mark.parametrize("dim", [1, 4095, 4096, 4097, 65537])
def test_validated_statistics_are_bitwise_identical(backend, dtype, layout, dim):
    rows = torch.rand(6, dim, dtype=dtype, generator=torch.Generator().manual_seed(752)).cuda()
    if layout == "channel_strided":
        rows = rows.T.contiguous().T
    elif layout == "spatial_strided":
        rows = rows.repeat_interleave(2, -1)[:, ::2]
    elif layout == "offset":
        rows = torch.cat((rows, rows), -1)[:, dim:]
    rows[0] = 0
    rows[1] = 1
    rows[2] *= .25
    original = rows.clone()
    guard = screening._ROUNDING_GUARD * torch.finfo(dtype).eps
    mean, maximum, state = backend.screening_statistics(rows, guard)
    global_maximum, (new_mean, new_maximum, new_state) = backend.validated_screening_statistics(rows, guard)
    assert global_maximum == 1.
    assert new_state[0] is None
    assert torch.equal(mean, new_mean)
    assert torch.equal(maximum, new_maximum)
    for before, after in zip(state[1:], new_state[1:]):
        assert torch.equal(before, after)
    active = maximum > .5
    before = backend.screening_candidates(rows, mean, maximum, guard, active, statistics=state, return_counts=True)
    after = backend.screening_candidates(rows, new_mean, new_maximum, guard, active,
                                         statistics=new_state, return_counts=True)
    for a, b in zip(before, after):
        assert torch.equal(a, b)
    assert torch.equal(original, rows)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["nan", "inf", "-inf", "below_zero", "above_one", "negative_subnormal"])
@pytest.mark.parametrize("position", [0, 4095, 4096, 8192])
@pytest.mark.parametrize("row", [0, 2])
def test_invalid_values_match_ordinary_validation(backend, dtype, kind, position, row):
    rows = torch.zeros(3, 8193, device="cuda", dtype=dtype)
    if kind == "below_zero":
        value = -torch.finfo(dtype).eps
    elif kind == "above_one":
        value = 1 + torch.finfo(dtype).eps
    elif kind == "negative_subnormal":
        value = torch.nextafter(rows.new_tensor(0.), rows.new_tensor(-1.))
    else:
        value = float(kind)
    rows[row, position] = value
    with pytest.raises(ValueError) as reference:
        validate_probability_tensor(rows[None])
    with pytest.raises(ValueError) as actual:
        backend.validated_screening_statistics(rows, 32 * torch.finfo(dtype).eps)
    assert str(actual.value) == str(reference.value)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_error_precedes_range_error_across_rows(backend, dtype, nonfinite):
    rows = torch.zeros(3, 8193, device="cuda", dtype=dtype)
    rows[0, 0], rows[1, -1], rows[2, 4096] = -1, 2, nonfinite
    with pytest.raises(ValueError, match="only finite"):
        backend.validated_screening_statistics(rows, 32 * torch.finfo(dtype).eps)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode,policy", [("multilabel", "max_score"), ("multiclass", "max_score"),
                                        ("multiclass", "void")])
def test_public_fused_path_skips_separate_value_scan(backend, monkeypatch, dtype, mode, policy):
    p = torch.rand(2, 3, 65537, generator=torch.Generator().manual_seed(789)).pow(12).to("cuda", dtype)
    p[:, 1] *= .25
    original = p.clone()
    options = dict(output_mode=mode, unassigned_policy=policy, safe_screening=True)
    with monkeypatch.context() as patch:
        patch.setattr(algo, "_rma_dice_validated_statistics", lambda p: None)
        reference = algo.rankseg_rma(p, **options)
    def reject(*args, **kwargs):
        pytest.fail("eligible fused path must not rescan values/statistics")
    monkeypatch.setattr(torch, "aminmax", reject)
    monkeypatch.setattr(algo, "_rma_dice_screening_statistics", reject)
    assert torch.equal(reference, algo.rankseg_rma(p, **options))
    assert torch.equal(p, original)


@pytest.mark.parametrize("mode", ["multiclass", "multilabel"])
@pytest.mark.parametrize("pruning", [0., .5, 1.])
def test_large_all_pruned_keeps_original_sum_and_skips_candidates(backend, monkeypatch, mode, pruning):
    p = torch.rand(1, 3, 65537, device="cuda", generator=torch.Generator(device="cuda").manual_seed(157)) * pruning
    options = dict(output_mode=mode, pruning_prob=pruning, safe_screening=True)
    with monkeypatch.context() as patch:
        patch.setattr(algo, "_rma_dice_validated_statistics", lambda p: None)
        reference = algo.rankseg_rma(p, **options)
    def reject(*args, **kwargs):
        pytest.fail("all-pruned input must not construct candidates or sort")
    monkeypatch.setattr(backend, "screening_candidates", reject)
    monkeypatch.setattr(torch, "sort", reject)
    assert torch.equal(reference, algo.rankseg_rma(p, **options))


@pytest.mark.parametrize("entry", ["direct", "functional", "module"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1., 2.])
def test_invalid_inputs_never_reach_candidates(backend, monkeypatch, entry, value):
    from rankseg import RankSEG
    from rankseg.functional import rankseg
    p = torch.zeros(1, 3, 65537, device="cuda")
    p[0, 2, -1] = value
    def reject(*args, **kwargs):
        pytest.fail("invalid probabilities reached candidate construction")
    monkeypatch.setattr(backend, "screening_candidates", reject)
    call = {"direct": algo.rankseg_rma, "functional": rankseg,
            "module": RankSEG(safe_screening=True)}[entry]
    with pytest.raises(ValueError, match="probs must"):
        call(p, **({} if entry == "module" else {"safe_screening": True}))


@pytest.mark.parametrize("options", [dict(safe_screening=False), dict(metric="iou"), dict(smooth=.2),
                                      dict(smooth=1e308)])
def test_unsupported_solver_paths_keep_ordinary_validation(backend, monkeypatch, options):
    def reject(*args, **kwargs):
        pytest.fail("unsupported path attempted fused validation")
    monkeypatch.setattr(algo, "_rma_dice_validated_statistics", reject)
    p = torch.zeros(1, 3, 65537, device="cuda")
    p[0, 0, -1] = float("nan")
    with pytest.raises(ValueError, match="only finite"):
        algo.rankseg_rma(p, **(dict(safe_screening=True) | options))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_keeps_pre_promotion_validation(backend, monkeypatch, dtype):
    def reject(*args, **kwargs):
        pytest.fail("low-precision input attempted fused validation")
    monkeypatch.setattr(algo, "_rma_dice_validated_statistics", reject)
    p = torch.zeros(1, 3, 65537, device="cuda", dtype=dtype)
    p[0, 0, -1] = float("inf")
    with pytest.raises(ValueError, match="only finite"):
        algo.rankseg_rma(p, safe_screening=True)


@pytest.mark.parametrize("shape", [(1, 1, 65537), (2, 3, 65536), (0, 3, 65537)])
def test_auto_small_or_empty_input_bypasses_fused_validation(backend, monkeypatch, shape):
    def reject(*args, **kwargs):
        pytest.fail("bypassed input attempted fused validation")
    monkeypatch.setattr(algo, "_rma_dice_validated_statistics", reject)
    p = torch.zeros(shape, device="cuda")
    result = algo.rankseg_rma(p, safe_screening="auto", output_mode="multilabel")
    assert result.shape == p.shape and not result.any()


def test_layout_requiring_copy_uses_ordinary_validation(backend, monkeypatch):
    p = torch.zeros(1, 3, 257, 257, device="cuda").transpose(-2, -1)
    p[0, -1, -1, -1] = float("nan")
    def reject(*args, **kwargs):
        pytest.fail("fused preparation must not copy probabilities")
    monkeypatch.setattr(backend, "validated_screening_statistics", reject)
    with pytest.raises(ValueError, match="only finite"):
        algo.rankseg_rma(p, safe_screening=True)


def test_no_triton_still_validates_values(monkeypatch):
    monkeypatch.setattr(screening, "_cuda_backend", lambda: None)
    p = torch.zeros(1, 3, 65537, device="cuda")
    p[0, 0, -1] = float("nan")
    with pytest.raises(ValueError, match="only finite"):
        algo.rankseg_rma(p, safe_screening=True)


def test_fused_validation_does_not_allocate_full_mask(backend, monkeypatch):
    p = torch.zeros(3, 8193, device="cuda")
    original = torch.empty
    def checked(shape, *args, **kwargs):
        if shape == p.shape and kwargs.get("dtype") == torch.bool:
            pytest.fail("validation must defer the full binary mask")
        return original(shape, *args, **kwargs)
    monkeypatch.setattr(torch, "empty", checked)
    maximum, (_, _, state) = backend.validated_screening_statistics(p, 32 * torch.finfo(p.dtype).eps)
    assert maximum == 0 and state[0] is None


def test_validation_on_nondefault_stream_and_adaptive_tiles(backend, monkeypatch):
    monkeypatch.setattr(backend, "_MAX_GRID_Y", 2)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        rows = torch.linspace(0, 1, 32769, device="cuda").repeat(3, 1)
        guard = 32 * torch.finfo(rows.dtype).eps
        reference = backend.screening_statistics(rows, guard)
        maximum, actual = backend.validated_screening_statistics(rows, guard)
        assert maximum == 1
        assert torch.equal(reference[0], actual[0])
        for a, b in zip(reference[2][1:], actual[2][1:]):
            assert torch.equal(a, b)
    stream.synchronize()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("direction", [-1., .5, 1.])
def test_validated_maximum_preserves_pruning_boundary(backend, dtype, direction):
    rows = torch.zeros(3, 4097, device="cuda", dtype=dtype)
    boundary = rows.new_tensor(.5)
    rows[-1, -1] = torch.nextafter(boundary, rows.new_tensor(direction))
    maximum, (_, row_maxima, _) = backend.validated_screening_statistics(rows, 32 * torch.finfo(dtype).eps)
    assert maximum == float(rows.max())
    assert torch.equal(row_maxima, rows.amax(-1))
    assert (maximum > .5) == (direction == 1.)


@pytest.mark.parametrize("options", [dict(metric=None), dict(output_mode=None), dict(output_mode="bad"),
                                      dict(unassigned_policy=None), dict(pruning_prob=-1), dict(smooth=-1),
                                      dict(smooth=True), dict(void_index=False), dict(safe_screening=1)])
def test_value_errors_still_precede_parameter_errors(backend, options):
    p = torch.zeros(1, 3, 65537, device="cuda")
    p[0, -1, -1] = float("nan")
    with pytest.raises(ValueError, match="only finite"):
        algo.rankseg_rma(p, **(dict(safe_screening=True) | options))


@pytest.mark.skipif(os.environ.get("RANKSEG_TEST_LARGE_CUDA") != "1", reason="opt-in large CUDA test")
def test_last_value_of_multigigabyte_input_is_validated(backend):
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < 6 * 1024**3:
        pytest.skip("requires 6 GiB free CUDA memory")
    p = torch.full((1, 3, 846 * 512 * 512), .125, device="cuda")
    p[0, -1, -1] = float("nan")
    with pytest.raises(ValueError, match="only finite"):
        algo.rankseg_rma(p, safe_screening=True)
