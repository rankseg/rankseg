"""Tri-state dispatch, exact threshold boundaries and unchanged objectives."""

import inspect
from types import SimpleNamespace

import pytest
import torch
from test_screening import _assert_epsilon_optimal

from rankseg import RankSEG
from rankseg import _rankseg_algo as algo
from rankseg import _screening as screening
from rankseg.functional import rankseg

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def test_rma_signature_defaults_to_false():
    assert inspect.signature(algo.rankseg_rma).parameters["safe_screening"].default is False


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("entry", [algo.rankseg_rma, rankseg, RankSEG])
@pytest.mark.parametrize("select", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("output_mode", ["multilabel", "multiclass"])
def test_omitted_option_is_exactly_false_without_screening(monkeypatch, device, entry, select, dtype, output_mode):
    p = torch.rand(2, 3, 257, generator=torch.Generator().manual_seed(61)).pow(9).to(device, dtype)
    monkeypatch.setattr(algo, "_RMA_CUDA_SCREENING_AUTO_MIN_ELEMENTS", 1 if select else p.numel() + 1)
    options = dict(output_mode=output_mode)
    def predict(extra):
        params = options | extra
        return entry(**params)(p) if entry is RankSEG else entry(p, **params)
    expected = predict(dict(safe_screening=False))

    def reject(*args, **kwargs):
        pytest.fail("default False must not initialize screening or its optimized full-sort path")

    for name in ("_rma_dice_use_screening", "_rma_dice_screened_masks", "_rma_dice_validated_statistics"):
        monkeypatch.setattr(algo, name, reject)
    monkeypatch.setattr(screening, "_cuda_backend", reject)
    assert torch.equal(predict({}), expected)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("entry", [algo.rankseg_rma, rankseg, RankSEG])
@pytest.mark.parametrize("case", ["empty", "all_pruned", "one_hot", "ties", "strided", "transposed"])
@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("mode,policy", [("multilabel", "max_score"), ("multiclass", "max_score"),
                                       ("multiclass", "void")])
def test_default_matches_false_for_edge_cases_and_layouts(
        monkeypatch, device, entry, case, vectorized, mode, policy):
    p = torch.rand(2, 3, 7, 10, generator=torch.Generator().manual_seed(903)).to(device)
    if case == "empty":
        p = p[:0]
    elif case == "all_pruned":
        p *= .5
    elif case == "one_hot":
        p = torch.zeros_like(p).scatter_(1, p.argmax(1, keepdim=True), 1)
    elif case == "ties":
        p.fill_(.6)
    elif case == "strided":
        p = p[..., ::2]
    elif case == "transposed":
        p = p.transpose(-1, -2)
    original = p.clone()
    # Mask reconstruction is internal, not a public solver argument. Exercise
    # both CUDA implementations without changing the public API.
    monkeypatch.setattr(algo, "_RMA_CUDA_VECTORIZED_MASK_MAX_DIM", 1000 if vectorized else 0)
    options = dict(output_mode=mode)
    if mode == "multiclass":
        options["unassigned_policy"] = policy

    def predict(extra):
        params = options | extra
        return entry(**params)(p) if entry is RankSEG else entry(p, **params)

    expected, actual = predict(dict(safe_screening=False)), predict({})
    assert torch.equal(actual, expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype and actual.device == p.device
    assert torch.equal(p, original)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("options", [dict(metric="iou"), dict(smooth=.2), dict(smooth=1e308)])
def test_default_does_not_change_other_rma_objectives(device, options):
    p = torch.rand(2, 3, 257, generator=torch.Generator().manual_seed(52)).to(device)
    assert torch.equal(algo.rankseg_rma(p, **options), algo.rankseg_rma(p, safe_screening=False, **options))


@pytest.mark.parametrize("solver", ["BA", "TRNA", "BA+TRNA"])
def test_rma_default_does_not_leak_into_other_solvers(monkeypatch, solver):
    def reject(*args, **kwargs):
        pytest.fail("non-RMA solver entered screening dispatch")
    monkeypatch.setattr(algo, "_rma_dice_use_screening", reject)
    p = torch.tensor([[[.9, .3, .1]]])
    actual = RankSEG(solver=solver, output_mode="multilabel")(p)
    assert torch.equal(actual, algo.rankdice_ba(p, solver=solver))


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps", "meta"])
@pytest.mark.parametrize("mode", [False, True, "auto"])
@pytest.mark.parametrize("size", [0, 1, 1_279_999, 1_280_000, 1_280_001])
@pytest.mark.parametrize("available", [False, True])
def test_dispatch_uses_only_metadata(monkeypatch, device, mode, size, available):
    calls = []

    def backend():
        calls.append(True)
        return object() if available else None

    monkeypatch.setattr(screening, "_cuda_backend", backend)
    # Deliberately no probability data, reductions, .cpu() or scalar reads.
    probs = SimpleNamespace(device=torch.device(device), numel=lambda: size)
    expected = mode is True or (mode == "auto" and (
        device == "cpu" or (device == "cuda" and size >= 1_280_000 and available)))
    assert algo._RMA_CUDA_SCREENING_AUTO_MIN_ELEMENTS == 1_280_000
    assert algo._rma_dice_use_screening(probs, mode) is expected
    assert bool(calls) == (mode == "auto" and device == "cuda" and size >= 1_280_000)


@pytest.mark.parametrize("entry", [algo.rankseg_rma, rankseg, RankSEG])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["multilabel", "multiclass"])
def test_cpu_auto_matches_forced_without_cuda_import(monkeypatch, entry, dtype, mode):
    def reject():
        pytest.fail("CPU must not load the CUDA backend")

    monkeypatch.setattr(screening, "_cuda_backend", reject)
    probs = torch.tensor([[[.9, .4, 0, 0], [.1, .6, 1, 0]]], dtype=dtype)
    expected = algo.rankseg_rma(probs, output_mode=mode, safe_screening=True)
    options = dict(output_mode=mode, safe_screening="auto")
    actual = entry(**options)(probs) if entry is RankSEG else entry(probs, **options)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("options", [dict(metric="iou"), dict(smooth=.2), dict(smooth=1e308)])
@pytest.mark.parametrize("mode", ["auto", True])
def test_ineligible_objectives_keep_original_path(monkeypatch, device, options, mode):
    def reject(*args, **kwargs):
        pytest.fail("ineligible objective must not use screening dispatch or backend")

    monkeypatch.setattr(algo, "_rma_dice_use_screening", reject)
    monkeypatch.setattr(screening, "_cuda_backend", reject)
    probs = torch.tensor([[[.9, .4, 0], [.1, .6, 1]]], device=device)
    assert torch.equal(algo.rankseg_rma(probs, safe_screening=mode, **options),
                       algo.rankseg_rma(probs, safe_screening=False, **options))


@CUDA
@pytest.mark.parametrize("shape", [(1, 1, 1_279_999), (1, 1, 1_280_000), (1, 1, 1_280_001),
                                   (2, 2, 319_999), (2, 2, 320_000), (2, 2, 320_001),
                                   (2, 5, 320, 400)])
def test_real_cuda_threshold_counts_batch_channels_and_spatial_dims(monkeypatch, shape):
    if screening._cuda_backend() is None:
        pytest.skip("Triton unavailable")
    generator = torch.Generator().manual_seed(952)
    # Isolate dispatch from CUDA float cumsum's nondeterministic rounding.
    # Integer-valued prefixes below 2**24 are exact in float32; randomized
    # large-input numerical behavior is checked independently below.
    probs = (torch.rand(shape, generator=generator) > .8).float().cuda()
    original = probs.clone()
    calls = []
    masks = algo._rma_dice_screened_masks

    def tracked(*args, **kwargs):
        calls.append(True)
        return masks(*args, **kwargs)

    monkeypatch.setattr(algo, "_rma_dice_screened_masks", tracked)
    actual = algo.rankseg_rma(probs, safe_screening="auto", output_mode="multilabel")
    selected = probs.numel() >= 1_280_000
    assert bool(calls) is selected
    _assert_epsilon_optimal(probs, actual)
    # Compare to the exact selected route, not to a different reduction tree.
    with monkeypatch.context() as patch:
        patch.setattr(algo, "_rma_dice_use_screening", lambda probs, mode: selected)
        expected = algo.rankseg_rma(probs, safe_screening=True, output_mode="multilabel")
    assert torch.equal(actual, expected)
    assert torch.equal(probs, original)


@CUDA
@pytest.mark.parametrize("seed", [952, 955, 960])
@pytest.mark.parametrize("option", [False, "auto"])
def test_large_cuda_full_sort_repeats_keep_original_objective_budget(seed, option):
    p = torch.rand(1, 1, 1_279_999, generator=torch.Generator().manual_seed(seed)).pow(12).cuda()
    original = p.clone()
    # Both original full sort and auto's optimized-full branch call CUDA
    # float cumsum. Repeated masks need not be bitwise equal near score ties;
    # every run must still satisfy the unchanged independent 4-eps budget.
    for _ in range(4):
        mask = algo.rankseg_rma(p, safe_screening=option, output_mode="multilabel")
        _assert_epsilon_optimal(p, mask)
    assert torch.equal(p, original)


@CUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["contiguous", "spatial_strided", "transpose"])
@pytest.mark.parametrize("mode,policy", [("multilabel", "max_score"), ("multiclass", "max_score"),
                                       ("multiclass", "void")])
@pytest.mark.parametrize("select", [False, True])
def test_auto_matches_selected_route_across_dtypes_layouts_and_policies(
        monkeypatch, dtype, layout, mode, policy, select):
    if screening._cuda_backend() is None:
        pytest.skip("Triton unavailable")
    p = torch.rand(2, 3, 17, 34, generator=torch.Generator().manual_seed(532)).pow(8).to("cuda", dtype)
    if layout == "spatial_strided":
        p = p[..., ::2]
    elif layout == "transpose":
        p = p.transpose(-2, -1)
    p[:, 0] *= .3  # Mix active and pruned rows.
    original = p.clone()
    monkeypatch.setattr(algo, "_RMA_CUDA_SCREENING_AUTO_MIN_ELEMENTS", p.numel() if select else p.numel() + 1)
    options = dict(output_mode=mode, unassigned_policy=policy)
    actual = algo.rankseg_rma(p, safe_screening="auto", **options)
    with monkeypatch.context() as patch:
        patch.setattr(algo, "_rma_dice_use_screening", lambda probs, mode: select)
        expected = algo.rankseg_rma(p, safe_screening=True, **options)
    assert torch.equal(actual, expected)
    assert torch.equal(p, original)
    binary = algo.rankseg_rma(p, safe_screening="auto", output_mode="multilabel")
    _assert_epsilon_optimal(p, binary)


@CUDA
def test_cuda_without_triton_auto_full_sort_but_true_still_screens(monkeypatch):
    monkeypatch.setattr(screening, "_cuda_backend", lambda: None)
    monkeypatch.setattr(algo, "_RMA_CUDA_SCREENING_AUTO_MIN_ELEMENTS", 1)
    p = torch.tensor([[[.9, .4, .2, 0]]], device="cuda")
    calls = []
    masks = algo._rma_dice_screened_masks

    def tracked(*args, **kwargs):
        calls.append(True)
        return masks(*args, **kwargs)

    monkeypatch.setattr(algo, "_rma_dice_screened_masks", tracked)
    auto = algo.rankseg_rma(p, safe_screening="auto", output_mode="multilabel")
    assert not calls
    forced = algo.rankseg_rma(p, safe_screening=True, output_mode="multilabel")
    assert calls
    _assert_epsilon_optimal(p, auto)
    _assert_epsilon_optimal(p, forced)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("shape", [(0, 3, 7), (2, 3, 7)])
@pytest.mark.parametrize("mode,policy", [("multilabel", "max_score"), ("multiclass", "max_score"),
                                       ("multiclass", "void")])
def test_auto_empty_and_all_pruned_preserve_original_results(monkeypatch, device, selected, shape, mode, policy):
    monkeypatch.setattr(algo, "_RMA_CUDA_SCREENING_AUTO_MIN_ELEMENTS", 0 if selected else 1_280_000)
    p = torch.zeros(shape, device=device)
    options = dict(output_mode=mode, unassigned_policy=policy)
    expected = algo.rankseg_rma(p, safe_screening=False, **options)

    def reject(*args, **kwargs):
        pytest.fail("empty/all-pruned inputs must skip candidate search")

    monkeypatch.setattr(algo, "_rma_dice_screened_masks", reject)
    assert torch.equal(algo.rankseg_rma(p, safe_screening="auto", **options), expected)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, 1.1])
def test_auto_still_rejects_invalid_probabilities(monkeypatch, device, selected, value):
    monkeypatch.setattr(algo, "_RMA_CUDA_SCREENING_AUTO_MIN_ELEMENTS", 1 if selected else 1_280_000)
    p = torch.ones(1, 1, 3, device=device)
    p[..., -1] = value
    with pytest.raises(ValueError, match="probs must"):
        algo.rankseg_rma(p, safe_screening="auto", output_mode="multilabel")


@pytest.mark.parametrize("mode", [False, "bad", 1, torch.tensor(True)])
def test_false_and_invalid_modes_never_consult_backend(monkeypatch, mode):
    def reject(*args, **kwargs):
        pytest.fail("False/invalid options must not initialize screening")

    monkeypatch.setattr(algo, "_rma_dice_use_screening", reject)
    p = torch.ones(1, 1, 3)
    if mode is False:
        assert algo.rankseg_rma(p, safe_screening=mode, output_mode="multilabel").all()
    else:
        with pytest.raises(TypeError, match="bool or 'auto'"):
            algo.rankseg_rma(p, safe_screening=mode, output_mode="multilabel")
