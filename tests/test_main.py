import importlib
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version

import pytest
import torch
import torch.nn.functional as F
from torchmetrics.functional.segmentation import dice_score

import rankseg
from rankseg import RankSEG, functional, rankseg_rma


def test_public_version_matches_distribution_metadata():
    assert rankseg.__version__ == version("rankseg")
    assert "__version__" in rankseg.__all__


def test_public_version_has_source_tree_fallback(monkeypatch):
    def missing_distribution(_name):
        raise PackageNotFoundError

    with monkeypatch.context() as patch:
        patch.setattr(importlib.metadata, "version", missing_distribution)
        assert importlib.reload(rankseg).__version__ == "0+unknown"

    assert importlib.reload(rankseg).__version__ == version("rankseg")


def _labels_to_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).movedim(-1, 1)


def _multiclass_to_one_hot(preds: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(preds, num_classes=num_classes).movedim(-1, 1)


def _assert_binary_tensor(tensor: torch.Tensor) -> None:
    assert bool(torch.all((tensor == 0) | (tensor == 1)))


def _mean_dice(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    return dice_score(preds, targets, num_classes=num_classes, average="none").nanmean(dim=0).nanmean().item()


@pytest.mark.parametrize(
    ("metric", "output_mode", "solver"),
    [
        ("dice", "multiclass", "RMA"),
        ("dice", "multilabel", "RMA"),
        ("dice", "multilabel", "BA"),
        ("dice", "multilabel", "TRNA"),
        ("dice", "multilabel", "BA+TRNA"),
        ("iou", "multiclass", "RMA"),
        ("iou", "multilabel", "RMA"),
        ("accuracy", "multiclass", "argmax"),
        ("accuracy", "multilabel", "TR"),
    ],
)
def test_rankseg_supported_solver_matrix(metric, output_mode, solver, demo_data):
    probs, labels = demo_data

    preds = RankSEG(metric=metric, solver=solver, output_mode=output_mode).predict(probs)

    expected_shape = labels.shape if output_mode == "multiclass" else probs.shape
    expected_dtype = torch.int64 if output_mode == "multiclass" else torch.bool
    assert preds.shape == expected_shape
    assert preds.dtype == expected_dtype


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("metric", "output_mode", "solver"),
    [
        ("dice", "multiclass", "RMA"),
        ("dice", "multilabel", "RMA"),
        ("dice", "multilabel", "BA"),
        ("dice", "multilabel", "TRNA"),
        ("dice", "multilabel", "BA+TRNA"),
        ("iou", "multiclass", "RMA"),
        ("iou", "multilabel", "RMA"),
        ("accuracy", "multiclass", "argmax"),
        ("accuracy", "multilabel", "TR"),
        ("accuracy", "multilabel", "argmax"),
    ],
)
def test_rankseg_output_contract_is_independent_of_solver_and_input_dtype(dtype, metric, output_mode, solver):
    probs = torch.tensor([[[0.8, 0.2, 0.6, 0.4], [0.2, 0.8, 0.4, 0.6]]], dtype=dtype)
    warning_context = (
        pytest.warns(UserWarning, match="non-overlapping one-hot masks")
        if solver == "argmax" and output_mode == "multilabel"
        else nullcontext()
    )

    with warning_context:
        preds = RankSEG(
            metric=metric,
            solver=solver,
            output_mode=output_mode,
            pruning_prob=0.0,
        ).predict(probs)

    expected_shape = (1, 4) if output_mode == "multiclass" else probs.shape
    expected_dtype = torch.int64 if output_mode == "multiclass" else torch.bool
    assert preds.shape == expected_shape
    assert preds.dtype == expected_dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_rankseg_output_contract_matches_on_cuda():
    probs_cpu = torch.tensor([[[0.8, 0.2, 0.6, 0.4], [0.2, 0.8, 0.4, 0.6]]], dtype=torch.float32)
    configurations = [
        ("dice", "multiclass", "RMA"),
        ("dice", "multilabel", "RMA"),
        ("dice", "multilabel", "BA"),
        ("dice", "multilabel", "TRNA"),
        ("dice", "multilabel", "BA+TRNA"),
        ("iou", "multiclass", "RMA"),
        ("iou", "multilabel", "RMA"),
        ("accuracy", "multiclass", "argmax"),
        ("accuracy", "multilabel", "TR"),
        ("accuracy", "multilabel", "argmax"),
    ]

    for metric, output_mode, solver in configurations:
        predictor = RankSEG(
            metric=metric,
            solver=solver,
            output_mode=output_mode,
            pruning_prob=0.0,
        )
        if solver == "argmax" and output_mode == "multilabel":
            with pytest.warns(UserWarning, match="non-overlapping one-hot masks"):
                preds_cpu = predictor.predict(probs_cpu)
            with pytest.warns(UserWarning, match="non-overlapping one-hot masks"):
                preds_cuda = predictor.predict(probs_cpu.cuda())
        else:
            preds_cpu = predictor.predict(probs_cpu)
            preds_cuda = predictor.predict(probs_cpu.cuda())

        expected_dtype = torch.int64 if output_mode == "multiclass" else torch.bool
        assert preds_cuda.device.type == "cuda"
        assert preds_cuda.dtype == expected_dtype
        assert torch.equal(preds_cuda.cpu(), preds_cpu)


def test_rankseg_ba_dice(demo_data):
    probs, labels = demo_data
    num_classes = probs.shape[1]
    labels_oh = _labels_to_one_hot(labels, num_classes)

    preds = RankSEG(metric="dice", solver="BA", output_mode="multilabel").predict(probs)
    mean_dice = _mean_dice(preds, labels_oh, num_classes)

    assert preds.shape == probs.shape
    assert preds.dtype == torch.bool
    _assert_binary_tensor(preds)
    assert torch.equal(preds, labels_oh.to(preds.dtype))
    assert mean_dice == pytest.approx(1.0)


def test_rankseg_rma_overlap(demo_data):
    probs, labels = demo_data
    num_classes = probs.shape[1]
    labels_oh = _labels_to_one_hot(labels, num_classes)

    preds = RankSEG(metric="dice", solver="RMA", output_mode="multilabel").predict(probs)
    mean_dice = _mean_dice(preds, labels_oh, num_classes)

    assert preds.shape == probs.shape
    assert preds.dtype == torch.bool
    _assert_binary_tensor(preds)
    assert torch.equal(preds, labels_oh.to(preds.dtype))
    assert mean_dice == pytest.approx(1.0)


def test_rankseg_rma_multiclass(demo_data):
    probs, labels = demo_data
    num_classes = probs.shape[1]
    labels_oh = _labels_to_one_hot(labels, num_classes)

    preds = RankSEG(metric="dice", solver="RMA", output_mode="multiclass").predict(probs)
    preds_oh = _multiclass_to_one_hot(preds, num_classes)
    mean_dice = _mean_dice(preds_oh, labels_oh, num_classes)

    assert preds.shape == labels.shape
    assert preds.dtype == torch.int64
    assert int(preds.min().item()) >= 0
    assert int(preds.max().item()) < num_classes
    assert torch.equal(preds, labels)
    assert mean_dice == pytest.approx(1.0)


def test_functional_rankseg_matches_rankseg_method(demo_data):
    probs, _ = demo_data
    rankseg = RankSEG(metric="dice", solver="RMA", output_mode="multiclass")

    preds_functional = functional.rankseg(probs, metric="dice", solver="RMA", output_mode="multiclass")
    preds_method = rankseg.predict(probs)

    assert torch.equal(preds_functional, preds_method)


def test_functional_rankseg_disables_autograd(monkeypatch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probs = torch.tensor(
        [[[0.8, 0.2, 0.6, 0.4], [0.2, 0.8, 0.4, 0.6]]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    expected = functional.rankseg(probs.detach(), metric="accuracy", solver="argmax", output_mode="multiclass")

    grad_states = []
    original_validate = functional.validate_probability_tensor

    def record_grad_state(*args, **kwargs):
        grad_states.append(torch.is_grad_enabled())
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(functional, "validate_probability_tensor", record_grad_state)
    with torch.enable_grad():
        actual = functional.rankseg(probs, metric="accuracy", solver="argmax", output_mode="multiclass")

    assert grad_states and not any(grad_states)
    assert not actual.requires_grad
    assert torch.equal(actual, expected)


def test_rankseg_call_matches_predict_method(demo_data):
    probs, _ = demo_data
    rankseg = RankSEG(metric="accuracy", solver="TR", output_mode="multilabel")

    preds_call = rankseg(probs)
    preds_method = rankseg.predict(probs)

    assert torch.equal(preds_call, preds_method)


def test_rankseg_acc_argmax(demo_data):
    probs, labels = demo_data

    preds = RankSEG(metric="acc", solver="argmax", output_mode="multiclass").predict(probs)

    assert preds.shape == labels.shape
    assert preds.dtype == torch.int64
    assert torch.equal(preds, labels)
    assert torch.equal(preds, torch.argmax(probs, dim=1))


def test_rankseg_acc_tr(demo_data):
    probs, labels = demo_data
    labels_oh = _labels_to_one_hot(labels, probs.shape[1])

    preds = RankSEG(metric="acc", solver="TR", output_mode="multilabel").predict(probs)
    expected = probs > 0.5

    assert preds.shape == probs.shape
    assert preds.dtype == torch.bool
    _assert_binary_tensor(preds)
    assert torch.equal(preds, labels_oh.to(preds.dtype))
    assert torch.equal(preds, expected)


def test_rankseg_rejects_unknown_output_mode(demo_data):
    probs, _ = demo_data

    with pytest.raises(ValueError, match="Unknown output mode"):
        RankSEG(metric="dice", solver="RMA", output_mode="invalid").predict(probs)


def test_rankseg_rejects_non_tensor_input():
    with pytest.raises(TypeError, match="torch.Tensor"):
        RankSEG(metric="dice", solver="RMA", output_mode="multilabel").predict([[[0.5]]])


def test_rankseg_rejects_inputs_with_too_few_dimensions():
    probs = torch.tensor([[0.2, 0.8]], dtype=torch.float32)

    with pytest.raises(ValueError, match="batch_size, num_class"):
        RankSEG(metric="dice", solver="RMA", output_mode="multilabel").predict(probs)


def test_rankseg_rejects_non_finite_probabilities():
    probs = torch.tensor([[[float("nan"), 0.5]]], dtype=torch.float32)

    with pytest.raises(ValueError, match="finite values"):
        RankSEG(metric="dice", solver="RMA", output_mode="multilabel").predict(probs)


def test_rankseg_rejects_out_of_range_probabilities():
    probs = torch.tensor([[[1.2, -0.1]]], dtype=torch.float32)

    with pytest.raises(ValueError, match="range \\[0, 1\\]"):
        RankSEG(metric="dice", solver="RMA", output_mode="multilabel").predict(probs)


@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.complex64])
def test_rankseg_rejects_non_real_floating_probability_dtypes(dtype):
    probs = torch.zeros((1, 1, 4), dtype=dtype)

    with pytest.raises(TypeError, match="real floating-point dtype"):
        RankSEG(metric="dice", solver="RMA", output_mode="multilabel").predict(probs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("solver", ["RMA", "BA"])
def test_rankseg_supports_real_floating_probability_dtypes(dtype, solver):
    probs = torch.tensor([[[0.8, 0.2, 0.6, 0.4]]], dtype=dtype)

    preds = RankSEG(
        metric="dice",
        solver=solver,
        output_mode="multilabel",
        pruning_prob=0.0,
    ).predict(probs)
    working_probs = probs.float() if dtype in (torch.float16, torch.bfloat16) else probs
    expected = RankSEG(
        metric="dice",
        solver=solver,
        output_mode="multilabel",
        pruning_prob=0.0,
    ).predict(working_probs)

    assert preds.shape == probs.shape
    assert preds.dtype == torch.bool
    _assert_binary_tensor(preds)
    assert torch.equal(preds, expected)


@pytest.mark.parametrize("solver", ["RMA", "BA"])
def test_rankseg_allows_empty_batches(solver):
    probs = torch.empty((0, 2, 4), dtype=torch.float32)

    preds = RankSEG(metric="dice", solver=solver, output_mode="multilabel").predict(probs)

    assert preds.shape == probs.shape
    assert preds.dtype == torch.bool


def test_rankseg_rejects_empty_class_dimension():
    probs = torch.empty((1, 0, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="at least one class"):
        RankSEG(metric="dice", solver="RMA", output_mode="multilabel").predict(probs)


@pytest.mark.parametrize("shape", [(1, 2, 0), (1, 2, 3, 0)])
def test_rankseg_rejects_empty_spatial_dimensions(shape):
    probs = torch.empty(shape, dtype=torch.float32)

    with pytest.raises(ValueError, match="spatial dimensions must be non-empty"):
        RankSEG(metric="dice", solver="RMA", output_mode="multilabel").predict(probs)


@pytest.mark.parametrize("name", ["metric", "solver", "output_mode"])
def test_rankseg_rejects_non_string_configuration(name, demo_data):
    probs, _ = demo_data
    config = {"metric": "dice", "solver": "RMA", "output_mode": "multilabel"}
    config[name] = None

    with pytest.raises(TypeError, match=rf"{name} must be a string"):
        RankSEG(**config).predict(probs)


@pytest.mark.parametrize("name", ["smooth", "pruning_prob"])
@pytest.mark.parametrize("value", [True, "0.5"])
def test_rankseg_rejects_non_real_numeric_parameters(name, value, demo_data):
    probs, _ = demo_data
    config = {name: value}

    with pytest.raises(TypeError, match=rf"{name} must be a real number"):
        RankSEG(metric="dice", solver="RMA", output_mode="multilabel", **config).predict(probs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("name", ["smooth", "pruning_prob"])
def test_rankseg_rejects_non_finite_numeric_parameters(name, value, demo_data):
    probs, _ = demo_data
    config = {name: value}

    with pytest.raises(ValueError, match=rf"{name} must be finite"):
        RankSEG(metric="dice", solver="RMA", output_mode="multilabel", **config).predict(probs)


def test_rankseg_rejects_negative_smooth(demo_data):
    probs, _ = demo_data

    with pytest.raises(ValueError, match="smooth must be greater than or equal to 0"):
        RankSEG(metric="dice", solver="RMA", output_mode="multilabel", smooth=-0.1).predict(probs)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_rankseg_rejects_out_of_range_pruning_prob(value, demo_data):
    probs, _ = demo_data

    with pytest.raises(ValueError, match="pruning_prob must be in the range \\[0, 1\\]"):
        RankSEG(metric="dice", solver="RMA", output_mode="multilabel", pruning_prob=value).predict(probs)


@pytest.mark.parametrize("value", [True, "0.1"])
def test_rankseg_rejects_non_real_eps(value):
    probs = torch.tensor([[[0.8, 0.2, 0.6, 0.4]]], dtype=torch.float32)

    with pytest.raises(TypeError, match="eps must be a real number"):
        RankSEG(metric="dice", solver="BA", output_mode="multilabel", eps=value).predict(probs)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.1])
def test_rankseg_rejects_out_of_range_eps(value):
    probs = torch.tensor([[[0.8, 0.2, 0.6, 0.4]]], dtype=torch.float32)

    with pytest.raises(ValueError, match="eps must be in the range \\(0, 1\\)"):
        RankSEG(metric="dice", solver="BA", output_mode="multilabel", eps=value).predict(probs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rankseg_rejects_non_finite_eps(value):
    probs = torch.tensor([[[0.8, 0.2, 0.6, 0.4]]], dtype=torch.float32)

    with pytest.raises(ValueError, match="eps must be finite"):
        RankSEG(metric="dice", solver="BA", output_mode="multilabel", eps=value).predict(probs)


def test_rankseg_rejects_single_channel_multiclass_input():
    probs = torch.tensor([[[0.8, 0.2, 0.6, 0.4]]], dtype=torch.float32)

    with pytest.raises(ValueError, match="Single-channel probabilities"):
        RankSEG(metric="dice", solver="RMA", output_mode="multiclass").predict(probs)


def test_rankseg_accuracy_rejects_solver_parameters(demo_data):
    probs, _ = demo_data

    with pytest.raises(TypeError, match="solver 'argmax' does not accept solver parameters: eps"):
        RankSEG(metric="accuracy", solver="argmax", output_mode="multiclass", eps=1e-4).predict(probs)


def test_rankseg_rejects_boolean_void_index(demo_data):
    probs, _ = demo_data

    with pytest.raises(TypeError, match="void_index must be an int"):
        RankSEG(
            metric="dice",
            solver="RMA",
            output_mode="multiclass",
            unassigned_policy="void",
            void_index=True,
        ).predict(probs)


def test_rankseg_rejects_exact_solver(demo_data):
    probs, _ = demo_data

    with pytest.raises(ValueError, match="Exact solver is not implemented yet"):
        RankSEG(metric="dice", solver="exact", output_mode="multilabel").predict(probs)


def test_rankseg_dice_rejects_unknown_solver(demo_data):
    probs, _ = demo_data
    rankseg = RankSEG(metric="dice", solver="invalid", output_mode="multilabel")

    with pytest.raises(ValueError, match=r"supported solvers: RMA, BA, TRNA, BA\+TRNA"):
        rankseg.predict(probs)

    assert rankseg.solver == "invalid"


def test_rankseg_dice_multiclass_rejects_ba(demo_data):
    probs, _ = demo_data
    rankseg = RankSEG(metric="dice", solver="BA", output_mode="multiclass")

    with pytest.raises(ValueError, match="supported solvers: RMA"):
        rankseg.predict(probs)

    assert rankseg.solver == "BA"


def test_rankseg_iou_rejects_ba(demo_data):
    probs, _ = demo_data
    rankseg = RankSEG(metric="IoU", solver="BA", output_mode="multiclass")

    with pytest.raises(ValueError, match="supported solvers: RMA"):
        rankseg.predict(probs)

    assert rankseg.solver == "BA"


def test_rankseg_exact_not_implemented_error_is_limited_to_dice(demo_data):
    probs, _ = demo_data

    with pytest.raises(ValueError, match="not supported for metric='iou'"):
        RankSEG(metric="IoU", solver="exact", output_mode="multiclass").predict(probs)


def test_rankseg_rma_rejects_ba_solver_parameter(demo_data):
    probs, _ = demo_data

    with pytest.raises(TypeError, match="solver 'RMA' does not accept solver parameters: eps"):
        RankSEG(metric="dice", solver="RMA", output_mode="multilabel", eps=1e-4).predict(probs)


def test_rankseg_ba_rejects_rma_solver_parameter(demo_data):
    probs, _ = demo_data

    with pytest.raises(TypeError, match="solver 'BA' does not accept solver parameters: void_index"):
        RankSEG(metric="dice", solver="BA", output_mode="multilabel", void_index=99).predict(probs)


def test_rankseg_rma_multilabel_rejects_unassigned_policy(demo_data):
    probs, _ = demo_data

    with pytest.raises(TypeError, match="solver 'RMA' does not accept solver parameters: unassigned_policy"):
        RankSEG(
            metric="dice",
            solver="RMA",
            output_mode="multilabel",
            unassigned_policy="void",
        ).predict(probs)


def test_rankseg_rma_void_index_requires_void_policy(demo_data):
    probs, _ = demo_data

    with pytest.raises(ValueError, match="void_index requires unassigned_policy='void'"):
        RankSEG(metric="dice", solver="RMA", output_mode="multiclass", void_index=99).predict(probs)


def test_rankseg_rma_multiclass_void_unassigned_policy():
    probs = torch.tensor(
        [
            [
                [[0.95, 0.01, 0.01, 0.01]],
                [[0.01, 0.95, 0.95, 0.10]],
                [[0.01, 0.55, 0.55, 0.01]],
            ]
        ],
        dtype=torch.float32,
    )

    rankseg = RankSEG(
        metric="dice",
        solver="RMA",
        output_mode="multiclass",
        pruning_prob=0.5,
        unassigned_policy="void",
        void_index=99,
    )
    preds = rankseg.predict(probs)

    assert rankseg.solver_params["unassigned_policy"] == "void"
    assert rankseg.solver_params["void_index"] == 99
    assert not hasattr(rankseg, "unassigned_policy")
    assert not hasattr(rankseg, "void_index")
    assert torch.equal(preds, torch.tensor([[[0, 1, 1, 99]]]))


def test_rankseg_high_level_rejects_void_index_that_collides_with_a_class():
    probs = torch.full((1, 3, 2), 0.2, dtype=torch.float32)

    with pytest.raises(ValueError, match=r"outside the valid class index range \[0, num_class\)"):
        RankSEG(
            metric="dice",
            solver="RMA",
            output_mode="multiclass",
            unassigned_policy="void",
            void_index=2,
        )(probs)


def test_rankseg_metric_is_case_insensitive_for_dice(demo_data):
    probs, _ = demo_data

    preds_upper = RankSEG(metric=" DICE ", solver="RMA", output_mode="multilabel").predict(probs)
    preds_lower = RankSEG(metric="dice", solver="RMA", output_mode="multilabel").predict(probs)

    assert torch.equal(preds_upper, preds_lower)


def test_rankseg_rma_metric_is_case_insensitive(demo_data):
    probs, _ = demo_data

    preds_mixed = rankseg_rma(probs, metric=" IoU ", output_mode="multiclass")
    preds_lower = rankseg_rma(probs, metric="iou", output_mode="multiclass")

    assert torch.equal(preds_mixed, preds_lower)


def test_rankseg_solver_is_case_insensitive_for_rma(demo_data):
    probs, _ = demo_data

    preds_mixed = RankSEG(metric="dice", solver=" rMa ", output_mode="multilabel").predict(probs)
    preds_canonical = RankSEG(metric="dice", solver="RMA", output_mode="multilabel").predict(probs)

    assert torch.equal(preds_mixed, preds_canonical)


def test_rankseg_solver_is_case_insensitive_for_ba_plus_trna(demo_data):
    probs, _ = demo_data

    preds_mixed = RankSEG(metric="dice", solver=" ba+trna ", output_mode="multilabel").predict(probs)
    preds_canonical = RankSEG(metric="dice", solver="BA+TRNA", output_mode="multilabel").predict(probs)

    assert torch.equal(preds_mixed, preds_canonical)


def test_rankseg_output_mode_is_case_insensitive(demo_data):
    probs, _ = demo_data

    preds_mixed = RankSEG(metric="accuracy", solver=" tr ", output_mode=" MultiLabel ").predict(probs)
    preds_canonical = RankSEG(metric="accuracy", solver="TR", output_mode="multilabel").predict(probs)

    assert torch.equal(preds_mixed, preds_canonical)


def test_rankseg_acc_single_class_thresholds_probs():
    probs = torch.tensor([[[[0.25, 0.75], [0.50, 0.51]]]], dtype=torch.float32)

    preds = RankSEG(metric=" Accuracy ", solver="TR", output_mode="multilabel").predict(probs)

    assert preds.shape == probs.shape
    assert preds.dtype == torch.bool
    assert torch.equal(preds, torch.tensor([[[[False, True], [False, True]]]]))


def test_rankseg_acc_multilabel_argmax_warns_and_returns_binary_masks(demo_data):
    probs, _ = demo_data
    rankseg = RankSEG(metric="acc", solver="argmax", output_mode="multilabel")
    expected = torch.zeros_like(probs, dtype=torch.bool)
    expected.scatter_(1, torch.argmax(probs, dim=1, keepdim=True), True)

    with pytest.warns(UserWarning, match="non-overlapping one-hot masks"):
        preds = rankseg.predict(probs)

    assert preds.dtype == torch.bool
    assert torch.equal(preds, expected)


def test_rankseg_acc_multilabel_rejects_invalid_solver(demo_data):
    probs, _ = demo_data
    rankseg = RankSEG(metric="accuracy", solver="invalid", output_mode="multilabel")

    with pytest.raises(ValueError, match="supported solvers: TR, argmax"):
        rankseg.predict(probs)


def test_rankseg_acc_multiclass_rejects_tr(demo_data):
    probs, _ = demo_data
    rankseg = RankSEG(metric="acc", solver="TR", output_mode="multiclass")

    with pytest.raises(ValueError, match="supported solvers: argmax"):
        rankseg.predict(probs)


def test_rankseg_acc_single_channel_rejects_non_tr_solver():
    probs = torch.tensor([[[[0.25, 0.75], [0.50, 0.51]]]], dtype=torch.float32)

    with pytest.raises(ValueError, match="single-channel input; supported solvers: TR"):
        RankSEG(metric="accuracy", solver="RMA", output_mode="multilabel").predict(probs)


def test_rankseg_rejects_unknown_metric(demo_data):
    probs, _ = demo_data

    with pytest.raises(ValueError, match="Unknown metric"):
        RankSEG(metric="f1", solver="RMA", output_mode="multiclass").predict(probs)
