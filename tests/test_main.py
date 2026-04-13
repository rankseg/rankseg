import pytest
import torch
import torch.nn.functional as F
from torchmetrics.functional.segmentation import dice_score

from rankseg import RankSEG, rankseg_rma


def _labels_to_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).movedim(-1, 1)


def _multiclass_to_one_hot(preds: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(preds, num_classes=num_classes).movedim(-1, 1)


def _assert_binary_tensor(tensor: torch.Tensor) -> None:
    assert bool(torch.all((tensor == 0) | (tensor == 1)))


def _mean_dice(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    return dice_score(preds, targets, num_classes=num_classes, average="none").nanmean(dim=0).nanmean().item()


def test_rankseg_ba_dice(demo_data):
    probs, labels = demo_data
    num_classes = probs.shape[1]
    labels_oh = _labels_to_one_hot(labels, num_classes)

    preds = RankSEG(metric="dice", solver="BA", output_mode="multilabel").predict(probs)
    mean_dice = _mean_dice(preds, labels_oh, num_classes)

    assert preds.shape == probs.shape
    assert preds.dtype in (torch.bool, torch.int32, torch.int64)
    _assert_binary_tensor(preds)
    assert int(preds.sum().item()) == 19
    assert mean_dice == pytest.approx(0.6026936173, abs=5e-6)


def test_rankseg_rma_overlap(demo_data):
    probs, labels = demo_data
    num_classes = probs.shape[1]
    labels_oh = _labels_to_one_hot(labels, num_classes)

    preds = RankSEG(metric="dice", solver="RMA", output_mode="multilabel").predict(probs)
    mean_dice = _mean_dice(preds, labels_oh, num_classes)

    assert preds.shape == probs.shape
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
    expected = torch.where(probs > 0.5, 1, 0)

    assert preds.shape == probs.shape
    assert preds.dtype == expected.dtype
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


def test_rankseg_rejects_exact_solver(demo_data):
    probs, _ = demo_data

    with pytest.raises(ValueError, match="Exact solver is not implemented yet"):
        RankSEG(metric="dice", solver="exact", output_mode="multilabel").predict(probs)


def test_rankseg_dice_invalid_solver_falls_back_to_rma(demo_data):
    probs, _ = demo_data
    rankseg = RankSEG(metric="dice", solver="invalid", output_mode="multilabel")
    expected = RankSEG(metric="dice", solver="RMA", output_mode="multilabel").predict(probs)

    with pytest.warns(UserWarning, match="uses `RMA`"):
        preds = rankseg.predict(probs)

    assert rankseg.solver == "invalid"
    assert torch.equal(preds, expected)


def test_rankseg_dice_multiclass_ba_falls_back_to_rma(demo_data):
    probs, labels = demo_data
    rankseg = RankSEG(metric="dice", solver="BA", output_mode="multiclass")
    expected = RankSEG(metric="dice", solver="RMA", output_mode="multiclass").predict(probs)

    with pytest.warns(UserWarning, match="uses `RMA`"):
        preds = rankseg.predict(probs)

    assert rankseg.solver == "BA"
    assert preds.shape == labels.shape
    assert torch.equal(preds, expected)


def test_rankseg_iou_invalid_solver_falls_back_to_rma(demo_data):
    probs, labels = demo_data
    rankseg = RankSEG(metric="IoU", solver="BA", output_mode="multiclass")
    expected = RankSEG(metric="IoU", solver="RMA", output_mode="multiclass").predict(probs)

    with pytest.warns(UserWarning, match="only support RMA solver for IoU metric"):
        preds = rankseg.predict(probs)

    assert rankseg.solver == "BA"
    assert preds.shape == labels.shape
    assert preds.dtype == expected.dtype
    assert torch.equal(preds, expected)


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
    assert torch.equal(preds, torch.tensor([[[[0, 1], [0, 1]]]]))


def test_rankseg_acc_multilabel_argmax_warns_and_returns_binary_masks(demo_data):
    probs, _ = demo_data
    rankseg = RankSEG(metric="acc", solver="argmax", output_mode="multilabel")
    expected = torch.zeros_like(probs)
    expected.scatter_(1, torch.argmax(probs, dim=1, keepdim=True), 1)

    with pytest.warns(UserWarning, match="produces non-overlapping predictions"):
        preds = rankseg.predict(probs)

    assert torch.equal(preds, expected)


def test_rankseg_acc_multilabel_invalid_solver_warns_and_falls_back_to_threshold(demo_data):
    probs, _ = demo_data
    rankseg = RankSEG(metric="accuracy", solver="invalid", output_mode="multilabel")
    expected = torch.where(probs > 0.5, 1, 0)

    with pytest.warns(UserWarning, match="uses the TR solver"):
        preds = rankseg.predict(probs)

    assert torch.equal(preds, expected)


def test_rankseg_acc_multiclass_non_argmax_warns_and_uses_argmax(demo_data):
    probs, _ = demo_data
    rankseg = RankSEG(metric="acc", solver="TR", output_mode="multiclass")
    expected = torch.argmax(probs, dim=1)

    with pytest.warns(UserWarning, match="uses the argmax solver"):
        preds = rankseg.predict(probs)

    assert torch.equal(preds, expected)


def test_rankseg_rejects_unknown_metric(demo_data):
    probs, _ = demo_data

    with pytest.raises(ValueError, match="Unknown metric"):
        RankSEG(metric="f1", solver="RMA", output_mode="multiclass").predict(probs)
