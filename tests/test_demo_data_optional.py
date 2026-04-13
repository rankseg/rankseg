from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torchmetrics.functional.segmentation import dice_score

from rankseg import RankSEG

DATA_DIR = Path(__file__).parent / "data"
DEMO_PROBS = DATA_DIR / "demo_probs.pt"
DEMO_LABELS = DATA_DIR / "demo_labels.pt"


def _labels_to_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    encoded_classes = max(num_classes + 1, int(labels.max().item()) + 1)
    return F.one_hot(labels, num_classes=encoded_classes).movedim(-1, 1)[:, :num_classes]


def _multiclass_to_one_hot(preds: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(preds, num_classes=num_classes).movedim(-1, 1)


def _mean_dice(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    return dice_score(preds, targets, num_classes=num_classes, average="none").nanmean(dim=0).nanmean().item()


@pytest.fixture(scope="session")
def demo_data_from_files():
    if not (DEMO_PROBS.exists() and DEMO_LABELS.exists()):
        pytest.skip("Optional demo-data regression tests require tests/data/demo_probs.pt and demo_labels.pt.")

    probs = torch.load(DEMO_PROBS, weights_only=True)
    labels = torch.load(DEMO_LABELS, weights_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return probs.to(device), labels.to(device)


def test_demo_data_dice_regression(demo_data_from_files):
    probs, labels = demo_data_from_files
    num_classes = probs.shape[1]
    labels_oh = _labels_to_one_hot(labels, num_classes)

    preds_ba = RankSEG(metric="dice", solver="BA", output_mode="multilabel").predict(probs)
    preds_rma_overlap = RankSEG(metric="dice", solver="RMA", output_mode="multilabel").predict(probs)
    preds_rma_multiclass = RankSEG(metric="dice", solver="RMA", output_mode="multiclass").predict(probs)
    preds_rma_multiclass_oh = _multiclass_to_one_hot(preds_rma_multiclass, num_classes)

    assert _mean_dice(preds_ba, labels_oh, num_classes) == pytest.approx(0.5592051148, abs=5e-4)
    assert _mean_dice(preds_rma_overlap, labels_oh, num_classes) == pytest.approx(0.5592266321, abs=5e-4)
    assert _mean_dice(preds_rma_multiclass_oh, labels_oh, num_classes) == pytest.approx(0.5595915914, abs=5e-4)


def test_demo_data_accuracy_regression(demo_data_from_files):
    probs, labels = demo_data_from_files

    preds_argmax = RankSEG(metric="acc", solver="argmax", output_mode="multiclass").predict(probs)
    preds_tr = RankSEG(metric="acc", solver="TR", output_mode="multilabel").predict(probs)

    assert preds_argmax.shape == labels.shape
    assert preds_argmax.dtype == torch.int64
    assert torch.equal(preds_argmax, torch.argmax(probs, dim=1))

    expected_tr = torch.where(probs > 0.5, 1, 0)
    assert preds_tr.shape == probs.shape
    assert preds_tr.dtype == expected_tr.dtype
    assert torch.equal(preds_tr, expected_tr)
