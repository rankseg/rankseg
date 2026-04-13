import pytest
import torch
import torch.nn.functional as F

from rankseg._rankseg_algo import rankdice_ba, rankseg_rma


def _demo_probs():
    labels = torch.tensor(
        [
            [[0, 0, 1, 1], [0, 2, 2, 1], [0, 2, 1, 1], [0, 0, 1, 2]],
            [[2, 2, 1, 1], [2, 2, 1, 0], [0, 1, 1, 0], [0, 0, 2, 2]],
        ],
        dtype=torch.int64,
    )
    labels_oh = F.one_hot(labels, num_classes=3).movedim(-1, 1).float()
    probs = labels_oh * 0.98 + (1 - labels_oh) * 0.01
    probs[0, :, 0, 0] = torch.tensor([0.80, 0.19, 0.01])
    probs[0, :, 1, 1] = torch.tensor([0.10, 0.05, 0.85])
    probs[1, :, 2, 2] = torch.tensor([0.10, 0.80, 0.10])
    probs[1, :, 0, 3] = torch.tensor([0.10, 0.80, 0.10])
    return probs


def test_rankseg_rma_binary_multiclass_returns_foreground_mask():
    probs = torch.tensor(
        [
            [
                [[0.80, 0.20], [0.30, 0.70]],
                [[0.20, 0.80], [0.70, 0.30]],
            ]
        ],
        dtype=torch.float32,
    )

    preds = rankseg_rma(probs, metric="dice", output_mode="multiclass", pruning_prob=0.0)

    assert preds.shape == (1, 2, 2)
    assert preds.dtype == torch.int64
    assert torch.equal(preds, torch.tensor([[[0, 1], [1, 0]]]))


def test_rankseg_rma_pruning_can_return_all_zero_masks():
    probs = torch.full((1, 3, 2, 2), 0.2, dtype=torch.float32)

    preds = rankseg_rma(probs, metric="iou", output_mode="multilabel", pruning_prob=0.5)

    assert preds.shape == probs.shape
    assert preds.dtype == torch.int64
    assert int(preds.sum().item()) == 0


def test_rankdice_ba_trna_returns_binary_masks():
    probs = _demo_probs()

    preds = rankdice_ba(probs, solver="TRNA")

    assert preds.shape == probs.shape
    assert preds.dtype == torch.bool
    assert int(preds.sum().item()) == 26


def test_rankdice_ba_plus_trna_executes_solver_selection():
    probs = _demo_probs()
    preds_trna = rankdice_ba(probs, solver="TRNA")

    preds = rankdice_ba(probs, solver="BA+TRNA")

    assert preds.shape == probs.shape
    assert preds.dtype == torch.bool
    assert torch.equal(preds, preds_trna)


def test_rankdice_ba_trna_supports_positive_smooth():
    probs = _demo_probs()

    preds = rankdice_ba(probs, solver="TRNA", smooth=0.2)

    assert preds.shape == probs.shape
    assert preds.dtype == torch.bool
    assert int(preds.sum().item()) == 26


def test_rankdice_ba_handles_zero_variance_probabilities():
    probs = torch.tensor(
        [
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0]],
            ]
        ],
        dtype=torch.float32,
    )

    preds = rankdice_ba(probs, solver="BA", pruning_prob=0.0)

    assert preds.shape == probs.shape
    assert preds.dtype == torch.bool
    assert int(preds.sum().item()) >= 0


def test_rankdice_ba_rejects_unknown_solver():
    probs = _demo_probs()

    with pytest.raises(ValueError, match="Unknown solver"):
        rankdice_ba(probs, solver="invalid")
