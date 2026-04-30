from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from rankseg.transformers import postprocess, restore_semantic_probs


def _model(config_name: str):
    return SimpleNamespace(config=type(config_name, (), {})())


def _assert_probs_sum_to_one(probs: torch.Tensor, *, atol: float = 1e-5) -> None:
    assert torch.allclose(probs.sum(dim=1), torch.ones_like(probs.sum(dim=1)), atol=atol)


def test_restore_semantic_probs_from_logits_resizes_then_softmax():
    logits = torch.tensor(
        [[[[2.0, 0.0], [0.0, 2.0]], [[0.0, 2.0], [2.0, 0.0]]]],
        dtype=torch.float32,
    )
    outputs = SimpleNamespace(logits=logits)

    probs = restore_semantic_probs(outputs, target_sizes=(4, 4))

    assert probs.shape == (1, 2, 4, 4)
    _assert_probs_sum_to_one(probs)


def test_restore_semantic_probs_from_query_outputs_normalizes_scores():
    class_queries_logits = torch.tensor([[[5.0, 1.0, -2.0], [1.0, 4.0, -2.0]]], dtype=torch.float32)
    masks_queries_logits = torch.tensor(
        [[[[4.0, 0.0], [0.0, 4.0]], [[0.0, 4.0], [4.0, 0.0]]]],
        dtype=torch.float32,
    )
    outputs = SimpleNamespace(
        class_queries_logits=class_queries_logits,
        masks_queries_logits=masks_queries_logits,
    )

    probs = restore_semantic_probs(outputs, target_sizes=(4, 4))

    assert probs.shape == (1, 2, 4, 4)
    _assert_probs_sum_to_one(probs)


def test_restore_semantic_probs_from_detr_requires_model():
    outputs = SimpleNamespace(
        logits=torch.randn(1, 2, 3),
        pred_masks=torch.randn(1, 2, 2, 2),
    )

    with pytest.raises(ValueError, match="`model` is required"):
        restore_semantic_probs(outputs, target_sizes=(4, 4))


def test_restore_semantic_probs_from_detr_drops_null_class():
    outputs = SimpleNamespace(
        logits=torch.tensor([[[6.0, 1.0, -5.0], [1.0, 6.0, -5.0]]], dtype=torch.float32),
        pred_masks=torch.tensor(
            [[[[4.0, 0.0], [0.0, 4.0]], [[0.0, 4.0], [4.0, 0.0]]]],
            dtype=torch.float32,
        ),
    )
    model = _model("DetrConfig")

    probs = restore_semantic_probs(outputs, model=model, target_sizes=(4, 4))

    assert probs.shape == (1, 2, 4, 4)
    _assert_probs_sum_to_one(probs)


def test_restore_semantic_probs_from_conditional_detr_keeps_all_classes():
    outputs = SimpleNamespace(
        logits=torch.tensor([[[5.0, 1.0], [1.0, 5.0]]], dtype=torch.float32),
        pred_masks=torch.tensor(
            [[[[4.0, 0.0], [0.0, 4.0]], [[0.0, 4.0], [4.0, 0.0]]]],
            dtype=torch.float32,
        ),
    )
    model = _model("ConditionalDetrConfig")

    probs = restore_semantic_probs(outputs, model=model, target_sizes=(4, 4))

    assert probs.shape == (1, 2, 4, 4)
    _assert_probs_sum_to_one(probs)


def test_restore_semantic_probs_from_semantic_seg_uses_sigmoid():
    outputs = SimpleNamespace(semantic_seg=torch.tensor([[[[0.0, 2.0], [-2.0, 0.0]]]], dtype=torch.float32))

    probs = restore_semantic_probs(outputs, target_sizes=(4, 4))

    assert probs.shape == (1, 1, 4, 4)
    assert float(probs.min()) >= 0.0
    assert float(probs.max()) <= 1.0


def test_restore_semantic_probs_from_mask2former_matches_official_pre_resize():
    class_queries_logits = torch.tensor([[[6.0, 1.0, -4.0], [1.0, 6.0, -4.0]]], dtype=torch.float32)
    masks_queries_logits = torch.tensor(
        [[[[8.0, -8.0], [-8.0, 8.0]], [[-8.0, 8.0], [8.0, -8.0]]]],
        dtype=torch.float32,
    )
    outputs = SimpleNamespace(
        class_queries_logits=class_queries_logits,
        masks_queries_logits=masks_queries_logits,
    )

    probs = restore_semantic_probs(outputs, model=_model("Mask2FormerConfig"), target_sizes=(5, 5))

    expected_masks = F.interpolate(masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False)
    expected_scores = torch.einsum(
        "bqc,bqhw->bchw",
        class_queries_logits.softmax(dim=-1)[..., :-1],
        expected_masks.sigmoid(),
    )
    expected_scores = F.interpolate(expected_scores, size=(5, 5), mode="bilinear", align_corners=False)
    expected_probs = expected_scores / expected_scores.sum(dim=1, keepdim=True).clamp_min(1e-12)

    assert probs.shape == (1, 2, 5, 5)
    assert torch.allclose(probs, expected_probs, atol=1e-5)


def test_restore_semantic_probs_from_semantic_seg_matches_sam3_official_order():
    semantic_seg = torch.tensor([[[[-4.0, 1.0], [2.0, 8.0]]]], dtype=torch.float32)
    outputs = SimpleNamespace(semantic_seg=semantic_seg)

    probs = restore_semantic_probs(outputs, target_sizes=(4, 5))

    expected = F.interpolate(semantic_seg.sigmoid(), size=(4, 5), mode="bilinear", align_corners=False)
    old_behavior = F.interpolate(semantic_seg, size=(4, 5), mode="bilinear", align_corners=False).sigmoid()

    assert torch.allclose(probs, expected, atol=1e-6)
    assert not torch.allclose(probs, old_behavior, atol=1e-6)


def test_postprocess_defaults_to_multiclass_for_multi_channel_probs():
    outputs = SimpleNamespace(
        logits=torch.tensor(
            [[[[3.0, 0.0], [0.0, 3.0]], [[0.0, 3.0], [3.0, 0.0]]]],
            dtype=torch.float32,
        )
    )

    preds = postprocess(outputs, target_sizes=(4, 4))

    assert preds.shape == (1, 4, 4)


def test_postprocess_defaults_to_multilabel_for_single_channel_probs():
    outputs = SimpleNamespace(semantic_seg=torch.tensor([[[[2.0, -2.0], [-2.0, 2.0]]]], dtype=torch.float32))

    preds = postprocess(outputs, target_sizes=(4, 4))

    assert preds.shape == (1, 1, 4, 4)


def test_restore_semantic_probs_rejects_patch_offsets_path():
    outputs = SimpleNamespace(
        class_queries_logits=torch.randn(1, 2, 3),
        masks_queries_logits=torch.randn(1, 2, 2, 2),
        patch_offsets=[torch.tensor([0, 0, 1, 1])],
    )

    with pytest.raises(ValueError, match="patch merge logic"):
        restore_semantic_probs(outputs, target_sizes=(4, 4))


def test_restore_semantic_probs_rejects_tuple_outputs():
    with pytest.raises(ValueError, match="Tuple-style outputs"):
        restore_semantic_probs((torch.randn(1, 2, 2, 2),), target_sizes=(4, 4))


def test_restore_semantic_probs_rejects_mixed_target_sizes():
    outputs = SimpleNamespace(logits=torch.randn(2, 2, 2, 2))

    with pytest.raises(ValueError, match="different original image sizes"):
        restore_semantic_probs(outputs, target_sizes=[(4, 4), (5, 5)])


def test_restore_semantic_probs_rejects_pred_masks_only_with_specific_message():
    outputs = SimpleNamespace(pred_masks=torch.randn(1, 2, 2, 2))

    with pytest.raises(ValueError, match="model-specific semantic reconstruction"):
        restore_semantic_probs(outputs, target_sizes=(4, 4))
