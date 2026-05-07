from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from rankseg.transformers import postprocess, restore_sam_mask_probs, restore_semantic_probs


def _model(config_name: str):
    return SimpleNamespace(config=type(config_name, (), {})())


def _outputs(class_name: str, **kwargs):
    return type(class_name, (SimpleNamespace,), {})(**kwargs)


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


def _sam_prompt_masks():
    return torch.tensor([[[[[4.0, -4.0], [-2.0, 2.0]]]]], dtype=torch.float32)


@pytest.mark.parametrize("class_name", ["SamImageSegmentationOutput", "SamHQImageSegmentationOutput"])
def test_restore_sam1_style_prompt_outputs_use_padded_geometry(class_name):
    pred_masks = _sam_prompt_masks()
    outputs = _outputs(class_name, pred_masks=pred_masks, iou_scores=torch.ones(1, 1, 1))

    probs = restore_sam_mask_probs(outputs, original_sizes=[(3, 5)], reshaped_input_sizes=[(8, 10)])

    expected = F.interpolate(pred_masks[0], size=(1024, 1024), mode="bilinear", align_corners=False)
    expected = expected[..., :8, :10]
    expected = F.interpolate(expected, size=(3, 5), mode="bilinear", align_corners=False).sigmoid()

    assert torch.allclose(probs[0], expected, atol=1e-6)


def test_restore_sam1_style_prompt_outputs_require_reshaped_input_sizes():
    outputs = _outputs("SamImageSegmentationOutput", pred_masks=_sam_prompt_masks(), iou_scores=torch.ones(1, 1, 1))

    with pytest.raises(ValueError, match="reshaped_input_sizes"):
        restore_sam_mask_probs(outputs, original_sizes=[(4, 4)])


def test_restore_sam2_prompt_outputs_resize_directly_and_ignore_reshaped_input_sizes():
    pred_masks = _sam_prompt_masks()
    outputs = _outputs("Sam2ImageSegmentationOutput", pred_masks=pred_masks, iou_scores=torch.ones(1, 1, 1))

    probs = restore_sam_mask_probs(outputs, original_sizes=[(3, 5)], reshaped_input_sizes=[(1, 1)])

    expected = F.interpolate(pred_masks[0], size=(3, 5), mode="bilinear", align_corners=False).sigmoid()
    assert torch.allclose(probs[0], expected, atol=1e-6)


def test_restore_sam2_prompt_outputs_apply_non_overlapping_constraints():
    pred_masks = torch.tensor(
        [[[[[5.0, 1.0], [5.0, 1.0]]], [[[1.0, 5.0], [1.0, 5.0]]]]],
        dtype=torch.float32,
    )
    outputs = _outputs(
        "Sam2ImageSegmentationOutput",
        pred_masks=pred_masks,
        iou_scores=torch.ones(1, 2, 1),
    )

    probs = restore_sam_mask_probs(
        outputs,
        original_sizes=[(2, 2)],
        apply_non_overlapping_constraints=True,
    )

    suppressed = torch.sigmoid(torch.tensor(-10.0))
    assert torch.isclose(probs[0][0, 0, 0, 1], suppressed)
    assert torch.isclose(probs[0][1, 0, 0, 0], suppressed)


def test_restore_sam_mask_probs_from_sam3_semantic_matches_official_order():
    semantic_seg = torch.tensor([[[[-4.0, 1.0], [2.0, 8.0]]]], dtype=torch.float32)
    outputs = _outputs("Sam3ImageSegmentationOutput", semantic_seg=semantic_seg)

    probs = restore_sam_mask_probs(outputs, target_sizes=[(4, 5)])

    expected = F.interpolate(semantic_seg.sigmoid(), size=(4, 5), mode="bilinear", align_corners=False)

    assert isinstance(probs, list)
    assert len(probs) == 1
    assert probs[0].shape == (1, 4, 5)
    assert torch.allclose(probs[0], expected[0], atol=1e-6)


def test_restore_sam_mask_probs_from_sam3_instance_matches_official_pre_threshold_steps():
    pred_logits = torch.tensor([[2.0, -2.0, 1.0]], dtype=torch.float32)
    presence_logits = torch.tensor([[2.0]], dtype=torch.float32)
    pred_boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.1, 0.2, 0.3, 0.4], [0.5, 0.25, 1.0, 0.75]]])
    pred_masks = torch.tensor(
        [[[[4.0, -4.0], [-2.0, 2.0]], [[-3.0, -3.0], [-3.0, -3.0]], [[1.0, 2.0], [3.0, 4.0]]]],
        dtype=torch.float32,
    )
    outputs = _outputs(
        "Sam3ImageSegmentationOutput",
        pred_logits=pred_logits,
        presence_logits=presence_logits,
        pred_boxes=pred_boxes,
        pred_masks=pred_masks,
    )

    results = restore_sam_mask_probs(outputs, target_sizes=[(4, 6)], threshold=0.5)

    scores = pred_logits.sigmoid() * presence_logits.sigmoid()
    keep = scores[0] > 0.5
    expected_boxes = pred_boxes[0, keep] * torch.tensor([6.0, 4.0, 6.0, 4.0])
    expected_masks = F.interpolate(
        pred_masks.sigmoid()[0, keep].unsqueeze(0),
        size=(4, 6),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    assert len(results) == 1
    assert torch.allclose(results[0]["scores"], scores[0, keep])
    assert torch.allclose(results[0]["boxes"], expected_boxes)
    assert torch.allclose(results[0]["mask_probs"], expected_masks)


def test_postprocess_dispatches_sam_prompt_outputs():
    outputs = _outputs(
        "Sam2ImageSegmentationOutput",
        pred_masks=torch.tensor([[[[[3.0, -3.0], [-3.0, 3.0]]]]], dtype=torch.float32),
        iou_scores=torch.ones(1, 1, 1),
    )

    preds = postprocess(outputs, original_sizes=[(2, 2)], rankseg_kwargs={"metric": "accuracy"})

    assert isinstance(preds, list)
    assert preds[0].shape == (1, 1, 2, 2)


def test_postprocess_sam3_full_outputs_default_to_instance_and_can_use_semantic():
    outputs = _outputs(
        "Sam3ImageSegmentationOutput",
        pred_logits=torch.tensor([[3.0, -3.0]], dtype=torch.float32),
        pred_boxes=torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.2, 0.2]]], dtype=torch.float32),
        pred_masks=torch.tensor([[[[3.0, -3.0], [-3.0, 3.0]], [[-3.0, -3.0], [-3.0, -3.0]]]], dtype=torch.float32),
        semantic_seg=torch.tensor([[[[3.0, -3.0], [-3.0, 3.0]]]], dtype=torch.float32),
    )

    instance_results = postprocess(
        outputs,
        target_sizes=[(2, 2)],
        threshold=0.5,
        rankseg_kwargs={"metric": "accuracy"},
    )
    semantic_preds = postprocess(
        outputs,
        sam_task="semantic",
        target_sizes=[(2, 2)],
        rankseg_kwargs={"metric": "accuracy"},
    )

    assert set(instance_results[0]) == {"scores", "boxes", "masks"}
    assert instance_results[0]["masks"].shape == (1, 2, 2)
    assert semantic_preds[0].shape == (2, 2)


def test_postprocess_rejects_unknown_sam_task():
    outputs = SimpleNamespace(semantic_seg=torch.randn(1, 1, 2, 2))

    with pytest.raises(ValueError, match="sam_task"):
        postprocess(outputs, sam_task="panoptic", target_sizes=[(2, 2)])


@pytest.mark.parametrize(
    "outputs",
    [
        SimpleNamespace(pred_masks=torch.randn(1, 1, 1, 2, 2), iou_scores=torch.ones(1, 1, 1)),
        SimpleNamespace(
            pred_logits=torch.tensor([[3.0, -3.0]], dtype=torch.float32),
            pred_boxes=torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.2, 0.2]]], dtype=torch.float32),
            pred_masks=torch.randn(1, 2, 2, 2),
        ),
    ],
)
def test_unknown_sam_like_outputs_do_not_enter_sam_path(outputs):
    with pytest.raises(ValueError, match="model-specific semantic reconstruction"):
        postprocess(outputs, original_sizes=[(2, 2)])


def test_sam_task_requires_supported_sam_output_class():
    outputs = SimpleNamespace(semantic_seg=torch.randn(1, 1, 2, 2))

    with pytest.raises(ValueError, match="supported transformers SAM structured outputs"):
        postprocess(outputs, sam_task="semantic", target_sizes=[(2, 2)])
