from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from rankseg.integration.sam import Sam1, Sam2, Sam3


def _outputs(class_name: str, **kwargs):
    return type(class_name, (SimpleNamespace,), {})(**kwargs)


def _sam_prompt_masks():
    return torch.tensor([[[[[4.0, -4.0], [-2.0, 2.0]]]]], dtype=torch.float32)


def _sam_prompt_masks_4d():
    return torch.tensor([[[[4.0, -4.0], [-2.0, 2.0]]]], dtype=torch.float32)


@pytest.mark.parametrize("class_name", ["SamImageSegmentationOutput", "SamHQImageSegmentationOutput"])
def test_sam1_prompt_outputs_use_padded_geometry(class_name):
    pred_masks = _sam_prompt_masks()
    outputs = _outputs(class_name, pred_masks=pred_masks, iou_scores=torch.ones(1, 1, 1))

    probs = Sam1().restore_mask_probs(outputs, original_sizes=[(3, 5)], reshaped_input_sizes=[(8, 10)])

    expected = F.interpolate(pred_masks[0], size=(1024, 1024), mode="bilinear", align_corners=False)
    expected = expected[..., :8, :10]
    expected = F.interpolate(expected, size=(3, 5), mode="bilinear", align_corners=False).sigmoid()

    assert torch.allclose(probs[0], expected, atol=1e-6)


def test_sam1_prompt_outputs_accept_official_4d_mask_shape():
    pred_masks = _sam_prompt_masks_4d()
    outputs = _outputs("SamImageSegmentationOutput", pred_masks=pred_masks, iou_scores=torch.ones(1, 1))

    probs = Sam1().restore_mask_probs(outputs, original_sizes=[(3, 5)], reshaped_input_sizes=[(8, 10)])

    expected = F.interpolate(pred_masks[0].unsqueeze(0), size=(1024, 1024), mode="bilinear", align_corners=False)
    expected = expected[..., :8, :10]
    expected = F.interpolate(expected, size=(3, 5), mode="bilinear", align_corners=False).sigmoid()

    assert isinstance(probs, list)
    assert probs[0].shape == (1, 1, 3, 5)
    assert torch.allclose(probs[0], expected, atol=1e-6)


def test_sam1_prompt_outputs_require_reshaped_input_sizes():
    outputs = _outputs("SamImageSegmentationOutput", pred_masks=_sam_prompt_masks(), iou_scores=torch.ones(1, 1, 1))

    with pytest.raises(ValueError, match="reshaped_input_sizes"):
        Sam1().restore_mask_probs(outputs, original_sizes=[(4, 4)], reshaped_input_sizes=None)


def test_sam2_prompt_outputs_resize_directly():
    pred_masks = _sam_prompt_masks()
    outputs = _outputs("Sam2ImageSegmentationOutput", pred_masks=pred_masks, iou_scores=torch.ones(1, 1, 1))

    probs = Sam2().restore_mask_probs(outputs, original_sizes=[(3, 5)])

    expected = F.interpolate(pred_masks[0], size=(3, 5), mode="bilinear", align_corners=False).sigmoid()
    assert isinstance(probs, list)
    assert probs[0].shape == (1, 1, 3, 5)
    assert torch.allclose(probs[0], expected, atol=1e-6)


def test_sam2_prompt_outputs_apply_non_overlapping_constraints():
    pred_masks = torch.tensor(
        [[[[[5.0, 1.0], [5.0, 1.0]]], [[[1.0, 5.0], [1.0, 5.0]]]]],
        dtype=torch.float32,
    )
    outputs = _outputs(
        "Sam2ImageSegmentationOutput",
        pred_masks=pred_masks,
        iou_scores=torch.ones(1, 2, 1),
    )

    probs = Sam2(apply_non_overlapping_constraints=True).restore_mask_probs(outputs, original_sizes=[(2, 2)])

    suppressed = torch.sigmoid(torch.tensor(-10.0))
    assert torch.isclose(probs[0][0, 0, 0, 1], suppressed)
    assert torch.isclose(probs[0][1, 0, 0, 0], suppressed)


def test_sam3_semantic_mask_probs_sigmoid_before_resize():
    semantic_seg = torch.tensor([[[[-4.0, 1.0], [2.0, 8.0]]]], dtype=torch.float32)
    outputs = _outputs("Sam3ImageSegmentationOutput", semantic_seg=semantic_seg)

    probs = Sam3().restore_semantic_mask_probs(outputs, target_sizes=[(4, 5)])

    expected = F.interpolate(semantic_seg.sigmoid(), size=(4, 5), mode="bilinear", align_corners=False)

    assert len(probs) == 1
    assert probs[0].shape == (1, 4, 5)
    assert torch.allclose(probs[0], expected[0], atol=1e-6)


def test_sam3_instance_mask_probs_threshold_and_scale_to_target_size():
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

    results = Sam3(threshold=0.5).restore_instance_mask_probs(outputs, target_sizes=[(4, 6)])

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


def test_sam3_instance_empty_results_match_official_shape_contract():
    outputs = _outputs(
        "Sam3ImageSegmentationOutput",
        pred_logits=torch.tensor([[-5.0, -6.0]], dtype=torch.float32),
        pred_boxes=torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.2, 0.2]]], dtype=torch.float32),
        pred_masks=torch.randn(1, 2, 2, 3),
    )

    results = Sam3(threshold=0.99).restore_instance_mask_probs(outputs, target_sizes=[(4, 6)])

    assert len(results) == 1
    assert set(results[0]) == {"scores", "boxes", "mask_probs"}
    assert results[0]["scores"].shape == (0,)
    assert results[0]["boxes"].shape == (0, 4)
    assert results[0]["mask_probs"].shape == (0, 2, 3)


def test_sam2_postprocess_predicts_rankseg_masks():
    outputs = _outputs(
        "Sam2ImageSegmentationOutput",
        pred_masks=torch.tensor([[[[[3.0, -3.0], [-3.0, 3.0]]]]], dtype=torch.float32),
        iou_scores=torch.ones(1, 1, 1),
    )

    preds = Sam2(rankseg_kwargs={"metric": "accuracy"}).postprocess(outputs, original_sizes=[(2, 2)])

    expected = torch.tensor([[[[1, 0], [0, 1]]]])
    assert torch.equal(preds[0], expected)


def test_sam3_postprocess_instance_predicts_rankseg_masks():
    pred_boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.2, 0.2]]], dtype=torch.float32)
    outputs = _outputs(
        "Sam3ImageSegmentationOutput",
        pred_logits=torch.tensor([[3.0, -3.0]], dtype=torch.float32),
        pred_boxes=pred_boxes,
        pred_masks=torch.tensor(
            [[[[3.0, -3.0], [-3.0, 3.0]], [[-3.0, -3.0], [-3.0, -3.0]]]],
            dtype=torch.float32,
        ),
    )

    results = Sam3(threshold=0.5, rankseg_kwargs={"metric": "accuracy"}).postprocess_instance(
        outputs,
        target_sizes=[(2, 2)],
    )

    assert torch.allclose(results[0]["scores"], torch.sigmoid(torch.tensor([3.0])))
    assert set(results[0]) == {"scores", "boxes", "masks"}
    assert torch.allclose(results[0]["boxes"], pred_boxes[0, :1] * torch.tensor([2.0, 2.0, 2.0, 2.0]))
    assert torch.equal(results[0]["masks"], torch.tensor([[[1, 0], [0, 1]]]))


def test_sam3_postprocess_semantic_predicts_rankseg_mask():
    outputs = _outputs(
        "Sam3ImageSegmentationOutput",
        semantic_seg=torch.tensor([[[[3.0, -3.0], [-3.0, 3.0]]]], dtype=torch.float32),
    )

    preds = Sam3(rankseg_kwargs={"metric": "accuracy"}).postprocess_semantic(outputs, target_sizes=[(2, 2)])

    assert torch.equal(preds[0], torch.tensor([[1, 0], [0, 1]]))
