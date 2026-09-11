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


@pytest.mark.parametrize(
    "original_sizes",
    [
        (3, 5),
        {"height": 3, "width": 5},
        torch.tensor([3, 5], dtype=torch.int32),
        torch.tensor([[3, 5]], dtype=torch.int64),
    ],
)
def test_sam_size_helpers_accept_supported_positive_integer_forms(original_sizes):
    outputs = _outputs(
        "Sam2ImageSegmentationOutput",
        pred_masks=_sam_prompt_masks(),
        iou_scores=torch.ones(1, 1, 1),
    )

    probs = Sam2().restore_mask_probs(outputs, original_sizes=original_sizes)

    assert probs[0].shape[-2:] == (3, 5)


@pytest.mark.parametrize(
    ("original_sizes", "exception", "message"),
    [
        ((3.5, 5), TypeError, "must be an integer"),
        ((True, 5), TypeError, "must be an integer"),
        (torch.tensor([3.0, 5.0]), TypeError, "integer dtype"),
        ((0, 5), ValueError, "must be positive"),
        ((3, -1), ValueError, "must be positive"),
        ((3, 4, 5), ValueError, "height, width"),
        ({"height": 3}, ValueError, "height, width"),
    ],
)
def test_sam_size_helpers_reject_invalid_values(original_sizes, exception, message):
    outputs = _outputs(
        "Sam2ImageSegmentationOutput",
        pred_masks=_sam_prompt_masks(),
        iou_scores=torch.ones(1, 1, 1),
    )

    with pytest.raises(exception, match=message):
        Sam2().restore_mask_probs(outputs, original_sizes=original_sizes)


def test_sam1_rejects_invalid_falsy_pad_size_instead_of_using_default():
    outputs = _outputs(
        "SamImageSegmentationOutput",
        pred_masks=_sam_prompt_masks(),
        iou_scores=torch.ones(1, 1, 1),
    )

    with pytest.raises(ValueError, match="pad_size"):
        Sam1(pad_size=()).restore_mask_probs(
            outputs,
            original_sizes=(2, 2),
            reshaped_input_sizes=(2, 2),
        )


def test_sam1_requires_reshaped_size_to_fit_inside_pad_size():
    outputs = _outputs(
        "SamImageSegmentationOutput",
        pred_masks=_sam_prompt_masks(),
        iou_scores=torch.ones(1, 1, 1),
    )

    with pytest.raises(ValueError, match="fit within `pad_size`"):
        Sam1(pad_size=(4, 4)).restore_mask_probs(
            outputs,
            original_sizes=(2, 2),
            reshaped_input_sizes=(5, 4),
        )


@pytest.mark.parametrize("class_name", ["SamImageSegmentationOutput", "Sam2ImageSegmentationOutput"])
def test_sam_prompt_probability_restoration_does_not_require_iou_scores(class_name):
    pred_masks = _sam_prompt_masks()
    outputs = _outputs(class_name, pred_masks=pred_masks)

    if class_name == "SamImageSegmentationOutput":
        probs = Sam1(pad_size=(2, 2)).restore_mask_probs(
            outputs,
            original_sizes=(2, 2),
            reshaped_input_sizes=(2, 2),
        )
    else:
        probs = Sam2().restore_mask_probs(outputs, original_sizes=(2, 2))

    assert torch.allclose(probs[0], pred_masks[0].sigmoid())


def test_sam_prompt_probability_restoration_ignores_unused_iou_scores():
    pred_masks = _sam_prompt_masks()
    without_iou = _outputs("Sam2ImageSegmentationOutput", pred_masks=pred_masks)
    with_unusable_iou = _outputs(
        "Sam2ImageSegmentationOutput",
        pred_masks=pred_masks,
        iou_scores="not used for mask restoration",
    )

    expected = Sam2().restore_mask_probs(without_iou, original_sizes=(3, 4))
    actual = Sam2().restore_mask_probs(with_unusable_iou, original_sizes=(3, 4))

    assert torch.equal(actual[0], expected[0])


def test_sam_adapters_accept_mapping_outputs():
    prompt_masks = _sam_prompt_masks()
    sam1_probs = Sam1(pad_size=(2, 2)).restore_mask_probs(
        {"pred_masks": prompt_masks},
        original_sizes=(2, 2),
        reshaped_input_sizes=(2, 2),
    )
    sam2_probs = Sam2().restore_mask_probs(
        {"pred_masks": prompt_masks},
        original_sizes=(3, 4),
    )
    sam3_instances = Sam3().restore_instance_mask_probs(
        {
            "pred_logits": torch.ones(1, 1),
            "pred_boxes": torch.zeros(1, 1, 4),
            "pred_masks": torch.zeros(1, 1, 2, 2),
        },
        target_sizes=(3, 4),
    )
    sam3_semantic = Sam3().restore_semantic_mask_probs(
        {"semantic_seg": torch.zeros(1, 1, 2, 2)},
        target_sizes=(3, 4),
    )

    assert sam1_probs[0].shape == (1, 1, 2, 2)
    assert sam2_probs[0].shape == (1, 1, 3, 4)
    assert sam3_instances[0]["mask_probs"].shape == (1, 3, 4)
    assert sam3_semantic[0].shape == (1, 3, 4)


@pytest.mark.parametrize(
    ("adapter", "class_name", "method", "kwargs"),
    [
        (Sam1(pad_size=(2, 2)), "Sam2ImageSegmentationOutput", "restore_mask_probs", {}),
        (Sam2(), "SamImageSegmentationOutput", "restore_mask_probs", {}),
        (Sam2(), "Sam3ImageSegmentationOutput", "restore_mask_probs", {}),
        (Sam3(), "SamImageSegmentationOutput", "restore_instance_mask_probs", {}),
        (Sam3(), "Sam2ImageSegmentationOutput", "restore_semantic_mask_probs", {}),
    ],
)
def test_sam_adapters_reject_identifiable_outputs_from_another_family(adapter, class_name, method, kwargs):
    base_output = type(class_name, (SimpleNamespace,), {})
    outputs = type(f"Wrapped{class_name}", (base_output,), {})()

    if isinstance(adapter, Sam1):
        kwargs = {"original_sizes": [(2, 2)], "reshaped_input_sizes": [(2, 2)]}
    elif isinstance(adapter, Sam2):
        kwargs = {"original_sizes": [(2, 2)]}

    with pytest.raises(ValueError, match="matching SAM-family adapter"):
        getattr(adapter, method)(outputs, **kwargs)


def test_sam_adapters_reject_tuple_outputs_explicitly():
    with pytest.raises(ValueError, match="Tuple-style outputs are not supported"):
        Sam2().restore_mask_probs((_sam_prompt_masks(),), original_sizes=(2, 2))


@pytest.mark.parametrize(
    ("pred_masks", "iou_scores", "message"),
    [
        (torch.randn(1, 2, 2), torch.ones(1), r"shape \(B, N, H, W\)"),
        (torch.empty(1, 0, 2, 2), torch.empty(1, 0), "non-batch dimensions must be non-empty"),
    ],
)
def test_sam_prompt_outputs_reject_invalid_shapes(pred_masks, iou_scores, message):
    outputs = _outputs("Sam2ImageSegmentationOutput", pred_masks=pred_masks, iou_scores=iou_scores)

    with pytest.raises(ValueError, match=message):
        Sam2().restore_mask_probs(outputs, original_sizes=[(2, 2)] * pred_masks.shape[0])


def test_sam2_requires_boolean_non_overlapping_option():
    with pytest.raises(TypeError, match="must be a bool"):
        Sam2(apply_non_overlapping_constraints="false")


def test_sam_prompt_postprocess_rejects_multiclass_candidate_mask_collapse():
    outputs = _outputs(
        "Sam2ImageSegmentationOutput",
        pred_masks=torch.randn(1, 1, 3, 2, 2),
    )
    adapter = Sam2(rankseg_kwargs={"metric": "dice", "output_mode": "multiclass"})

    with pytest.raises(ValueError, match="independent binary masks"):
        adapter.postprocess(outputs, original_sizes=[(2, 2)])


def test_sam_postprocess_requires_exact_multilabel_output_mode():
    outputs = _outputs(
        "Sam2ImageSegmentationOutput",
        pred_masks=torch.zeros(1, 1, 1, 2, 2),
    )
    adapter = Sam2(rankseg_kwargs={"metric": "dice", "output_mode": "Multilabel"})

    with pytest.raises(ValueError, match="output_mode='multilabel'"):
        adapter.postprocess(outputs, original_sizes=[(2, 2)])


@pytest.mark.parametrize("output_mode", [None, 1])
def test_sam_postprocess_requires_string_multilabel_output_mode(output_mode):
    outputs = _outputs(
        "Sam3ImageSegmentationOutput",
        semantic_seg=torch.zeros(1, 1, 2, 2),
    )
    adapter = Sam3(rankseg_kwargs={"metric": "dice", "output_mode": output_mode})

    with pytest.raises(TypeError, match="string 'multilabel'"):
        adapter.postprocess_semantic(outputs)


def test_sam3_semantic_mask_probs_sigmoid_before_resize():
    semantic_seg = torch.tensor([[[[-4.0, 1.0], [2.0, 8.0]]]], dtype=torch.float32)
    outputs = _outputs("Sam3ImageSegmentationOutput", semantic_seg=semantic_seg)

    probs = Sam3().restore_semantic_mask_probs(outputs, target_sizes=[(4, 5)])

    expected = F.interpolate(semantic_seg.sigmoid(), size=(4, 5), mode="bilinear", align_corners=False)

    assert len(probs) == 1
    assert probs[0].shape == (1, 4, 5)
    assert torch.allclose(probs[0], expected[0], atol=1e-6)


@pytest.mark.parametrize(
    ("threshold", "exception", "message"),
    [
        (True, TypeError, "real number"),
        ("0.3", TypeError, "real number"),
        (float("nan"), ValueError, "finite"),
        (float("inf"), ValueError, "finite"),
        (-0.1, ValueError, r"range \[0, 1\]"),
        (1.1, ValueError, r"range \[0, 1\]"),
    ],
)
def test_sam3_rejects_invalid_threshold(threshold, exception, message):
    with pytest.raises(exception, match=message):
        Sam3(threshold=threshold)


def test_sam3_instance_threshold_is_strict():
    outputs = _outputs(
        "Sam3ImageSegmentationOutput",
        pred_logits=torch.tensor([[0.0]]),
        pred_boxes=torch.ones(1, 1, 4),
        pred_masks=torch.ones(1, 1, 2, 2),
    )

    results = Sam3(threshold=0.5).restore_instance_mask_probs(outputs)

    assert results[0]["scores"].numel() == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pred_logits", torch.randn(1, 2, 1), r"shape \(B, Q\)"),
        ("pred_boxes", torch.randn(1, 2, 3), r"shape \(B, Q, 4\)"),
        ("pred_masks", torch.randn(1, 2, 3), r"shape \(B, Q, H, W\)"),
        ("pred_boxes", torch.randn(1, 3, 4), "matching batch and query dimensions"),
        ("pred_masks", torch.empty(1, 2, 0, 3), "spatial dimensions must be non-empty"),
        ("presence_logits", torch.randn(1), r"shape \(B, 1\)"),
    ],
)
def test_sam3_instance_outputs_reject_invalid_shapes(field, value, message):
    values = {
        "pred_logits": torch.randn(1, 2),
        "pred_boxes": torch.randn(1, 2, 4),
        "pred_masks": torch.randn(1, 2, 3, 3),
    }
    values[field] = value
    outputs = _outputs("Sam3ImageSegmentationOutput", **values)

    with pytest.raises(ValueError, match=message):
        Sam3().restore_instance_mask_probs(outputs)


def test_sam3_instance_outputs_require_one_device():
    outputs = _outputs(
        "Sam3ImageSegmentationOutput",
        pred_logits=torch.randn(1, 2),
        pred_boxes=torch.empty(1, 2, 4, device="meta"),
        pred_masks=torch.randn(1, 2, 3, 3),
    )

    with pytest.raises(ValueError, match="same device"):
        Sam3().restore_instance_mask_probs(outputs)


@pytest.mark.parametrize(
    ("semantic_seg", "message"),
    [
        (torch.randn(1, 2, 2), r"shape \(B, 1, H, W\)"),
        (torch.randn(1, 2, 2, 2), r"shape \(B, 1, H, W\)"),
        (torch.empty(1, 1, 0, 2), "spatial dimensions must be non-empty"),
    ],
)
def test_sam3_semantic_outputs_reject_invalid_shapes(semantic_seg, message):
    outputs = _outputs("Sam3ImageSegmentationOutput", semantic_seg=semantic_seg)

    with pytest.raises(ValueError, match=message):
        Sam3().restore_semantic_mask_probs(outputs)


def test_sam3_original_sizes_errors_name_the_used_argument():
    outputs = _outputs(
        "Sam3ImageSegmentationOutput",
        semantic_seg=torch.randn(1, 1, 2, 2),
    )

    with pytest.raises(TypeError, match="original_sizes"):
        Sam3().restore_semantic_mask_probs(outputs, original_sizes=(4.5, 5))


def test_sam3_target_sizes_take_precedence_over_original_sizes():
    outputs = _outputs(
        "Sam3ImageSegmentationOutput",
        semantic_seg=torch.randn(1, 1, 2, 2),
    )

    probs = Sam3().restore_semantic_mask_probs(
        outputs,
        target_sizes=(3, 4),
        original_sizes=(8, 9),
    )

    assert probs[0].shape == (1, 3, 4)


def test_sam_helpers_accept_empty_batches():
    sam1_outputs = _outputs(
        "SamImageSegmentationOutput",
        pred_masks=torch.empty(0, 1, 1, 2, 2),
        iou_scores=torch.empty(0, 1, 1),
    )
    sam2_outputs = _outputs(
        "Sam2ImageSegmentationOutput",
        pred_masks=torch.empty(0, 1, 1, 2, 2),
        iou_scores=torch.empty(0, 1, 1),
    )
    sam3_instance_outputs = _outputs(
        "Sam3ImageSegmentationOutput",
        pred_logits=torch.empty(0, 2),
        pred_boxes=torch.empty(0, 2, 4),
        pred_masks=torch.empty(0, 2, 2, 2),
        presence_logits=torch.empty(0, 1),
    )
    sam3_semantic_outputs = _outputs(
        "Sam3ImageSegmentationOutput",
        semantic_seg=torch.empty(0, 1, 2, 2),
    )

    assert (
        Sam1(pad_size=(2, 2)).restore_mask_probs(
            sam1_outputs,
            original_sizes=[],
            reshaped_input_sizes=[],
        )
        == []
    )
    assert Sam2().restore_mask_probs(sam2_outputs, original_sizes=[]) == []
    assert Sam3().restore_instance_mask_probs(sam3_instance_outputs, target_sizes=[]) == []
    assert Sam3().restore_semantic_mask_probs(sam3_semantic_outputs, target_sizes=[]) == []


def test_sam_helpers_validate_empty_size_tensor_dtype():
    outputs = _outputs(
        "Sam3ImageSegmentationOutput",
        semantic_seg=torch.empty(0, 1, 2, 2),
    )

    with pytest.raises(TypeError, match="integer dtype"):
        Sam3().restore_semantic_mask_probs(outputs, target_sizes=torch.empty(0, 2))

    assert (
        Sam3().restore_semantic_mask_probs(
            outputs,
            target_sizes=torch.empty(0, 2, dtype=torch.int64),
        )
        == []
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sam_probability_restoration_remains_differentiable(dtype):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam1_masks = _sam_prompt_masks().to(device=device, dtype=dtype).requires_grad_()
    sam1_outputs = _outputs(
        "SamImageSegmentationOutput",
        pred_masks=sam1_masks,
        iou_scores=torch.ones((1, 1, 1), device=device, dtype=dtype),
    )
    sam1_probs = Sam1(pad_size=(2, 2)).restore_mask_probs(
        sam1_outputs,
        original_sizes=[(2, 2)],
        reshaped_input_sizes=[(2, 2)],
    )
    sam1_probs[0].sum().backward()

    sam2_masks = _sam_prompt_masks().to(device=device, dtype=dtype).requires_grad_()
    sam2_outputs = _outputs(
        "Sam2ImageSegmentationOutput",
        pred_masks=sam2_masks,
        iou_scores=torch.ones((1, 1, 1), device=device, dtype=dtype),
    )
    sam2_probs = Sam2().restore_mask_probs(sam2_outputs, original_sizes=[(2, 2)])
    sam2_probs[0].sum().backward()

    semantic_logits = torch.randn((1, 1, 2, 2), device=device, dtype=dtype, requires_grad=True)
    semantic_outputs = _outputs("Sam3ImageSegmentationOutput", semantic_seg=semantic_logits)
    semantic_probs = Sam3().restore_semantic_mask_probs(semantic_outputs, target_sizes=[(3, 3)])
    semantic_probs[0].sum().backward()

    instance_logits = torch.tensor([[2.0]], device=device, dtype=dtype, requires_grad=True)
    instance_boxes = torch.ones((1, 1, 4), device=device, dtype=dtype, requires_grad=True)
    instance_masks = torch.randn((1, 1, 2, 2), device=device, dtype=dtype, requires_grad=True)
    instance_outputs = _outputs(
        "Sam3ImageSegmentationOutput",
        pred_logits=instance_logits,
        pred_boxes=instance_boxes,
        pred_masks=instance_masks,
    )
    instance_results = Sam3(threshold=0.5).restore_instance_mask_probs(
        instance_outputs,
        target_sizes=[(3, 3)],
    )
    sum(instance_results[0][name].sum() for name in ("scores", "boxes", "mask_probs")).backward()

    for tensor in (sam1_masks, sam2_masks, semantic_logits, instance_logits, instance_boxes, instance_masks):
        assert tensor.grad is not None
        assert bool(torch.isfinite(tensor.grad).all())


def test_sam_postprocess_entry_points_disable_autograd(monkeypatch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grad_states = []

    original_sam1_restore = Sam1.restore_mask_probs
    original_sam2_restore = Sam2.restore_mask_probs
    original_instance_restore = Sam3.restore_instance_mask_probs
    original_semantic_restore = Sam3.restore_semantic_mask_probs

    def record_sam1(self, *args, **kwargs):
        grad_states.append(torch.is_grad_enabled())
        return original_sam1_restore(self, *args, **kwargs)

    def record_sam2(self, *args, **kwargs):
        grad_states.append(torch.is_grad_enabled())
        return original_sam2_restore(self, *args, **kwargs)

    def record_instance(self, *args, **kwargs):
        grad_states.append(torch.is_grad_enabled())
        return original_instance_restore(self, *args, **kwargs)

    def record_semantic(self, *args, **kwargs):
        grad_states.append(torch.is_grad_enabled())
        return original_semantic_restore(self, *args, **kwargs)

    monkeypatch.setattr(Sam1, "restore_mask_probs", record_sam1)
    monkeypatch.setattr(Sam2, "restore_mask_probs", record_sam2)
    monkeypatch.setattr(Sam3, "restore_instance_mask_probs", record_instance)
    monkeypatch.setattr(Sam3, "restore_semantic_mask_probs", record_semantic)

    sam1_outputs = _outputs(
        "SamImageSegmentationOutput",
        pred_masks=_sam_prompt_masks().to(device).requires_grad_(),
        iou_scores=torch.ones((1, 1, 1), device=device),
    )
    sam2_outputs = _outputs(
        "Sam2ImageSegmentationOutput",
        pred_masks=_sam_prompt_masks().to(device).requires_grad_(),
        iou_scores=torch.ones((1, 1, 1), device=device),
    )
    instance_outputs = _outputs(
        "Sam3ImageSegmentationOutput",
        pred_logits=torch.tensor([[2.0]], device=device, requires_grad=True),
        pred_boxes=torch.ones((1, 1, 4), device=device, requires_grad=True),
        pred_masks=torch.randn((1, 1, 2, 2), device=device, requires_grad=True),
    )
    semantic_outputs = _outputs(
        "Sam3ImageSegmentationOutput",
        semantic_seg=torch.randn((1, 1, 2, 2), device=device, requires_grad=True),
    )

    with torch.enable_grad():
        sam1_preds = Sam1(
            pad_size=(2, 2),
            rankseg_kwargs={"metric": "accuracy", "solver": "TR"},
        ).postprocess(sam1_outputs, original_sizes=[(2, 2)], reshaped_input_sizes=[(2, 2)])
        sam2_preds = Sam2(rankseg_kwargs={"metric": "accuracy", "solver": "TR"}).postprocess(
            sam2_outputs,
            original_sizes=[(2, 2)],
        )
        instance_results = Sam3(
            threshold=0.5,
            rankseg_kwargs={"metric": "accuracy", "solver": "TR"},
        ).postprocess_instance(instance_outputs, target_sizes=[(2, 2)])
        semantic_preds = Sam3(rankseg_kwargs={"metric": "accuracy", "solver": "TR"}).postprocess_semantic(
            semantic_outputs,
            target_sizes=[(2, 2)],
        )

    assert grad_states == [False, False, False, False]
    for tensor in (sam1_preds[0], sam2_preds[0], instance_results[0]["masks"], semantic_preds[0]):
        assert not tensor.requires_grad
    assert not instance_results[0]["scores"].requires_grad
    assert not instance_results[0]["boxes"].requires_grad


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

    preds = Sam2(rankseg_kwargs={"metric": "accuracy", "solver": "TR"}).postprocess(outputs, original_sizes=[(2, 2)])

    expected = torch.tensor([[[[True, False], [False, True]]]])
    assert preds[0].dtype == torch.bool
    assert torch.equal(preds[0], expected)


def test_sam3_postprocess_instance_empty_masks_are_boolean():
    outputs = _outputs(
        "Sam3ImageSegmentationOutput",
        pred_logits=torch.tensor([[-5.0, -6.0]], dtype=torch.float32),
        pred_boxes=torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.2, 0.2]]], dtype=torch.float32),
        pred_masks=torch.randn(1, 2, 2, 3),
    )

    results = Sam3(threshold=0.99).postprocess_instance(outputs, target_sizes=[(4, 6)])

    assert results[0]["masks"].shape == (0, 2, 3)
    assert results[0]["masks"].dtype == torch.bool


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

    results = Sam3(threshold=0.5, rankseg_kwargs={"metric": "accuracy", "solver": "TR"}).postprocess_instance(
        outputs,
        target_sizes=[(2, 2)],
    )

    assert torch.allclose(results[0]["scores"], torch.sigmoid(torch.tensor([3.0])))
    assert set(results[0]) == {"scores", "boxes", "masks"}
    assert torch.allclose(results[0]["boxes"], pred_boxes[0, :1] * torch.tensor([2.0, 2.0, 2.0, 2.0]))
    assert results[0]["masks"].dtype == torch.bool
    assert torch.equal(results[0]["masks"], torch.tensor([[[True, False], [False, True]]]))


def test_sam3_postprocess_semantic_predicts_rankseg_mask():
    outputs = _outputs(
        "Sam3ImageSegmentationOutput",
        semantic_seg=torch.tensor([[[[3.0, -3.0], [-3.0, 3.0]]]], dtype=torch.float32),
    )

    preds = Sam3(rankseg_kwargs={"metric": "accuracy", "solver": "TR"}).postprocess_semantic(
        outputs, target_sizes=[(2, 2)]
    )

    assert preds[0].dtype == torch.bool
    assert torch.equal(preds[0], torch.tensor([[True, False], [False, True]]))


@pytest.mark.parametrize(
    ("input_dtype", "expected_dtype"),
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
@pytest.mark.parametrize("family", ["sam1", "sam2", "sam3_semantic", "sam3_instance"])
def test_sam_probability_restoration_uses_stable_working_dtype(family, input_dtype, expected_dtype):
    if family in {"sam1", "sam2"}:
        outputs = _outputs(
            "SamImageSegmentationOutput" if family == "sam1" else "Sam2ImageSegmentationOutput",
            pred_masks=torch.randn(1, 1, 1, 2, 2, dtype=input_dtype),
            iou_scores=torch.ones(1, 1, 1, dtype=input_dtype),
        )
        if family == "sam1":
            restored = Sam1(pad_size=(2, 2)).restore_mask_probs(
                outputs,
                original_sizes=(3, 3),
                reshaped_input_sizes=(2, 2),
            )[0]
        else:
            restored = Sam2().restore_mask_probs(outputs, original_sizes=(3, 3))[0]
        tensors = (restored,)
    elif family == "sam3_semantic":
        outputs = _outputs(
            "Sam3ImageSegmentationOutput",
            semantic_seg=torch.randn(1, 1, 2, 2, dtype=input_dtype),
        )
        tensors = (Sam3().restore_semantic_mask_probs(outputs, target_sizes=(3, 3))[0],)
    else:
        outputs = _outputs(
            "Sam3ImageSegmentationOutput",
            pred_logits=torch.ones(1, 1, dtype=input_dtype),
            pred_boxes=torch.ones(1, 1, 4, dtype=input_dtype),
            pred_masks=torch.ones(1, 1, 2, 2, dtype=input_dtype),
        )
        result = Sam3(threshold=0.0).restore_instance_mask_probs(outputs, target_sizes=(3, 3))[0]
        tensors = (result["scores"], result["boxes"], result["mask_probs"])

    assert all(tensor.dtype == expected_dtype for tensor in tensors)
    assert all(bool(torch.isfinite(tensor).all()) for tensor in tensors)


@pytest.mark.parametrize("family", ["sam1", "sam2", "sam3_semantic"])
def test_sam_float64_precision_preserves_binary_threshold_boundary(family):
    epsilon = 1e-8
    if family in {"sam1", "sam2"}:
        outputs = _outputs(
            "SamImageSegmentationOutput" if family == "sam1" else "Sam2ImageSegmentationOutput",
            pred_masks=torch.tensor([[[[[epsilon]]]]], dtype=torch.float64),
            iou_scores=torch.ones(1, 1, 1, dtype=torch.float64),
        )
        if family == "sam1":
            adapter = Sam1(
                pad_size=(1, 1),
                rankseg_kwargs={"metric": "accuracy", "solver": "TR"},
            )
            probs = adapter.restore_mask_probs(
                outputs,
                original_sizes=(1, 1),
                reshaped_input_sizes=(1, 1),
            )
            preds = adapter.postprocess(
                outputs,
                original_sizes=(1, 1),
                reshaped_input_sizes=(1, 1),
            )
        else:
            adapter = Sam2(rankseg_kwargs={"metric": "accuracy", "solver": "TR"})
            probs = adapter.restore_mask_probs(outputs, original_sizes=(1, 1))
            preds = adapter.postprocess(outputs, original_sizes=(1, 1))
    else:
        outputs = _outputs(
            "Sam3ImageSegmentationOutput",
            semantic_seg=torch.tensor([[[[epsilon]]]], dtype=torch.float64),
        )
        adapter = Sam3(rankseg_kwargs={"metric": "accuracy", "solver": "TR"})
        probs = adapter.restore_semantic_mask_probs(outputs, target_sizes=(1, 1))
        preds = adapter.postprocess_semantic(outputs, target_sizes=(1, 1))

    assert probs[0].dtype == torch.float64
    assert probs[0].item() > 0.5
    assert preds[0].item()


def test_sam3_instance_float64_precision_preserves_threshold_boundary():
    epsilon = 1e-8
    outputs = _outputs(
        "Sam3ImageSegmentationOutput",
        pred_logits=torch.tensor([[epsilon]], dtype=torch.float64),
        pred_boxes=torch.ones(1, 1, 4, dtype=torch.float64),
        pred_masks=torch.ones(1, 1, 1, 1, dtype=torch.float64),
    )

    result = Sam3(threshold=0.5).restore_instance_mask_probs(outputs)[0]

    assert result["scores"].dtype == torch.float64
    assert result["scores"].numel() == 1
    assert result["scores"].item() > 0.5


def test_sam3_instance_mixed_float_dtypes_preserve_each_result_precision():
    outputs = _outputs(
        "Sam3ImageSegmentationOutput",
        pred_logits=torch.ones(1, 1, dtype=torch.float32),
        presence_logits=torch.ones(1, 1, dtype=torch.float64),
        pred_boxes=torch.ones(1, 1, 4, dtype=torch.float64),
        pred_masks=torch.ones(1, 1, 1, 1, dtype=torch.float32),
    )

    result = Sam3(threshold=0.0).restore_instance_mask_probs(outputs)[0]

    assert result["scores"].dtype == torch.float64
    assert result["boxes"].dtype == torch.float64
    assert result["mask_probs"].dtype == torch.float32


@pytest.mark.parametrize(
    ("family", "field", "value"),
    [
        ("sam1", "pred_masks", torch.ones(1, 1, 1, 1, 1, dtype=torch.int64)),
        ("sam2", "pred_masks", torch.ones(1, 1, 1, 1, 1, dtype=torch.bool)),
        ("sam3_instance", "pred_boxes", torch.ones(1, 1, 4, dtype=torch.complex64)),
        ("sam3_semantic", "semantic_seg", torch.ones(1, 1, 1, 1, dtype=torch.int64)),
    ],
)
def test_sam_restoration_rejects_non_real_floating_outputs(family, field, value):
    if family in {"sam1", "sam2"}:
        values = {
            "pred_masks": torch.ones(1, 1, 1, 1, 1),
            "iou_scores": torch.ones(1, 1, 1),
        }
        values[field] = value
        outputs = _outputs(
            "SamImageSegmentationOutput" if family == "sam1" else "Sam2ImageSegmentationOutput",
            **values,
        )
    elif family == "sam3_instance":
        values = {
            "pred_logits": torch.ones(1, 1),
            "pred_boxes": torch.ones(1, 1, 4),
            "pred_masks": torch.ones(1, 1, 1, 1),
        }
        values[field] = value
        outputs = _outputs("Sam3ImageSegmentationOutput", **values)
    else:
        outputs = _outputs("Sam3ImageSegmentationOutput", **{field: value})

    with pytest.raises(TypeError, match="real floating-point dtype"):
        if family == "sam1":
            Sam1(pad_size=(1, 1)).restore_mask_probs(
                outputs,
                original_sizes=(1, 1),
                reshaped_input_sizes=(1, 1),
            )
        elif family == "sam2":
            Sam2().restore_mask_probs(outputs, original_sizes=(1, 1))
        elif family == "sam3_instance":
            Sam3().restore_instance_mask_probs(outputs)
        else:
            Sam3().restore_semantic_mask_probs(outputs)


@pytest.mark.parametrize(
    ("family", "field", "value"),
    [
        ("sam1", "pred_masks", torch.tensor([[[[[float("nan")]]]]])),
        ("sam2", "pred_masks", torch.tensor([[[[[float("inf")]]]]])),
        ("sam3_instance", "pred_logits", torch.tensor([[float("nan")]])),
        ("sam3_instance", "pred_boxes", torch.tensor([[[0.0, 0.0, float("inf"), 1.0]]])),
        ("sam3_instance", "pred_masks", torch.tensor([[[[float("-inf")]]]])),
        ("sam3_instance", "presence_logits", torch.tensor([[float("nan")]])),
        ("sam3_semantic", "semantic_seg", torch.tensor([[[[float("nan")]]]])),
    ],
)
def test_sam_restoration_rejects_nonfinite_outputs(family, field, value):
    if family in {"sam1", "sam2"}:
        outputs = {"pred_masks": value}
    elif family == "sam3_instance":
        values = {
            "pred_logits": torch.ones(1, 1),
            "pred_boxes": torch.ones(1, 1, 4),
            "pred_masks": torch.ones(1, 1, 1, 1),
        }
        values[field] = value
        outputs = values
    else:
        outputs = {field: value}

    with pytest.raises(ValueError, match=rf"outputs.{field}.*finite"):
        if family == "sam1":
            Sam1(pad_size=(1, 1)).restore_mask_probs(
                outputs,
                original_sizes=(1, 1),
                reshaped_input_sizes=(1, 1),
            )
        elif family == "sam2":
            Sam2().restore_mask_probs(outputs, original_sizes=(1, 1))
        elif family == "sam3_instance":
            Sam3().restore_instance_mask_probs(outputs)
        else:
            Sam3().restore_semantic_mask_probs(outputs)
