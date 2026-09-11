from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import rankseg.integration.transformers as transformers_integration
from rankseg.integration.transformers import postprocess, restore_semantic_probs


def _model(config_name: str):
    return SimpleNamespace(config=type(config_name, (), {})())


def _outputs(class_name: str, **kwargs):
    return type(class_name, (SimpleNamespace,), {})(**kwargs)


def _assert_probs_sum_to_one(probs: torch.Tensor, *, atol: float = 1e-5) -> None:
    assert torch.allclose(probs.sum(dim=0), torch.ones_like(probs.sum(dim=0)), atol=atol)


def test_restore_semantic_probs_from_logits_resizes_then_softmax():
    logits = torch.tensor(
        [[[[2.0, 0.0], [0.0, 2.0]], [[0.0, 2.0], [2.0, 0.0]]]],
        dtype=torch.float32,
    )
    outputs = SimpleNamespace(logits=logits)

    probs = restore_semantic_probs(outputs, target_sizes=[(4, 4)])

    assert len(probs) == 1
    assert probs[0].shape == (2, 4, 4)
    _assert_probs_sum_to_one(probs[0])


@pytest.mark.parametrize("num_channels", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_restore_semantic_probs_remains_differentiable(num_channels, dtype):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logits = torch.randn((1, num_channels, 2, 2), device=device, dtype=dtype, requires_grad=True)

    probs = restore_semantic_probs(SimpleNamespace(logits=logits), target_sizes=[(3, 3)])
    probs[0].square().sum().backward()

    assert probs[0].requires_grad
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())


def test_restore_semantic_probs_from_single_channel_logits_resizes_then_sigmoids():
    logits = torch.tensor([[[[-4.0, 1.0], [2.0, 8.0]]]], dtype=torch.float32)
    outputs = SimpleNamespace(logits=logits)

    probs = restore_semantic_probs(outputs, target_sizes=[(4, 5)])

    resized_logits = F.interpolate(logits, size=(4, 5), mode="bilinear", align_corners=False)
    expected = resized_logits.sigmoid()
    activation_before_resize = F.interpolate(logits.sigmoid(), size=(4, 5), mode="bilinear", align_corners=False)

    assert len(probs) == 1
    assert probs[0].shape == (1, 4, 5)
    assert torch.allclose(probs[0], expected[0], atol=1e-6)
    assert not torch.allclose(probs[0], activation_before_resize[0], atol=1e-6)


def test_postprocess_single_channel_logits_uses_sigmoid_threshold_boundary():
    logits = torch.tensor([[[[-2.0, 0.0], [2.0, -0.1]]]], dtype=torch.float32)
    outputs = SimpleNamespace(logits=logits)

    preds = postprocess(
        outputs,
        rankseg_kwargs={"metric": "accuracy", "solver": "TR"},
    )

    expected = torch.tensor([[[False, False], [True, False]]])
    assert len(preds) == 1
    assert preds[0].dtype == torch.bool
    assert torch.equal(preds[0], expected)


def test_postprocess_disables_autograd_during_probability_restoration(monkeypatch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logits = torch.randn((1, 2, 2, 2), device=device, requires_grad=True)
    outputs = SimpleNamespace(logits=logits)
    expected = postprocess(
        outputs,
        target_sizes=[(3, 3)],
        rankseg_kwargs={"metric": "accuracy", "solver": "argmax"},
    )

    grad_states = []
    original_restore = transformers_integration.restore_semantic_probs

    def record_grad_state(*args, **kwargs):
        grad_states.append(torch.is_grad_enabled())
        return original_restore(*args, **kwargs)

    monkeypatch.setattr(transformers_integration, "restore_semantic_probs", record_grad_state)
    with torch.enable_grad():
        actual = transformers_integration.postprocess(
            outputs,
            target_sizes=[(3, 3)],
            rankseg_kwargs={"metric": "accuracy", "solver": "argmax"},
        )

    assert grad_states == [False]
    assert not actual[0].requires_grad
    assert torch.equal(actual[0], expected[0])


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

    probs = restore_semantic_probs(outputs, target_sizes=[(4, 4)])

    assert len(probs) == 1
    assert probs[0].shape == (2, 4, 4)
    _assert_probs_sum_to_one(probs[0])


@pytest.mark.parametrize("family", ["query", "detr"])
def test_query_normalization_preserves_tiny_nonzero_float64_scores(family):
    class_logits = torch.tensor([[[-700.0, -701.0, 0.0]]], dtype=torch.float64)
    mask_logits = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
    if family == "query":
        outputs = SimpleNamespace(class_queries_logits=class_logits, masks_queries_logits=mask_logits)
    else:
        outputs = SimpleNamespace(logits=class_logits, pred_masks=mask_logits)

    probs = restore_semantic_probs(outputs)[0]

    semantic_scores = class_logits.softmax(dim=-1)[..., :-1] * mask_logits.sigmoid()[..., 0, 0, None]
    expected = semantic_scores[0, 0] / semantic_scores[0, 0].sum()
    assert torch.equal(probs[:, 0, 0], expected)
    assert probs[:, 0, 0].sum().item() == 1.0


@pytest.mark.parametrize("family", ["query", "detr"])
def test_query_normalization_maps_exact_zero_total_to_zero(family):
    class_logits = torch.tensor([[[-1000.0, -1001.0, 0.0]]], dtype=torch.float64)
    mask_logits = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
    if family == "query":
        outputs = SimpleNamespace(class_queries_logits=class_logits, masks_queries_logits=mask_logits)
    else:
        outputs = SimpleNamespace(logits=class_logits, pred_masks=mask_logits)

    probs = restore_semantic_probs(outputs)[0]

    assert torch.equal(probs, torch.zeros_like(probs))
    assert bool(torch.isfinite(probs).all())


@pytest.mark.parametrize("family", ["query", "detr"])
def test_restore_semantic_probs_from_single_class_query_outputs_preserves_spatial_scores(family):
    class_logits = torch.tensor([[[2.0, 0.0]]], dtype=torch.float32)
    mask_logits = torch.tensor([[[[-3.0, 0.0], [1.0, 3.0]]]], dtype=torch.float32)
    if family == "query":
        outputs = SimpleNamespace(class_queries_logits=class_logits, masks_queries_logits=mask_logits)
        model = None
    else:
        outputs = SimpleNamespace(logits=class_logits, pred_masks=mask_logits)
        model = _model("DetrConfig")

    probs = restore_semantic_probs(outputs, model=model)

    expected = class_logits.softmax(dim=-1)[..., :1]
    expected = torch.einsum("bqc,bqhw->bchw", expected, mask_logits.sigmoid())
    assert probs[0].shape == (1, 2, 2)
    assert torch.allclose(probs[0], expected[0], atol=1e-6)
    assert not torch.equal(probs[0], torch.ones_like(probs[0]))


@pytest.mark.parametrize("family", ["query", "detr"])
def test_restore_semantic_probs_from_single_class_query_outputs_clamps_overlap_excess(family):
    class_logits = torch.tensor([[[20.0, -20.0], [20.0, -20.0]]], dtype=torch.float32)
    mask_logits = torch.full((1, 2, 2, 2), 20.0)
    if family == "query":
        outputs = SimpleNamespace(class_queries_logits=class_logits, masks_queries_logits=mask_logits)
        model = None
    else:
        outputs = SimpleNamespace(logits=class_logits, pred_masks=mask_logits)
        model = _model("DetrConfig")

    probs = restore_semantic_probs(outputs, model=model)

    assert torch.equal(probs[0], torch.ones_like(probs[0]))
    assert bool(((probs[0] >= 0) & (probs[0] <= 1)).all())


@pytest.mark.parametrize("family", ["query", "detr"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_single_class_query_probability_restoration_remains_differentiable(family, dtype):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_logits = torch.tensor([[[0.4, -0.4]]], dtype=dtype, device=device, requires_grad=True)
    mask_logits = torch.tensor([[[[-0.8, -0.2], [0.2, 0.8]]]], dtype=dtype, device=device, requires_grad=True)
    if family == "query":
        outputs = SimpleNamespace(class_queries_logits=class_logits, masks_queries_logits=mask_logits)
        model = None
    else:
        outputs = SimpleNamespace(logits=class_logits, pred_masks=mask_logits)
        model = _model("DetrConfig")

    probs = restore_semantic_probs(outputs, model=model)
    probs[0].sum().backward()

    assert class_logits.grad is not None
    assert mask_logits.grad is not None
    assert bool(torch.isfinite(class_logits.grad).all())
    assert bool(torch.isfinite(mask_logits.grad).all())


def test_restore_semantic_probs_from_detr_does_not_require_model():
    outputs = SimpleNamespace(
        logits=torch.tensor([[[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]]]),
        pred_masks=torch.tensor(
            [[[[2.0, -2.0], [-2.0, 2.0]], [[-2.0, 2.0], [2.0, -2.0]]]],
        ),
    )

    probs = restore_semantic_probs(outputs, target_sizes=[(4, 4)])

    expected_scores = torch.einsum(
        "bqc,bqhw->bchw",
        outputs.logits.softmax(dim=-1)[..., :-1],
        outputs.pred_masks.sigmoid(),
    )
    expected_scores = F.interpolate(expected_scores, size=(4, 4), mode="bilinear", align_corners=False)
    expected = expected_scores / expected_scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
    assert len(probs) == 1
    assert torch.allclose(probs[0], expected[0], atol=1e-6)


def test_restore_semantic_probs_from_detr_drops_null_class():
    outputs = SimpleNamespace(
        logits=torch.tensor([[[6.0, 1.0, -5.0], [1.0, 6.0, -5.0]]], dtype=torch.float32),
        pred_masks=torch.tensor(
            [[[[4.0, 0.0], [0.0, 4.0]], [[0.0, 4.0], [4.0, 0.0]]]],
            dtype=torch.float32,
        ),
    )
    probs = restore_semantic_probs(outputs, target_sizes=[(4, 4)])

    assert len(probs) == 1
    assert probs[0].shape == (2, 4, 4)
    _assert_probs_sum_to_one(probs[0])


def test_restore_semantic_probs_from_conditional_detr_drops_null_class():
    outputs = SimpleNamespace(
        logits=torch.tensor([[[5.0, 1.0, -5.0], [1.0, 5.0, -5.0]]], dtype=torch.float32),
        pred_masks=torch.tensor(
            [[[[4.0, 0.0], [0.0, 4.0]], [[0.0, 4.0], [4.0, 0.0]]]],
            dtype=torch.float32,
        ),
    )
    probs = restore_semantic_probs(outputs, target_sizes=[(4, 4)])

    assert len(probs) == 1
    assert probs[0].shape == (2, 4, 4)
    _assert_probs_sum_to_one(probs[0])


def test_conditional_detr_dominant_null_class_cannot_be_predicted():
    outputs = SimpleNamespace(
        logits=torch.tensor([[[0.0, -1.0, 8.0]]], dtype=torch.float32),
        pred_masks=torch.full((1, 1, 2, 2), 8.0),
    )
    probs = restore_semantic_probs(outputs)
    preds = postprocess(
        outputs,
        rankseg_kwargs={"metric": "accuracy", "solver": "argmax"},
    )

    assert probs[0].shape == (2, 2, 2)
    assert torch.equal(preds[0], torch.zeros((2, 2), dtype=torch.int64))


@pytest.mark.parametrize("identifier", ["exact_config", "derived_config", "model_type", "derived_output"])
def test_restore_semantic_probs_from_mask2former_matches_official_pre_resize(identifier):
    class_queries_logits = torch.tensor([[[6.0, 1.0, -4.0], [1.0, 6.0, -4.0]]], dtype=torch.float32)
    masks_queries_logits = torch.tensor(
        [[[[8.0, -8.0], [-8.0, 8.0]], [[-8.0, 8.0], [8.0, -8.0]]]],
        dtype=torch.float32,
    )
    fields = {
        "class_queries_logits": class_queries_logits,
        "masks_queries_logits": masks_queries_logits,
    }
    model = None
    if identifier == "exact_config":
        outputs = SimpleNamespace(**fields)
        model = _model("Mask2FormerConfig")
    elif identifier == "derived_config":
        base_config = type("Mask2FormerConfig", (), {})
        derived_config = type("ProjectMask2FormerConfig", (base_config,), {})
        outputs = SimpleNamespace(**fields)
        model = SimpleNamespace(config=derived_config())
    elif identifier == "model_type":
        outputs = SimpleNamespace(**fields)
        model = SimpleNamespace(config=SimpleNamespace(model_type="mask2former"))
    else:
        base_output = type("Mask2FormerForUniversalSegmentationOutput", (SimpleNamespace,), {})
        derived_output = type("WrappedMask2FormerOutput", (base_output,), {})
        outputs = derived_output(**fields)

    probs = restore_semantic_probs(outputs, model=model, target_sizes=[(5, 5)])

    expected_masks = F.interpolate(masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False)
    expected_scores = torch.einsum(
        "bqc,bqhw->bchw",
        class_queries_logits.softmax(dim=-1)[..., :-1],
        expected_masks.sigmoid(),
    )
    expected_scores = F.interpolate(expected_scores, size=(5, 5), mode="bilinear", align_corners=False)
    expected_probs = expected_scores / expected_scores.sum(dim=1, keepdim=True).clamp_min(1e-12)

    assert len(probs) == 1
    assert probs[0].shape == (2, 5, 5)
    assert torch.allclose(probs[0], expected_probs[0], atol=1e-5)


def test_transformers_helper_rejects_sam_output_subclasses():
    base_output = type("Sam2ImageSegmentationOutput", (SimpleNamespace,), {})
    derived_output = type("WrappedSam2Output", (base_output,), {})
    outputs = derived_output(logits=torch.zeros(1, 1, 2, 2))

    with pytest.raises(ValueError, match="SAM-family outputs require"):
        restore_semantic_probs(outputs)


@pytest.mark.parametrize(
    "outputs",
    [
        {"semantic_seg": torch.randn(1, 1, 2, 2)},
        SimpleNamespace(semantic_seg=torch.randn(1, 1, 2, 2)),
        {
            "pred_logits": torch.randn(1, 2),
            "pred_boxes": torch.randn(1, 2, 4),
            "pred_masks": torch.randn(1, 2, 2, 2),
        },
        {
            "iou_scores": torch.randn(1, 1, 1),
            "pred_masks": torch.randn(1, 1, 1, 2, 2),
        },
    ],
)
def test_transformers_helper_rejects_structurally_identifiable_sam_outputs(outputs):
    with pytest.raises(ValueError, match="rankseg.integration.sam"):
        restore_semantic_probs(outputs)


def test_postprocess_rejects_sam_outputs_with_adapter_message():
    outputs = _outputs(
        "Sam3ImageSegmentationOutput",
        semantic_seg=torch.randn(1, 1, 2, 2),
        pred_logits=torch.randn(1, 2),
        pred_boxes=torch.randn(1, 2, 4),
        pred_masks=torch.randn(1, 2, 2, 2),
    )

    with pytest.raises(ValueError, match="rankseg.integration.sam"):
        postprocess(outputs, target_sizes=[(4, 4)])


def test_postprocess_defaults_to_multiclass_for_multi_channel_probs():
    outputs = SimpleNamespace(
        logits=torch.tensor(
            [[[[3.0, 0.0], [0.0, 3.0]], [[0.0, 3.0], [3.0, 0.0]]]],
            dtype=torch.float32,
        )
    )

    preds = postprocess(outputs, target_sizes=[(4, 4)])

    assert len(preds) == 1
    assert preds[0].shape == (4, 4)
    assert preds[0].dtype == torch.int64


def test_postprocess_defaults_to_multilabel_for_single_channel_probs():
    outputs = SimpleNamespace(logits=torch.tensor([[[[2.0, -2.0], [-2.0, 2.0]]]], dtype=torch.float32))

    preds = postprocess(outputs, target_sizes=[(4, 4)])

    assert len(preds) == 1
    assert preds[0].shape == (1, 4, 4)
    assert preds[0].dtype == torch.bool


@pytest.mark.parametrize("identifier", ["native", "derived", "field_none", "field_value", "model_type"])
def test_restore_semantic_probs_rejects_all_eomt_paths(identifier):
    fields = {
        "class_queries_logits": torch.randn(1, 2, 3),
        "masks_queries_logits": torch.randn(1, 2, 2, 2),
    }
    model = None
    if identifier == "native":
        outputs = _outputs("EomtForUniversalSegmentationOutput", **fields)
    elif identifier == "derived":
        base_output = type("EomtForUniversalSegmentationOutput", (SimpleNamespace,), {})
        outputs = type("WrappedEomtOutput", (base_output,), {})(**fields)
    elif identifier == "field_none":
        outputs = {**fields, "patch_offsets": None}
    elif identifier == "field_value":
        outputs = SimpleNamespace(**fields, patch_offsets=[torch.tensor([0, 0, 1, 1])])
    else:
        outputs = SimpleNamespace(**fields)
        model = SimpleNamespace(config={"model_type": "eomt"})

    with pytest.raises(ValueError, match="processor-specific resizing.*patch merge"):
        restore_semantic_probs(outputs, model=model, target_sizes=[(4, 4)])


def test_restore_semantic_probs_rejects_tuple_outputs():
    with pytest.raises(ValueError, match="Tuple-style outputs"):
        restore_semantic_probs((torch.randn(1, 2, 2, 2),), target_sizes=[(4, 4)])


def test_restore_semantic_probs_accepts_per_image_target_sizes():
    outputs = SimpleNamespace(logits=torch.randn(2, 2, 2, 2))

    probs = restore_semantic_probs(outputs, target_sizes=[(4, 4), (5, 5)])

    assert len(probs) == 2
    assert probs[0].shape == (2, 4, 4)
    assert probs[1].shape == (2, 5, 5)


def test_restore_semantic_probs_accepts_integer_tensor_target_sizes():
    outputs = SimpleNamespace(logits=torch.randn(2, 2, 2, 2))

    probs = restore_semantic_probs(outputs, target_sizes=torch.tensor([[3, 4], [5, 6]], dtype=torch.int64))

    assert [prob.shape for prob in probs] == [(2, 3, 4), (2, 5, 6)]


@pytest.mark.parametrize(
    ("target_sizes", "exception", "message"),
    [
        ([(4.5, 5)], TypeError, "must be an integer"),
        ([(True, 5)], TypeError, "must be an integer"),
        (torch.tensor([[4.0, 5.0]]), TypeError, "integer dtype"),
        ([(0, 5)], ValueError, "must be positive"),
        ([(4, -1)], ValueError, "must be positive"),
        ([(4, 5, 6)], ValueError, "positive-integer"),
    ],
)
def test_restore_semantic_probs_rejects_invalid_target_sizes(target_sizes, exception, message):
    outputs = SimpleNamespace(logits=torch.randn(1, 2, 2, 2))

    with pytest.raises(exception, match=message):
        restore_semantic_probs(outputs, target_sizes=target_sizes)


@pytest.mark.parametrize(
    ("logits", "message"),
    [
        (torch.randn(1, 2, 2), r"shape \(B, C, H, W\)"),
        (torch.empty(1, 0, 2, 2), "at least one class channel"),
        (torch.empty(1, 2, 0, 2), "spatial dimensions must be non-empty"),
    ],
)
def test_restore_semantic_probs_rejects_invalid_dense_logits_shape(logits, message):
    with pytest.raises(ValueError, match=message):
        restore_semantic_probs(SimpleNamespace(logits=logits))


@pytest.mark.parametrize(
    ("class_logits", "mask_logits", "message"),
    [
        (torch.randn(1, 2), torch.randn(1, 2, 3, 3), r"shape \(B, Q, C \+ 1\)"),
        (torch.randn(1, 2, 3), torch.randn(1, 2, 3), r"shape \(B, Q, H, W\)"),
        (torch.randn(1, 2, 1), torch.randn(1, 2, 3, 3), "semantic class and the no-object class"),
        (torch.randn(2, 2, 3), torch.randn(1, 2, 3, 3), "matching batch and query dimensions"),
        (torch.randn(1, 3, 3), torch.randn(1, 2, 3, 3), "matching batch and query dimensions"),
        (torch.randn(1, 2, 3), torch.empty(1, 2, 0, 3), "spatial dimensions must be non-empty"),
    ],
)
def test_restore_semantic_probs_rejects_invalid_query_shapes(class_logits, mask_logits, message):
    outputs = SimpleNamespace(class_queries_logits=class_logits, masks_queries_logits=mask_logits)

    with pytest.raises(ValueError, match=message):
        restore_semantic_probs(outputs)


def test_restore_semantic_probs_rejects_query_tensors_on_different_devices():
    outputs = SimpleNamespace(
        class_queries_logits=torch.randn(1, 2, 3),
        masks_queries_logits=torch.empty(1, 2, 3, 3, device="meta"),
    )

    with pytest.raises(ValueError, match="same device"):
        restore_semantic_probs(outputs)


@pytest.mark.parametrize(
    "outputs",
    [
        SimpleNamespace(logits=torch.empty(0, 2, 3, 3)),
        SimpleNamespace(
            class_queries_logits=torch.empty(0, 2, 3),
            masks_queries_logits=torch.empty(0, 2, 3, 3),
        ),
    ],
)
def test_restore_semantic_probs_accepts_empty_batch(outputs):
    assert restore_semantic_probs(outputs, target_sizes=[]) == []


def test_restore_semantic_probs_validates_empty_target_size_tensor_dtype():
    outputs = SimpleNamespace(logits=torch.empty(0, 2, 3, 3))

    with pytest.raises(TypeError, match="integer dtype"):
        restore_semantic_probs(outputs, target_sizes=torch.empty(0, 2))

    assert restore_semantic_probs(outputs, target_sizes=torch.empty(0, 2, dtype=torch.int64)) == []


def test_restore_semantic_probs_rejects_pred_masks_only_with_specific_message():
    outputs = SimpleNamespace(pred_masks=torch.randn(1, 2, 2, 2))

    with pytest.raises(ValueError, match="model-specific semantic reconstruction"):
        restore_semantic_probs(outputs, target_sizes=[(4, 4)])


@pytest.mark.parametrize(
    ("input_dtype", "expected_dtype"),
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
@pytest.mark.parametrize("family", ["dense", "query", "detr"])
def test_transformers_probability_restoration_uses_stable_working_dtype(family, input_dtype, expected_dtype):
    if family == "dense":
        outputs = SimpleNamespace(logits=torch.randn(1, 2, 2, 2, dtype=input_dtype))
        model = None
    elif family == "query":
        outputs = SimpleNamespace(
            class_queries_logits=torch.randn(1, 2, 3, dtype=input_dtype),
            masks_queries_logits=torch.randn(1, 2, 2, 2, dtype=input_dtype),
        )
        model = None
    else:
        outputs = SimpleNamespace(
            logits=torch.randn(1, 2, 3, dtype=input_dtype),
            pred_masks=torch.randn(1, 2, 2, 2, dtype=input_dtype),
        )
        model = _model("DetrConfig")

    probs = restore_semantic_probs(outputs, model=model, target_sizes=[(3, 3)])

    assert probs[0].dtype == expected_dtype
    assert bool(torch.isfinite(probs[0]).all())


def test_transformers_dense_float64_precision_changes_binary_and_multiclass_boundaries():
    epsilon = 1e-8
    binary_outputs = SimpleNamespace(logits=torch.tensor([[[[epsilon]]]], dtype=torch.float64))
    multiclass_outputs = SimpleNamespace(
        logits=torch.tensor([[[[0.0]], [[epsilon]]]], dtype=torch.float64),
    )

    binary_probs = restore_semantic_probs(binary_outputs)
    binary_preds = postprocess(
        binary_outputs,
        rankseg_kwargs={"metric": "accuracy", "solver": "TR"},
    )
    multiclass_probs = restore_semantic_probs(multiclass_outputs)
    multiclass_preds = postprocess(
        multiclass_outputs,
        rankseg_kwargs={"metric": "accuracy", "solver": "argmax"},
    )

    assert binary_probs[0].dtype == torch.float64
    assert binary_probs[0].item() > 0.5
    assert binary_preds[0].item()
    assert multiclass_probs[0].dtype == torch.float64
    assert multiclass_probs[0][1, 0, 0] > multiclass_probs[0][0, 0, 0]
    assert multiclass_preds[0].item() == 1


@pytest.mark.parametrize("family", ["query", "detr"])
def test_transformers_query_float64_precision_preserves_near_tie_class(family):
    epsilon = 1e-8
    class_logits = torch.tensor([[[0.0, epsilon, -100.0]]], dtype=torch.float64)
    mask_logits = torch.full((1, 1, 1, 1), 100.0, dtype=torch.float64)
    if family == "query":
        outputs = SimpleNamespace(class_queries_logits=class_logits, masks_queries_logits=mask_logits)
        model = None
    else:
        outputs = SimpleNamespace(logits=class_logits, pred_masks=mask_logits)
        model = _model("DetrConfig")

    probs = restore_semantic_probs(outputs, model=model)
    preds = postprocess(
        outputs,
        model=model,
        rankseg_kwargs={"metric": "accuracy", "solver": "argmax"},
    )

    assert probs[0].dtype == torch.float64
    assert probs[0][1, 0, 0] > probs[0][0, 0, 0]
    assert preds[0].item() == 1


@pytest.mark.parametrize("family", ["query", "detr"])
def test_transformers_query_mixed_float_dtypes_promote_to_float64(family):
    class_logits = torch.randn(1, 2, 3, dtype=torch.float64)
    mask_logits = torch.randn(1, 2, 2, 2, dtype=torch.float32)
    if family == "query":
        outputs = SimpleNamespace(class_queries_logits=class_logits, masks_queries_logits=mask_logits)
        model = None
    else:
        outputs = SimpleNamespace(logits=class_logits, pred_masks=mask_logits)
        model = _model("DetrConfig")

    probs = restore_semantic_probs(outputs, model=model)

    assert probs[0].dtype == torch.float64


@pytest.mark.parametrize(
    "outputs",
    [
        SimpleNamespace(logits=torch.ones(1, 2, 2, 2, dtype=torch.int64)),
        SimpleNamespace(
            class_queries_logits=torch.ones(1, 2, 3, dtype=torch.complex64),
            masks_queries_logits=torch.ones(1, 2, 2, 2),
        ),
        SimpleNamespace(
            logits=torch.ones(1, 2, 3),
            pred_masks=torch.ones(1, 2, 2, 2, dtype=torch.int64),
        ),
    ],
)
def test_transformers_restoration_rejects_non_real_floating_logits(outputs):
    model = _model("DetrConfig") if hasattr(outputs, "pred_masks") else None

    with pytest.raises(TypeError, match="real floating-point dtype"):
        restore_semantic_probs(outputs, model=model)


@pytest.mark.parametrize(
    ("outputs", "field"),
    [
        (SimpleNamespace(logits=torch.tensor([[[[float("nan")]]]])), "outputs.logits"),
        (
            SimpleNamespace(logits=torch.tensor([[[[float("inf")]], [[0.0]]]])),
            "outputs.logits",
        ),
        (
            SimpleNamespace(
                class_queries_logits=torch.tensor([[[float("inf"), 0.0, -1.0]]]),
                masks_queries_logits=torch.zeros(1, 1, 1, 1),
            ),
            "outputs.class_queries_logits",
        ),
        (
            SimpleNamespace(
                class_queries_logits=torch.zeros(1, 1, 2),
                masks_queries_logits=torch.tensor([[[[float("nan")]]]]),
            ),
            "outputs.masks_queries_logits",
        ),
        (
            SimpleNamespace(
                logits=torch.zeros(1, 1, 2),
                pred_masks=torch.tensor([[[[float("inf")]]]]),
            ),
            "outputs.pred_masks",
        ),
    ],
)
def test_transformers_restoration_rejects_nonfinite_outputs(outputs, field):
    with pytest.raises(ValueError, match=rf"{field}.*finite"):
        restore_semantic_probs(outputs)
