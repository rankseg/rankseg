"""Helpers for standard Hugging Face Transformers segmentation outputs.

This module restores semantic probability maps from supported Transformers
output objects and applies :class:`rankseg.RankSEG` as the final prediction
step. SAM-family outputs are handled by :mod:`rankseg.integration.sam`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import torch
import torch.nn.functional as F

from .._rankseg import RankSEG
from .._validation import SUPPORTED_PROB_DTYPES

_TARGET_SIZES_ERROR = (
    "`target_sizes` must be None, a list of positive-integer (height, width) pairs, "
    "or an integer tensor of shape (B, 2)."
)
_SAM_OUTPUT_CLASSES = {
    "SamImageSegmentationOutput",
    "SamHQImageSegmentationOutput",
    "Sam2ImageSegmentationOutput",
    "Sam3ImageSegmentationOutput",
    "Sam3LiteTextImageSegmentationOutput",
}


def _get_output_value(outputs, name: str):
    if isinstance(outputs, (tuple, list)):
        raise ValueError("Tuple-style outputs are not supported. Pass structured outputs with named fields.")

    value = getattr(outputs, name, None)
    if value is not None:
        return value

    if isinstance(outputs, Mapping):
        return outputs.get(name)

    return None


def _has_output_field(outputs, name: str) -> bool:
    if isinstance(outputs, Mapping):
        return name in outputs
    return hasattr(outputs, name)


def _normalize_target_sizes(target_sizes, batch_size: int) -> list[tuple[int, int]] | None:
    if target_sizes is None:
        return None

    if isinstance(target_sizes, torch.Tensor):
        if target_sizes.ndim != 2 or target_sizes.shape[1] != 2:
            raise ValueError(_TARGET_SIZES_ERROR)
        if target_sizes.dtype == torch.bool or target_sizes.is_floating_point() or target_sizes.is_complex():
            raise TypeError("`target_sizes` tensor must have an integer dtype.")
        target_sizes = target_sizes.tolist()

    if isinstance(target_sizes, Sequence) and not isinstance(target_sizes, (str, bytes)):
        if len(target_sizes) != batch_size:
            raise ValueError("`target_sizes` must contain one (height, width) pair per batch item.")
        normalized = []
        for size in target_sizes:
            if isinstance(size, torch.Tensor):
                if size.ndim != 1 or size.numel() != 2:
                    raise ValueError(_TARGET_SIZES_ERROR)
                size = size.tolist()
            elif isinstance(size, Sequence) and not isinstance(size, (str, bytes)):
                if len(size) != 2:
                    raise ValueError(_TARGET_SIZES_ERROR)
            else:
                raise ValueError(_TARGET_SIZES_ERROR)

            if any(isinstance(value, bool) or not isinstance(value, Integral) for value in size):
                raise TypeError("Each `target_sizes` height and width must be an integer.")
            if any(value <= 0 for value in size):
                raise ValueError("Each `target_sizes` height and width must be positive.")
            normalized.append((int(size[0]), int(size[1])))
        return normalized

    raise ValueError(_TARGET_SIZES_ERROR)


def _validate_dense_logits(logits: torch.Tensor, name: str) -> None:
    _validate_float_tensor(logits, name)
    if logits.ndim != 4:
        raise ValueError(f"`{name}` must have shape (B, C, H, W).")
    if logits.shape[1] == 0:
        raise ValueError(f"`{name}` must contain at least one class channel.")
    if any(size == 0 for size in logits.shape[-2:]):
        raise ValueError(f"`{name}` spatial dimensions must be non-empty.")
    _validate_finite_tensor(logits, name)


def _validate_query_logits(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    *,
    class_name: str,
    mask_name: str,
) -> None:
    _validate_float_tensor(class_logits, class_name)
    _validate_float_tensor(mask_logits, mask_name)
    if class_logits.ndim != 3:
        raise ValueError(f"`{class_name}` must have shape (B, Q, C + 1).")
    if mask_logits.ndim != 4:
        raise ValueError(f"`{mask_name}` must have shape (B, Q, H, W).")
    if class_logits.shape[-1] < 2:
        raise ValueError(f"`{class_name}` must contain at least one semantic class and the no-object class.")
    if any(size == 0 for size in mask_logits.shape[-2:]):
        raise ValueError(f"`{mask_name}` spatial dimensions must be non-empty.")
    if class_logits.shape[:2] != mask_logits.shape[:2]:
        raise ValueError(f"`{class_name}` and `{mask_name}` must have matching batch and query dimensions.")
    if class_logits.device != mask_logits.device:
        raise ValueError(f"`{class_name}` and `{mask_name}` must be on the same device.")
    _validate_finite_tensor(class_logits, class_name)
    _validate_finite_tensor(mask_logits, mask_name)


def _validate_float_tensor(tensor: torch.Tensor, name: str) -> None:
    if tensor.dtype not in SUPPORTED_PROB_DTYPES:
        raise TypeError(f"`{name}` must have a real floating-point dtype.")


def _validate_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if tensor.device.type == "meta":
        raise ValueError(f"`{name}` must contain materialized, finite values.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"`{name}` must contain only finite values.")


def _as_working_float(tensor: torch.Tensor) -> torch.Tensor:
    working_dtype = torch.promote_types(torch.float32, tensor.dtype)
    return tensor.to(dtype=working_dtype)


def _as_common_working_float(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    working_dtype = torch.float32
    for tensor in tensors:
        working_dtype = torch.promote_types(working_dtype, tensor.dtype)
    return tuple(tensor.to(dtype=working_dtype) for tensor in tensors)


def _resize_spatial(tensor: torch.Tensor, target_sizes: list[tuple[int, int]] | None) -> list[torch.Tensor]:
    if target_sizes is None:
        return list(tensor.unbind(dim=0))
    resized = []
    for idx, target_size in enumerate(target_sizes):
        sample = tensor[idx : idx + 1]
        if sample.shape[-2:] != target_size:
            sample = F.interpolate(sample, size=target_size, mode="bilinear", align_corners=False)
        resized.append(sample[0])
    return resized


def _normalize_scores(scores: list[torch.Tensor]) -> list[torch.Tensor]:
    normalized = []
    for score in scores:
        if score.shape[0] == 1:
            # Dividing a single channel by itself would erase all spatial
            # confidence and turn every nonzero score into one. Query-based
            # semantic scores are sums, so clamp possible overlap excess.
            normalized.append(score.clamp(min=0.0, max=1.0))
        else:
            total = score.sum(dim=0, keepdim=True)
            # Normalize every positive total, however small. A fixed epsilon
            # would incorrectly preserve the absolute scale of tiny float64
            # scores instead of producing a class distribution. Only a true
            # zero total has no class information and remains all-zero.
            denominator = torch.where(total > 0, total, torch.ones_like(total))
            normalized.append(score / denominator)
    return normalized


def _activate_dense_logits(logits: list[torch.Tensor]) -> list[torch.Tensor]:
    return [logit.sigmoid() if logit.shape[0] == 1 else logit.softmax(dim=0) for logit in logits]


def _predict_probs(probs: list[torch.Tensor], rankseg_kwargs) -> list[torch.Tensor]:
    if not probs:
        return []
    first = probs[0]
    predictor = RankSEG(
        **_rankseg_kwargs(
            rankseg_kwargs,
            default_output_mode="multilabel" if first.shape[0] == 1 else "multiclass",
        )
    )
    return [predictor.predict(prob.unsqueeze(0))[0] for prob in probs]


def _resize_single_size(tensor: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    if tensor.shape[-2:] == target_size:
        return tensor
    return F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)


def _matches_output_class(outputs, class_name: str) -> bool:
    return any(base.__name__ == class_name for base in type(outputs).__mro__)


def _matches_model_config(model, *, class_name: str, model_type: str) -> bool:
    config = getattr(model, "config", None)
    if config is None:
        return False
    if any(base.__name__ == class_name for base in type(config).__mro__):
        return True
    if isinstance(config, Mapping):
        return config.get("model_type") == model_type
    return getattr(config, "model_type", None) == model_type


def _reject_sam_outputs(outputs) -> None:
    is_sam_output_class = any(_matches_output_class(outputs, class_name) for class_name in _SAM_OUTPUT_CLASSES)
    pred_masks = _get_output_value(outputs, "pred_masks")
    has_prompt_mask_fields = pred_masks is not None and _get_output_value(outputs, "iou_scores") is not None
    has_sam3_instance_fields = (
        pred_masks is not None
        and _get_output_value(outputs, "pred_logits") is not None
        and _get_output_value(outputs, "pred_boxes") is not None
    )
    has_sam3_semantic_field = _get_output_value(outputs, "semantic_seg") is not None
    if is_sam_output_class or has_prompt_mask_fields or has_sam3_instance_fields or has_sam3_semantic_field:
        raise ValueError(
            "SAM-family outputs require the explicit adapters from `rankseg.integration.sam` "
            "(`Sam1`, `Sam2`, or `Sam3`)."
        )


def _rankseg_kwargs(rankseg_kwargs, *, default_output_mode: str) -> dict:
    if rankseg_kwargs is None:
        rankseg_kwargs = {}
    elif not isinstance(rankseg_kwargs, dict):
        raise ValueError("`rankseg_kwargs` must be a dictionary.")

    if "output_mode" not in rankseg_kwargs:
        rankseg_kwargs = {**rankseg_kwargs, "output_mode": default_output_mode}
    return rankseg_kwargs


def restore_semantic_probs(outputs, *, model=None, target_sizes=None) -> list[torch.Tensor]:
    """Restore per-image semantic probability maps from supported outputs.

    :param outputs: Structured model output with fields such as ``logits``,
        ``class_queries_logits``, or ``masks_queries_logits``.
    :param model: Optional Transformers model instance used to identify
        model-family-specific restoration behavior when the structured output
        type does not identify it. It is not required for DETR-style
        ``logits`` + ``pred_masks`` outputs.
    :param target_sizes: Optional list or tensor with one ``(height, width)``
        pair per batch item. If omitted, probabilities keep the model output
        spatial size.
    :returns: One ``(C, H, W)`` probability tensor per input image.
    :raises ValueError: If the output structure is unsupported or belongs to a
        SAM-family model.
    :raises TypeError: If expected output fields are not tensors.
    """

    _reject_sam_outputs(outputs)

    class_queries_logits = _get_output_value(outputs, "class_queries_logits")
    masks_queries_logits = _get_output_value(outputs, "masks_queries_logits")
    if class_queries_logits is not None and masks_queries_logits is not None:
        if (
            _has_output_field(outputs, "patch_offsets")
            or _matches_output_class(outputs, "EomtForUniversalSegmentationOutput")
            or _matches_model_config(model, class_name="EomtConfig", model_type="eomt")
        ):
            raise ValueError(
                "EOMT outputs require processor-specific resizing, unpadding, and optional patch merge logic using "
                "the processor `size` and patch metadata. This helper does not support EOMT in this release."
            )
        if not isinstance(class_queries_logits, torch.Tensor) or not isinstance(masks_queries_logits, torch.Tensor):
            raise TypeError("`class_queries_logits` and `masks_queries_logits` must be torch.Tensor values.")
        _validate_query_logits(
            class_queries_logits,
            masks_queries_logits,
            class_name="outputs.class_queries_logits",
            mask_name="outputs.masks_queries_logits",
        )
        class_queries_logits, masks_queries_logits = _as_common_working_float(
            class_queries_logits,
            masks_queries_logits,
        )
        target_sizes = _normalize_target_sizes(target_sizes, class_queries_logits.shape[0])
        if _matches_model_config(
            model,
            class_name="Mask2FormerConfig",
            model_type="mask2former",
        ) or _matches_output_class(outputs, "Mask2FormerForUniversalSegmentationOutput"):
            masks_queries_logits = _resize_single_size(masks_queries_logits, (384, 384))
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()
        semantic_scores = torch.einsum("bqc,bqhw->bchw", masks_classes, masks_probs)
        semantic_scores = _resize_spatial(semantic_scores, target_sizes)
        return _normalize_scores(semantic_scores)

    logits = _get_output_value(outputs, "logits")
    pred_masks = _get_output_value(outputs, "pred_masks")
    if logits is not None and pred_masks is not None:
        if not isinstance(logits, torch.Tensor) or not isinstance(pred_masks, torch.Tensor):
            raise TypeError("`logits` and `pred_masks` must be torch.Tensor values.")

        _validate_query_logits(
            logits,
            pred_masks,
            class_name="outputs.logits",
            mask_name="outputs.pred_masks",
        )
        logits, pred_masks = _as_common_working_float(logits, pred_masks)
        target_sizes = _normalize_target_sizes(target_sizes, logits.shape[0])
        # DETR segmentation variants, including Conditional DETR, append a
        # no-object class to every query's classification logits.
        masks_classes = logits.softmax(dim=-1)[..., :-1]
        masks_probs = pred_masks.sigmoid()
        semantic_scores = torch.einsum("bqc,bqhw->bchw", masks_classes, masks_probs)
        semantic_scores = _resize_spatial(semantic_scores, target_sizes)
        return _normalize_scores(semantic_scores)

    if logits is not None:
        if not isinstance(logits, torch.Tensor):
            raise TypeError("`outputs.logits` must be a torch.Tensor.")
        _validate_dense_logits(logits, "outputs.logits")
        target_sizes = _normalize_target_sizes(target_sizes, logits.shape[0])
        resized_logits = _resize_spatial(_as_working_float(logits), target_sizes)
        return _activate_dense_logits(resized_logits)

    if pred_masks is not None:
        raise ValueError(
            "Outputs with only `pred_masks` require model-specific semantic reconstruction and are not supported."
        )

    raise ValueError("Unsupported outputs structure for semantic probability restoration.")


@torch.no_grad()
def postprocess(outputs, *, model=None, target_sizes=None, rankseg_kwargs=None) -> list[torch.Tensor]:
    """Convert supported Transformers outputs into RankSEG predictions.

    :param outputs: Structured model output returned by a supported
        Transformers semantic segmentation model.
    :param model: Optional Transformers model instance used by
        :func:`restore_semantic_probs`.
    :param target_sizes: Optional list or tensor with one ``(height, width)``
        pair per batch item.
    :param rankseg_kwargs: Optional keyword arguments forwarded to
        :class:`rankseg.RankSEG`. ``output_mode`` is inferred when omitted.
    :returns: One prediction tensor per input image.

    This discrete inference entry point disables gradient recording. Call
    :func:`restore_semantic_probs` directly when differentiable restored
    probabilities are required.
    """

    probs = restore_semantic_probs(outputs, model=model, target_sizes=target_sizes)
    return _predict_probs(probs, rankseg_kwargs)
