from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from ._rankseg import RankSEG


def _get_output_value(outputs, name: str):
    if isinstance(outputs, (tuple, list)):
        raise ValueError("Tuple-style outputs are not supported. Pass structured outputs with named fields.")

    value = getattr(outputs, name, None)
    if value is not None:
        return value

    if isinstance(outputs, Mapping):
        return outputs.get(name)

    return None


def _normalize_target_size(target_sizes, batch_size: int) -> tuple[int, int]:
    if target_sizes is None:
        raise ValueError("`target_sizes` is required and must describe the original image size.")

    if isinstance(target_sizes, torch.Tensor):
        if target_sizes.ndim == 1:
            target_sizes = tuple(int(v) for v in target_sizes.tolist())
        elif target_sizes.ndim == 2:
            target_sizes = [tuple(int(v) for v in row.tolist()) for row in target_sizes]
        else:
            raise ValueError("`target_sizes` must be a tuple, list of tuples, or a tensor of shape (2) or (B, 2).")

    if isinstance(target_sizes, Sequence) and not isinstance(target_sizes, (str, bytes)):
        if len(target_sizes) == 2 and all(not isinstance(v, Sequence) for v in target_sizes):
            return int(target_sizes[0]), int(target_sizes[1])

        normalized = [tuple(int(v) for v in size) for size in target_sizes]
        if len(normalized) != batch_size:
            raise ValueError("`target_sizes` must match the batch size when passing per-sample sizes.")
        first = normalized[0]
        if any(size != first for size in normalized[1:]):
            raise ValueError("Batches with different original image sizes are not supported. Call the helper per sample.")
        return first

    raise ValueError("`target_sizes` must be a tuple, list of tuples, or a tensor of shape (2) or (B, 2).")


def _resize_spatial(tensor: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    if tensor.shape[-2:] == target_size:
        return tensor
    return F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _matches_output_class(outputs, class_name: str) -> bool:
    return type(outputs).__name__ == class_name


def _matches_config_class(model, class_name: str) -> bool:
    return type(getattr(model, "config", None)).__name__ == class_name


def restore_semantic_probs(outputs, *, model=None, target_sizes):
    semantic_seg = _get_output_value(outputs, "semantic_seg")
    if semantic_seg is not None:
        if not isinstance(semantic_seg, torch.Tensor):
            raise TypeError("`outputs.semantic_seg` must be a torch.Tensor.")
        target_size = _normalize_target_size(target_sizes, semantic_seg.shape[0])
        semantic_probs = semantic_seg.float().sigmoid()
        return _resize_spatial(semantic_probs, target_size)

    class_queries_logits = _get_output_value(outputs, "class_queries_logits")
    masks_queries_logits = _get_output_value(outputs, "masks_queries_logits")
    patch_offsets = _get_output_value(outputs, "patch_offsets")
    if class_queries_logits is not None and masks_queries_logits is not None:
        if patch_offsets is not None:
            raise ValueError(
                "Outputs with `patch_offsets` require official patch merge logic and are not supported by this helper."
            )
        if not isinstance(class_queries_logits, torch.Tensor) or not isinstance(masks_queries_logits, torch.Tensor):
            raise TypeError("`class_queries_logits` and `masks_queries_logits` must be torch.Tensor values.")
        target_size = _normalize_target_size(target_sizes, class_queries_logits.shape[0])
        if _matches_config_class(model, "Mask2FormerConfig") or _matches_output_class(
            outputs, "Mask2FormerForUniversalSegmentationOutput"
        ):
            masks_queries_logits = F.interpolate(
                masks_queries_logits.float(), size=(384, 384), mode="bilinear", align_corners=False
            )
        else:
            masks_queries_logits = masks_queries_logits.float()
        masks_classes = class_queries_logits.float().softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()
        semantic_scores = torch.einsum("bqc,bqhw->bchw", masks_classes, masks_probs)
        semantic_scores = _resize_spatial(semantic_scores, target_size)
        return _normalize_scores(semantic_scores)

    logits = _get_output_value(outputs, "logits")
    pred_masks = _get_output_value(outputs, "pred_masks")
    if logits is not None and pred_masks is not None:
        if not isinstance(logits, torch.Tensor) or not isinstance(pred_masks, torch.Tensor):
            raise TypeError("`logits` and `pred_masks` must be torch.Tensor values.")
        if model is None:
            raise ValueError("`model` is required for outputs with both `logits` and `pred_masks`.")

        target_size = _normalize_target_size(target_sizes, logits.shape[0])
        config_name = type(getattr(model, "config", None)).__name__
        masks_classes = logits.float().softmax(dim=-1)
        if config_name != "ConditionalDetrConfig":
            masks_classes = masks_classes[..., :-1]
        masks_probs = pred_masks.float().sigmoid()
        semantic_scores = torch.einsum("bqc,bqhw->bchw", masks_classes, masks_probs)
        semantic_scores = _resize_spatial(semantic_scores, target_size)
        return _normalize_scores(semantic_scores)

    if logits is not None:
        if not isinstance(logits, torch.Tensor):
            raise TypeError("`outputs.logits` must be a torch.Tensor.")
        target_size = _normalize_target_size(target_sizes, logits.shape[0])
        resized_logits = _resize_spatial(logits.float(), target_size)
        return resized_logits.softmax(dim=1)

    if pred_masks is not None:
        raise ValueError(
            "Outputs with only `pred_masks` require model-specific semantic reconstruction and are not supported."
        )

    raise ValueError("Unsupported outputs structure for semantic probability restoration.")


def postprocess(outputs, *, model=None, target_sizes, rankseg_kwargs=None):
    if rankseg_kwargs is None:
        rankseg_kwargs = {}
    elif not isinstance(rankseg_kwargs, dict):
        raise ValueError("`rankseg_kwargs` must be a dictionary.")

    probs = restore_semantic_probs(outputs, model=model, target_sizes=target_sizes)

    if "output_mode" not in rankseg_kwargs:
        rankseg_kwargs = {
            **rankseg_kwargs,
            "output_mode": "multilabel" if probs.shape[1] == 1 else "multiclass",
        }

    predictor = RankSEG(**rankseg_kwargs)
    return predictor.predict(probs)
