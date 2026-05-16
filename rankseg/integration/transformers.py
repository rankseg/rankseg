from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from .._rankseg import RankSEG

_TARGET_SIZES_ERROR = (
    "`target_sizes` must be None, a list of (height, width) pairs, "
    "or a tensor of shape (B, 2)."
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


def _normalize_target_sizes(target_sizes, batch_size: int) -> list[tuple[int, int]] | None:
    if target_sizes is None:
        return None

    if isinstance(target_sizes, torch.Tensor):
        if target_sizes.ndim != 2 or target_sizes.shape[1] != 2:
            raise ValueError(_TARGET_SIZES_ERROR)
        normalized = [tuple(int(v) for v in row.tolist()) for row in target_sizes]
        if len(normalized) != batch_size:
            raise ValueError("`target_sizes` must contain one (height, width) pair per batch item.")
        return normalized

    if isinstance(target_sizes, Sequence) and not isinstance(target_sizes, (str, bytes)):
        if len(target_sizes) != batch_size:
            raise ValueError("`target_sizes` must contain one (height, width) pair per batch item.")
        normalized = []
        for size in target_sizes:
            if isinstance(size, torch.Tensor):
                if size.ndim != 1 or size.numel() != 2:
                    raise ValueError(_TARGET_SIZES_ERROR)
                normalized.append(tuple(int(v) for v in size.tolist()))
            elif isinstance(size, Sequence) and not isinstance(size, (str, bytes)):
                if len(size) != 2:
                    raise ValueError(_TARGET_SIZES_ERROR)
                normalized.append((int(size[0]), int(size[1])))
            else:
                raise ValueError(_TARGET_SIZES_ERROR)
        if len(normalized) != batch_size:
            raise ValueError("`target_sizes` must contain one (height, width) pair per batch item.")
        return normalized

    raise ValueError(_TARGET_SIZES_ERROR)


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
    return [score / score.sum(dim=0, keepdim=True).clamp_min(1e-12) for score in scores]


def _softmax_scores(scores: list[torch.Tensor]) -> list[torch.Tensor]:
    return [score.softmax(dim=0) for score in scores]


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
    return type(outputs).__name__ == class_name


def _matches_config_class(model, class_name: str) -> bool:
    return type(getattr(model, "config", None)).__name__ == class_name


def _reject_sam_outputs(outputs) -> None:
    if type(outputs).__name__ in _SAM_OUTPUT_CLASSES:
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
    _reject_sam_outputs(outputs)

    semantic_seg = _get_output_value(outputs, "semantic_seg")
    if semantic_seg is not None:
        if not isinstance(semantic_seg, torch.Tensor):
            raise TypeError("`outputs.semantic_seg` must be a torch.Tensor.")
        target_sizes = _normalize_target_sizes(target_sizes, semantic_seg.shape[0])
        semantic_probs = semantic_seg.float().sigmoid()
        return _resize_spatial(semantic_probs, target_sizes)

    class_queries_logits = _get_output_value(outputs, "class_queries_logits")
    masks_queries_logits = _get_output_value(outputs, "masks_queries_logits")
    patch_offsets = _get_output_value(outputs, "patch_offsets")
    if class_queries_logits is not None and masks_queries_logits is not None:
        if patch_offsets is not None:
            raise ValueError(
                "Outputs with `patch_offsets` require EOMT patch merge logic using the processor `size` and patch "
                "metadata. This helper does not support EOMT patch merge in this release."
            )
        if not isinstance(class_queries_logits, torch.Tensor) or not isinstance(masks_queries_logits, torch.Tensor):
            raise TypeError("`class_queries_logits` and `masks_queries_logits` must be torch.Tensor values.")
        target_sizes = _normalize_target_sizes(target_sizes, class_queries_logits.shape[0])
        if _matches_config_class(model, "Mask2FormerConfig") or _matches_output_class(
            outputs, "Mask2FormerForUniversalSegmentationOutput"
        ):
            masks_queries_logits = _resize_single_size(masks_queries_logits.float(), (384, 384))
        else:
            masks_queries_logits = masks_queries_logits.float()
        masks_classes = class_queries_logits.float().softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()
        semantic_scores = torch.einsum("bqc,bqhw->bchw", masks_classes, masks_probs)
        semantic_scores = _resize_spatial(semantic_scores, target_sizes)
        return _normalize_scores(semantic_scores)

    logits = _get_output_value(outputs, "logits")
    pred_masks = _get_output_value(outputs, "pred_masks")
    if logits is not None and pred_masks is not None:
        if not isinstance(logits, torch.Tensor) or not isinstance(pred_masks, torch.Tensor):
            raise TypeError("`logits` and `pred_masks` must be torch.Tensor values.")
        if model is None:
            raise ValueError("`model` is required for outputs with both `logits` and `pred_masks`.")

        target_sizes = _normalize_target_sizes(target_sizes, logits.shape[0])
        config_name = type(getattr(model, "config", None)).__name__
        masks_classes = logits.float().softmax(dim=-1)
        if config_name != "ConditionalDetrConfig":
            masks_classes = masks_classes[..., :-1]
        masks_probs = pred_masks.float().sigmoid()
        semantic_scores = torch.einsum("bqc,bqhw->bchw", masks_classes, masks_probs)
        semantic_scores = _resize_spatial(semantic_scores, target_sizes)
        return _normalize_scores(semantic_scores)

    if logits is not None:
        if not isinstance(logits, torch.Tensor):
            raise TypeError("`outputs.logits` must be a torch.Tensor.")
        target_sizes = _normalize_target_sizes(target_sizes, logits.shape[0])
        resized_logits = _resize_spatial(logits.float(), target_sizes)
        return _softmax_scores(resized_logits)

    if pred_masks is not None:
        raise ValueError(
            "Outputs with only `pred_masks` require model-specific semantic reconstruction and are not supported."
        )

    raise ValueError("Unsupported outputs structure for semantic probability restoration.")


def postprocess(outputs, *, model=None, target_sizes=None, rankseg_kwargs=None) -> list[torch.Tensor]:
    probs = restore_semantic_probs(outputs, model=model, target_sizes=target_sizes)
    return _predict_probs(probs, rankseg_kwargs)
