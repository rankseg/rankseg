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
            raise ValueError(
                "Batches with different original image sizes are not supported. Call the helper per sample."
            )
        return first

    raise ValueError("`target_sizes` must be a tuple, list of tuples, or a tensor of shape (2) or (B, 2).")


def _resize_spatial(tensor: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    if tensor.shape[-2:] == target_size:
        return tensor
    return F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _normalize_size_list(sizes, batch_size: int, name: str) -> list[tuple[int, int]]:
    if sizes is None:
        raise ValueError(f"`{name}` is required.")

    if isinstance(sizes, torch.Tensor):
        if sizes.ndim == 1:
            sizes = [tuple(int(v) for v in sizes.tolist())]
        elif sizes.ndim == 2:
            sizes = [tuple(int(v) for v in row.tolist()) for row in sizes]
        else:
            raise ValueError(f"`{name}` must be a tuple, list of tuples, or a tensor of shape (2) or (B, 2).")

    if isinstance(sizes, Mapping):
        sizes = [(int(sizes["height"]), int(sizes["width"]))]

    if isinstance(sizes, Sequence) and not isinstance(sizes, (str, bytes)):
        if len(sizes) == 2 and all(not isinstance(v, Sequence) for v in sizes):
            normalized = [(int(sizes[0]), int(sizes[1]))]
        else:
            normalized = [tuple(int(v) for v in size) for size in sizes]

        if len(normalized) != batch_size:
            raise ValueError(f"`{name}` must match the batch size.")
        return normalized

    raise ValueError(f"`{name}` must be a tuple, list of tuples, or a tensor of shape (2) or (B, 2).")


def _normalize_hw_size(size, name: str) -> tuple[int, int]:
    if isinstance(size, Mapping):
        return int(size["height"]), int(size["width"])
    if isinstance(size, torch.Tensor):
        if size.ndim != 1 or size.numel() != 2:
            raise ValueError(f"`{name}` must describe one (height, width) size.")
        size = size.tolist()
    if isinstance(size, Sequence) and not isinstance(size, (str, bytes)) and len(size) == 2:
        return int(size[0]), int(size[1])
    raise ValueError(f"`{name}` must describe one (height, width) size.")


def _matches_output_class(outputs, class_name: str) -> bool:
    return type(outputs).__name__ == class_name


def _matches_config_class(model, class_name: str) -> bool:
    return type(getattr(model, "config", None)).__name__ == class_name


def _sam_output_family(outputs) -> str | None:
    output_class = type(outputs).__name__
    if output_class in {"SamImageSegmentationOutput", "SamHQImageSegmentationOutput"}:
        return "sam1_prompt"
    if output_class == "Sam2ImageSegmentationOutput":
        return "sam2_prompt"
    if output_class in {"Sam3ImageSegmentationOutput", "Sam3LiteTextImageSegmentationOutput"}:
        return "sam3"
    return None


def _is_sam3_instance_outputs(outputs) -> bool:
    return (
        _get_output_value(outputs, "pred_logits") is not None
        and _get_output_value(outputs, "pred_boxes") is not None
        and _get_output_value(outputs, "pred_masks") is not None
    )


def _is_sam_prompt_outputs(outputs) -> bool:
    return _get_output_value(outputs, "pred_masks") is not None and _get_output_value(outputs, "iou_scores") is not None


def _normalize_sam_task(sam_task: str | None) -> str | None:
    if sam_task not in (None, "instance", "semantic", "prompt"):
        raise ValueError("`sam_task` must be one of None, 'instance', 'semantic', or 'prompt'.")
    return sam_task


def _apply_sam_non_overlapping_constraints(masks: torch.Tensor) -> torch.Tensor:
    batch_size = masks.size(0)
    if batch_size == 1:
        return masks

    max_obj_inds = torch.argmax(masks, dim=0, keepdim=True)
    batch_obj_inds = torch.arange(batch_size, device=masks.device)[:, None, None, None]
    keep = max_obj_inds == batch_obj_inds
    return torch.where(keep, masks, torch.clamp(masks, max=-10.0))


def _scale_sam3_boxes(boxes: torch.Tensor, target_sizes: list[tuple[int, int]]) -> torch.Tensor:
    image_height = torch.tensor([size[0] for size in target_sizes], device=boxes.device, dtype=boxes.dtype)
    image_width = torch.tensor([size[1] for size in target_sizes], device=boxes.device, dtype=boxes.dtype)
    scale_factor = torch.stack([image_width, image_height, image_width, image_height], dim=1)
    return boxes * scale_factor.unsqueeze(1)


def _restore_sam1_prompt_mask_probs(
    pred_masks: torch.Tensor,
    *,
    original_sizes,
    reshaped_input_sizes,
    pad_size,
) -> list[torch.Tensor]:
    original_sizes = _normalize_size_list(original_sizes, pred_masks.shape[0], "original_sizes")
    reshaped_input_sizes = _normalize_size_list(reshaped_input_sizes, pred_masks.shape[0], "reshaped_input_sizes")
    pad_size = _normalize_hw_size(pad_size or {"height": 1024, "width": 1024}, "pad_size")

    output_masks = []
    for idx, original_size in enumerate(original_sizes):
        masks = pred_masks[idx].float()
        masks = F.interpolate(masks, pad_size, mode="bilinear", align_corners=False)
        masks = masks[..., : reshaped_input_sizes[idx][0], : reshaped_input_sizes[idx][1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        output_masks.append(masks.sigmoid())
    return output_masks


def _restore_sam2_prompt_mask_probs(
    pred_masks: torch.Tensor,
    *,
    original_sizes,
    apply_non_overlapping_constraints: bool,
) -> list[torch.Tensor]:
    original_sizes = _normalize_size_list(original_sizes, pred_masks.shape[0], "original_sizes")

    output_masks = []
    for idx, original_size in enumerate(original_sizes):
        masks = pred_masks[idx].float()
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        if apply_non_overlapping_constraints:
            masks = _apply_sam_non_overlapping_constraints(masks)
        output_masks.append(masks.sigmoid())
    return output_masks


def _restore_sam3_semantic_mask_probs(outputs, *, target_sizes=None, original_sizes=None):
    semantic_seg = _get_output_value(outputs, "semantic_seg")
    if semantic_seg is None:
        raise ValueError("`outputs.semantic_seg` is required for SAM3 semantic mask restoration.")
    if not isinstance(semantic_seg, torch.Tensor):
        raise TypeError("`outputs.semantic_seg` must be a torch.Tensor.")

    semantic_probs = semantic_seg.float().sigmoid()
    sizes = target_sizes if target_sizes is not None else original_sizes
    if sizes is None:
        return [semantic_probs[idx] for idx in range(semantic_probs.shape[0])]

    target_sizes = _normalize_size_list(sizes, semantic_probs.shape[0], "target_sizes")
    return [
        F.interpolate(
            semantic_probs[idx].unsqueeze(0),
            size=target_sizes[idx],
            mode="bilinear",
            align_corners=False,
        )[0]
        for idx in range(semantic_probs.shape[0])
    ]


def _restore_sam3_instance_mask_probs(
    outputs,
    *,
    target_sizes=None,
    original_sizes=None,
    threshold: float,
) -> list[dict[str, torch.Tensor]]:
    pred_logits = _get_output_value(outputs, "pred_logits")
    pred_boxes = _get_output_value(outputs, "pred_boxes")
    pred_masks = _get_output_value(outputs, "pred_masks")
    presence_logits = _get_output_value(outputs, "presence_logits")

    if not isinstance(pred_logits, torch.Tensor) or not isinstance(pred_boxes, torch.Tensor):
        raise TypeError("`outputs.pred_logits` and `outputs.pred_boxes` must be torch.Tensor values.")
    if not isinstance(pred_masks, torch.Tensor):
        raise TypeError("`outputs.pred_masks` must be a torch.Tensor.")

    batch_size = pred_logits.shape[0]
    sizes = target_sizes if target_sizes is not None else original_sizes
    target_size_list = None if sizes is None else _normalize_size_list(sizes, batch_size, "target_sizes")

    batch_scores = pred_logits.float().sigmoid()
    if presence_logits is not None:
        if not isinstance(presence_logits, torch.Tensor):
            raise TypeError("`outputs.presence_logits` must be a torch.Tensor.")
        batch_scores = batch_scores * presence_logits.float().sigmoid()

    batch_masks = pred_masks.float().sigmoid()
    batch_boxes = pred_boxes.float()
    if target_size_list is not None:
        batch_boxes = _scale_sam3_boxes(batch_boxes, target_size_list)

    results = []
    for idx, (scores, boxes, masks) in enumerate(zip(batch_scores, batch_boxes, batch_masks)):
        keep = scores > threshold
        scores = scores[keep]
        boxes = boxes[keep]
        masks = masks[keep]

        if target_size_list is not None and len(masks) > 0:
            masks = F.interpolate(
                masks.unsqueeze(0),
                size=target_size_list[idx],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        results.append({"scores": scores, "boxes": boxes, "mask_probs": masks})

    return results


def restore_sam_mask_probs(
    outputs,
    *,
    sam_task=None,
    target_sizes=None,
    original_sizes=None,
    reshaped_input_sizes=None,
    threshold=0.3,
    pad_size=None,
    apply_non_overlapping_constraints=False,
):
    sam_task = _normalize_sam_task(sam_task)
    sam_family = _sam_output_family(outputs)

    if sam_family is None:
        raise ValueError("Unsupported SAM output class. Pass the original transformers SAM structured output.")

    if sam_family == "sam1_prompt":
        if sam_task not in (None, "prompt"):
            raise ValueError("SAM1-style outputs only support `sam_task=None` or `sam_task='prompt'`.")
        if not _is_sam_prompt_outputs(outputs):
            raise ValueError("SAM1-style outputs require `outputs.pred_masks` and `outputs.iou_scores`.")
        pred_masks = _get_output_value(outputs, "pred_masks")
        if not isinstance(pred_masks, torch.Tensor):
            raise TypeError("`outputs.pred_masks` must be a torch.Tensor.")
        return _restore_sam1_prompt_mask_probs(
            pred_masks,
            original_sizes=original_sizes,
            reshaped_input_sizes=reshaped_input_sizes,
            pad_size=pad_size,
        )

    if sam_family == "sam2_prompt":
        if sam_task not in (None, "prompt"):
            raise ValueError("SAM2 outputs only support `sam_task=None` or `sam_task='prompt'`.")
        if not _is_sam_prompt_outputs(outputs):
            raise ValueError("SAM2 outputs require `outputs.pred_masks` and `outputs.iou_scores`.")
        pred_masks = _get_output_value(outputs, "pred_masks")
        if not isinstance(pred_masks, torch.Tensor):
            raise TypeError("`outputs.pred_masks` must be a torch.Tensor.")
        return _restore_sam2_prompt_mask_probs(
            pred_masks,
            original_sizes=original_sizes,
            apply_non_overlapping_constraints=apply_non_overlapping_constraints,
        )

    if sam_task in (None, "instance") and _is_sam3_instance_outputs(outputs):
        return _restore_sam3_instance_mask_probs(
            outputs,
            target_sizes=target_sizes,
            original_sizes=original_sizes,
            threshold=threshold,
        )

    if sam_task == "semantic":
        return _restore_sam3_semantic_mask_probs(
            outputs,
            target_sizes=target_sizes,
            original_sizes=original_sizes,
        )

    if sam_task is None and _get_output_value(outputs, "semantic_seg") is not None:
        return _restore_sam3_semantic_mask_probs(
            outputs,
            target_sizes=target_sizes,
            original_sizes=original_sizes,
        )

    raise ValueError(f"Unsupported SAM3 outputs structure for `sam_task` {sam_task!r}.")


def _rankseg_kwargs(rankseg_kwargs, *, default_output_mode: str) -> dict:
    if rankseg_kwargs is None:
        rankseg_kwargs = {}
    elif not isinstance(rankseg_kwargs, dict):
        raise ValueError("`rankseg_kwargs` must be a dictionary.")

    if "output_mode" not in rankseg_kwargs:
        rankseg_kwargs = {**rankseg_kwargs, "output_mode": default_output_mode}
    return rankseg_kwargs


def _predict_sam_mask_probs(mask_probs, rankseg_kwargs):
    predictor = RankSEG(**_rankseg_kwargs(rankseg_kwargs, default_output_mode="multilabel"))

    if isinstance(mask_probs, torch.Tensor):
        return predictor.predict(mask_probs)

    if isinstance(mask_probs, list) and (not mask_probs or isinstance(mask_probs[0], torch.Tensor)):
        preds = []
        for probs in mask_probs:
            if probs.ndim == 3:
                preds.append(predictor.predict(probs.unsqueeze(0))[0])
            else:
                preds.append(predictor.predict(probs))
        return preds

    results = []
    for result in mask_probs:
        masks = result["mask_probs"]
        if len(masks) == 0:
            preds = masks.to(dtype=torch.long)
        else:
            preds = predictor.predict(masks.unsqueeze(1)).squeeze(1)
        results.append({"scores": result["scores"], "boxes": result["boxes"], "masks": preds})
    return results


def _predict_sam_semantic_mask_probs(mask_probs, rankseg_kwargs):
    predictor = RankSEG(**_rankseg_kwargs(rankseg_kwargs, default_output_mode="multilabel"))

    if isinstance(mask_probs, torch.Tensor):
        mask_probs = [mask_probs[idx] for idx in range(mask_probs.shape[0])]

    preds = []
    for probs in mask_probs:
        if probs.ndim == 2:
            probs = probs.unsqueeze(0)
        pred = predictor.predict(probs.unsqueeze(0))[0]
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]
        preds.append(pred)
    return preds


def _postprocess_sam_outputs(
    outputs,
    *,
    sam_task=None,
    target_sizes=None,
    original_sizes=None,
    reshaped_input_sizes=None,
    rankseg_kwargs=None,
    threshold=0.3,
    pad_size=None,
    apply_non_overlapping_constraints=False,
):
    sam_task = _normalize_sam_task(sam_task)
    sam_family = _sam_output_family(outputs)
    mask_probs = restore_sam_mask_probs(
        outputs,
        sam_task=sam_task,
        target_sizes=target_sizes,
        original_sizes=original_sizes,
        reshaped_input_sizes=reshaped_input_sizes,
        threshold=threshold,
        pad_size=pad_size,
        apply_non_overlapping_constraints=apply_non_overlapping_constraints,
    )
    if sam_task == "semantic" or (sam_family == "sam3" and sam_task is None and not _is_sam3_instance_outputs(outputs)):
        return _predict_sam_semantic_mask_probs(mask_probs, rankseg_kwargs)
    return _predict_sam_mask_probs(mask_probs, rankseg_kwargs)


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


def postprocess(
    outputs,
    *,
    model=None,
    sam_task=None,
    target_sizes=None,
    original_sizes=None,
    reshaped_input_sizes=None,
    rankseg_kwargs=None,
    threshold=0.3,
    pad_size=None,
    apply_non_overlapping_constraints=False,
):
    sam_family = _sam_output_family(outputs)
    if sam_family is not None:
        return _postprocess_sam_outputs(
            outputs,
            sam_task=sam_task,
            target_sizes=target_sizes,
            original_sizes=original_sizes,
            reshaped_input_sizes=reshaped_input_sizes,
            rankseg_kwargs=rankseg_kwargs,
            threshold=threshold,
            pad_size=pad_size,
            apply_non_overlapping_constraints=apply_non_overlapping_constraints,
        )
    if sam_task is not None:
        _normalize_sam_task(sam_task)
        raise ValueError("`sam_task` can only be used with supported transformers SAM structured outputs.")

    probs = restore_semantic_probs(outputs, model=model, target_sizes=target_sizes)

    predictor = RankSEG(
        **_rankseg_kwargs(
            rankseg_kwargs,
            default_output_mode="multilabel" if probs.shape[1] == 1 else "multiclass",
        )
    )
    return predictor.predict(probs)
