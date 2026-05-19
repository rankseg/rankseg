"""Adapters for SAM-family outputs from Hugging Face Transformers.

SAM models expose family-specific output geometry, so this module uses explicit
adapter classes instead of the generic Transformers semantic segmentation
helper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from .._rankseg import RankSEG


def _get_output_value(outputs, name: str):
    if isinstance(outputs, (tuple, list)):
        raise ValueError("Tuple-style outputs are not supported. Pass structured outputs with named fields.")

    value = getattr(outputs, name, None)
    if value is not None:
        return value

    if isinstance(outputs, Mapping):
        return outputs.get(name)

    return None


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


def _require_output_class(outputs, class_names: set[str], family_name: str) -> None:
    if type(outputs).__name__ not in class_names:
        supported = ", ".join(sorted(class_names))
        raise ValueError(f"{family_name} requires one of these transformers output classes: {supported}.")


def _require_prompt_outputs(outputs, family_name: str) -> torch.Tensor:
    pred_masks = _get_output_value(outputs, "pred_masks")
    iou_scores = _get_output_value(outputs, "iou_scores")
    if pred_masks is None or iou_scores is None:
        raise ValueError(f"{family_name} outputs require `outputs.pred_masks` and `outputs.iou_scores`.")
    if not isinstance(pred_masks, torch.Tensor):
        raise TypeError("`outputs.pred_masks` must be a torch.Tensor.")
    return pred_masks


def _rankseg_kwargs(rankseg_kwargs, *, default_output_mode: str) -> dict:
    if rankseg_kwargs is None:
        rankseg_kwargs = {}
    elif not isinstance(rankseg_kwargs, dict):
        raise ValueError("`rankseg_kwargs` must be a dictionary.")

    if "output_mode" not in rankseg_kwargs:
        rankseg_kwargs = {**rankseg_kwargs, "output_mode": default_output_mode}
    return rankseg_kwargs


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


def _as_interpolation_input(masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim == 3:
        return masks.unsqueeze(0)
    return masks


def _predict_mask_probs(mask_probs, rankseg_kwargs):
    predictor = RankSEG(**_rankseg_kwargs(rankseg_kwargs, default_output_mode="multilabel"))

    if isinstance(mask_probs, torch.Tensor):
        return predictor.predict(mask_probs)

    preds = []
    for probs in mask_probs:
        if probs.ndim == 3:
            preds.append(predictor.predict(probs.unsqueeze(0))[0])
        else:
            preds.append(predictor.predict(probs))
    return preds


def _predict_instance_mask_probs(mask_probs, rankseg_kwargs):
    predictor = RankSEG(**_rankseg_kwargs(rankseg_kwargs, default_output_mode="multilabel"))

    results = []
    for result in mask_probs:
        masks = result["mask_probs"]
        if len(masks) == 0:
            preds = masks.to(dtype=torch.long)
        else:
            preds = predictor.predict(masks.unsqueeze(1)).squeeze(1)
        results.append({"scores": result["scores"], "boxes": result["boxes"], "masks": preds})
    return results


def _predict_semantic_mask_probs(mask_probs, rankseg_kwargs):
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


class Sam1:
    """Adapter for SAM1 and SAM-HQ prompt mask outputs.

    :param rankseg_kwargs: Optional keyword arguments forwarded to
        :class:`rankseg.RankSEG`.
    :param pad_size: Optional padded model input size as ``(height, width)`` or
        ``{"height": h, "width": w}``. Defaults to ``1024 x 1024``.
    """

    def __init__(self, *, rankseg_kwargs=None, pad_size=None):
        self.rankseg_kwargs = rankseg_kwargs
        self.pad_size = pad_size

    def restore_mask_probs(self, outputs, *, original_sizes, reshaped_input_sizes) -> list[torch.Tensor]:
        """Restore SAM1 prompt mask probabilities to original image sizes.

        :param outputs: ``SamImageSegmentationOutput`` or
            ``SamHQImageSegmentationOutput`` with ``pred_masks`` and
            ``iou_scores`` fields.
        :param original_sizes: One original ``(height, width)`` size per batch
            item.
        :param reshaped_input_sizes: One reshaped input ``(height, width)`` size
            per batch item before padding.
        :returns: Restored mask probability tensors for each batch item.
        """

        _require_output_class(outputs, {"SamHQImageSegmentationOutput", "SamImageSegmentationOutput"}, "SAM1")
        pred_masks = _require_prompt_outputs(outputs, "SAM1")
        original_sizes = _normalize_size_list(original_sizes, pred_masks.shape[0], "original_sizes")
        reshaped_input_sizes = _normalize_size_list(reshaped_input_sizes, pred_masks.shape[0], "reshaped_input_sizes")
        pad_size = _normalize_hw_size(self.pad_size or {"height": 1024, "width": 1024}, "pad_size")

        output_masks = []
        for idx, original_size in enumerate(original_sizes):
            masks = _as_interpolation_input(pred_masks[idx].float())
            masks = F.interpolate(masks, pad_size, mode="bilinear", align_corners=False)
            masks = masks[..., : reshaped_input_sizes[idx][0], : reshaped_input_sizes[idx][1]]
            masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
            output_masks.append(masks.sigmoid())
        return output_masks

    def postprocess(self, outputs, *, original_sizes, reshaped_input_sizes):
        """Restore SAM1 mask probabilities and convert them to predictions."""

        mask_probs = self.restore_mask_probs(
            outputs,
            original_sizes=original_sizes,
            reshaped_input_sizes=reshaped_input_sizes,
        )
        return _predict_mask_probs(mask_probs, self.rankseg_kwargs)


class Sam2:
    """Adapter for SAM2 prompt mask outputs.

    :param rankseg_kwargs: Optional keyword arguments forwarded to
        :class:`rankseg.RankSEG`.
    :param apply_non_overlapping_constraints: Whether to suppress lower-scoring
        overlapping masks before converting logits to probabilities.
    """

    def __init__(self, *, rankseg_kwargs=None, apply_non_overlapping_constraints=False):
        self.rankseg_kwargs = rankseg_kwargs
        self.apply_non_overlapping_constraints = apply_non_overlapping_constraints

    def restore_mask_probs(self, outputs, *, original_sizes) -> list[torch.Tensor]:
        """Restore SAM2 prompt mask probabilities to original image sizes.

        :param outputs: ``Sam2ImageSegmentationOutput`` with ``pred_masks`` and
            ``iou_scores`` fields.
        :param original_sizes: One original ``(height, width)`` size per batch
            item.
        :returns: Restored mask probability tensors for each batch item.
        """

        _require_output_class(outputs, {"Sam2ImageSegmentationOutput"}, "SAM2")
        pred_masks = _require_prompt_outputs(outputs, "SAM2")
        original_sizes = _normalize_size_list(original_sizes, pred_masks.shape[0], "original_sizes")

        output_masks = []
        for idx, original_size in enumerate(original_sizes):
            masks = _as_interpolation_input(pred_masks[idx].float())
            masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
            if self.apply_non_overlapping_constraints:
                masks = _apply_sam_non_overlapping_constraints(masks)
            output_masks.append(masks.sigmoid())
        return output_masks

    def postprocess(self, outputs, *, original_sizes):
        """Restore SAM2 mask probabilities and convert them to predictions."""

        mask_probs = self.restore_mask_probs(outputs, original_sizes=original_sizes)
        return _predict_mask_probs(mask_probs, self.rankseg_kwargs)


class Sam3:
    """Adapter for SAM3 instance and semantic mask outputs.

    :param rankseg_kwargs: Optional keyword arguments forwarded to
        :class:`rankseg.RankSEG`.
    :param threshold: Minimum instance confidence used by
        :meth:`restore_instance_mask_probs`.
    """

    def __init__(self, *, rankseg_kwargs=None, threshold=0.3):
        self.rankseg_kwargs = rankseg_kwargs
        self.threshold = threshold

    def restore_instance_mask_probs(
        self,
        outputs,
        *,
        target_sizes=None,
        original_sizes=None,
    ) -> list[dict[str, torch.Tensor]]:
        """Restore SAM3 instance masks, boxes, and scores.

        :param outputs: ``Sam3ImageSegmentationOutput`` or
            ``Sam3LiteTextImageSegmentationOutput`` with instance mask fields.
        :param target_sizes: Optional output ``(height, width)`` sizes. Used in
            preference to ``original_sizes`` when both are provided.
        :param original_sizes: Optional fallback output sizes.
        :returns: Per-image dictionaries with ``scores``, ``boxes``, and
            ``mask_probs`` tensors.
        """

        _require_output_class(
            outputs,
            {"Sam3ImageSegmentationOutput", "Sam3LiteTextImageSegmentationOutput"},
            "SAM3",
        )
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
            keep = scores > self.threshold
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

    def restore_semantic_mask_probs(self, outputs, *, target_sizes=None, original_sizes=None) -> list[torch.Tensor]:
        """Restore SAM3 semantic mask probabilities.

        :param outputs: ``Sam3ImageSegmentationOutput`` or
            ``Sam3LiteTextImageSegmentationOutput`` with ``semantic_seg``.
        :param target_sizes: Optional output ``(height, width)`` sizes. Used in
            preference to ``original_sizes`` when both are provided.
        :param original_sizes: Optional fallback output sizes.
        :returns: One semantic probability tensor per input image.
        """

        _require_output_class(
            outputs,
            {"Sam3ImageSegmentationOutput", "Sam3LiteTextImageSegmentationOutput"},
            "SAM3",
        )
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

    def postprocess_instance(self, outputs, *, target_sizes=None, original_sizes=None):
        """Restore SAM3 instance probabilities and convert masks to predictions."""

        mask_probs = self.restore_instance_mask_probs(
            outputs,
            target_sizes=target_sizes,
            original_sizes=original_sizes,
        )
        return _predict_instance_mask_probs(mask_probs, self.rankseg_kwargs)

    def postprocess_semantic(self, outputs, *, target_sizes=None, original_sizes=None):
        """Restore SAM3 semantic probabilities and convert them to predictions."""

        mask_probs = self.restore_semantic_mask_probs(
            outputs,
            target_sizes=target_sizes,
            original_sizes=original_sizes,
        )
        return _predict_semantic_mask_probs(mask_probs, self.rankseg_kwargs)
