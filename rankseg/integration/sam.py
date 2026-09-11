"""Adapters for SAM-family outputs from Hugging Face Transformers.

SAM models expose family-specific output geometry, so this module uses explicit
adapter classes instead of the generic Transformers semantic segmentation
helper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import torch
import torch.nn.functional as F

from .._rankseg import RankSEG
from .._validation import SUPPORTED_PROB_DTYPES, validate_finite_real

_SIZE_ERROR = "`{name}` must be one (height, width) pair or one pair per batch item."
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


def _validate_adapter_family(outputs, accepted_class_names: set[str], family_name: str) -> None:
    output_mro_names = {base.__name__ for base in type(outputs).__mro__}
    identified_sam_classes = output_mro_names & _SAM_OUTPUT_CLASSES
    if identified_sam_classes and not identified_sam_classes & accepted_class_names:
        output_name = sorted(identified_sam_classes)[0]
        raise ValueError(f"{family_name} cannot process `{output_name}`; use the matching SAM-family adapter.")


def _normalize_size_pair(size, name: str) -> tuple[int, int]:
    if isinstance(size, Mapping):
        if "height" not in size or "width" not in size:
            raise ValueError(_SIZE_ERROR.format(name=name))
        size = (size["height"], size["width"])
    if isinstance(size, torch.Tensor):
        if size.ndim != 1 or size.numel() != 2:
            raise ValueError(_SIZE_ERROR.format(name=name))
        if size.dtype == torch.bool or size.is_floating_point() or size.is_complex():
            raise TypeError(f"`{name}` tensor must have an integer dtype.")
        size = size.tolist()
    if not isinstance(size, Sequence) or isinstance(size, (str, bytes)) or len(size) != 2:
        raise ValueError(_SIZE_ERROR.format(name=name))
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in size):
        raise TypeError(f"Each `{name}` height and width must be an integer.")
    if any(value <= 0 for value in size):
        raise ValueError(f"Each `{name}` height and width must be positive.")
    return int(size[0]), int(size[1])


def _normalize_size_list(sizes, batch_size: int, name: str) -> list[tuple[int, int]]:
    if sizes is None:
        raise ValueError(f"`{name}` is required.")

    if isinstance(sizes, torch.Tensor):
        if sizes.ndim == 1 and sizes.numel() == 2:
            sizes = [sizes]
        elif sizes.ndim == 2 and sizes.shape[1] == 2:
            if sizes.dtype == torch.bool or sizes.is_floating_point() or sizes.is_complex():
                raise TypeError(f"`{name}` tensor must have an integer dtype.")
            sizes = list(sizes.unbind(dim=0))
        else:
            raise ValueError(_SIZE_ERROR.format(name=name))
    elif isinstance(sizes, Mapping):
        sizes = [sizes]
    elif isinstance(sizes, Sequence) and not isinstance(sizes, (str, bytes)):
        is_flat_size = len(sizes) > 0 and all(
            not isinstance(value, (Mapping, Sequence, torch.Tensor)) for value in sizes
        )
        sizes = [sizes] if is_flat_size else list(sizes)
    else:
        raise ValueError(_SIZE_ERROR.format(name=name))

    if len(sizes) != batch_size:
        raise ValueError(f"`{name}` must match the batch size.")
    return [_normalize_size_pair(size, name) for size in sizes]


def _normalize_hw_size(size, name: str) -> tuple[int, int]:
    return _normalize_size_pair(size, name)


def _require_prompt_masks(outputs, family_name: str) -> torch.Tensor:
    pred_masks = _get_output_value(outputs, "pred_masks")
    if pred_masks is None:
        raise ValueError(f"{family_name} outputs require `outputs.pred_masks`.")
    if not isinstance(pred_masks, torch.Tensor):
        raise TypeError("`outputs.pred_masks` must be a torch.Tensor.")
    _validate_float_tensor(pred_masks, "outputs.pred_masks")
    if pred_masks.ndim not in (4, 5):
        raise ValueError("`outputs.pred_masks` must have shape (B, N, H, W) or (B, P, N, H, W).")
    if any(size == 0 for size in pred_masks.shape[1:]):
        raise ValueError("`outputs.pred_masks` non-batch dimensions must be non-empty.")
    _validate_finite_tensor(pred_masks, "outputs.pred_masks")
    return pred_masks


def _validate_sam3_instance_outputs(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_masks: torch.Tensor,
    presence_logits: torch.Tensor | None,
) -> None:
    _validate_float_tensor(pred_logits, "outputs.pred_logits")
    _validate_float_tensor(pred_boxes, "outputs.pred_boxes")
    _validate_float_tensor(pred_masks, "outputs.pred_masks")
    if pred_logits.ndim != 2:
        raise ValueError("`outputs.pred_logits` must have shape (B, Q).")
    if pred_boxes.ndim != 3 or pred_boxes.shape[-1] != 4:
        raise ValueError("`outputs.pred_boxes` must have shape (B, Q, 4).")
    if pred_masks.ndim != 4:
        raise ValueError("`outputs.pred_masks` must have shape (B, Q, H, W).")
    if any(size == 0 for size in pred_masks.shape[-2:]):
        raise ValueError("`outputs.pred_masks` spatial dimensions must be non-empty.")
    if pred_logits.shape[:2] != pred_boxes.shape[:2] or pred_logits.shape[:2] != pred_masks.shape[:2]:
        raise ValueError("SAM3 instance logits, boxes, and masks must have matching batch and query dimensions.")
    if pred_logits.device != pred_boxes.device or pred_logits.device != pred_masks.device:
        raise ValueError("SAM3 instance logits, boxes, and masks must be on the same device.")
    if presence_logits is not None:
        _validate_float_tensor(presence_logits, "outputs.presence_logits")
        if presence_logits.ndim != 2 or presence_logits.shape != (pred_logits.shape[0], 1):
            raise ValueError("`outputs.presence_logits` must have shape (B, 1).")
        if presence_logits.device != pred_logits.device:
            raise ValueError("`outputs.presence_logits` and other SAM3 instance outputs must be on the same device.")
    _validate_finite_tensor(pred_logits, "outputs.pred_logits")
    _validate_finite_tensor(pred_boxes, "outputs.pred_boxes")
    _validate_finite_tensor(pred_masks, "outputs.pred_masks")
    if presence_logits is not None:
        _validate_finite_tensor(presence_logits, "outputs.presence_logits")


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


def _rankseg_kwargs(rankseg_kwargs, *, default_output_mode: str) -> dict:
    if rankseg_kwargs is None:
        rankseg_kwargs = {}
    elif not isinstance(rankseg_kwargs, dict):
        raise ValueError("`rankseg_kwargs` must be a dictionary.")

    output_mode = rankseg_kwargs.get("output_mode", default_output_mode)
    if not isinstance(output_mode, str):
        raise TypeError("SAM adapters require `output_mode` to be the string 'multilabel'.")
    if output_mode != "multilabel":
        raise ValueError(
            "SAM adapters require output_mode='multilabel' because prompt, instance, and semantic outputs are "
            "independent binary masks rather than semantic class channels."
        )
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
            preds = masks.to(dtype=torch.bool)
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

        :param outputs: A structured output or mapping with a ``pred_masks``
            field, such as ``SamImageSegmentationOutput`` or
            ``SamHQImageSegmentationOutput``. ``iou_scores`` is not needed to
            restore mask probabilities.
        :param original_sizes: One original ``(height, width)`` size per batch
            item.
        :param reshaped_input_sizes: One reshaped input ``(height, width)`` size
            per batch item before padding.
        :returns: Restored mask probability tensors for each batch item.
        """

        _validate_adapter_family(
            outputs,
            {"SamImageSegmentationOutput", "SamHQImageSegmentationOutput"},
            "SAM1",
        )
        pred_masks = _require_prompt_masks(outputs, "SAM1")
        original_sizes = _normalize_size_list(original_sizes, pred_masks.shape[0], "original_sizes")
        reshaped_input_sizes = _normalize_size_list(reshaped_input_sizes, pred_masks.shape[0], "reshaped_input_sizes")
        configured_pad_size = {"height": 1024, "width": 1024} if self.pad_size is None else self.pad_size
        pad_size = _normalize_hw_size(configured_pad_size, "pad_size")
        if any(
            reshaped_height > pad_size[0] or reshaped_width > pad_size[1]
            for reshaped_height, reshaped_width in reshaped_input_sizes
        ):
            raise ValueError("Each `reshaped_input_sizes` pair must fit within `pad_size`.")

        output_masks = []
        for idx, original_size in enumerate(original_sizes):
            masks = _as_interpolation_input(_as_working_float(pred_masks[idx]))
            masks = F.interpolate(masks, pad_size, mode="bilinear", align_corners=False)
            masks = masks[..., : reshaped_input_sizes[idx][0], : reshaped_input_sizes[idx][1]]
            masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
            output_masks.append(masks.sigmoid())
        return output_masks

    @torch.no_grad()
    def postprocess(self, outputs, *, original_sizes, reshaped_input_sizes):
        """Restore SAM1 probabilities and produce inference-only predictions.

        Gradient recording is disabled. Call :meth:`restore_mask_probs`
        directly when differentiable restored probabilities are required.
        """

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
        if not isinstance(apply_non_overlapping_constraints, bool):
            raise TypeError("`apply_non_overlapping_constraints` must be a bool.")
        self.rankseg_kwargs = rankseg_kwargs
        self.apply_non_overlapping_constraints = apply_non_overlapping_constraints

    def restore_mask_probs(self, outputs, *, original_sizes) -> list[torch.Tensor]:
        """Restore SAM2 prompt mask probabilities to original image sizes.

        :param outputs: A structured output or mapping with a ``pred_masks``
            field, such as ``Sam2ImageSegmentationOutput``. ``iou_scores`` is
            not needed to restore mask probabilities.
        :param original_sizes: One original ``(height, width)`` size per batch
            item.
        :returns: Restored mask probability tensors for each batch item.
        """

        _validate_adapter_family(outputs, {"Sam2ImageSegmentationOutput"}, "SAM2")
        pred_masks = _require_prompt_masks(outputs, "SAM2")
        original_sizes = _normalize_size_list(original_sizes, pred_masks.shape[0], "original_sizes")

        output_masks = []
        for idx, original_size in enumerate(original_sizes):
            masks = _as_interpolation_input(_as_working_float(pred_masks[idx]))
            masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
            if self.apply_non_overlapping_constraints:
                masks = _apply_sam_non_overlapping_constraints(masks)
            output_masks.append(masks.sigmoid())
        return output_masks

    @torch.no_grad()
    def postprocess(self, outputs, *, original_sizes):
        """Restore SAM2 probabilities and produce inference-only predictions.

        Gradient recording is disabled. Call :meth:`restore_mask_probs`
        directly when differentiable restored probabilities are required.
        """

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
        threshold = validate_finite_real("threshold", threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in the range [0, 1]")
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

        :param outputs: A structured output or mapping with SAM3 instance mask
            fields, such as ``Sam3ImageSegmentationOutput`` or
            ``Sam3LiteTextImageSegmentationOutput``.
        :param target_sizes: Optional output ``(height, width)`` sizes. Used in
            preference to ``original_sizes`` when both are provided.
        :param original_sizes: Optional fallback output sizes.
        :returns: Per-image dictionaries with ``scores``, ``boxes``, and
            ``mask_probs`` tensors.
        """

        _validate_adapter_family(
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
        if presence_logits is not None and not isinstance(presence_logits, torch.Tensor):
            raise TypeError("`outputs.presence_logits` must be a torch.Tensor.")

        _validate_sam3_instance_outputs(pred_logits, pred_boxes, pred_masks, presence_logits)

        batch_size = pred_logits.shape[0]
        sizes = target_sizes if target_sizes is not None else original_sizes
        size_name = "target_sizes" if target_sizes is not None else "original_sizes"
        target_size_list = None if sizes is None else _normalize_size_list(sizes, batch_size, size_name)

        batch_scores = _as_working_float(pred_logits).sigmoid()
        if presence_logits is not None:
            batch_scores = batch_scores * _as_working_float(presence_logits).sigmoid()

        batch_masks = _as_working_float(pred_masks).sigmoid()
        batch_boxes = _as_working_float(pred_boxes)
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

        :param outputs: A structured output or mapping with ``semantic_seg``,
            such as ``Sam3ImageSegmentationOutput`` or
            ``Sam3LiteTextImageSegmentationOutput``.
        :param target_sizes: Optional output ``(height, width)`` sizes. Used in
            preference to ``original_sizes`` when both are provided.
        :param original_sizes: Optional fallback output sizes.
        :returns: One semantic probability tensor per input image.
        """

        _validate_adapter_family(
            outputs,
            {"Sam3ImageSegmentationOutput", "Sam3LiteTextImageSegmentationOutput"},
            "SAM3",
        )
        semantic_seg = _get_output_value(outputs, "semantic_seg")
        if semantic_seg is None:
            raise ValueError("`outputs.semantic_seg` is required for SAM3 semantic mask restoration.")
        if not isinstance(semantic_seg, torch.Tensor):
            raise TypeError("`outputs.semantic_seg` must be a torch.Tensor.")
        if semantic_seg.ndim != 4 or semantic_seg.shape[1] != 1:
            raise ValueError("`outputs.semantic_seg` must have shape (B, 1, H, W).")
        if any(size == 0 for size in semantic_seg.shape[-2:]):
            raise ValueError("`outputs.semantic_seg` spatial dimensions must be non-empty.")
        _validate_float_tensor(semantic_seg, "outputs.semantic_seg")
        _validate_finite_tensor(semantic_seg, "outputs.semantic_seg")

        semantic_probs = _as_working_float(semantic_seg).sigmoid()
        sizes = target_sizes if target_sizes is not None else original_sizes
        if sizes is None:
            return [semantic_probs[idx] for idx in range(semantic_probs.shape[0])]

        size_name = "target_sizes" if target_sizes is not None else "original_sizes"
        target_sizes = _normalize_size_list(sizes, semantic_probs.shape[0], size_name)
        return [
            F.interpolate(
                semantic_probs[idx].unsqueeze(0),
                size=target_sizes[idx],
                mode="bilinear",
                align_corners=False,
            )[0]
            for idx in range(semantic_probs.shape[0])
        ]

    @torch.no_grad()
    def postprocess_instance(self, outputs, *, target_sizes=None, original_sizes=None):
        """Restore SAM3 instances and produce inference-only predictions.

        Gradient recording is disabled. Call
        :meth:`restore_instance_mask_probs` directly when differentiable
        restored values are required.
        """

        mask_probs = self.restore_instance_mask_probs(
            outputs,
            target_sizes=target_sizes,
            original_sizes=original_sizes,
        )
        return _predict_instance_mask_probs(mask_probs, self.rankseg_kwargs)

    @torch.no_grad()
    def postprocess_semantic(self, outputs, *, target_sizes=None, original_sizes=None):
        """Restore SAM3 probabilities and produce inference-only predictions.

        Gradient recording is disabled. Call
        :meth:`restore_semantic_mask_probs` directly when differentiable
        restored probabilities are required.
        """

        mask_probs = self.restore_semantic_mask_probs(
            outputs,
            target_sizes=target_sizes,
            original_sizes=original_sizes,
        )
        return _predict_semantic_mask_probs(mask_probs, self.rankseg_kwargs)
