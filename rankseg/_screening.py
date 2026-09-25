"""Experimental, conservative screening for unsmoothed Dice RMA.

Per-class binary search and optional fused assignment dispatch live here.
Original probabilities, class pruning, and multiclass eligibility must remain
independent of pixel screening.
Computed scores use direct argmax (first exact maximum), without epsilon
comparisons or full-sort retries. Candidate density/length never triggers a
full-sort retry; grouping limits only the padding added to candidate rows.
"""

from functools import lru_cache

import torch

# Bound the padded sort workspace, not the size of the original output mask.
_MAX_PADDED_ELEMENTS = 1_048_576
_MAX_BATCHED_ROWS = 128
_MAX_PADDING_RATIO = 2
_ROUNDING_GUARD = 32


@lru_cache(maxsize=1)
def _cuda_backend():
    """Load optional kernels only on the CUDA screening path."""
    try:
        import triton  # noqa: F401
    except ImportError:
        return None
    from . import _screening_cuda
    return _screening_cuda


def _score_argmax(values, means, n_forced, forced_mass, lengths, backend):
    prefix = values.cumsum(-1)
    if backend is not None:
        return backend.score_argmax(prefix, means, n_forced, forced_mass, lengths)
    prefix = torch.cat((values.new_zeros((*values.shape[:-1], 1)), prefix), dim=-1)
    prefix += forced_mass[..., None]
    ell = torch.arange(values.shape[-1] + 1, device=values.device)
    scores = 2 * prefix / (means[..., None] + (n_forced[..., None] + ell) + 1)
    if lengths is not None:
        scores.masked_fill_(ell > lengths[..., None], -torch.inf)
    return scores.argmax(dim=-1)


def _solve_cuda_candidate_group(backend, rows, packed_indices, group, starts, lengths,
                                means, n_forced, forced_mass, masks):
    values, group_means, group_counts, group_mass, metadata = backend.gather_candidate_group(
        rows, packed_indices, group, starts, lengths, means, n_forced, forced_mass,
    )
    sorted_probs, order = values.sort(dim=-1, descending=True)
    del values
    opt_ell = _score_argmax(sorted_probs, group_means, group_counts, group_mass,
                            metadata[2] if len(group) > 1 else None, backend)
    del sorted_probs
    backend.scatter_candidate_group(masks, packed_indices, metadata, order, opt_ell)


def _candidate_groups(counts):
    """Bucket similar nonzero lengths without padding to the original D."""
    ordered = sorted((length, row) for row, length in enumerate(counts) if length)
    group = []
    minimum = 0
    for length, row in ordered:
        if group and (
            length > _MAX_PADDING_RATIO * minimum
            or length * (len(group) + 1) > _MAX_PADDED_ELEMENTS
            or len(group) >= _MAX_BATCHED_ROWS
        ):
            yield group
            group = []
        if not group:
            minimum = length
        group.append(row)
    if group:
        yield group


def _rma_dice_screening_statistics(probs):
    """Prepare fused CUDA statistics; None selects the portable reductions."""
    backend = _cuda_backend() if probs.is_cuda else None
    if backend is None:
        return None
    means, maxima, state = backend.screening_statistics(
        probs.reshape(-1, probs.shape[-1]), _ROUNDING_GUARD * torch.finfo(probs.dtype).eps,
    )
    return means.reshape(probs.shape[:2]), maxima.reshape(probs.shape[:2]), state


def _rma_dice_validated_statistics(probs):
    """Fuse value validation for inputs that can be flattened without a copy.

    The caller checks structure, float32/float64 CUDA dtype and dispatch first.
    None leaves ordinary validation/statistics in place; no input is skipped.
    """
    try:
        rows = probs.view(probs.shape[0] * probs.shape[1], -1)
    except RuntimeError:
        return None
    backend = _cuda_backend()
    if backend is None:
        return None
    maximum, (means, maxima, state) = backend.validated_screening_statistics(
        rows, _ROUNDING_GUARD * torch.finfo(probs.dtype).eps,
    )
    return maximum, (means.reshape(probs.shape[:2]), maxima.reshape(probs.shape[:2]), state)


def _rma_dice_nonoverlap(masks, probs, means, active, policy, void_index):
    backend = _cuda_backend() if probs.is_cuda else None
    if backend is None:
        return None
    return backend.dice_nonoverlap_from_masks(masks, probs, means, active, policy == "void", void_index)


def _rma_dice_screened_masks(probs, pb_mean, active_mask, *, maxima=None, statistics=None):
    """Return binary masks for validated (B, C, D) float32/float64 inputs.

    All undecided candidates are solved, regardless of their length/density.
    Long rows run alone without padding; group limits never discard candidates.
    A rounding margin retains boundary probabilities; near-tied objective
    scores do not trigger a retry.
    """
    batch_size, num_classes, dim = probs.shape
    flat_probs = probs.reshape(-1, dim)
    backend = _cuda_backend() if probs.is_cuda else None
    # Prepared statistics apply class pruning on device for every row, without
    # copying full probability rows or synchronizing to discover active indices.
    # The legacy supplied-statistics path also supports this for a single row.
    device_pruning = backend is not None and (batch_size * num_classes == 1 or statistics is not None)
    active_indices = None if device_pruning else active_mask.reshape(-1).nonzero(as_tuple=True)[0]
    if not device_pruning and not active_indices.numel():
        return torch.zeros_like(probs, dtype=torch.bool)
    all_active = device_pruning or active_indices.numel() == batch_size * num_classes
    # Avoid scanning the inactive classes again at every screening step.
    # Specialized row copies avoid CUDA advanced-index launch failures on
    # extremely wide rows. nonzero gives unique, ascending row indices.
    rows = flat_probs if all_active else flat_probs.index_select(0, active_indices)
    means = pb_mean.reshape(-1) if all_active else pb_mean.reshape(-1)[active_indices]
    guard = _ROUNDING_GUARD * torch.finfo(probs.dtype).eps
    if maxima is None:
        maxima = rows.amax(dim=1)
    else:
        maxima = maxima.reshape(-1) if all_active else maxima.reshape(-1)[active_indices]

    # Move both certificates inward to keep near-boundary pixels undecided.
    # The flat scatter below must update this storage, not a reshape copy of
    # a channel-strided input's non-contiguous elementwise output.
    if backend is not None:
        masks, undecided, n_forced, forced_mass, partial_counts = backend.screening_candidates(
            rows, means, maxima, guard, active_mask.reshape(-1) if device_pruning else None,
            statistics=statistics, return_counts=True, materialize_candidates=False,
        )
        candidate_offsets = partial_counts.cumsum(-1)
        counts = candidate_offsets[:, -1]
        raw_lengths = counts.cpu().tolist()
    else:
        masks = (rows > 0.5 + guard).contiguous()
        n_forced = masks.sum(dim=1)
        forced_mass = torch.where(masks, rows, 0).sum(dim=1)
        lower = torch.maximum(
            torch.maximum(maxima / (means + 2), forced_mass / (n_forced + means + 1)),
            means / (dim + means + 1),
        )
        lower = (lower - guard).clamp(min=0, max=0.5)
        undecided = (rows >= lower[:, None]) & ~masks

    # A single row needs no padded workspace. CUDA reuses fused block counts;
    # the portable path gets the candidate count from nonzero.
    if rows.shape[0] == 1:
        if backend is None:
            indices = undecided[0].nonzero(as_tuple=True)[0]
            count = indices.numel()
        else:
            count = raw_lengths[0]
        if count:
            if backend is not None:
                indices = backend.pack_candidates(undecided, candidate_offsets, [count])
                _solve_cuda_candidate_group(backend, rows, indices, [0], [0], [count],
                                            means, n_forced, forced_mass, masks)
            else:
                values, order = rows[0, indices].sort(descending=True)
                opt_ell = _score_argmax(values, means[0], n_forced[0], forced_mass[0], None, backend)
                masks[0, indices[order]] = torch.arange(count, device=probs.device) < opt_ell
        if all_active:
            return masks.reshape_as(probs)
        output = torch.zeros_like(flat_probs, dtype=torch.bool)
        output.index_copy_(0, active_indices, masks)
        return output.reshape_as(probs)

    if backend is None:
        counts = undecided.sum(dim=1)
        raw_lengths = counts.cpu().tolist()
    # One metadata transfer drives bounded grouping; never call item() per
    # pixel or use a GPU-selected opt_ell as a Python slice boundary.
    lengths = raw_lengths
    if backend is None:
        packed_indices = undecided.reshape(-1).nonzero(as_tuple=True)[0]
    else:
        packed_indices = backend.pack_candidates(undecided, candidate_offsets, lengths)
    del undecided
    active_probs = rows.reshape(-1) if backend is None else None
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)

    for group in _candidate_groups(lengths):
        if backend is not None:
            _solve_cuda_candidate_group(
                backend, rows, packed_indices, group, [offsets[row] for row in group],
                [lengths[row] for row in group], means, n_forced, forced_mass, masks,
            )
            continue
        group_rows = torch.tensor(group, device=probs.device)
        group_lengths = [lengths[row] for row in group]
        width = max(group_lengths)
        total = sum(group_lengths)
        indices = torch.cat([packed_indices[offsets[row] : offsets[row + 1]] for row in group])
        if len(group) == 1:
            row = group[0]
            values, order = active_probs[indices].sort(descending=True)
            opt_ell = _score_argmax(values, means[row], n_forced[row], forced_mass[row], None, backend)
            masks.reshape(-1)[indices[order]] = torch.arange(width, device=probs.device) < opt_ell
            continue
        repeats = torch.tensor(group_lengths, device=probs.device)
        owner = torch.repeat_interleave(torch.arange(len(group), device=probs.device), repeats, output_size=total)
        starts = repeats.cumsum(0) - repeats
        position = torch.arange(total, device=probs.device) - torch.repeat_interleave(starts, repeats, output_size=total)

        # -1 sorts after every real probability, including zero. All invalid
        # prefixes are explicitly excluded before argmax.
        padded = probs.new_full((len(group), width), -1)
        padded[owner, position] = active_probs[indices]
        sorted_probs, order = padded.sort(dim=1, descending=True)
        opt_ell = _score_argmax(sorted_probs, means[group_rows], n_forced[group_rows],
                                forced_mass[group_rows], repeats, backend)

        # This permutation covers the padded WORKSPACE, not the output image.
        # Scatter back only real M entries, whose global indices are unique.
        # No padding index can overwrite H or another candidate's decision.
        selected = torch.arange(width, device=probs.device) < opt_ell[:, None]
        in_original_order = torch.empty_like(selected).scatter_(1, order, selected)
        masks.reshape(-1).scatter_(0, indices, in_original_order[owner, position])

    if all_active:
        return masks.reshape_as(probs)
    output = torch.zeros_like(flat_probs, dtype=torch.bool)
    output.index_copy_(0, active_indices, masks)
    return output.reshape_as(probs)
