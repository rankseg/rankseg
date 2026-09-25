"""Optional fused CUDA kernels for unsmoothed Dice screening only.

Imported lazily: neither CPU use nor the default full-sort solver needs Triton.
Keep the original full-image mean and original probabilities for multiclass
assignment. Kernels optimize binary searches and multiclass statistics/scoring.
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

_BLOCK = 4096
_MAX_GRID_Y = 65535
_SCORE_REDUCTION_BLOCK = 4096
# Two uint8 states are reserved: 0=unassigned and 1=overlapping.
_MAX_CACHED_UNIQUE_CHANNELS = 254


class _CandidateBounds(NamedTuple):
    """Replay the exact candidate predicate without storing a dense M mask.

    Probabilities must remain unchanged until packing. Integer block counts
    encode pruning; packing retains every unresolved candidate in active rows.
    """

    probabilities: torch.Tensor
    lower: torch.Tensor
    guard: float
    block: int


@triton.jit
def _divide(numerator, denominator):
    if numerator.dtype == tl.float64:
        return numerator / denominator
    else:
        return tl.div_rn(numerator, denominator)


@triton.jit
def _forced_partials(P, Mask, Count, Mass, Active, Sum, Maximum, Errors, D, S0, S1, PARTS: tl.constexpr,
                     GUARD: tl.constexpr, PRUNE: tl.constexpr, STATS: tl.constexpr,
                     VALIDATE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    x = block * BLOCK + tl.arange(0, BLOCK)
    p = tl.load(P + row.to(tl.int64) * S0 + x * S1, x < D, other=0)
    # Do not fold 0.5 + GUARD into a float32 literal for float64 inputs.
    active = tl.load(Active + row) if PRUNE else True
    forced = (x < D) & active & (p > tl.full((), 0.5, p.dtype) + GUARD)
    if not STATS:
        tl.store(Mask + row.to(tl.int64) * D + x, forced, x < D)
    tl.store(Count + row * PARTS + block, tl.sum(forced.to(tl.int64), 0))
    tl.store(Mass + row * PARTS + block, tl.sum(tl.where(forced, p, 0), 0))
    if STATS:
        tl.store(Sum + row * PARTS + block, tl.sum(p, 0))
        tl.store(Maximum + row * PARTS + block, tl.max(p, 0))
    if VALIDATE:
        # Explicit NaN handling: do not depend on floating-point max's NaN
        # propagation. Nonfinite errors take precedence over range errors.
        nonfinite = (p != p) | (tl.abs(p) == float("inf"))
        invalid = (p < 0) | (p > 1)
        error = tl.where(x < D, tl.where(nonfinite, 2, invalid.to(tl.int32)), 0)
        tl.store(Errors + row * PARTS + block, tl.max(error, 0))


@triton.jit
def _forced_final(Count, Mass, Mean, Maximum, N, H, Lower, D,
                  Sum, Maxima, Errors, Validation, ROWS, PARTS: tl.constexpr, GUARD: tl.constexpr,
                  STATS: tl.constexpr, VALIDATE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x = tl.arange(0, BLOCK)
    n = tl.sum(tl.load(Count + row * PARTS + x, x < PARTS, other=0), 0)
    h = tl.sum(tl.load(Mass + row * PARTS + x, x < PARTS, other=0), 0)
    if STATS:
        mu = tl.sum(tl.load(Sum + row * PARTS + x, x < PARTS, other=0), 0)
        maximum = tl.max(tl.load(Maxima + row * PARTS + x, x < PARTS, other=0), 0)
        tl.store(Mean + row, mu)
        tl.store(Maximum + row, maximum)
    else:
        mu = tl.load(Mean + row)
        maximum = tl.load(Maximum + row)
    lower = tl.maximum(_divide(maximum, mu + 2), _divide(h, n.to(mu.dtype) + mu + 1))
    lower = tl.maximum(lower, _divide(mu, D + mu + 1))
    lower = tl.minimum(tl.maximum(lower - GUARD, 0), 0.5)
    tl.store(N + row, n)
    tl.store(H + row, h)
    tl.store(Lower + row, lower)
    if VALIDATE:
        error = tl.max(tl.load(Errors + row * PARTS + x, x < PARTS, other=0), 0)
        tl.store(Validation + row, error)
        tl.store(Validation + ROWS + row, maximum)


@triton.jit
def _undecided(P, Mask, Lower, M, Active, Counts, D, S0, S1, PARTS: tl.constexpr,
               PRUNE: tl.constexpr, PREPARED: tl.constexpr, COUNT: tl.constexpr,
               MATERIALIZE: tl.constexpr, GUARD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    active = tl.load(Active + row) if PRUNE else True
    if active:
        p = tl.load(P + row.to(tl.int64) * S0 + x * S1, x < D, other=0)
        if PREPARED:
            forced = (x < D) & (p > tl.full((), 0.5, p.dtype) + GUARD)
            tl.store(Mask + row.to(tl.int64) * D + x, forced, x < D)
        else:
            forced = tl.load(Mask + row.to(tl.int64) * D + x, x < D, other=False)
        lower = tl.load(Lower + row)
        candidate = (x < D) & (p >= lower) & ~forced
        if MATERIALIZE:
            tl.store(M + row.to(tl.int64) * D + x, candidate, x < D)
        if COUNT:
            tl.store(Counts + row * PARTS + tl.program_id(1), tl.sum(candidate.to(tl.int64), 0))
    else:
        tl.store(Mask + row.to(tl.int64) * D + x, False, x < D)
        if MATERIALIZE:
            tl.store(M + row.to(tl.int64) * D + x, False, x < D)
        if COUNT:
            tl.store(Counts + row * PARTS + tl.program_id(1), 0)


@triton.jit
def _pack_candidates(Source, Lower, Offsets, RowStarts, Indices, D, S0, S1,
                     PARTS: tl.constexpr, RECOMPUTE: tl.constexpr,
                     GUARD: tl.constexpr, BLOCK: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    start = tl.load(RowStarts + row)
    end = tl.load(RowStarts + row + 1)
    # Pruned and fully resolved rows have zero retained length.
    if start < end:
        previous = tl.load(Offsets + row * PARTS + block - 1, block > 0, other=0)
        current = tl.load(Offsets + row * PARTS + block)
        if previous < current:
            x = block * BLOCK + tl.arange(0, BLOCK)
            if RECOMPUTE:
                # Only nonempty blocks of retained rows reread probabilities.
                # Reuse the stored bound and the original dtype/comparisons;
                # recomputing statistics here could change boundary membership.
                p = tl.load(Source + row.to(tl.int64) * S0 + x * S1, x < D, other=0)
                lower = tl.load(Lower + row)
                forced = p > tl.full((), 0.5, p.dtype) + GUARD
                candidate = (x < D) & (p >= lower) & ~forced
            else:
                candidate = tl.load(Source + row.to(tl.int64) * D + x, x < D, other=False)
            rank = tl.cumsum(candidate.to(tl.int32), 0) - 1
            tl.store(Indices + start + previous + rank, row.to(tl.int64) * D + x, candidate)


def pack_candidates(candidates, offsets, lengths):
    """Stable row-major compaction with a single, already-known host count.

    No atomics, dynamic nonzero allocation, or pixel-wise host transfers.
    Zero lengths omit rows; block offsets still describe each row's
    complete candidate set. Nonempty rows must retain their full count.
    A _CandidateBounds input replays the predicate only in nonempty blocks;
    the original dense-mask input remains supported for diagnostics.
    """
    recompute = isinstance(candidates, _CandidateBounds)
    source = candidates.probabilities if recompute else candidates
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    indices = torch.empty(starts[-1], device=source.device, dtype=torch.int64)
    if not starts[-1]:
        return indices
    row_starts = torch.tensor(starts, device=source.device, dtype=torch.int64)
    rows, dim = source.shape
    block = candidates.block if recompute else max(_BLOCK, triton.next_power_of_2(triton.cdiv(dim, _MAX_GRID_Y)))
    with torch.cuda.device(source.device):
        _pack_candidates[(rows, offsets.shape[1])](
            source, candidates.lower if recompute else source, offsets, row_starts, indices,
            dim, *source.stride(), offsets.shape[1], recompute,
            candidates.guard if recompute else 0., block,
        )
    return indices


@triton.jit
def _gather_candidate_group(P, Packed, Metadata, Mean, N, H, Values, GroupMean, GroupN, GroupH,
                            D, S0, S1, MS, NS, HS, W, GROUP_ROWS, BLOCK: tl.constexpr):
    block, group_row = tl.program_id(0), tl.program_id(1)
    row = tl.load(Metadata + group_row)
    start = tl.load(Metadata + GROUP_ROWS + group_row)
    length = tl.load(Metadata + 2 * GROUP_ROWS + group_row)
    x = block * BLOCK + tl.arange(0, BLOCK)
    valid = (x < W) & (x < length)
    index = tl.load(Packed + start + x, valid, other=0)
    # Packed indices use logical row-major positions, not storage offsets.
    # Read the original strides directly, without flattening/copying P.
    p = tl.load(P + row * S0 + (index - row * D) * S1, valid, other=-1)
    tl.store(Values + group_row.to(tl.int64) * W + x, p, x < W)
    if block == 0:
        tl.store(GroupMean + group_row, tl.load(Mean + row * MS))
        tl.store(GroupN + group_row, tl.load(N + row * NS))
        tl.store(GroupH + group_row, tl.load(H + row * HS))


def gather_candidate_group(rows, packed_indices, group, starts, lengths, means, n_forced, forced_mass):
    """Gather one bounded group in its original candidate order, padding by -1.

    Row/start/length metadata are already known on the host. Each nonempty
    group keeps the original width and row ordering; singleton values remain
    1-D so sort/cumsum use the same shapes as before. No owner/position array,
    concatenated candidate-index copy, or dense probability copy is needed.
    Statistics are copied, never recomputed; their vector strides are respected.
    """
    n_rows, width = len(group), max(lengths)
    metadata = torch.tensor([group, starts, lengths], device=rows.device, dtype=torch.int64)
    values = rows.new_empty((n_rows, width) if n_rows > 1 else (width,))
    group_means, group_mass = rows.new_empty(n_rows), rows.new_empty(n_rows)
    group_counts = n_forced.new_empty(n_rows)
    with torch.cuda.device(rows.device):
        _gather_candidate_group[(triton.cdiv(width, _BLOCK), n_rows)](
            rows, packed_indices, metadata, means, n_forced, forced_mass,
            values, group_means, group_counts, group_mass,
            rows.shape[1], *rows.stride(), means.stride(0), n_forced.stride(0), forced_mass.stride(0),
            width, n_rows, _BLOCK,
        )
    return values, group_means, group_counts, group_mass, metadata


@triton.jit
def _scatter_candidate_group(Mask, Packed, Metadata, Order, OptEll, W,
                             GROUP_ROWS, BLOCK: tl.constexpr):
    block, group_row = tl.program_id(0), tl.program_id(1)
    start = tl.load(Metadata + GROUP_ROWS + group_row)
    length = tl.load(Metadata + 2 * GROUP_ROWS + group_row)
    x = block * BLOCK + tl.arange(0, BLOCK)
    position = tl.load(Order + group_row.to(tl.int64) * W + x, x < W, other=0)
    # Only real candidates may write. Padding cannot overwrite H or another
    # row; sort's permutation and unique packed indices give exactly one write
    # per candidate, without atomics or a materialized inverse permutation.
    valid = (x < W) & (position < length)
    index = tl.load(Packed + start + position, valid, other=0)
    selected = x < tl.load(OptEll + group_row)
    tl.store(Mask + index, selected, valid)


def scatter_candidate_group(masks, packed_indices, metadata, order, opt_ell):
    """Write only group candidates into a dense mask using the sort permutation.

    Masks, order and opt_ell are contiguous; untouched H and pruned
    positions retain their values. No selected/inverse-permutation workspace.
    """
    n_rows, width = metadata.shape[1], order.shape[-1]
    with torch.cuda.device(masks.device):
        _scatter_candidate_group[(triton.cdiv(width, _BLOCK), n_rows)](
            masks, packed_indices, metadata, order, opt_ell, width, n_rows, _BLOCK,
        )


def _statistics(rows, means, maxima, guard, active=None, validate=False):
    """Reduce full-image statistics, optionally reusing supplied sum/max."""
    n_rows, dim = rows.shape
    block = _BLOCK
    parts = triton.cdiv(dim, block)
    # CUDA's second grid dimension is bounded. Grow tiles only when needed
    # for launch validity; this is not a workload/performance dispatch rule.
    if parts > _MAX_GRID_Y:
        block = triton.next_power_of_2(triton.cdiv(dim, _MAX_GRID_Y))
        parts = triton.cdiv(dim, block)
    count = torch.empty((n_rows, parts), device=rows.device, dtype=torch.int64)
    mass = rows.new_empty((n_rows, parts))
    # Validated preparation defers the full mask allocation until candidate
    # construction. Invalid and globally pruned inputs never need this buffer.
    masks = None if validate else torch.empty(rows.shape, device=rows.device, dtype=torch.bool)
    n_forced = count.new_empty(n_rows)
    forced_mass = rows.new_empty(n_rows)
    lower = rows.new_empty(n_rows)
    fused = means is None
    if fused:
        means, maxima = rows.new_empty(n_rows), rows.new_empty(n_rows)
        partial_sum, partial_max = torch.empty_like(mass), torch.empty_like(mass)
    else:
        means, maxima = means.contiguous(), maxima.contiguous()
        partial_sum = partial_max = mass  # unused when STATS=False
    errors = torch.empty((n_rows, parts), device=rows.device, dtype=torch.int32) if validate else count
    validation = rows.new_empty((2, n_rows)) if validate else mass
    mask_pointer = masks if masks is not None else mass  # unused when STATS=True
    active_pointer = active if active is not None else mask_pointer
    with torch.cuda.device(rows.device):
        _forced_partials[(n_rows, parts)](
            rows, mask_pointer, count, mass, active_pointer, partial_sum, partial_max, errors,
            dim, *rows.stride(), parts, guard, active is not None, fused, validate, block,
            enable_fp_fusion=False,
        )
        _forced_final[(n_rows,)](
            count, mass, means, maxima, n_forced, forced_mass, lower, dim,
            partial_sum, partial_max, errors, validation, n_rows, parts, guard, fused, validate,
            triton.next_power_of_2(parts),
            enable_fp_fusion=False,
        )
    result = (means, maxima, (masks, n_forced, forced_mass, lower))
    return (*result, validation) if validate else result


def screening_statistics(rows, guard):
    """Compute sum, max and H statistics in one full probability scan.

    The sum includes ALL pixels, including eventual pruned classes/negatives.
    The output mask buffer is initialized, with class pruning applied, when
    screening_candidates consumes the prepared state.
    """
    return _statistics(rows, None, None, guard)


def validated_screening_statistics(rows, guard):
    """Validate probabilities during the unchanged full-image statistics scan.

    Input structure and working dtype must be checked by the caller. The one
    validation host transfer also returns the global maximum for pruning.
    No candidate construction or full-size mask allocation precedes validation.
    """
    means, maxima, state, validation = _statistics(rows, None, None, guard, validate=True)
    errors, row_maxima = validation.cpu().tolist()
    error = max(errors)
    if error == 2:
        raise ValueError("probs must contain only finite values")
    if error == 1:
        raise ValueError("probs must be in the range [0, 1]")
    return max(row_maxima), (means, maxima, state)


def screening_candidates(rows, means, maxima, guard, active=None, statistics=None, return_counts=False,
                         materialize_candidates=True):
    """Build candidate metadata, optionally retaining a dense diagnostic mask.

    The compact form requires block counts and retains the exact stored lower
    bound. It must be packed before modifying its source probabilities.
    """
    if not materialize_candidates and not return_counts:
        raise ValueError("compact candidates require return_counts=True")
    prepared = statistics is not None
    if not prepared:
        _, _, statistics = _statistics(rows, means, maxima, guard, active)
    masks, n_forced, forced_mass, lower = statistics
    if masks is None:
        masks = torch.empty(rows.shape, device=rows.device, dtype=torch.bool)
    n_rows, dim = rows.shape
    block = max(_BLOCK, triton.next_power_of_2(triton.cdiv(dim, _MAX_GRID_Y)))
    parts = triton.cdiv(dim, block)
    candidates = torch.empty_like(masks) if materialize_candidates else _CandidateBounds(rows, lower, guard, block)
    counts = torch.empty((n_rows, parts), device=rows.device, dtype=torch.int64) if return_counts else masks
    active_pointer = active if active is not None else masks
    with torch.cuda.device(rows.device):
        _undecided[(n_rows, parts)](
            rows, masks, lower, candidates if materialize_candidates else masks,
            active_pointer, counts, dim, *rows.stride(), parts,
            active is not None, prepared, return_counts, materialize_candidates, guard, block,
        )
    result = (masks, candidates, n_forced, forced_mass)
    return (*result, counts) if return_counts else result


@triton.jit
def _score_partials(Prefix, Mean, N, H, Length, Values, Indices, W,
                    PARTS: tl.constexpr, RAGGED: tl.constexpr, BLOCK: tl.constexpr):
    # Candidate rows no longer have an absolute length cap. Put their blocks
    # on grid.x, whose limit is much larger than grid.y's 65535.
    row = tl.program_id(1)
    ell = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(Length + row) if RAGGED else W
    prefix = tl.load(Prefix + row.to(tl.int64) * W + ell - 1,
                     (ell > 0) & (ell <= length), other=0)
    mu = tl.load(Mean + row)
    n = tl.load(N + row)
    h = tl.load(H + row)
    score = _divide(2 * (prefix + h), mu + (n + ell).to(mu.dtype) + 1)
    score = tl.where(ell <= length, score, float("-inf"))
    best = tl.max(score, 0)
    first = tl.min(tl.where((ell <= length) & (score == best), ell, W + 1), 0)
    tl.store(Values + row * PARTS + tl.program_id(0), best)
    tl.store(Indices + row * PARTS + tl.program_id(0), first)


@triton.jit
def _reduce_score_partials(Values, Indices, ReducedValues, ReducedIndices,
                           PARTS, OUT_PARTS, BLOCK: tl.constexpr):
    block, row = tl.program_id(0), tl.program_id(1)
    x = block * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(Values + row.to(tl.int64) * PARTS + x, x < PARTS, other=float("-inf"))
    indices = tl.load(Indices + row.to(tl.int64) * PARTS + x, x < PARTS, other=9223372036854775807)
    best = tl.max(values, 0)
    first = tl.min(tl.where(values == best, indices, 9223372036854775807), 0)
    tl.store(ReducedValues + row.to(tl.int64) * OUT_PARTS + block, best)
    tl.store(ReducedIndices + row.to(tl.int64) * OUT_PARTS + block, first)


@triton.jit
def _score_final(Values, Indices, Result, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x = tl.arange(0, BLOCK)
    values = tl.load(Values + row * PARTS + x, x < PARTS, other=float("-inf"))
    indices = tl.load(Indices + row * PARTS + x, x < PARTS, other=9223372036854775807)
    best = tl.max(values, 0)
    first = tl.min(tl.where(values == best, indices, 9223372036854775807), 0)
    tl.store(Result + row, first)


def score_argmax(prefix, means, n_forced, forced_mass, lengths):
    """Score all valid prefixes (including ell=0) and reduce to first argmax.

    No full score tensor, epsilon comparison, host scalar or per-row launch.
    Prefix is a contiguous one- or two-dimensional cumsum workspace; scalar
    statistics/lengths are contiguous. Very long rows reduce block maxima in
    bounded stages; this preserves the first exact maximum without full sort.
    """
    width = prefix.shape[-1]
    n_rows = prefix.numel() // width
    parts = triton.cdiv(width + 1, _BLOCK)
    values = prefix.new_empty((n_rows, parts))
    indices = torch.empty((n_rows, parts), device=prefix.device, dtype=torch.int64)
    with torch.cuda.device(prefix.device):
        _score_partials[(parts, n_rows)](
            prefix, means, n_forced, forced_mass, lengths if lengths is not None else n_forced,
            values, indices, width, parts, lengths is not None, _BLOCK, enable_fp_fusion=False,
        )
        if parts == 1:
            return indices.reshape(prefix.shape[:-1])
        # Do not compile a final reduction tile proportional to the entire
        # candidate count. Carry original indices through bounded max/min
        # stages, including ties spanning different blocks. Normal small
        # candidate rows keep the existing single final reduction.
        while parts > _SCORE_REDUCTION_BLOCK:
            next_parts = triton.cdiv(parts, _SCORE_REDUCTION_BLOCK)
            reduced_values = prefix.new_empty((n_rows, next_parts))
            reduced_indices = indices.new_empty((n_rows, next_parts))
            _reduce_score_partials[(next_parts, n_rows)](
                values, indices, reduced_values, reduced_indices,
                parts, next_parts, _SCORE_REDUCTION_BLOCK,
            )
            values, indices, parts = reduced_values, reduced_indices, next_parts
        result = indices.new_empty(n_rows)
        _score_final[(n_rows,)](
            values, indices, result, parts, triton.next_power_of_2(parts),
        )
    return result.reshape(prefix.shape[:-1])


@triton.jit
def _unique_partials(P, Mask, Classes, Count, Mass, C: tl.constexpr, D, S0, S1, S2,
                     PARTS, CHANNELS: tl.constexpr, BLOCK: tl.constexpr, CACHE_UNIQUE: tl.constexpr):
    b = tl.program_id(1)
    block = tl.program_id(0)
    x = block * BLOCK + tl.arange(0, BLOCK)
    c = tl.arange(0, CHANNELS)
    row = b * C + c
    valid = (c[:, None] < C) & (x[None, :] < D)
    selected = tl.load(Mask + row[:, None].to(tl.int64) * D + x[None, :], valid, other=False)
    if CACHE_UNIQUE:
        # A single exact integer reduction carries both the count (low eight
        # bits) and the sum of class codes. C<=254 prevents a count carry; even
        # selecting every class gives a tag below 2**24, safely within int32.
        # The high bits identify a class only when count==1; never interpret
        # an overlapping pixel's summed codes as a class index.
        tag = tl.sum(tl.where(selected, ((c[:, None] + 2) << 8) + 1, 0), 0)
        count = tag & 255
        status = tl.where(count == 0, 0, tl.where(count == 1, tag >> 8, 1))
    else:
        count = tl.sum(selected.to(tl.int32), 0)
        # Saturating at two also supports C=256 without uint8 overflow.
        status = tl.minimum(count, 2)
    tl.store(Classes + b.to(tl.int64) * D + x, status, x < D)
    unique = selected & (count[None, :] == 1)
    p = tl.load(P + b.to(tl.int64) * S0 + c[:, None].to(tl.int64) * S1 + x[None, :] * S2,
                valid & unique, other=0)
    offset = row.to(tl.int64) * PARTS + block
    tl.store(Count + offset, tl.sum(unique.to(tl.int32), 1), c < C)
    tl.store(Mass + offset, tl.sum(p, 1), c < C)


@triton.jit
def _unique_reduce(Count, Mass, N, H, PARTS, OUT_PARTS, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    x = block * BLOCK + tl.arange(0, BLOCK)
    n = tl.load(Count + row * PARTS + x, x < PARTS, other=0).to(tl.int64)
    h = tl.load(Mass + row * PARTS + x, x < PARTS, other=0)
    tl.store(N + row * OUT_PARTS + block, tl.sum(n, 0))
    tl.store(H + row * OUT_PARTS + block, tl.sum(h, 0))


def unique_statistics(masks, probs, *, cache_unique=False):
    """Return uint8 status and per-class unique counts/probability sums.

    By default status is 0/1/2+ multiplicity. With cache_unique and C<=254,
    reuse the same byte as 0=unassigned, 1=overlapping, 2+class=unique; C=255/256
    retain multiplicity. Pass the same option to dice_nonoverlap to decode it.

    No B*C*D numeric copies, unique mask or probability-product workspace.
    Counts use exact integer reduction; nonnegative probability sums retain
    the input precision but can differ in rounding from torch.sum.
    Caller bounds C <= 256 and batch <= _MAX_GRID_Y, and supplies dense masks.
    Spatial tiles use grid.x; bounded hierarchical reductions also support
    volumes larger than 65535 tiles without growing a giant reduction kernel.
    """
    batch, channels, dim = probs.shape
    cache_unique = cache_unique and channels <= _MAX_CACHED_UNIQUE_CHANNELS
    block = 1024 if channels <= 8 else 64
    parts = triton.cdiv(dim, block)
    classes = torch.empty((batch, dim), dtype=torch.uint8, device=probs.device)
    counts = torch.empty((batch * channels, parts), dtype=torch.int32, device=probs.device)
    mass = probs.new_empty((batch * channels, parts))
    with torch.cuda.device(probs.device):
        _unique_partials[(parts, batch)](
            probs, masks, classes, counts, mass, channels, dim, *probs.stride(), parts,
            triton.next_power_of_2(channels), block, cache_unique, enable_fp_fusion=False,
            num_warps=8 if channels > 64 else 4,
        )
        while True:
            out_parts = triton.cdiv(parts, _BLOCK)
            n = torch.empty((batch * channels, out_parts), dtype=torch.int64, device=probs.device)
            h = probs.new_empty((batch * channels, out_parts))
            _unique_reduce[(batch * channels, out_parts)](
                counts, mass, n, h, parts, out_parts, min(_BLOCK, triton.next_power_of_2(parts)),
                enable_fp_fusion=False,
            )
            if out_parts == 1:
                return classes, n.reshape(batch, channels), h.reshape(batch, channels)
            counts, mass, parts = n, h, out_parts


def dice_nonoverlap_from_masks(masks, probs, means, active, void=False, void_index=255):
    """Fuse statistics before assignment; None leaves the portable path intact."""
    batch, channels, _ = probs.shape
    if channels > 256 or batch > _MAX_GRID_Y:
        return None
    if not batch:
        return torch.empty((batch, probs.shape[-1]), device=probs.device, dtype=torch.int64)
    masks = masks.contiguous()
    classes, n, h = unique_statistics(masks, probs, cache_unique=True)
    return dice_nonoverlap(masks, probs, classes, means, n, h, active, void, void_index,
                           cache_unique=True)


@triton.jit
def _assign_scored_classes(P, Mask, Classes, Mean, N, H, Active, Output, C: tl.constexpr, D,
                           S0, S1, S2, VOID: tl.constexpr, VOID_INDEX: tl.constexpr,
                           CHANNELS: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(1)
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = tl.arange(0, CHANNELS)
    row = b * C + c
    active = tl.load(Active + row, c < C, other=False)
    if not VOID:
        active = (c < C) & (active | (tl.sum(active.to(tl.int32), 0) == 0))
    classes = tl.load(Classes + b.to(tl.int64) * D + x, x < D, other=0)
    selected = tl.load(Mask + row[:, None].to(tl.int64) * D + x[None, :],
                       (c[:, None] < C) & (x[None, :] < D), other=False)
    eligible = (c[:, None] < C) & tl.where(classes[None, :] > 0, selected, active[:, None])
    p = tl.load(P + b.to(tl.int64) * S0 + c[:, None].to(tl.int64) * S1 + x[None, :] * S2,
                (c[:, None] < C) & (x[None, :] < D), other=0)
    mu = tl.load(H + row, c < C, other=0)
    n = tl.load(N + row, c < C, other=0).to(mu.dtype)
    mean = tl.load(Mean + row, c < C, other=0)
    denom = n + mean + 1
    score = 2 * (_divide(mu[:, None] + p, denom[:, None] + 1) - _divide(mu, denom)[:, None])
    score = tl.where(eligible, score, -float("inf"))
    best = tl.max(score, 0)
    winner = tl.min(tl.where(score == best[None, :], c[:, None], CHANNELS), 0).to(tl.int64)
    if VOID:
        winner = tl.where(classes == 0, VOID_INDEX, winner)
    tl.store(Output + b.to(tl.int64) * D + x, winner, x < D)


@triton.jit(noinline=True)
def _assign_scored_classes_isolated(P, Mask, Classes, Mean, N, H, Active, Output, C: tl.constexpr, D,
                                    S0, S1, S2, VOID: tl.constexpr, VOID_INDEX: tl.constexpr,
                                    CHANNELS: tl.constexpr, BLOCK: tl.constexpr):
    # A device call (not another kernel launch) keeps the high-class score
    # workspace separate from the unique branch and avoids register spills.
    _assign_scored_classes(P, Mask, Classes, Mean, N, H, Active, Output, C, D,
                           S0, S1, S2, VOID, VOID_INDEX, CHANNELS, BLOCK)


@triton.jit
def _assign_classes(P, Mask, Classes, Mean, N, H, Active, Output, C: tl.constexpr, D,
                    S0, S1, S2, VOID: tl.constexpr, VOID_INDEX: tl.constexpr,
                    CHANNELS: tl.constexpr, BLOCK: tl.constexpr, CACHE_UNIQUE: tl.constexpr):
    b = tl.program_id(1)
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = tl.arange(0, CHANNELS)
    row = b * C + c
    classes = tl.load(Classes + b.to(tl.int64) * D + x, x < D, other=0)
    unique = classes >= 2 if CACHE_UNIQUE else classes == 1
    # A unique binary selection has exactly one eligible class regardless of
    # its incremental score. Do not confuse this with a screened candidate or
    # a raw-probability argmax. Ignore padding when testing the whole tile.
    if tl.sum(((x < D) & ~unique).to(tl.int32), 0) == 0:
        if CACHE_UNIQUE:
            # Unique class identity was recorded during statistics: no second
            # class-by-pixel mask read or class reduction is needed here.
            winner = classes.to(tl.int64) - 2
        else:
            selected = tl.load(Mask + row[:, None].to(tl.int64) * D + x[None, :],
                               (c[:, None] < C) & (x[None, :] < D), other=False)
            winner = tl.max(tl.where(selected, c[:, None], 0), 0).to(tl.int64)
        tl.store(Output + b.to(tl.int64) * D + x, winner, x < D)
    else:
        if CHANNELS > 64:
            _assign_scored_classes_isolated(P, Mask, Classes, Mean, N, H, Active, Output, C, D,
                                            S0, S1, S2, VOID, VOID_INDEX, CHANNELS, BLOCK)
        else:
            _assign_scored_classes(P, Mask, Classes, Mean, N, H, Active, Output, C, D,
                                   S0, S1, S2, VOID, VOID_INDEX, CHANNELS, BLOCK)


def dice_nonoverlap(masks, probs, classes, means, n, h, active, void=False, void_index=255,
                    *, cache_unique=False):
    """Fuse assignment, optionally using unique_statistics' cached-class status.

    The default retains ordinary count/status inputs. Cached codes are used
    only for C<=254; higher supported class counts keep the original path.
    """
    batch, channels, dim = probs.shape
    if channels > 256 or batch > _MAX_GRID_Y:
        # Bound the class-by-pixel tile/register workspace; use portable
        # assignment for arbitrarily many classes, without changing semantics.
        return None
    output = torch.empty((batch, dim), device=probs.device, dtype=torch.int64)
    if not batch:
        return output
    cache_unique = cache_unique and channels <= _MAX_CACHED_UNIQUE_CHANNELS
    masks, means, active = masks.contiguous(), means.contiguous(), active.contiguous()
    with torch.cuda.device(probs.device):
        assignment_block = 256 if channels <= 8 else 32
        # Use grid axis 0 for spatial tiles: CUDA grid.y cannot cover a very
        # wide liver volume when using small assignment tiles.
        _assign_classes[(triton.cdiv(dim, assignment_block), batch)](
            probs, masks, classes, means, n, h, active, output, channels, dim, *probs.stride(),
            void, void_index, triton.next_power_of_2(channels), assignment_block, cache_unique,
            enable_fp_fusion=False,
        )
    return output
