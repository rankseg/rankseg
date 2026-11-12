# Author: Ben Dai <bendai@cuhk.edu.hk>, Zixun Wang <zixunwang@link.cuhk.edu.hk>
# License: BSD 3 clause

import warnings
from collections.abc import Sequence
from numbers import Integral, Real
from typing import Literal, Union

import torch
import torch.nn.functional as F

from rankseg._screening import (
    _rma_dice_nonoverlap, _rma_dice_screened_masks, _rma_dice_screening_statistics,
    _rma_dice_validated_statistics,
)
from rankseg._validation import (
    _validate_probability_values, validate_finite_real, validate_integral, validate_probability_tensor,
)
from rankseg.distribution import RefinedNormalPB

_TRNA_ACCELERATOR_PMF_CHUNK_SIZE = 128
_SCALED_SCORE_SMOOTH_THRESHOLD = 1e6
_RMA_CUDA_VECTORIZED_MASK_MAX_DIM = 524_288
# Decimal probability count B*C*D, not spatial size or bytes. This empirical
# auto policy trades modest latency increases for lower sorting memory.
_RMA_CUDA_SCREENING_AUTO_MIN_ELEMENTS = 1_280_000
_DEP_MAX_ITERS = 10
_DEP_TOL = 1e-6
# Bound the padded FFT workspace (elements per chunk), not the input size.
_DEP_FFT_CHUNK_NUMEL = 2**27


def _rma_dice_use_screening(probs: torch.Tensor, mode: Union[bool, Literal["auto"]]) -> bool:
    """Metadata-only dispatch for eligible, nonempty Dice inputs.

    True forces the portable or fused implementation. Auto uses screening on
    CPU, and on sufficiently large CUDA inputs with the optional backend.
    No probability scan, device synchronization or online timing is added.
    """
    if mode is True:
        return True
    if mode != "auto":
        return False
    if probs.device.type == "cpu":
        return True
    if probs.device.type == "cuda" and probs.numel() >= _RMA_CUDA_SCREENING_AUTO_MIN_ELEMENTS:
        from rankseg._screening import _cuda_backend

        return _cuda_backend() is not None
    return False


def _uses_host_refined_normal(device: torch.device) -> bool:
    """Return whether refined-normal work must be staged through the CPU."""
    return device.type != "cpu"


def _count_selected_pixels(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Count selected pixels without discarding the caller's working precision."""
    if dtype == torch.float64:
        # Integer reduction is exact before conversion and is faster than a
        # direct float64 reduction on the supported CPU/CUDA paths.
        return mask.sum(dim=2, keepdim=True).to(dtype=dtype)
    return mask.sum(dim=2, keepdim=True, dtype=dtype)


@torch.no_grad()
def rankdice_ba(
    probs: torch.Tensor, solver: str = "BA", smooth: float = 0.0, eps: float = 1e-4, pruning_prob: float = 0.5
) -> torch.Tensor:
    r"""
    Produce the predicted segmentation by `rankdice` based on the estimated output probability.

    Parameters
    ----------
    probs : Tensor, shape (batch_size, num_class, \*image_shape)
        The estimated probability tensor. Must use a real floating-point dtype
        with finite values in the range [0, 1]. Predictions are returned on the
        same device as this tensor.

    solver : str, {'TRNA', 'BA', 'BA+TRNA'}
        The approximate algorithm used to implement `RankDice`.
        Values are matched case-insensitively after stripping leading and
        trailing whitespace.
        `TRNA` indicates the truncated refined normal approximation (T-RNA), and
        `BA` indicates the blind approximation (BA),
        `BA+TRNA` indicates a combination of both BA and TRNA.

        - we use Cohen's d to determine if we use BA or TRNA
        - if Cohen's d is less than 0.2, we use BA; otherwise, we use TRNA

        With accelerator input, ranking and scoring remain on that device. The SciPy
        confidence interval and refined-normal PMFs are computed on the CPU;
        PMFs are copied back to the accelerator in bounded chunks, avoiding a transfer
        and synchronization for every TRNA search step.

    smooth : float, default=0.0
        A smooth parameter in the Dice metric.

    eps : float, default=1e-4
        Tail probability used to retain the central ``1 - eps`` refined-normal
        confidence interval. PMF support outside that interval is omitted, and
        the retained PMF is normalized. Must be finite and lie strictly between
        0 and 1.

    pruning_prob : float, default=0.5
        The threshold for pruning, if all probabilities are less than or equal to `pruning_prob`,
        we skip the class.

    Returns
    -------
    preds : Tensor, shape (batch_size, num_class, \*image_shape)
        The predicted segmentation based on `rankdice`, with dtype
        ``torch.bool``.

    Notes
    -----
    This is a discrete inference-time post-processing operation. Gradient
    recording is disabled internally, even when ``probs.requires_grad`` is
    ``True``. Float16 and bfloat16 inputs are promoted to float32 for the
    calculation; float32 and float64 inputs retain their original precision.

    References
    ----------
    :cite:p:`dai2023rankseg` Dai, B., & Li, C. (2023). RankSEG: a consistent ranking-based framework for
    segmentation. Journal of Machine Learning Research, 24(224), 1-50.
    """

    validate_probability_tensor(probs)
    if not isinstance(solver, str):
        raise TypeError("solver must be a string")
    solver = solver.strip().upper()
    if solver not in {"BA", "TRNA", "BA+TRNA"}:
        raise ValueError("Unknown solver: %s" % solver)
    smooth = validate_finite_real("smooth", smooth)
    if smooth < 0:
        raise ValueError("smooth must be greater than or equal to 0")
    eps = validate_finite_real("eps", eps)
    if not 0 < eps < 1:
        raise ValueError("eps must be in the range (0, 1)")
    pruning_prob = validate_finite_real("pruning_prob", pruning_prob)
    if not 0 <= pruning_prob <= 1:
        raise ValueError("pruning_prob must be in the range [0, 1]")

    # Half-precision inputs use float32 working tensors for numerical
    # stability. Preserve float32 and float64 so ranking, pruning, and scoring
    # respect the precision explicitly supplied by the caller.
    if probs.dtype in (torch.float16, torch.bfloat16):
        probs = probs.float()
    batch_size, num_class, *image_shape = probs.shape

    probs = torch.flatten(probs, start_dim=2, end_dim=-1)
    dim = probs.shape[-1]
    device = probs.device
    uses_host_refined_normal = _uses_host_refined_normal(device)
    ## initialize
    preds = torch.zeros(batch_size, num_class, dim, dtype=torch.bool, device=device)
    # prob_cutoff = torch.zeros(batch_size, num_class, device=device)
    ## precomputed constants
    discount = torch.arange(2 * dim + 1, dtype=probs.dtype, device=device)

    ## ranking (batch_size, num_class, dim);
    ## TBO: torch.sort is super memory consuming, anything to improve?
    sorted_prob, top_index = torch.sort(probs, dim=-1, descending=True)

    ## free memory
    del probs

    ## Compute ALL pruning masks upfront
    mask_skip = sorted_prob[:, :, 0] <= pruning_prob  # (batch, num_class)
    # mask_prune_tau = up_tau < lq ## since all prob are very small

    ## compute cumsum and the search upper bound
    cumsum_prob = torch.cumsum(sorted_prob, axis=-1)
    # Lemma 3 defines d0 as the first tau in {1, ..., dim} satisfying
    # cumsum(tau) >= (tau + smooth + dim) * p_(tau+1).  Compare by
    # multiplication to avoid division by zero.  The last candidate, tau=dim,
    # is the fallback because p_(dim+1) is conventionally zero.
    if dim == 1:
        up_tau = torch.ones((batch_size, num_class), dtype=torch.int64, device=device)
    else:
        search_denom = discount[1:dim] + smooth + dim
        if smooth > _SCALED_SCORE_SMOOTH_THRESHOLD:
            # Division remains defined when a very large finite smooth value
            # overflows the float32 denominator to infinity; multiplication
            # would instead form inf * 0 = nan at deterministic tails.
            stop_search = cumsum_prob[:, :, :-1] / search_denom >= sorted_prob[:, :, 1:]
        else:
            stop_search = cumsum_prob[:, :, :-1] >= search_denom * sorted_prob[:, :, 1:]
        first_stop = torch.argmax(stop_search.to(torch.int64), dim=-1) + 1
        up_tau = torch.where(stop_search.any(dim=-1), first_stop, dim)
        del stop_search

    # Refined-normal statistics and SciPy intervals remain on CPU for every
    # accelerator backend. Keep ranking indices on the input device so final
    # mask construction does not copy them back.
    if uses_host_refined_normal:
        mask_skip = mask_skip.cpu()
        sorted_prob = sorted_prob.cpu()

    ## compute statistics of pb distribution (batch_size, num_class)
    var_eps = torch.finfo(sorted_prob.dtype).eps
    pb_mean = sorted_prob.sum(axis=-1)
    pb_var = torch.sum(sorted_prob * (1 - sorted_prob), axis=-1)
    pb_var_safe = torch.clamp(pb_var, min=var_eps)
    pb_scale = torch.sqrt(pb_var_safe)
    pb_m3 = torch.sum(sorted_prob * (1 - sorted_prob) * (1 - 2 * sorted_prob), axis=-1)
    pb_skew = pb_m3 / pb_var_safe ** (3 / 2)
    # For s >= 1, maximize s * (score - 1) rather than a score close to 1.
    # Waiting until s > 1e6 already loses float32 candidate differences at
    # and below that boundary. Keep s < 1 (especially s == 0) on its original
    # path so scaling does not shrink scores or introduce tiny-s division.
    use_scaled_scores = smooth >= 1.0

    for k in range(num_class):
        active_indices = torch.where(~mask_skip[:, k])[0]
        if active_indices.numel() == 0:
            continue
        active_positions = torch.arange(active_indices.numel())
        if solver == "BA+TRNA":
            cohens_d = 1.0 / torch.clamp(pb_scale[active_indices, k], min=1e-8)
            use_ba_mask = cohens_d < 0.2
            ba_positions = active_positions[use_ba_mask]
            trna_positions = active_positions[~use_ba_mask]
        elif solver == "BA":
            ba_positions = active_positions
            trna_positions = active_positions[:0]
        else:
            ba_positions = active_positions[:0]
            trna_positions = active_positions

        ## compute the PMF of the evaluation interval
        RNPB_rv = RefinedNormalPB(
            dim=dim, loc=pb_mean[active_indices, k], scale=pb_scale[active_indices, k], skew=pb_skew[active_indices, k]
        )

        # Step 1: truncate the evaluation interval [lq, uq] such that P(lq <= X <= uq) = 1 - eps
        lq, uq = RNPB_rv.interval(eps)
        # RefinedNormalPB.interval returns inclusive integer endpoints.
        max_CI = torch.max(uq - lq + 1).item()
        supp = torch.arange(max_CI) + lq.unsqueeze(-1)
        # Step 2: compute the PMF of the evaluation interval
        pmf_supp = RNPB_rv.pmf(supp)
        if uses_host_refined_normal:
            pmf_supp = pmf_supp.to(device)
        # pmf_supp = pmf_supp / torch.sum(pmf_supp, axis=1, keepdim=True)

        for active_pos_tensor in ba_positions:
            active_pos = int(active_pos_tensor.item())
            b = int(active_indices[active_pos].item())
            ## compute the PMF of the evaluation interval
            CI_tmp = (uq[active_pos] - lq[active_pos] + 1).item()
            pmf_tmp = pmf_supp[active_pos, :CI_tmp].view(1, 1, -1)
            pmf_tmp = pmf_tmp / torch.sum(pmf_tmp)
            ## use convolutional layer to compute (13) in RankSEG JMLR paper
            low_tmp = int(lq[active_pos].item())
            up_tau_tmp = int(up_tau[b, k].item())
            up_tmp = int(uq[active_pos].item()) + up_tau_tmp

            pi = torch.zeros(up_tau_tmp + 1, dtype=sorted_prob.dtype, device=device)
            # compute (13)
            # with torch.backends.cudnn.flags(enabled=False, deterministic=True, benchmark=True):
            # add cudnn even slower
            if use_scaled_scores:
                # Maximize smooth * (pi - 1), which is affine-equivalent to
                # pi but retains score differences when pi rounds to 1.
                denom_offset = discount[low_tmp:up_tmp].view(1, 1, -1)
                ma_tmp = F.conv1d(1.0 / (1.0 + (denom_offset + 2) / smooth), pmf_tmp)
                nu_range = F.conv1d(
                    -(denom_offset + 1) / (1.0 + (denom_offset + 1) / smooth),
                    pmf_tmp,
                )
            else:
                # left, right in (13) of the paper
                right_denom_tmp = (discount[low_tmp:up_tmp] + smooth + 1).view(1, 1, -1)
                ma_tmp = F.conv1d(1.0 / (right_denom_tmp + 1), pmf_tmp)
                nu_range = F.conv1d(smooth / right_denom_tmp, pmf_tmp)
            w_range = 2.0 * ma_tmp * cumsum_prob[b, k, :up_tau_tmp]
            ## compute score for the range: tilde pi in the paper
            pi[1:] = (w_range + nu_range).flatten()
            if use_scaled_scores:
                support = discount[low_tmp : low_tmp + CI_tmp]
                pi[0] = -torch.sum((support / (1.0 + support / smooth)) * pmf_tmp)
            elif smooth > 0:
                pi[0] = smooth * torch.sum((1.0 / (discount[low_tmp : low_tmp + CI_tmp] + smooth)) * pmf_tmp)
            else:
                # The Dice convention in the paper defines 0 / 0 as zero.
                pi[0] = 0.0
            ## find the optimal tau
            opt_tau = int(torch.argmax(pi).item())

            preds[b, k, top_index[b, k, :opt_tau]] = True
            # prob_cutoff[b,k] = sorted_prob[b,k,opt_tau]

        for active_pos_tensor in trna_positions:
            active_pos = int(active_pos_tensor.item())
            b = int(active_indices[active_pos].item())
            ## compute (12) in RankSEG JMLR paper
            ## compute v(x) when tau = 0
            CI_tmp = (uq[active_pos] - lq[active_pos] + 1).item()
            full_pmf = pmf_supp[active_pos, :CI_tmp].view(1, 1, -1)
            full_pmf = full_pmf / torch.sum(full_pmf)

            low_tmp = int(lq[active_pos].item())
            high_tmp = int(uq[active_pos].item()) + 1
            up_tau_tmp = int(up_tau[b, k].item())
            scores = torch.empty(up_tau_tmp + 1, dtype=sorted_prob.dtype, device=device)
            if use_scaled_scores:
                support = discount[low_tmp:high_tmp]
                scores[0] = -torch.sum((support / (1.0 + support / smooth)) * full_pmf)
            elif smooth > 0:
                scores[0] = smooth * torch.sum((1.0 / (discount[low_tmp:high_tmp] + smooth)) * full_pmf)
            else:
                scores[0] = 0.0

            w_vec = torch.zeros(CI_tmp, dtype=sorted_prob.dtype, device=device)

            def compute_trna_pmf(tau):
                ## compute the pmf of Gamma_{-j}
                # compute moments
                pb_mean_tmp = pb_mean[b, k] - sorted_prob[b, k, tau - 1]
                pb_var_tmp = pb_var[b, k] - sorted_prob[b, k, tau - 1] * (1 - sorted_prob[b, k, tau - 1])
                pb_var_tmp_safe = torch.clamp(pb_var_tmp, min=var_eps)
                pb_scale_tmp = torch.sqrt(pb_var_tmp_safe)
                pb_m3_tmp = pb_m3[b, k] - sorted_prob[b, k, tau - 1] * (1 - sorted_prob[b, k, tau - 1]) * (
                    1 - 2 * sorted_prob[b, k, tau - 1]
                )
                pb_skew_tmp = pb_m3_tmp / pb_var_tmp_safe ** (3 / 2)
                # eval pmf
                RNPB_rv_tmp = RefinedNormalPB(dim=dim - 1, loc=pb_mean_tmp, scale=pb_scale_tmp, skew=pb_skew_tmp)
                return RNPB_rv_tmp.pmf(torch.arange(low_tmp, high_tmp))

            def record_trna_score(tau, excluded_pmf):
                nonlocal w_vec
                excluded_pmf = excluded_pmf / torch.sum(excluded_pmf)
                ## compute w_vec according to (9)
                w_vec = w_vec + sorted_prob[b, k, tau - 1] * excluded_pmf

                ## compute omega_tau according to (12)
                support = discount[low_tmp:high_tmp]
                if use_scaled_scores:
                    ma_tmp = torch.sum(2.0 / (1.0 + (support + tau + 1) / smooth) * w_vec)
                    nu_offset = support + tau
                    nu_tmp = -torch.sum((nu_offset / (1.0 + nu_offset / smooth)) * full_pmf)
                else:
                    ma_tmp = torch.sum(2.0 / (support + tau + smooth + 1) * w_vec)
                    nu_tmp = smooth * torch.sum((1.0 / (support + tau + smooth)) * full_pmf)
                scores[tau] = ma_tmp + nu_tmp

            if uses_host_refined_normal:
                for chunk_start in range(1, up_tau_tmp + 1, _TRNA_ACCELERATOR_PMF_CHUNK_SIZE):
                    chunk_stop = min(chunk_start + _TRNA_ACCELERATOR_PMF_CHUNK_SIZE, up_tau_tmp + 1)
                    pmf_chunk = torch.stack([compute_trna_pmf(tau) for tau in range(chunk_start, chunk_stop)]).to(
                        device
                    )
                    for chunk_offset, tau in enumerate(range(chunk_start, chunk_stop)):
                        record_trna_score(tau, pmf_chunk[chunk_offset])
            else:
                for tau in range(1, up_tau_tmp + 1):
                    record_trna_score(tau, compute_trna_pmf(tau))

            opt_tau = int(torch.argmax(scores).item())
            preds[b, k, top_index[b, k, :opt_tau]] = True
            # prob_cutoff[b,k] = sorted_prob[b,k,opt_tau]

    return preds.reshape(batch_size, num_class, *image_shape)


@torch.no_grad()
def rankseg_rma(
    probs: torch.Tensor,
    metric: str = "dice",
    smooth: float = 0.0,
    output_mode: str = "multiclass",
    pruning_prob: float = 0.5,
    unassigned_policy: str = "max_score",
    void_index: int = 255,
    safe_screening: Union[bool, Literal["auto"]] = False,
) -> torch.Tensor:
    r"""
    Produce the predicted segmentation by `rankdice` based on the estimated output probability.

    Parameters
    ----------
    probs : Tensor, shape (batch_size, num_class, \*image_shape)
        The estimated probability tensor. Must use a real floating-point dtype
        with finite values in the range [0, 1].

    metric : str, default='dice'
        The metric aim to optimize, either 'iou' or 'dice'.

    output_mode : {'multiclass', 'multilabel'}, default='multiclass'
        Controls overlap behavior of the predictions.
        - 'multiclass': non-overlapping; each pixel belongs to exactly one class.
        - 'multilabel': overlapping; pixels can belong to multiple classes (binary mask per class).

    smooth : float, default=0.0
        A smooth parameter in the segmentation metric. When positive, the RMA
        volume search also compares the score of an empty mask (``tau=0``).

    pruning_prob : float, default=0.5
        The threshold for pruning, if all probabilities are less than or equal to `pruning_prob`,
        we skip the class.

    unassigned_policy : {'max_score', 'void'}, default='max_score'
        Policy for pixels that are not selected as positive by any class when
        converting individual binary masks to a multiclass prediction. This
        option is only applicable when ``output_mode='multiclass'``;
        ``'void'`` is rejected for multilabel output.

        RankSEG-RMA first builds one binary mask per class. In multiclass mode,
        a pixel can then be:

        - selected by exactly one class, in which case that class is used;
        - selected by multiple classes, in which case the class with the largest
          incremental score among the classes that selected the pixel is used
          (a non-selecting class cannot win);
        - selected by no class, in which case this policy is used.

        The incremental score is the RMA objective change from assigning the
        pixel after fixing pixels selected by exactly one class; it is not the
        raw pixel probability. Equal scores select the lowest eligible class
        index.

        If 'max_score', unassigned pixels are assigned to the active class with
        the largest incremental score. If pruning removes every class in a
        sample, all classes remain eligible for this fallback so that
        multiclass output can still assign a class to every pixel.

        If 'void', unassigned pixels are assigned to `void_index`.

    void_index : int, default=255
        Label used for unassigned pixels when `unassigned_policy == 'void'`.
        A non-default value is rejected unless ``output_mode='multiclass'`` and
        ``unassigned_policy='void'``.

        The value must fit in ``torch.int64`` and lie outside the valid class
        index range ``[0, num_class)`` so abstentions remain distinguishable
        from class predictions. The default value, 255, is therefore suitable
        only when there are fewer than 256 classes.

    safe_screening : bool or {'auto'}, default=False
        Control experimental two-sided screening for Dice with ``smooth=0``.
        False (the default) retains the original path; True forces screening regardless of
        input size. 'auto' screens CPU inputs, and CUDA inputs with Triton and
        at least 1,280,000 probability values (batch * classes * spatial size).
        Other CUDA inputs use optimized full sort; other devices retain the
        original path in auto mode. The threshold is empirical, not a speed
        guarantee. No probability scan or timing is added for dispatch.
        Sort only unresolved pixels, using bounded padded groups for different
        per-class lengths. Direct argmax prefers the smallest searched volume
        only on exactly equal computed maxima, without close-score retries.
        Optional Triton kernels fuse CUDA screening statistics and scoring;
        otherwise True uses pure PyTorch. Candidate size never triggers a
        full-sort retry. Other metric/smooth combinations
        retain the original full sort. Class pruning and multiclass assignment
        rules are unchanged, but binary and multiclass masks can differ.
        This can reduce sorting memory, but is not faster for every input.

    Returns
    -------
    preds : Tensor
        If ``output_mode == 'multilabel'``, returns boolean masks of shape
        (batch_size, num_class, \*image_shape) with dtype ``torch.bool``.
        Otherwise, returns class-index maps with dtype ``torch.int64`` and
        shape (batch_size, \*image_shape).

    Notes
    -----
    This is a discrete inference-time post-processing operation. Gradient
    recording is disabled internally, even when ``probs.requires_grad`` is
    ``True``.

    References
    ----------
    :cite:p:`wang2025rankseg` Wang, Z., & Dai, B. (2025). RankSEG-RMA: An Efficient Segmentation Algorithm via Reciprocal Moment Approximation. Advances in Neural Information Processing Systems (NeurIPS 2025).
    """

    validate_probability_tensor(probs, check_values=False)
    batch_size, num_classes, *image_shape = probs.shape
    auto_screening = isinstance(safe_screening, str) and safe_screening == "auto"
    screened_dice = (
        (safe_screening is True or (auto_screening and probs.device.type in ("cpu", "cuda")))
        and isinstance(metric, str) and metric.strip().lower() == "dice"
        and isinstance(smooth, Real) and not isinstance(smooth, bool) and smooth == 0
    )
    use_screening = screened_dice and batch_size > 0 and _rma_dice_use_screening(probs, safe_screening)
    prepared = None
    maximum = None
    # Preserve value-error precedence and the ordinary validation path for
    # unsupported modes/dtypes. Metadata-only gating never inspects GPU values.
    if use_screening and probs.is_cuda and probs.dtype in (torch.float32, torch.float64):
        validated = _rma_dice_validated_statistics(probs)
        if validated is not None:
            maximum, prepared = validated
        del validated
    if maximum is None:
        bounds = _validate_probability_values(probs)
        maximum = bounds[1] if bounds is not None else None
    if not isinstance(metric, str):
        raise TypeError("metric must be a string")
    if not isinstance(output_mode, str):
        raise TypeError("output_mode must be a string")
    if not isinstance(unassigned_policy, str):
        raise TypeError("unassigned_policy must be a string")
    if not isinstance(safe_screening, bool) and not auto_screening:
        raise TypeError("safe_screening must be a bool or 'auto'")
    smooth = validate_finite_real("smooth", smooth)
    if smooth < 0:
        raise ValueError("smooth must be greater than or equal to 0")
    pruning_prob = validate_finite_real("pruning_prob", pruning_prob)
    if not 0 <= pruning_prob <= 1:
        raise ValueError("pruning_prob must be in the range [0, 1]")
    if probs.dtype in (torch.float16, torch.bfloat16):
        probs = probs.float()

    metric = metric.strip().lower()
    output_mode = output_mode.strip().lower()
    unassigned_policy = unassigned_policy.strip().lower()
    if metric not in ["iou", "dice"]:
        raise ValueError("metric should be iou or dice")
    if output_mode not in ["multiclass", "multilabel"]:
        raise ValueError("output_mode should be multiclass or multilabel")
    if output_mode == "multiclass" and probs.shape[1] == 1:
        raise ValueError(
            "Single-channel probabilities cannot produce a binary multiclass label map; "
            "use output_mode='multilabel' or provide background and foreground channels"
        )
    if unassigned_policy not in ["max_score", "void"]:
        raise ValueError("unassigned_policy should be max_score or void")
    void_index = validate_integral("void_index", void_index)
    if output_mode == "multilabel" and unassigned_policy != "max_score":
        raise ValueError("unassigned_policy='void' is only supported when output_mode='multiclass'")
    if void_index != 255 and (output_mode != "multiclass" or unassigned_policy != "void"):
        raise ValueError("a non-default void_index requires output_mode='multiclass' and unassigned_policy='void'")
    if output_mode == "multiclass" and unassigned_policy == "void":
        int64_info = torch.iinfo(torch.int64)
        if not int64_info.min <= void_index <= int64_info.max:
            raise ValueError("void_index must be representable as torch.int64")
        if 0 <= void_index < probs.shape[1]:
            raise ValueError("void_index must lie outside the valid class index range [0, num_class)")

    def compute_opt_tau(
        metric: str,
        pb_mean: torch.Tensor,
        cumsum_prob: torch.Tensor,
        dim: int,
        smooth: float,
    ):
        """Compute optimal tau and cutpoint based on the selected metric."""
        device = pb_mean.device
        if screened_dice:
            # This opt-in path has no smoothing terms. Reuse the prefix
            # workspace and account for the empty candidate without padding
            # another full-sized score tensor.
            offsets = torch.arange(2, dim + 2, device=device)
            scores = cumsum_prob.mul_(2).div_(pb_mean.unsqueeze(-1) + offsets)
            best, index = scores.max(dim=-1)
            return (index + 1).masked_fill_(best == 0, 0)
        taus = torch.arange(1, dim + 1, device=device).view(1, 1, -1)
        use_scaled_scores = smooth > _SCALED_SCORE_SMOOTH_THRESHOLD
        if metric == "dice":
            denom_offset = pb_mean.unsqueeze(-1) + taus
            if use_scaled_scores:
                metric_values = 2.0 * cumsum_prob / (1.0 + (denom_offset + 1) / smooth)
                metric_values -= denom_offset / (1.0 + denom_offset / smooth)
            else:
                discount = denom_offset + 1.0 + smooth
                metric_values = 2.0 * cumsum_prob / discount
                metric_values += smooth / (discount - 1)
        elif metric == "iou":
            denom_offset = pb_mean.unsqueeze(-1) - cumsum_prob + taus
            if use_scaled_scores:
                metric_values = (2.0 * cumsum_prob - pb_mean.unsqueeze(-1) - taus) / (1.0 + denom_offset / smooth)
            else:
                metric_values = (cumsum_prob + smooth) / (denom_offset + smooth)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported metric: {metric}")

        # The RMA paper considers tau in {1, ..., dim} for the unsmoothed
        # objective.  A positive smooth value also gives the empty mask a
        # nonzero score, so include its exact RMA score in the extended API.
        if smooth > 0:
            if use_scaled_scores:
                empty_values = -pb_mean / (1.0 + pb_mean / smooth)
            else:
                empty_values = smooth / (pb_mean + smooth)
        else:
            empty_values = torch.zeros_like(pb_mean)
        metric_values = torch.cat([empty_values.unsqueeze(-1), metric_values], dim=-1)

        # Get optimal tau indices, now including tau=0.
        opt_tau = torch.argmax(metric_values, dim=-1)
        # cutpoint = sorted_prob[torch.arange(batch_size)[:, None], torch.arange(num_class), opt_tau - 1]
        return opt_tau

    def convert_to_nonoverlap(
        overlap_preds: torch.Tensor,
        probs: torch.Tensor,
        metric: str,
        active_mask: torch.Tensor,
        pb_mean: torch.Tensor,
        smooth: float,
        pruning_prob: float,
        unassigned_policy: str,
        void_index: int,
    ) -> torch.Tensor:
        batch_size, _, dim = probs.size()

        if screened_dice:
            # Dispatch before materializing dense count/unique/score tensors.
            fused = _rma_dice_nonoverlap(
                overlap_preds, probs, pb_mean, active_mask, unassigned_policy, void_index,
            )
            if fused is not None:
                return fused

        class_counts = overlap_preds.sum(dim=1)
        unassigned_mask = class_counts == 0
        single_pred_mask = class_counts == 1
        has_selected_mask = class_counts > 0
        safe_to_predict = overlap_preds & single_pred_mask.unsqueeze(1)

        mu = (probs * safe_to_predict).sum(dim=2, keepdim=True)
        opt_tau = _count_selected_pixels(safe_to_predict, probs.dtype)

        use_scaled_scores = smooth > _SCALED_SCORE_SMOOTH_THRESHOLD

        if metric == "dice":
            denom_offset = opt_tau + pb_mean.unsqueeze(2) + 1
            if smooth == 0:
                # Factor the exact objective increment rather than subtracting
                # two almost equal overall scores. Large volumes can otherwise
                # lose or reverse float32 class-score differences. Threshold
                # and scale are per class, not per pixel; keep the class scale
                # because assignment compares increments across classes.
                raw_increment_scores = probs - mu / denom_offset
                raw_increment_scores.mul_(2.0 / (denom_offset + 1))
            elif use_scaled_scores:
                # Maximize smooth times the exact objective increment. Scaling
                # preserves the class ordering while retaining differences
                # that would round away when smooth dominates the denominator.
                scaled_base_denom = 1.0 + (denom_offset - 1) / smooth
                scaled_denom = 1.0 + denom_offset / smooth
                scaled_next_denom = 1.0 + (denom_offset + 1) / smooth
                raw_increment_scores = 2 * (probs * scaled_denom - mu / smooth) / (
                    scaled_denom * scaled_next_denom
                ) - 1.0 / (scaled_base_denom * scaled_denom)
            else:
                denom = denom_offset + smooth
                base_denom = denom - 1
                safe_base_denom = torch.where(base_denom == 0, torch.ones_like(base_denom), base_denom)
                raw_increment_scores = 2 * (probs * denom - mu) / (denom * (denom + 1))
                raw_increment_scores -= smooth / (safe_base_denom * denom)
                # A positive Python float can underflow to zero in a lower-
                # precision working dtype. For an empty, certainly-empty
                # class, adding a zero-probability pixel changes Dice from 1
                # to approximately 0, so its limiting increment is -1.
                raw_increment_scores = torch.where(
                    base_denom == 0,
                    torch.full_like(raw_increment_scores, -1.0),
                    raw_increment_scores,
                )
        else:
            denom_offset = opt_tau + pb_mean.unsqueeze(2) - mu
            if use_scaled_scores:
                scaled_denom = 1.0 + denom_offset / smooth
                scaled_next_denom = 1.0 + (denom_offset - probs + 1) / smooth
                raw_increment_scores = (probs * scaled_denom - (1.0 + mu / smooth) * (1.0 - probs)) / (
                    scaled_denom * scaled_next_denom
                )
            else:
                denom = denom_offset + smooth
                # With an empty mask, smooth=0, and an identically-zero class,
                # the base IoU is 0/0. Use the metric convention 0/0 := 0 so
                # that a NaN cannot make the zero-probability class win the
                # all-classes-pruned max-score fallback.
                safe_denom = torch.where(denom == 0, torch.ones_like(denom), denom)
                base_scores = (mu + smooth) / safe_denom
                # (a+p)/(b+1-p) - a/b = (p-(a/b)*(1-p))/(b+1-p).
                # Avoid cancellation between overall IoU scores. At b=0 the
                # validated input is an empty zero-probability class (p=a=0),
                # so the safe base score also preserves the 0/0 := 0 rule.
                remaining_prob = 1 - probs
                raw_increment_scores = probs - base_scores * remaining_prob
                raw_increment_scores.div_(denom + remaining_prob)
                del remaining_prob

        if unassigned_policy == "max_score":
            all_classes_pruned = ~active_mask.any(dim=1, keepdim=True)
            unassigned_eligible_classes = active_mask | all_classes_pruned
        else:
            unassigned_eligible_classes = active_mask

        # At an overlap, only classes whose binary masks selected the pixel are
        # eligible. At an unassigned pixel, max_score considers active classes,
        # or every class when pruning removed all classes from that sample.
        # Reuse the binary-mask storage on CPU to avoid allocating another
        # class-by-pixel mask; accelerators are faster with one vectorized mask.
        if probs.device.type == "cpu":
            eligible_mask = overlap_preds
            if unassigned_policy == "max_score":
                for b in range(batch_size):
                    eligible_mask[b, :, unassigned_mask[b]] = unassigned_eligible_classes[b].unsqueeze(1)
        else:
            eligible_mask = torch.where(
                has_selected_mask.unsqueeze(1),
                overlap_preds,
                unassigned_eligible_classes.unsqueeze(2),
            )
        eligible_increment_scores = torch.where(
            eligible_mask,
            raw_increment_scores,
            float("-inf"),
        )

        # Eligibility makes this argmax the final class map for single-mask,
        # overlapping, and max-score unassigned pixels alike.
        nonoverlap_predicts = eligible_increment_scores.argmax(dim=1)
        if unassigned_policy == "void":
            nonoverlap_predicts[unassigned_mask] = void_index

        return nonoverlap_predicts

    def full_sort_masks(probs: torch.Tensor, pb_mean: torch.Tensor):
        """Keep the original full-sort search and mask reconstruction together."""
        batch_size, num_classes, dim = probs.shape
        device = probs.device
        sorted_prob, top_index = torch.sort(probs, dim=-1, descending=True)
        active_mask = sorted_prob[:, :, 0] > pruning_prob
        cumsum_prob = torch.cumsum(sorted_prob, dim=-1)
        opt_tau = compute_opt_tau(metric, pb_mean, cumsum_prob, dim, smooth)
        use_vectorized_mask = (
            device.type == "cuda" and (num_classes > 1 or screened_dice)
            and dim <= _RMA_CUDA_VECTORIZED_MASK_MAX_DIM
        )
        if use_vectorized_mask:
            rank_positions = torch.arange(dim, device=device).view(1, 1, -1)
            selected_by_rank = rank_positions < opt_tau.unsqueeze(-1)
            selected_by_rank &= active_mask.unsqueeze(-1)
            # Each top_index row is a full permutation, so scatter writes every
            # output position exactly once and an uninitialized destination is safe.
            overlap_preds = torch.empty_like(selected_by_rank).scatter_(2, top_index, selected_by_rank)
        else:
            overlap_preds = torch.zeros(batch_size, num_classes, dim, dtype=torch.bool, device=device)
            for b in range(batch_size):
                for c in range(num_classes):
                    if not active_mask[b, c]:
                        continue
                    overlap_preds[b, c, top_index[b, c, : opt_tau[b, c]]] = True
        return overlap_preds, active_mask

    probs = torch.flatten(probs, start_dim=2, end_dim=-1)
    all_pruned = screened_dice and maximum is not None and maximum <= pruning_prob
    if all_pruned and output_mode == "multilabel":
        # Value validation already copied the global maximum to the host.
        # Reuse it: no extra reduction/synchronization or candidate pipeline.
        return torch.zeros((batch_size, num_classes, *image_shape), device=probs.device, dtype=torch.bool)
    if all_pruned:
        # Retain the original torch.sum reduction for all-pruned max_score.
        # Reusing the fused sum here could change near-tied fallback labels.
        del prepared
        pb_mean = probs.sum(dim=-1)
        overlap_preds = torch.zeros_like(probs, dtype=torch.bool)
        active_mask = torch.zeros_like(pb_mean, dtype=torch.bool)
    elif use_screening:
        if prepared is None:
            prepared = _rma_dice_screening_statistics(probs)
        if prepared is None:
            pb_mean, maxima = probs.sum(dim=-1), probs.amax(dim=-1)
            statistics = None
        else:
            pb_mean, maxima, statistics = prepared
        active_mask = maxima > pruning_prob
        overlap_preds = _rma_dice_screened_masks(
            probs, pb_mean, active_mask, maxima=maxima,
            statistics=statistics,
        )
        del prepared, statistics
    else:
        pb_mean = probs.sum(dim=-1)
        overlap_preds, active_mask = full_sort_masks(probs, pb_mean)

    if output_mode == "multilabel":
        preds = overlap_preds.reshape(batch_size, num_classes, *image_shape)
    else:
        nonoverlap_preds = convert_to_nonoverlap(
            overlap_preds,
            probs,
            metric,
            active_mask,
            pb_mean,
            smooth,
            pruning_prob,
            unassigned_policy,
            void_index,
        )
        preds = nonoverlap_preds.reshape(batch_size, *image_shape)

    return preds if output_mode == "multilabel" else preds.long()


def _dep_half_widths(kernel_size, image_shape):
    """Return per-axis kernel half widths, clipped to the largest displacement."""
    ndim = len(image_shape)
    if kernel_size is None:
        return tuple(n - 1 for n in image_shape)
    if isinstance(kernel_size, Integral) and not isinstance(kernel_size, bool):
        sizes = (int(kernel_size),) * ndim
    elif isinstance(kernel_size, Sequence) and not isinstance(kernel_size, str):
        sizes = tuple(validate_integral("kernel_size", size) for size in kernel_size)
        if len(sizes) != ndim:
            raise ValueError(f"kernel_size must have one entry per spatial dimension ({ndim})")
    else:
        raise TypeError("kernel_size must be None, an int, or a sequence of ints")
    if any(size <= 0 or size % 2 == 0 for size in sizes):
        raise ValueError("kernel_size entries must be positive odd integers")
    # Displacements beyond n - 1 never occur inside the image, so a larger
    # kernel is equivalent to the full kernel.
    return tuple(min(size // 2, n - 1) for size, n in zip(sizes, image_shape))


def _dep_gaussian_convolve(x: torch.Tensor, theta: float, half_widths) -> torch.Tensor:
    """Zero-padded linear convolution of x (rows, \\*image_shape) with a truncated Gaussian.

    The kernel exp(-||delta||^2 / (2 theta^2)) is separable, so an N-D
    convolution is a sequence of 1-D FFT convolutions, one per spatial axis.
    Padding each axis to n + half suffices to avoid circular wrap-around.
    """
    for axis, half in enumerate(half_widths):
        dim = axis + 1
        n = x.shape[dim]
        length = n + half
        offsets = torch.arange(length, device=x.device)
        signed = torch.where(offsets <= half, offsets, offsets - length).to(x.dtype)
        weights = torch.exp(-signed.square() / (2 * theta**2))
        weights.masked_fill_(signed.abs() > half, 0)
        view_shape = [1] * x.ndim
        view_shape[dim] = -1
        weights_fft = torch.fft.rfft(weights).view(view_shape)
        x = torch.fft.irfft(torch.fft.rfft(x, n=length, dim=dim) * weights_fft, n=length, dim=dim)
        x = x.narrow(dim, 0, n)
    return x


def _dep_volume_means(probs: torch.Tensor, theta: float, half_widths) -> torch.Tensor:
    r"""Estimate the per-pixel conditional volume mean under spatially localized dependence.

    For probability rows of shape (rows, \*image_shape), return
    ``q + (nu / p) * (nu * kappa)`` flattened to (rows, dim), where
    ``q = sum(p)``, ``nu = sqrt(p (1 - p))`` and ``kappa`` is the Gaussian kernel.
    """
    rows, *image_shape = probs.shape
    dim = probs[0].numel() if rows else 0
    nu = torch.sqrt(probs * (1 - probs) + 1e-10)
    padded_numel = dim
    for n, half in zip(image_shape, half_widths):
        padded_numel = padded_numel // n * (n + half)
    chunk = max(1, _DEP_FFT_CHUNK_NUMEL // max(padded_numel, 1))
    nu_conv = torch.empty_like(nu)
    for start in range(0, rows, chunk):
        nu_conv[start:start + chunk] = _dep_gaussian_convolve(nu[start:start + chunk], theta, half_widths)
    # Rounding in the FFT can produce tiny negative values for a
    # nonnegative convolution.
    nu_conv.clamp_(min=0)
    probs = probs.flatten(1)
    q = probs.sum(dim=-1, keepdim=True)
    return q + (nu.flatten(1) / (probs + 1e-8)) * nu_conv.flatten(1)


def _dep_fixed_point_masks(
    probs: torch.Tensor,
    mu: torch.Tensor,
    sorted_prob: torch.Tensor,
    sorted_index: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    r"""Binary masks for rows of shape (rows, dim) by the DEP fixed-point search.

    Each iteration ranks pixels by the current score, maximizes the
    second-order reciprocal moment approximation of
    ``sum_{j in S} p_j / (|S| + mu_j)`` over the volume, and re-ranks by
    ``p / (mu + tau)``, where tau is the zero-based optimal index as in the
    reference implementation. A row stops once its best objective changes by
    less than ``_DEP_TOL``, after at most ``_DEP_MAX_ITERS`` iterations.
    Returns the masks and the number of rows that did not converge, which keep
    their last iterate.
    """
    rows, dim = probs.shape
    device, dtype = probs.device, probs.dtype
    taus = torch.arange(1, dim + 1, device=device, dtype=dtype)
    final_index = torch.empty_like(sorted_index)
    final_tau = torch.zeros(rows, dtype=torch.long, device=device)
    best_prev = torch.full((rows,), -1.0, dtype=dtype, device=device)
    score = None
    pending = torch.arange(rows, device=device)
    for it in range(_DEP_MAX_ITERS):
        if it == 0:
            index, ranked_prob = sorted_index, sorted_prob
        else:
            index = torch.sort(score, dim=-1, descending=True)[1]
            ranked_prob = torch.gather(probs[pending], -1, index)
        row_mu = mu[pending]
        ranked_mu = torch.gather(row_mu, -1, index)

        mean = torch.cumsum(ranked_mu, dim=-1).div_(taus)
        z0 = torch.cumsum(ranked_prob, dim=-1)
        weighted = ranked_prob * ranked_mu
        z1 = torch.cumsum(weighted, dim=-1)
        z2 = torch.cumsum(weighted.mul_(ranked_mu), dim=-1)
        del weighted, ranked_prob, ranked_mu
        discount = (taus + mean).reciprocal_()
        first = (z1 - mean * z0).mul_(discount.square())
        second = (z2 - 2 * mean * z1 + mean.square() * z0).mul_(discount.pow(3))
        objective = z0.mul_(discount).sub_(first).add_(second)
        del mean, z1, z2, discount, first, second

        opt_tau = torch.argmax(objective, dim=-1)
        best = objective.gather(-1, opt_tau.unsqueeze(-1)).squeeze(-1)
        del objective
        final_index[pending] = index
        final_tau[pending] = opt_tau + 1

        converged = (best - best_prev[pending]).abs() < _DEP_TOL
        best_prev[pending] = best
        score = probs[pending] / (row_mu + opt_tau.unsqueeze(-1))
        score = score[~converged]
        pending = pending[~converged]
        if pending.numel() == 0:
            break

    rank_positions = torch.arange(dim, device=device).view(1, -1)
    selected_by_rank = rank_positions < final_tau.unsqueeze(-1)
    masks = torch.empty_like(selected_by_rank).scatter_(1, final_index, selected_by_rank)
    return masks, pending.numel()


def rankseg_dep(
    probs: torch.Tensor,
    metric: str = "dice",
    smooth: float = 0.0,
    output_mode: str = "multiclass",
    pruning_prob: float = 0.5,
    theta: float = 300.0,
    kernel_size=None,
    unassigned_policy: str = "max_score",
    void_index: int = 255,
) -> torch.Tensor:
    r"""
    Produce the predicted segmentation by RankSEG under spatially localized label dependence (DEP).

    RankSEG-RMA assumes that pixel labels are conditionally independent given
    the image. DEP relaxes this assumption: labels of nearby pixels are
    correlated with strength given by a Gaussian kernel of bandwidth `theta`.
    The dependence enters through a per-pixel conditional volume mean, and
    each class mask is found by a fixed-point search over a second-order
    reciprocal moment approximation of the Dice objective.

    Parameters
    ----------
    probs : Tensor, shape (batch_size, num_class, \*image_shape)
        The estimated probability tensor. Must use a real floating-point dtype
        with finite values in the range [0, 1]. image_shape may have any number
        of dimensions (for example 2D images or 3D volumes).

    metric : str, default='dice'
        The metric to optimize. Only 'dice' is supported; 'iou' raises an error.

    smooth : float, default=0.0
        Accepted for interface compatibility with RMA. Only 0 is supported.

    output_mode : {'multiclass', 'multilabel'}, default='multiclass'
        Controls overlap behavior of the predictions.
        - 'multiclass': non-overlapping; each pixel belongs to exactly one class.
        - 'multilabel': overlapping; pixels can belong to multiple classes (binary mask per class).

    pruning_prob : float, default=0.5
        The threshold for pruning, if all probabilities are less than or equal to `pruning_prob`,
        we skip the class.

    theta : float, default=300.0
        Bandwidth, in pixels, of the Gaussian dependence kernel
        ``exp(-||delta||^2 / (2 theta^2))``. Must be finite and positive.

    kernel_size : None, int or sequence of int, default=None
        Spatial support of the dependence kernel. None uses every displacement
        within the image (no truncation). An int truncates every spatial axis
        to that odd size; a sequence gives one odd size per spatial axis.

    unassigned_policy : {'max_score', 'void'}, default='max_score'
        Policy for pixels that are not selected by any class in multiclass
        mode. Overlapping and unassigned pixels follow the same rules as RMA:
        an overlap goes to the selecting class with the largest incremental
        score; with 'max_score' an unassigned pixel goes to the active class
        (or any class, if pruning removes every class in a sample) with the
        largest incremental score; with 'void' it is assigned `void_index`.
        The incremental score is the DEP objective change from assigning the
        pixel after fixing pixels selected by exactly one class.

    void_index : int, default=255
        Label used for unassigned pixels when `unassigned_policy == 'void'`.
        The same restrictions as for RMA apply.

    Returns
    -------
    preds : Tensor
        If ``output_mode == 'multilabel'``, returns boolean masks of shape
        (batch_size, num_class, \*image_shape) with dtype ``torch.bool``.
        Otherwise, returns class-index maps with dtype ``torch.int64`` and
        shape (batch_size, \*image_shape).

    Notes
    -----
    This is a discrete inference-time post-processing operation. Gradient
    recording is disabled internally, even when ``probs.requires_grad`` is
    ``True``. A RuntimeWarning is issued if the fixed-point search does not
    converge within 10 iterations; the last iterate is used.

    References
    ----------
    :cite:p:`wang2026rankseg` Wang, Z., & Dai, B. (2026). On the Relaxation of Conditional Independence Assumption for Image Segmentation. Advances in Neural Information Processing Systems (NeurIPS 2026).
    """

    validate_probability_tensor(probs, check_values=False)
    _validate_probability_values(probs)
    if not isinstance(metric, str):
        raise TypeError("metric must be a string")
    if not isinstance(output_mode, str):
        raise TypeError("output_mode must be a string")
    if not isinstance(unassigned_policy, str):
        raise TypeError("unassigned_policy must be a string")
    smooth = validate_finite_real("smooth", smooth)
    if smooth != 0:
        raise ValueError("the DEP solver supports only smooth=0")
    pruning_prob = validate_finite_real("pruning_prob", pruning_prob)
    if not 0 <= pruning_prob <= 1:
        raise ValueError("pruning_prob must be in the range [0, 1]")
    theta = validate_finite_real("theta", theta)
    if theta <= 0:
        raise ValueError("theta must be positive")
    if probs.dtype in (torch.float16, torch.bfloat16):
        probs = probs.float()

    metric = metric.strip().lower()
    output_mode = output_mode.strip().lower()
    unassigned_policy = unassigned_policy.strip().lower()
    if metric == "iou":
        raise ValueError("the DEP solver supports only metric='dice'; IoU is not supported")
    if metric != "dice":
        raise ValueError("metric should be dice")
    if output_mode not in ["multiclass", "multilabel"]:
        raise ValueError("output_mode should be multiclass or multilabel")
    if output_mode == "multiclass" and probs.shape[1] == 1:
        raise ValueError(
            "Single-channel probabilities cannot produce a binary multiclass label map; "
            "use output_mode='multilabel' or provide background and foreground channels"
        )
    if unassigned_policy not in ["max_score", "void"]:
        raise ValueError("unassigned_policy should be max_score or void")
    void_index = validate_integral("void_index", void_index)
    if output_mode == "multilabel" and unassigned_policy != "max_score":
        raise ValueError("unassigned_policy='void' is only supported when output_mode='multiclass'")
    if void_index != 255 and (output_mode != "multiclass" or unassigned_policy != "void"):
        raise ValueError("a non-default void_index requires output_mode='multiclass' and unassigned_policy='void'")
    if output_mode == "multiclass" and unassigned_policy == "void":
        int64_info = torch.iinfo(torch.int64)
        if not int64_info.min <= void_index <= int64_info.max:
            raise ValueError("void_index must be representable as torch.int64")
        if 0 <= void_index < probs.shape[1]:
            raise ValueError("void_index must lie outside the valid class index range [0, num_class)")

    batch_size, num_classes, *image_shape = probs.shape
    half_widths = _dep_half_widths(kernel_size, image_shape)

    def convert_to_nonoverlap(
        overlap_preds: torch.Tensor,
        probs: torch.Tensor,
        mu: torch.Tensor,
        active_mask: torch.Tensor,
        mu_mask: torch.Tensor,
    ) -> torch.Tensor:
        class_counts = overlap_preds.sum(dim=1)
        unassigned_mask = class_counts == 0
        has_selected_mask = class_counts > 0
        safe_to_predict = overlap_preds & (class_counts == 1).unsqueeze(1)

        # With t pixels fixed for a class and d_j = t + mu_j + 1, the objective
        # sum_{i in P} 2 p_i / d_i changes by
        #   2 p_j / (d_j + 1) - sum_{i in P} 2 p_i / (d_i (d_i + 1))
        # when pixel j is added. Factoring the difference avoids subtracting
        # two almost equal overall objectives.
        opt_tau = safe_to_predict.sum(dim=2, keepdim=True).to(probs.dtype)
        denom = opt_tau + mu + 1
        next_denom = denom + 1
        penalty = (2 * probs / (denom * next_denom) * safe_to_predict).sum(dim=2, keepdim=True)
        raw_increment_scores = (2 * probs / next_denom).sub_(penalty)
        del denom, next_denom, penalty

        if unassigned_policy == "max_score":
            all_classes_pruned = ~active_mask.any(dim=1, keepdim=True)
            unassigned_eligible_classes = active_mask | all_classes_pruned
        else:
            unassigned_eligible_classes = active_mask
        eligible_mask = torch.where(
            has_selected_mask.unsqueeze(1),
            overlap_preds,
            unassigned_eligible_classes.unsqueeze(2),
        )
        eligible_increment_scores = torch.where(
            eligible_mask & mu_mask.unsqueeze(2),
            raw_increment_scores,
            float("-inf"),
        )
        nonoverlap_predicts = eligible_increment_scores.argmax(dim=1)
        if unassigned_policy == "void":
            nonoverlap_predicts[unassigned_mask] = void_index
        return nonoverlap_predicts

    flat_probs = torch.flatten(probs, start_dim=2, end_dim=-1)
    dim = flat_probs.shape[-1]
    sorted_prob, top_index = torch.sort(flat_probs, dim=-1, descending=True)
    active_mask = sorted_prob[:, :, 0] > pruning_prob

    # Volume means are needed for active classes and, for the multiclass
    # max-score fallback, for every class of a sample with no active class.
    mu_mask = active_mask.clone()
    if output_mode == "multiclass" and unassigned_policy == "max_score":
        mu_mask |= ~active_mask.any(dim=1, keepdim=True)
    mu_rows = mu_mask.flatten().nonzero().squeeze(1)
    row_probs = probs.reshape(batch_size * num_classes, *image_shape)[mu_rows]
    row_mu = _dep_volume_means(row_probs, theta, half_widths)
    del row_probs

    overlap_preds = torch.zeros(batch_size * num_classes, dim, dtype=torch.bool, device=flat_probs.device)
    active_rows = active_mask.flatten().nonzero().squeeze(1)
    if active_rows.numel() > 0:
        # Positions of the active rows within mu_rows; both are sorted.
        active_in_mu = torch.searchsorted(mu_rows, active_rows)
        overlap_preds[active_rows], unconverged = _dep_fixed_point_masks(
            flat_probs.reshape(-1, dim)[active_rows],
            row_mu[active_in_mu],
            sorted_prob.reshape(-1, dim)[active_rows],
            top_index.reshape(-1, dim)[active_rows],
        )
        if unconverged:
            warnings.warn(
                f"DEP fixed-point iteration did not converge for {unconverged} of {active_rows.numel()} "
                f"class masks within {_DEP_MAX_ITERS} iterations; using the last iterate",
                RuntimeWarning,
                stacklevel=2,
            )
    del sorted_prob, top_index
    overlap_preds = overlap_preds.view(batch_size, num_classes, dim)

    if output_mode == "multilabel":
        return overlap_preds.reshape(batch_size, num_classes, *image_shape)

    mu = torch.zeros(batch_size * num_classes, dim, dtype=flat_probs.dtype, device=flat_probs.device)
    mu[mu_rows] = row_mu
    del row_mu
    nonoverlap_preds = convert_to_nonoverlap(
        overlap_preds, flat_probs, mu.view(batch_size, num_classes, dim), active_mask, mu_mask,
    )
    return nonoverlap_preds.reshape(batch_size, *image_shape).long()
