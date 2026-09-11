# Author: Ben Dai <bendai@cuhk.edu.hk>, Zixun Wang <zixunwang@link.cuhk.edu.hk>
# License: BSD 3 clause

import torch
import torch.nn.functional as F

from rankseg._validation import validate_finite_real, validate_integral, validate_probability_tensor
from rankseg.distribution import RefinedNormalPB

_TRNA_ACCELERATOR_PMF_CHUNK_SIZE = 128
_SCALED_SCORE_SMOOTH_THRESHOLD = 1e6
_RMA_CUDA_VECTORIZED_MASK_MAX_DIM = 524_288


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
    use_scaled_scores = smooth > _SCALED_SCORE_SMOOTH_THRESHOLD

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

    validate_probability_tensor(probs)
    if not isinstance(metric, str):
        raise TypeError("metric must be a string")
    if not isinstance(output_mode, str):
        raise TypeError("output_mode must be a string")
    if not isinstance(unassigned_policy, str):
        raise TypeError("unassigned_policy must be a string")
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
        sorted_prob: torch.Tensor,
        pb_mean: torch.Tensor,
        smooth: float,
        pruning_prob: float,
        unassigned_policy: str,
        void_index: int,
    ) -> torch.Tensor:
        batch_size, _, dim = probs.size()

        class_counts = overlap_preds.sum(dim=1)
        unassigned_mask = class_counts == 0
        single_pred_mask = class_counts == 1
        has_selected_mask = class_counts > 0
        safe_to_predict = overlap_preds & single_pred_mask.unsqueeze(1)

        mu = (probs * safe_to_predict).sum(dim=2, keepdim=True)
        opt_tau = _count_selected_pixels(safe_to_predict, probs.dtype)

        active_mask = sorted_prob[:, :, 0] > pruning_prob
        use_scaled_scores = smooth > _SCALED_SCORE_SMOOTH_THRESHOLD

        if metric == "dice":
            denom_offset = opt_tau + pb_mean.unsqueeze(2) + 1
            if smooth == 0:
                # Keep the common unsmoothed path minimal. This is the exact
                # change in 2 * mu / (tau + E[Y] + 1) after adding the pixel.
                raw_increment_scores = 2 * ((mu + probs) / (denom_offset + 1) - mu / denom_offset)
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
                raw_increment_scores = (mu + probs + smooth) / (denom - probs + 1) - base_scores

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

    device = probs.device
    batch_size, num_classes, *image_shape = probs.shape

    probs = torch.flatten(probs, start_dim=2, end_dim=-1)
    dim = probs.shape[-1]

    sorted_prob, top_index = torch.sort(probs, dim=-1, descending=True)
    pb_mean = probs.sum(dim=-1)
    cumsum_prob = torch.cumsum(sorted_prob, dim=-1)

    opt_tau = compute_opt_tau(metric, pb_mean, cumsum_prob, dim, smooth)
    use_vectorized_mask = device.type == "cuda" and num_classes > 1 and dim <= _RMA_CUDA_VECTORIZED_MASK_MAX_DIM
    if use_vectorized_mask:
        rank_positions = torch.arange(dim, device=device).view(1, 1, -1)
        selected_by_rank = rank_positions < opt_tau.unsqueeze(-1)
        selected_by_rank &= (sorted_prob[:, :, 0] > pruning_prob).unsqueeze(-1)
        # Each top_index row is a full permutation, so scatter writes every
        # output position exactly once and an uninitialized destination is safe.
        overlap_preds = torch.empty_like(selected_by_rank).scatter_(2, top_index, selected_by_rank)
    else:
        overlap_preds = torch.zeros(batch_size, num_classes, dim, dtype=torch.bool, device=device)
        for b in range(batch_size):
            for c in range(num_classes):
                if sorted_prob[b, c, 0] <= pruning_prob:
                    continue
                overlap_preds[b, c, top_index[b, c, : opt_tau[b, c]]] = True

    if output_mode == "multilabel":
        preds = overlap_preds.reshape(batch_size, num_classes, *image_shape)
    else:
        nonoverlap_preds = convert_to_nonoverlap(
            overlap_preds,
            probs,
            metric,
            sorted_prob,
            pb_mean,
            smooth,
            pruning_prob,
            unassigned_policy,
            void_index,
        )
        preds = nonoverlap_preds.reshape(batch_size, *image_shape)

    return preds if output_mode == "multilabel" else preds.long()
