# Author: Ben Dai <bendai@cuhk.edu.hk>, Zixun Wang <zixunwang@link.cuhk.edu.hk>
# License: BSD 3 clause

import torch
import torch.nn.functional as F
from rankseg.distribution import RefinedNormalPB

def rankdice_ba(probs: torch.Tensor, 
                   solver: str='BA', 
                   smooth: float=0.0, 
                   eps: float=1e-4,
                   pruning_prob: float=0.5) -> torch.Tensor:
    """
    Produce the predicted segmentation by `rankdice` based on the estimated output probability.

    Parameters
    ----------
    probs : Tensor, shape (batch_size, num_class, \*image_shape)
        The estimated probability tensor. 
    
    solver : str, {'exact', 'TRNA', 'BA', 'BA+TRNA'}
        The approximate algorithm used to implement `RankDice`. 
        `exact` indicates exact evaluation (under development),
        `TRNA` indicates the truncated refined normal approximation (T-RNA), and 
        `BA` indicates the blind approximation (BA),
        `BA+TRNA` indicates a combination of both BA and TRNA.
        
        - we use Cohen's d to determine if we use BA or TRNA
        - if Cohen's d is less than 0.2, we use BA; otherwise, we use TRNA
    
    smooth : float, default=0.0
        A smooth parameter in the Dice metric.
    
    eps : float, default=1e-4
        The threshold for truncation of the pmf of posisson-binomial distribution, 
        if the probability is less than `eps`, we truncate it to 0.

    pruning_prob : float, default=0.5
        The threshold for pruning, if all probabilities are less than `pruning_prob`, 
        we skip the class.

    Returns
    -------
    preds : Tensor, shape (batch_size, num_class, \*image_shape)
        The predicted segmentation based on `rankdice`.

    References
    ----------
    :cite:p:`dai2023rankseg` Dai, B., & Li, C. (2023). RankSEG: a consistent ranking-based framework for
    segmentation. Journal of Machine Learning Research, 24(224), 1-50.
    """

    batch_size, num_class, *image_shape = probs.shape

    probs = torch.flatten(probs, start_dim=2, end_dim=-1)
    dim = probs.shape[-1]
    device = probs.device
    ## initialize
    preds = torch.zeros(batch_size, num_class, dim, dtype=torch.bool, device=device)
    # prob_cutoff = torch.zeros(batch_size, num_class, device=device)
    ## precomputed constants
    discount = torch.arange(2*dim+1, device=device)

    ## ranking (batch_size, num_class, dim); 
    ## TBO: torch.sort is super memory consuming, anything to improve?
    sorted_prob, top_index = torch.sort(probs, dim=-1, descending=True)

    ## free memory
    del probs
    
    ## Compute ALL pruning masks upfront
    mask_skip = (sorted_prob[:,:,0] < pruning_prob)  # (batch, num_class)
    # mask_prune_tau = up_tau < lq ## since all prob are very small

    ## compute cumsum and ratio
    cumsum_prob = torch.cumsum(sorted_prob, axis=-1)
    ratio_prob = cumsum_prob[:,:,:-1] / (sorted_prob[:,:,1:]+1e-5)

    ## move to cpu for saving GPU memory
    if device.type == 'cuda':
        mask_skip = mask_skip.cpu()
        sorted_prob, top_index = sorted_prob.cpu(), top_index.cpu()

    ## compute statistics of pb distribution (batch_size, num_class)
    pb_mean = sorted_prob.sum(axis=-1) 
    pb_var = torch.sum(sorted_prob*(1-sorted_prob), axis=-1)
    pb_scale = torch.sqrt(pb_var)
    pb_m3 = torch.sum(sorted_prob*(1-sorted_prob)*(1 - 2*sorted_prob), axis=-1)
    pb_skew = pb_m3 / pb_var**(3/2)

    ## compute up_tau (according to Lemma 1)
    ## we only need to search tau over the range [0, up_tau] (d0 in (13) in RankSEG JMLR paper)
    up_tau = torch.argmax(torch.where(ratio_prob - discount[1:dim] - smooth - dim > 0, 1, 0), axis=-1)
    ## if up_tau == 0, it means that the ratio_prob is always less than discount[1:dim] + smooth + dim
    ## in this case, we set up_tau to be dim-1; we cannot prune the grid search
    up_tau = torch.where(up_tau == 0, dim-1, up_tau)

    del ratio_prob

    for k in range(num_class):
        active_indices = torch.where(~mask_skip[:, k])[0]
        if len(active_indices) == 0:
            continue
        ba_indices, trna_indices = [], []
        if solver == 'BA+TRNA':
            cohens_d = 1.0 / torch.clamp(pb_scale[active_indices, k], min=1e-8)
            use_ba_mask = cohens_d < 0.2
            ba_indices = active_indices[use_ba_mask]
            trna_indices = active_indices[~use_ba_mask]
        elif solver == 'BA':
            ba_indices = active_indices
        elif solver == 'TRNA':
            trna_indices = active_indices
        # elif solver == 'exact':
        #     trna_indices = active_indices
        else:
            raise ValueError('Unknown solver: %s' % solver)

        ## compute the PMF of the evaluation interval
        RNPB_rv = RefinedNormalPB(dim=dim,
                                 loc=pb_mean[active_indices,k],
                                 scale=pb_scale[active_indices,k],
                                 skew=pb_skew[active_indices,k])
        
        # Step 1: truncate the evaluation interval [lq, uq] such that P(lq <= X <= uq) = 1 - p
        lq, uq = RNPB_rv.interval(eps)
        max_CI = torch.max(uq - lq).item()
        supp = torch.arange(max_CI) + lq.unsqueeze(-1)
        # Step 2: compute the PMF of the evaluation interval
        pmf_supp = RNPB_rv.pmf(supp)
        if device.type == 'cuda':
            pmf_supp = pmf_supp.to(device)
        # pmf_supp = pmf_supp / torch.sum(pmf_supp, axis=1, keepdim=True)

        for b_idx, b in enumerate(ba_indices):
            ## compute the PMF of the evaluation interval
            CI_tmp = (uq[b_idx] - lq[b_idx]).item()
            pmf_tmp = pmf_supp[b_idx, :CI_tmp].view(1,1,-1)
            pmf_tmp = pmf_tmp / torch.sum(pmf_tmp)
            ## use convolutional layer to compute (13) in RankSEG JMLR paper
            low_tmp, up_tmp = lq[b_idx], uq[b_idx] + up_tau[b,k] - 1
            # left, right in (13) of the paper
            right_denom_tmp = (discount[low_tmp:up_tmp]+smooth+1).view(1,1,-1)

            pi = torch.zeros(up_tau[b,k]+1)
            # compute (13)
            # with torch.backends.cudnn.flags(enabled=False, deterministic=True, benchmark=True):
            # add cudnn even slower
            ma_tmp = F.conv1d(1.0/(right_denom_tmp+1), pmf_tmp)
            nu_range = F.conv1d(smooth/right_denom_tmp, pmf_tmp)
            w_range = 2.0*ma_tmp*cumsum_prob[b,k,:up_tau[b,k]]
            ## compute score for the range: tilde pi in the paper
            pi[1:] = (w_range + nu_range).flatten()
            pi[0] = smooth*torch.sum( (1./(discount[low_tmp:low_tmp+CI_tmp]+smooth)) * pmf_tmp)
            ## find the optimal tau
            opt_tau = torch.argmax(pi)
            best_score = pi[opt_tau]

            preds[b, k, top_index[b,k,:opt_tau]] = True
            # prob_cutoff[b,k] = sorted_prob[b,k,opt_tau]

        for b_idx, b in enumerate(trna_indices):
            ## compute (12) in RankSEG JMLR paper
            best_score, opt_tau = 0.0, 0
            ## compute v(x) when tau = 0
            CI_tmp = (uq[b_idx] - lq[b_idx]).item()
            pmf_tmp = pmf_supp[b_idx, :CI_tmp].view(1,1,-1)
            pmf_tmp = pmf_tmp / torch.sum(pmf_tmp)

            if smooth > 0:
                best_score = smooth*torch.sum((1./(discount[low_tmp:low_tmp+max_CI]+smooth))*pmf_tmp)
            
            w_vec = torch.zeros(CI_tmp, dtype=torch.float32, device=device)
            
            for tau in range(1, up_tau[b,k]+1):
                ## compute the pmf of Gamma_{-j}
                # compute moments
                pb_mean_tmp = pb_mean[b,k] - sorted_prob[b,k,tau-1]
                pb_var_tmp = pb_var[b,k] - sorted_prob[b,k,tau-1]*(1 - sorted_prob[b,k,tau-1])
                pb_scale_tmp = torch.sqrt(pb_var_tmp)
                pb_m3_tmp = pb_m3[b,k] - sorted_prob[b,k,tau-1]*(1 - sorted_prob[b,k,tau-1])*(1 - 2*sorted_prob[b,k,tau-1])
                pb_skew_tmp = pb_m3_tmp / pb_var_tmp**(3/2)
                # eval pmf
                RNPB_rv_tmp = RefinedNormalPB(dim=dim-1, loc=pb_mean_tmp, scale=pb_scale_tmp, skew=pb_skew_tmp)
                pmf_tmp = RNPB_rv_tmp.pmf(torch.arange(lq[b_idx], uq[b_idx]))
                if device.type == 'cuda':
                    pmf_tmp = pmf_tmp.to(device)
                pmf_tmp = pmf_tmp / torch.sum(pmf_tmp)
                ## compute w_vec according to (9)
                w_vec = w_vec + sorted_prob[b,k,tau-1] * pmf_tmp
                
                ## compute omega_tau according to (12)
                ma_tmp = torch.sum(2./(discount[lq[b_idx]:uq[b_idx]]+tau+smooth+2)*w_vec)
                nu_tmp = smooth*torch.sum((1./(discount[lq[b_idx]:uq[b_idx]]+tau+smooth+1))*pmf_tmp)
                score_tmp = ma_tmp + nu_tmp

                if score_tmp > best_score:
                    best_score = score_tmp
                    opt_tau = tau

                preds[b, k, top_index[b,k,:opt_tau]] = True
                # prob_cutoff[b,k] = sorted_prob[b,k,opt_tau]
    
    return preds.reshape(batch_size, num_class, *image_shape)


def rankseg_rma(
        probs: torch.Tensor,
        metric: str="dice",
        smooth: float=0.0,
        output_mode: str='multiclass',
        pruning_prob: float=0.5,
    ) -> torch.Tensor:
    """
    Produce the predicted segmentation by `rankdice` based on the estimated output probability.

    Parameters
    ----------
    probs : Tensor, shape (batch_size, num_class, \*image_shape)
        The estimated probability tensor.

    metric : str, default='dice'
        The metric aim to optimize, either 'iou' or 'dice'.

    output_mode : {'multiclass', 'multilabel'}, default='multiclass'
        Controls overlap behavior of the predictions.
        - 'multiclass': non-overlapping; each pixel belongs to exactly one class.
        - 'multilabel': overlapping; pixels can belong to multiple classes (binary mask per class).

    smooth : float, default=0.0
        A smooth parameter in the Dice metric.

    pruning_prob : float, default=0.5
        The threshold for pruning, if all probabilities are less than `pruning_prob`, 
        we skip the class.

    Returns
    -------
    preds : Tensor
        Shape (batch_size, num_class, \*image_shape) if output_mode == 'multilabel',
        otherwise shape (batch_size, \*image_shape)

    References
    ----------
    :cite:p:`wang2025rankseg` Wang, Z., & Dai, B. (2025). RankSEG-RMA: An Efficient Segmentation Algorithm via Reciprocal Moment Approximation. arXiv preprint arXiv:2510.15362.
    """

    assert metric in ['iou', 'dice'], 'metric should be iou or dice'

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
        if metric == 'dice':
            discount = pb_mean.unsqueeze(-1) + taus + 1.0 + smooth
            metric_values = 2.0 * cumsum_prob / discount
            metric_values += smooth / (discount - 1)
        elif metric == 'iou':
            discount = pb_mean.unsqueeze(-1) - cumsum_prob + taus + smooth
            metric_values = (cumsum_prob + smooth) / discount
        else:
            raise ValueError(f'Unsupported metric: {metric}')

        # Get optimal tau indices
        opt_tau = torch.argmax(metric_values, dim=-1) + 1
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
        ) -> torch.Tensor:
        num_class = overlap_preds.size(0)
        nonoverlap_predict = torch.zeros_like(overlap_preds[0], dtype=torch.uint8)
        assert num_class <= 256, 'num_class should be less than 256, when using uint8'
        overlap_mask = overlap_preds.sum(0) > 1
        increment_score = torch.zeros_like(probs, dtype=torch.float32)
        for c in range(num_class):
            if sorted_prob[c][0] <= pruning_prob:
                continue
            safe_to_predict = overlap_preds[c] & ~overlap_mask
            nonoverlap_predict[safe_to_predict] = c
            mu = probs[c][safe_to_predict].sum().item()
            opt_tau_this_c = safe_to_predict.sum().item()
            if metric == 'dice':
                increment_score[c] = 2 * ((mu + probs[c]) / (opt_tau_this_c + pb_mean[c] + 2 + smooth) - mu / (opt_tau_this_c + pb_mean[c] + 1 + smooth))
                increment_score[c] += smooth * (1 / (opt_tau_this_c + pb_mean[c] + 1 + smooth) - 1 / (opt_tau_this_c + pb_mean[c] + smooth))
            else:
                increment_score[c] = (mu + probs[c] + smooth) / (opt_tau_this_c + pb_mean[c] - mu - probs[c] + 1 + smooth) - (mu + smooth) / (opt_tau_this_c + pb_mean[c] - mu)
        increment_argmax_mask = increment_score.argmax(0)
        nonoverlap_predict[overlap_mask] = increment_argmax_mask[overlap_mask].type(torch.uint8)
        return nonoverlap_predict

    return_binary_masks = (output_mode == 'multilabel')
    is_binary = (probs.shape[1] == 2) and not return_binary_masks

    if is_binary:
        probs = probs[:, 1:2, ...]
        num_classes = 1

    device = probs.device
    batch_size, num_classes, *image_shape = probs.shape

    probs = torch.flatten(probs, start_dim=2, end_dim=-1)
    dim = probs.shape[-1]

    sorted_prob, top_index = torch.sort(probs, dim=-1, descending=True)
    pb_mean = probs.sum(dim=-1)
    cumsum_prob = torch.cumsum(sorted_prob, dim=-1)

    overlap_preds = torch.zeros(batch_size, num_classes, dim, dtype=torch.bool, device=device)
    opt_tau = compute_opt_tau(metric, pb_mean, cumsum_prob, dim, smooth)
    for b in range(batch_size):
        for c in range(num_classes):
            if sorted_prob[b, c, 0] <= pruning_prob:  # TODO: review this prune
                continue
            overlap_preds[b, c, top_index[b, c, :opt_tau[b, c]]] = True

    if return_binary_masks:
        preds = overlap_preds.reshape(batch_size, num_classes, *image_shape)
    else:
        if is_binary:
            nonoverlap_preds = overlap_preds[:, 0, ...].long()
        else:
            nonoverlap_preds = torch.zeros(batch_size, dim, dtype=torch.uint8, device=device)
            for b in range(batch_size):
                nonoverlap_preds[b] = convert_to_nonoverlap(
                    overlap_preds[b],
                    probs=probs[b],
                    metric=metric,
                    sorted_prob=sorted_prob[b],
                    pb_mean=pb_mean[b],
                    smooth=smooth,
                    pruning_prob=pruning_prob,
                )
        preds = nonoverlap_preds.reshape(batch_size, *image_shape)

    return preds.long()
