# Author: Ben Dai <bendai@cuhk.edu.hk>
# License: BSD 3 clause

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from scipy.stats import rv_continuous
from rankseg._distribution import RefinedNormalPB

def rankdice_batch_(probs, 
                   solver='auto', 
                   smooth=0., 
                   pruning=True, 
                   verbose=0):
    """
    Produce the predicted segmentation by `rankdice` based on the estimated output probability.

    Parameters
    ----------
    probs: Tensor, shape (batch_size, num_class, width, height)
        The estimated probability tensor. 
    
    solver: str, {'exact', 'TRNA', 'BA'}
        The approximate algorithm used to implement `RankDice`. 
        `exact` indicates exact evaluation (under development),  
        `TRNA` indicates the truncated refined normal approximation (T-RNA), and 
        `BA` indicates the blind approximation (BA),
        `auto` indicates automatic selection of the solver: 
            - we use Cohen's d to determine if we use BA or TRNA
            - if Cohen's d is less than 0.2, we use BA; otherwise, we use TRNA
    
    smooth: float, default=0.0
        A smooth parameter in the Dice metric.
    
    verbose: bool, default=0
        Whether print the results for each batch and class.

    Return
    ------
    predict: Tensor, shape (batch_size, num_class, width, height)
        The predicted segmentation based on `rankdice`.

    tau_rd: Tensor, shape (batch_size, num_class)
        The total number of segmentation pixels

    cutpoint_rd: Tensor, shape (batch_size, num_class)
        The cutpoint of probabilties of segmentation pixels and non-segmentation pixels

    Reference
    ---------
    
    """
    ## for tests
    ## probs = torch.load('./tests/data/demo_probs.pt')
    ## labels = torch.load('./tests/data/demo_labels.pt')

    batch_size, num_class, width, height = probs.shape
    probs = torch.flatten(probs, start_dim=2, end_dim=-1)
    dim = probs.shape[-1]
    device = probs.device
    ## initialize
    preds = torch.zeros(batch_size, num_class, dim, dtype=torch.bool, device=device)
    cutpoint_rd = torch.zeros(batch_size, num_class, device=device)
    ## precomputed constants
    discount = torch.arange(2*dim+1, device=device)

    ## ranking (batch_size, num_class, dim)
    sorted_prob, top_index = torch.sort(probs, dim=-1, descending=True)
    
    ## Compute ALL pruning masks upfront
    mask_skip = (sorted_prob[:,:,0] < 0.5) & pruning  # (batch, num_class)
    # mask_prune_tau = up_tau < lq ## since all prob are very small

    ## compute cumsum and ratio
    cumsum_prob = torch.cumsum(sorted_prob, axis=-1)
    ratio_prob = cumsum_prob[:,:,:-1] / (sorted_prob[:,:,1:]+1e-5)

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

    for k in range(num_class):
        active_indices = torch.where(~mask_skip[:, k])[0]
        if len(active_indices) == 0:
            continue
        ba_indices, trna_indices = [], []
        if solver == 'auto':
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
        lq, uq = RNPB_rv.interval(1e-4)
        max_CI = torch.max(uq - lq).item()
        supp = torch.arange(max_CI) + lq.unsqueeze(-1)
        # Step 2: compute the PMF of the evaluation interval
        pmf_supp = RNPB_rv.pdf(supp)
        pmf_supp = pmf_supp / torch.sum(pmf_supp, axis=1, keepdim=True)

        for b_idx, b in enumerate(ba_indices):
            pmf_tmp = pmf_supp[b_idx].view(1,1,-1)
            ## use convolutional layer to compute (13) in RankSEG JMLR paper
            low_tmp, up_tmp = lq[b_idx], lq[b_idx] + max_CI + up_tau[b,k] - 1
            # left, right in (13) of the paper
            right_denom_tmp = (discount[low_tmp:up_tmp]+smooth+1).view(1,1,-1)

            pi = torch.zeros(up_tau[b,k]+1)
            # compute (13)
            with torch.backends.cudnn.flags(enabled=False, deterministic=True, benchmark=True):
                ma_tmp = F.conv1d(1.0/(right_denom_tmp+1), pmf_tmp)
                nu_range = F.conv1d(smooth/right_denom_tmp, pmf_tmp)
            w_range = 2.0*ma_tmp*cumsum_prob[b,k,:up_tau[b,k]]
            ## compute score for the range: tilde pi in the paper
            pi[1:] = (w_range + nu_range).flatten()
            pi[0] = smooth*torch.sum( (1./(discount[low_tmp:low_tmp+max_CI]+smooth)) * pmf_tmp)
            ## find the optimal tau
            opt_tau = torch.argmax(pi)
            best_score = pi[opt_tau]

            preds[b, k, top_index[b,k,:opt_tau]] = True
            cutpoint_rd[b,k] = sorted_prob[b,k,opt_tau]

            # print('TEST sample-%d; class-%d; mean_pb: %.1f; up_tau:%d; tau_best: %d; score_best: %.4f' %(b_idx, k, pb_mean[b,k], up_tau[b,k], opt_tau, best_score))

        for b_idx, b in enumerate(trna_indices):
            ## compute (12) in RankSEG JMLR paper
            best_score, opt_tau = 0.0, 0
            ## compute v(x) when tau = 0
            if smooth > 0:
                best_score = smooth*torch.sum((1./(discount[low_tmp:low_tmp+max_CI]+smooth))*pmf_tmp)
            CI_tmp = (uq[b,k] - lq[b,k]).item()
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
                pmf_tmp = RNPB_rv_tmp.pdf(torch.arange(lq[b,k], uq[b,k]))
                pmf_tmp = pmf_tmp / torch.sum(pmf_tmp)
                ## compute w_vec according to (9)
                w_vec = w_vec + sorted_prob[b,k,tau-1] * pmf_tmp
                ## compute omega_tau according to (12)
                ma_tmp = torch.sum(2./(discount[lq[b,k]:uq[b,k]]+tau+smooth+2)*w_vec)
                nu_tmp = smooth*torch.sum((1./(discount[lq[b,k]:uq[b,k]]+tau+smooth+1))*pmf_supp[b,k,:CI_tmp])
                score_tmp = ma_tmp + nu_tmp

                if score_tmp > best_score:
                    best_score = score_tmp
                    opt_tau = tau

                preds[b, k, top_index[b,k,:opt_tau]] = True
                cutpoint_rd[b,k] = sorted_prob[b,k,opt_tau]
    
    return preds.reshape(batch_size, num_class, width, height), cutpoint_rd