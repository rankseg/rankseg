# Author: Ben Dai <bendai@cuhk.edu.hk>
# License: BSD 3 clause

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from scipy.stats import rv_continuous


def rank_dice(output, device, app=2, smooth=0., 
              allow_overlap=True, 
              truncate_mean=False, 
              pruning=True, 
              verbose=0):
    """
    Produce the predicted segmentation by `rankdice` based on the estimated output probability.

    Parameters
    ----------
    output: Tensor, shape (batch_size, num_class, width, height)
        The estimated probability tensor. 

    device: String, {'cpu', 'cuda'}
        Device class of `torch.device`.
    
    app: int, {0, 1, 2}
        The approximate algo used to implement `rankdice`. `0` indicates exact evaluation, `1` indicates the truncated refined normal approximation (T-RNA), and `2` indicates the blind approximation (BA).
    
    smooth: float, default=0.0
        A smooth parameter in the Dice metric.
    
    allow_overlap: bool, default=True
        Whether allow the overlapping in the resulting segmentation.
    
    truncate_mean: bool, default=True
        Whether truncate mean the mean in refined normal approx.
    
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
    batch_size, num_class, width, height = output.shape
    output = torch.flatten(output, start_dim=-2, end_dim=-1)
    dim = output.shape[-1]
    predict = torch.zeros(batch_size, num_class, dim, dtype=torch.bool)
    tau_rd = torch.zeros(batch_size, num_class)
    cutpoint_rd = torch.zeros(batch_size, num_class)
    predict = predict.to(device)
    discount = torch.arange(2*dim+1, device=device)

    ## ranking
    sorted_prob, top_index = torch.sort(output, dim=-1, descending=True)
    cumsum_prob = torch.cumsum(sorted_prob, axis=-1)
    ratio_prob = cumsum_prob[:,:,:-1] / (sorted_prob[:,:,1:]+1e-5)
    ## compute statistics
    if truncate_mean == True:
        pb_mean = 1.*( sorted_prob >= .5).sum(axis=-1)
    else:
        pb_mean = sorted_prob.sum(axis=-1)

    pb_var = torch.sum(sorted_prob*(1-sorted_prob), axis=-1)
    pb_m3 = torch.sum(sorted_prob*(1-sorted_prob)*(1 - 2*sorted_prob), axis=-1)

    up_tau = torch.argmax(torch.where(ratio_prob - discount[1:dim] - smooth - dim > 0, 1, 0), axis=-1)
    up_tau = torch.where(up_tau == 0, dim-1, up_tau)

    # up_tau = torch.argmax(( ratio_prob - 1.  > 1.8*pb_mean.view(batch_size, num_class,1) )*1, axis=-1)
    ## normal approx domain
    # low_class, up_class = app_action_set(pb_mean=pb_mean,
    #                                     pb_var=pb_var,
    #                                     dim=dim)
    low_class, up_class = app_action_set(pb_mean=pb_mean,
                                        pb_var=pb_var,
                                        pb_m3=pb_m3,
                                        device=device,
                                        dim=dim)

    for k in range(num_class):
        ## searching for optimal vol for each sample and each class
        for b in range(batch_size):
            if (sorted_prob[b,k,0] <= .5) and pruning:
                ## pruning for predicted TP = FP = 0
                continue
            ## If the mean is too small, do not use blind approx
            elif pb_mean[b,k] <= 50:
                ## mean is too small; it is too risky to use BA
                up_tau[b,k] = 5*pb_mean[b,k] + 1
                app_tmp = 1
                print('Sample-%d; Class-%d use T-RNA since mean_pb: %.1f' %(b, k, pb_mean[b,k]))
            else:
                app_tmp = app
            # print('TEST sample-%d; class-%d; mean_tau: %d; low_tau: %d, up_tau: %d' %(b, k, int(pb_mean[b,k]), low_class[b,k], up_tau[b,k]))
            if up_tau[b,k] <= low_class[b,k]:
                opt_tau = up_tau[b,k]
                predict[b, k, top_index[b,k,:opt_tau]] = True
                tau_rd[b,k] = opt_tau
                cutpoint_rd[b,k] = sorted_prob[b,k,opt_tau]
                print('Sample-%d; Class-%d: up_tau <= low_class, set tau_rd = %d' %(b, k, opt_tau))
                continue
            if app_tmp > 1:
                pmf_tmp = PB_RNA(pb_mean[b,k],
                                pb_var[b,k],
                                pb_m3[b,k],
                                device=device,
                                up=up_class[b,k], low=low_class[b,k])
                pmf_tmp = pmf_tmp / torch.sum(pmf_tmp)
                
                # print('sum of pmf_tmp: %.3f' %torch.sum(pmf_tmp))
                ## grid search for tau
                # tau_range = torch.arange(low_class[b,k]+1, up_tau[b,k]+1, device=device)
                # tau_range = tau_range.repeat(up_class[b,k]-low_class[b,k],1)
                # score_range = torch.sum(1./(discount[low_class[b,k]:up_class[b,k]]+tau_range.T)*pmf_tmp, axis=-1)*cumsum_prob[b,k,low_class[b,k]:up_tau[b,k]]
                # opt_add = torch.argmax(score_range)
                # best_score, opt_tau = score_range[opt_add], low_class[b,k]+opt_add + 1
                # print('TEST sample-%d; class-%d; tau_best: %d; score_best: %.4f' %(b, k, opt_tau, best_score))
                
                # # use convolutional layer
                low_tmp, up_tmp = low_class[b,k], up_class[b,k]+up_tau[b,k]-1
                # with torch.backends.cudnn.flags(enabled=False, deterministic=True, benchmark=True):
                ma_tmp = F.conv1d( (2./(discount[low_tmp:up_tmp]+smooth+2)).view(1,1,-1), pmf_tmp.view(1,1,-1))
                nu_range = F.conv1d( (smooth/(discount[low_tmp:up_tmp]+smooth+1)).view(1,1,-1), pmf_tmp.view(1,1,-1))
                
                w_range = ma_tmp*cumsum_prob[b,k,:up_tau[b,k]]
                score_range = w_range + nu_range
                score_range = score_range.flatten()
                # opt_add = torch.argmax(score_range)
                opt_tau = torch.argmax(score_range)+1
                best_score = score_range[opt_tau-1]
                score_zero = smooth*torch.sum( (1./(discount[low_class[b,k]:up_class[b,k]]+smooth)) * pmf_tmp )
                if best_score <= score_zero:
                    best_score = score_zero
                    opt_tau = 0
                if verbose == 1:
                    print('TEST sample-%d; class-%d; mean_pb: %.1f; up_tau:%d; tau_best: %d; score_best: %.4f' %(b, k, pb_mean[b,k], up_tau[b,k], opt_tau, best_score))
            if app_tmp <= 1:
                sorted_prob_tmp = sorted_prob[b,k]
                pmf_tmp_zero = PB_RNA(pb_mean[b,k],
                                pb_var[b,k],
                                pb_m3[b,k],
                                device=device,
                                up=up_class[b,k], low=low_class[b,k])
                pmf_tmp_zero = pmf_tmp_zero / torch.sum(pmf_tmp_zero)
                best_score, opt_tau = 0., 0
                if smooth > 0:
                    best_score = smooth*torch.sum((1./(discount[low_class[b,k]:up_class[b,k]]+smooth))*pmf_tmp_zero)
                w_old = torch.zeros(up_class[b,k]-low_class[b,k], dtype=torch.float32, device=device)
                for tau in range(1, up_tau[b,k]+1):
                    pb_mean_tmp = torch.maximum(pb_mean[b,k] - sorted_prob[b,k,tau-1], torch.tensor(0))
                    pb_var_tmp = pb_var[b,k] - sorted_prob[b,k,tau-1]*(1 - sorted_prob[b,k,tau-1])
                    pb_m3_tmp = pb_m3[b,k] - sorted_prob[b,k,tau-1]*(1 - sorted_prob[b,k,tau-1])*(1 - 2*sorted_prob[b,k,tau-1])
                    
                    pmf_tmp = PB_RNA(pb_mean_tmp,
                                    pb_var_tmp,
                                    pb_m3_tmp,
                                    device=device,
                                    up=up_class[b,k], low=low_class[b,k])
                    
                    pmf_tmp = pmf_tmp / torch.sum(pmf_tmp)
                    w_old = w_old + sorted_prob[b,k,tau-1] * pmf_tmp
                    omega_tmp = torch.sum(2./(discount[low_class[b,k]:up_class[b,k]]+tau+smooth+2)*w_old)
                    nu_tmp = smooth*torch.sum((1./(discount[low_class[b,k]:up_class[b,k]]+tau+smooth+1))*pmf_tmp_zero)
                    score_tmp = omega_tmp + nu_tmp
                    if score_tmp > best_score:
                        opt_tau = tau
                        best_score = score_tmp
                if verbose == 1:
                    print('(tau, score_tmp): (%d, %.4f)' %(tau, score_tmp))
                    print('sample-%d; class-%d; mean_tau: %d; up_tau: %d; tau_best: %d; score_best: %.4f' %(b, k, int(pb_mean[b,k]), up_tau[b,k], opt_tau, best_score))
            predict[b, k, top_index[b,k,:opt_tau]] = True
            tau_rd[b,k] = opt_tau
            cutpoint_rd[b,k] = sorted_prob[b,k,opt_tau]
    return predict.reshape(batch_size, num_class, width, height), tau_rd, cutpoint_rd

def app_action_set(pb_mean, pb_var, pb_m3, device, dim, tol=1e-4):
    refined_normal = RN_rv()
    skew = (pb_m3 / pb_var**(3/2)).cpu()
    low_quantile = torch.tensor(refined_normal.ppf(tol, skew=skew), device=device)
    up_quantile = torch.tensor(refined_normal.ppf(1-tol, skew=skew), device=device)
    lower = torch.maximum(torch.floor(torch.sqrt(pb_var)*low_quantile + pb_mean) - 1, torch.tensor(0))
    upper = torch.minimum(torch.ceil(torch.sqrt(pb_var)*up_quantile + pb_mean), torch.tensor(dim))
    return lower.type(torch.int), upper.type(torch.int)

# def app_action_set(pb_mean, pb_var, dim, tol=1e-4):
#     label_app = torch.where(pb_var <= .1618/1e-1, True, False)
#     # label_app = torch.where(pb_var <= .1618/1e-1, True, False)
#     normal_rv = scipy.stats.norm(0, 1)
#     quantile = normal_rv.ppf(tol)
#     quantile = torch.tensor(quantile)
#     lower = torch.maximum(torch.floor(torch.sqrt(pb_var)*quantile + pb_mean) - 1, torch.tensor(0))
#     upper = torch.minimum(torch.ceil(-torch.sqrt(pb_var)*quantile + pb_mean), torch.tensor(dim))
#     lower[label_app] = 0
#     upper[label_app] = dim
#     return lower.type(torch.int), upper.type(torch.int)

def PB_RNA(pb_mean, pb_var, pb_m3, device, up, low=0, top=None):
    candidate = torch.arange(low-1, up, device=device)
    norm_rv = torch.distributions.normal.Normal(0,1)
    ## x = (k + 1/2 - mu) / sqrt(var)
    candidate_val = (candidate + 0.5 - pb_mean) / torch.sqrt(pb_var)

    ## refined normal approximation
    ## G(x) = Phi(x) + gamma (1 - x^2) phi(x) / 6 
    cdf_tmp = norm_rv.cdf(candidate_val) + pb_m3*(1 - candidate_val**2)*torch.exp(norm_rv.log_prob(candidate_val)) / 6 / pb_var**(3/2)
    pmf_tmp = cdf_tmp[1:] - cdf_tmp[:-1]

    if top != None:
        if top < (up-low):
            split_val = torch.topk(pmf_tmp, top)[0][-1]
            pmf_tmp[pmf_tmp < split_val] = float(0.)
            pmf_tmp = pmf_tmp / pmf_tmp.sum()
    return torch.minimum(torch.maximum(pmf_tmp, torch.tensor(0)), torch.tensor(1))

class RN_rv(rv_continuous):
    def _argcheck(self, skew):
        return np.isfinite(skew)
    def _cdf(self, x, skew):
        return scipy.stats.norm.cdf(x) + skew*(1 - x**2)*scipy.stats.norm.pdf(x)/6
