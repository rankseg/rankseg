import torch
import torch.nn.functional as F
from rankseg import RefinedNormal

smooth = 0.1

## simluate output
output = torch.rand(32, 1, 64, 128)
batch_size, num_class, width, height = output.shape # (batch_size, num_class, width, height)
## flatten (width, height) output
output = torch.flatten(output, start_dim=-2, end_dim=-1) # (batch_size, num_class, width*height)
dim = output.shape[-1]

preds = torch.zeros(batch_size, num_class, dim, dtype=torch.bool)
tau_rd = torch.zeros(batch_size, num_class)
cutpoint_rd = torch.zeros(batch_size, num_class)

## precompute discount
discount = torch.arange(2*dim+1)

## ranking
sorted_prob, top_index = torch.sort(output, dim=-1, descending=True)
cumsum_prob = torch.cumsum(sorted_prob, axis=-1)
ratio_prob = cumsum_prob[:,:,:-1] / (sorted_prob[:,:,1:]+1e-5)

## compute statistics of pb distribution
pb_mean = sorted_prob.sum(axis=-1) # (batch_size, num_class)
pb_var = torch.sum(sorted_prob*(1-sorted_prob), axis=-1)
pb_m3 = torch.sum(sorted_prob*(1-sorted_prob)*(1 - 2*sorted_prob), axis=-1)
pb_skew = pb_m3 / pb_var**(3/2)

## compute up_tau (according to Lemma 1)
## we only need to search tau over the range [0, up_tau]
up_tau = torch.argmax(torch.where(ratio_prob - discount[1:dim] - smooth - dim > 0, 1, 0), axis=-1)
## if up_tau == 0, it means that the ratio_prob is always less than discount[1:dim] + smooth + dim
## in this case, we set up_tau to be dim-1; we cannot prune the search
up_tau = torch.where(up_tau == 0, dim-1, up_tau)

low_class, up_class = app_action_set(pb_mean=pb_mean,
                                    pb_var=pb_var,
                                    pb_m3=pb_m3,
                                    device=device,
                                    dim=dim)

def app_action_set(pb_mean, pb_var, pb_m3, device, dim, tol=1e-4):
    refined_normal = RN_rv()
    skew = (pb_m3 / pb_var**(3/2)).cpu() + 1e-5
    low_quantile = torch.tensor(refined_normal.ppf(tol, skew=skew), device=device)
    up_quantile = torch.tensor(refined_normal.ppf(1-tol, skew=skew), device=device)
    lower = torch.maximum(torch.floor(torch.sqrt(pb_var)*low_quantile + pb_mean) - 1, torch.tensor(0))
    upper = torch.minimum(torch.ceil(torch.sqrt(pb_var)*up_quantile + pb_mean), torch.tensor(dim))
    return lower.type(torch.int), upper.type(torch.int)