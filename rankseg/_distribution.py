# Author: Ben Dai <bendai@cuhk.edu.hk>

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from torch.distributions import Distribution, constraints
from torch.distributions.normal import Normal
from torch.distributions.utils import broadcast_all

## example: https://pytorch-forecasting.readthedocs.io/en/v0.8.3/_modules/torch/distributions/negative_binomial.html

class RefinedNorm(Distribution):
    """Refined Normal distribution to approximate Poisson binomial distribution."""
    arg_constraints = {'skew': constraints.real}
    support = constraints.real
    has_rsample = False
    
    def __init__(self, skew, validate_args=None):
        self.skew = skew
        batch_shape = self.skew.shape
        super(RefinedNorm, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def cdf(self, x):
        norm = Normal(0, 1)
        prob = norm.cdf(x) + self.skew*(1 - x**2)*norm.log_prob(x).exp()/6
        return np.clip(prob, 0, 1)

    def pdf(self, x):
        norm = Normal(0, 1)
        pdf_value = norm.log_prob(x).exp()
        return pdf_value + self.skew/6*pdf_value*(3*x - x**3)

    def log_prob(self, x):
        return torch.log(self.pdf(x))

    def icdf(self, p, max_iter=1000, tol=1e-6):
        """Inverse CDF (quantile function) using Newton-Raphson method.
        
        Parameters
        ----------
        p : torch.Tensor
            Probability values (0 < p < 1)
        max_iter : int, optional
            Maximum number of iterations (default: 50)
        tol : float, optional
            Tolerance for convergence (default: 1e-6)
            
        Returns
        -------
        torch.Tensor
            Quantile values corresponding to probabilities p
        """
        # Clamp probabilities to valid range
        p = torch.clamp(p, 1e-8, 1 - 1e-8)
        
        # Initialize with normal quantiles as starting point
        norm = Normal(0, 1)
        x = norm.icdf(p)
        
        # Newton-Raphson iterations
        for _ in range(max_iter):
            fx = self.cdf(x) - p
            fpx = self.pdf(x)
            
            # Avoid division by zero
            fpx = torch.clamp(fpx, min=1e-10)
            x_new = x - fx / fpx

            # Check convergence
            if torch.max(torch.abs(x_new - x)) < tol:
                break
            x = x_new
        return x

## test
refined_norm = RefinedNorm(torch.tensor([1,2,3]))
print(refined_norm.cdf(torch.tensor([.1,.2,-.3])))
print(refined_norm.icdf(torch.tensor([.1,.2,.3])))


class RNA_BP(object):
    """Refined Normal Approximation for Poisson binomial distribution."""
    def __init__(self, pb_mean, pb_var, pb_m3, device):
        self.pb_mean = pb_mean
        self.pb_var = pb_var
        self.pb_m3 = pb_m3
        self.skew = (pb_m3 / pb_var**(3/2)).to(device)
        self.device = device
    
    def cdf(self, x):
        pass

    def pmf(self, x):
        pass

class RN_rv(scipy.stats.rv_continuous):
    """Refined Normal distribution to approximate Poisson binomial distribution.

    This class extends the continuous random variable class from SciPy to implement
    a modified normal distribution with a skewness correction term. The distribution
    is particularly effective for approximating the Poisson binomial distribution
    (the sum of independent but non-identical Bernoulli random variables).
    
    The CDF is defined as:
    F(x; skew) = Φ(x) + skew * (1 - x²) * φ(x) / 6
    
    where:
    - Φ(x) is the standard normal CDF
    - φ(x) is the standard normal PDF
    - skew is the skewness parameter
    
    Parameters
    ----------
    skew : float
        Skewness parameter controlling the third moment correction term.
        Must be a finite value.
    
    Notes
    -----
    This refined approximation offers improved accuracy over the standard normal
    approximation.
    
    In the context of RankSEG, this distribution is used to efficiently compute
    approximate the Poisson binomial distribution.

    References
    ----------
    .. [1] Volkova, A.Y., 1996. A refinement of the central limit theorem for sums 
           of independent random indicators. Theory of Probability and its 
           Applications 40, 791-794.
    """
    def _argcheck(self, skew):
        return np.isfinite(skew)

    def _cdf(self, x, skew):
        prob = scipy.stats.norm.cdf(x) + skew*(1 - x**2)*scipy.stats.norm.pdf(x)/6
        return np.clip(prob, 0, 1)
    
    def _pdf(self, x, skew):
        return scipy.stats.norm.pdf(x) + skew/6*scipy.stats.norm.pdf(x)*(3*x - x**3)

## test
refined_norm = RN_rv()
print(refined_norm.cdf([.1,.2,-.3], skew=[1,2,3]))
print(refined_norm.ppf([.1,.2,.3], skew=[1,2,3]))

def app_action_set(pb_mean, pb_var, pb_m3, device, dim, tol=1e-4):
    """Compute approximate action set bounds for Poisson binomial distribution.
    
    Uses the refined normal approximation to efficiently determine the range of
    likely values for a Poisson binomial distribution, avoiding full computation
    of all probability mass values.
    
    Parameters
    ----------
    pb_mean : torch.Tensor
        Mean of the Poisson binomial distribution
    pb_var : torch.Tensor
        Variance of the Poisson binomial distribution
    pb_m3 : torch.Tensor
        Third moment of the Poisson binomial distribution
    device : torch.device
        Device to place output tensors on
    dim : int
        Maximum dimension/upper bound for the action set
    tol : float, optional
        Tolerance for quantile computation (default: 1e-4)
        
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Lower and upper bounds as integer tensors
    """
    refined_norm = RN_rv()

    # Compute skewness with numerical stability
    skew = (pb_m3 / pb_var**(3/2)).cpu() + 1e-5

    low_quantile = torch.tensor(refined_norm.ppf(tol, skew=skew), device=device)
    up_quantile = torch.tensor(refined_norm.ppf(1-tol, skew=skew), device=device)
    
    # Compute bounds with proper broadcasting
    lower = torch.maximum(torch.floor(torch.sqrt(pb_var)*low_quantile + pb_mean) - 1, torch.tensor(0))
    upper = torch.minimum(torch.ceil(torch.sqrt(pb_var)*up_quantile + pb_mean), torch.tensor(dim))
    return lower.type(torch.int), upper.type(torch.int)

def PB_RNA(pb_mean, pb_var, pb_m3, device, up, low=0):
    """Compute the probability mass function of the Poisson binomial distribution using the refined normal approximation.
    
    Parameters
    ----------
    pb_mean : torch.Tensor
        Mean of the Poisson binomial distribution
    pb_var : torch.Tensor
        Variance of the Poisson binomial distribution
    pb_m3 : torch.Tensor
        Third moment of the Poisson binomial distribution
    device : torch.device
        Device to place output tensors on
    up : int
        Upper bound for the action set
    low : int, optional
        Lower bound for the action set (default: 0)
    
    Returns
    -------
    torch.Tensor
        Probability mass function of the Poisson binomial distribution
    """
    skew = (pb_m3 / pb_var**(3/2)).to(device)

    pb_range = torch.arange(low-1, up, device=device)
    pb_na_score = (pb_range + 0.5 - pb_mean) / torch.sqrt(pb_var)
    
    pb_cdf = RN_rv().cdf(pb_na_score, skew=skew)
    pb_pmf = pb_cdf[1:] - pb_cdf[:-1]

    return torch.clamp(pb_pmf, 0.0, 1.0)

