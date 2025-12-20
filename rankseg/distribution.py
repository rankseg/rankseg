# Author: Ben Dai <bendai@cuhk.edu.hk>
from typing import Optional, Union
import numpy as np
import scipy
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Distribution, constraints
from torch.distributions.normal import Normal
from torch.distributions.utils import broadcast_all


_Number = (int, float, bool)


class RefinedNormalPB(Distribution):
    r"""
    Refined Normal distribution to approximate Poisson binomial distribution.
    
    The CDF is defined as:
    
    .. math::
        F(k; skew) = G( (k + 0.5 - loc) / scale ); \quad G(x) = \Phi(x) + skew * (1 - x^2) * \phi(x) / 6
    
    where:

    - \Phi(x) is the standard normal CDF
    - \phi(x) is the standard normal PDF
    - skew is the skewness parameter
    
    The PDF is defined as:

    Args:
        skew: Skewness parameter controlling the third moment correction term.
        validate_args: Whether to validate the arguments.
    """
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive, "skew": constraints.real}
    support = constraints.real
    has_rsample = False
    
    def __init__(
        self,
        dim: Union[Tensor, int],
        loc: Union[Tensor, float],
        scale: Union[Tensor, float],
        skew: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ):
        self.dim, self.loc, self.scale, self.skew = broadcast_all(dim, loc, scale, skew)
        if isinstance(loc, _Number) and isinstance(scale, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(RefinedNormalPB, _instance)
        batch_shape = torch.Size(batch_shape)
        new.dim = self.dim.expand(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.skew = self.skew.expand(batch_shape)
        super(RefinedNormalPB, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def cdf(self, value):
        # Handle broadcasting when value has extra dimensions
        # e.g., value: (batch, class, d) vs loc/scale/skew: (batch, class)
        loc = self.loc
        scale = self.scale
        skew = self.skew
        
        # Add dimensions to match value's shape for broadcasting
        while loc.ndim < value.ndim:
            loc = loc.unsqueeze(-1)
            scale = scale.unsqueeze(-1)
            skew = skew.unsqueeze(-1)

        # if self._validate_args:
        #     self._validate_sample(value)
        
        x = (value + 0.5 - loc) / scale
        
        ## to be optimized: directly compute the CDF without define Normal class
        norm = Normal(0, 1)
        prob = norm.cdf(x) + skew*(1 - x**2)*norm.log_prob(x).exp()/6
        return torch.clip(prob, min=0.0, max=1.0)

    def pdf(self, value):
        # Handle broadcasting when value has extra dimensions
        # e.g., value: (batch, class, d) vs loc/scale/skew: (batch, class)
        loc = self.loc
        scale = self.scale
        skew = self.skew
        
        # Add dimensions to match value's shape for broadcasting
        while loc.ndim < value.ndim:
            loc = loc.unsqueeze(-1)
            scale = scale.unsqueeze(-1)
            skew = skew.unsqueeze(-1)
        
        x = (value + 0.5 - loc) / scale
        ## to be optimized: directly compute the PDF without define Normal class
        norm = Normal(0, 1)
        pdf_value = norm.log_prob(x).exp()
        g_value = pdf_value + skew/6*pdf_value*(x**3 - 3*x)
        return torch.clip(g_value / scale, min=0.0)

    def pmf(self, value):
        # P(X = value) = F(value) - F(value-1)
        pmf_tmp = self.cdf(value) - self.cdf(value-1)
        return torch.clip(pmf_tmp, min=0.0, max=1.0)

    # def log_prob(self, x):
    #     return torch.log(self.pdf(x))

    def icdf(self, p, max_iter=1000, tol=1e-6):
        ## To be optimized: Brent’s method is better for root finding.
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

    def interval(self, p):
        """
        Compute the confidence interval [lq, uq] such that P(lq <= X <= uq) = 1 - p.
        """
        scipy_refined_normal = RefinedNormal()
        lq, uq = scipy_refined_normal.interval(1-p, skew=self.skew)
        lq, uq = torch.Tensor(lq), torch.Tensor(uq)
        lq = torch.clip(torch.floor(self.scale*lq + self.loc) - 1, min=0)
        uq = torch.clip(torch.ceil(self.scale*uq + self.loc), max=self.dim)
        return lq.int(), uq.int()
        
class RefinedNormal(scipy.stats.rv_continuous):
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
    approximation by incorporating a skewness correction term :cite:p:`volkova1996refinement`.
    
    In the context of RankSEG, this distribution is used to efficiently
    approximate the Poisson binomial distribution.

    References
    ----------
    :cite:p:`volkova1996refinement` Volkova, A.Y., 1996. A refinement of the central limit theorem for sums 
    of independent random indicators. Theory of Probability and its 
    Applications 40, 791-794.
    """
    def _argcheck(self, skew):
        return np.isfinite(skew)

    def _cdf(self, x, skew):
        prob = scipy.stats.norm.cdf(x) + skew*(1 - x**2)*scipy.stats.norm.pdf(x)/6
        return np.clip(prob, 0, 1)
    
    # def _pdf(self, x, skew):
    #     return scipy.stats.norm.pdf(x) + skew/6*scipy.stats.norm.pdf(x)*(3*x - x**3)

