# Author: Ben Dai <bendai@cuhk.edu.hk>
from numbers import Real
from typing import Optional, Union

import numpy as np
import scipy
import torch
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.distributions.normal import Normal
from torch.distributions.utils import broadcast_all

from rankseg._validation import SUPPORTED_PROB_DTYPES, validate_finite_real, validate_integral


def _validate_interval_probability(p):
    if isinstance(p, Tensor):
        if p.dtype == torch.bool or p.is_complex():
            raise TypeError("p must contain real numbers")
        if p.numel() == 0:
            raise ValueError("p must not be empty")
        if not bool(torch.isfinite(p).all()):
            raise ValueError("p must contain only finite values")
        if bool(torch.any((p < 0) | (p > 1))):
            raise ValueError("p must be in the range [0, 1]")
        return p.detach().to(device="cpu", dtype=torch.float64).numpy()

    try:
        scipy_p = np.asarray(p)
    except (TypeError, ValueError) as error:
        raise TypeError("p must contain real numbers") from error
    if scipy_p.dtype.kind not in "fiu" or scipy_p.dtype.kind == "b":
        raise TypeError("p must contain real numbers")
    if scipy_p.size == 0:
        raise ValueError("p must not be empty")
    if not bool(np.isfinite(scipy_p).all()):
        raise ValueError("p must contain only finite values")
    if bool(np.any((scipy_p < 0) | (scipy_p > 1))):
        raise ValueError("p must be in the range [0, 1]")
    return scipy_p.astype(np.float64, copy=False)


def _validate_icdf_probability(p):
    if not isinstance(p, Tensor):
        raise TypeError("p must be a torch.Tensor")
    if p.dtype not in SUPPORTED_PROB_DTYPES:
        raise TypeError("p must have a real floating-point dtype")
    if p.numel() == 0:
        raise ValueError("p must not be empty")
    if not bool(torch.isfinite(p).all()):
        raise ValueError("p must contain only finite values")
    if bool(torch.any((p <= 0) | (p >= 1))):
        raise ValueError("p must be in the range (0, 1)")
    return p


def _validate_real_parameter_type(name, value):
    if isinstance(value, Tensor):
        if value.dtype == torch.bool or value.is_complex():
            raise TypeError(f"{name} must contain real numbers")
        return
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must contain real numbers")


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

    .. math::
        f(k; skew) = \frac{\phi(x)}{scale}
        \left[1 + \frac{skew}{6}(x^3 - 3x)\right],
        \quad x = \frac{k + 0.5 - loc}{scale}.

    Parameters
    ----------
    dim : torch.Tensor or int
        Finite, nonnegative integer upper bound used when clipping integer
        confidence intervals.
    loc : torch.Tensor or float
        Finite location parameter of the refined normal approximation.
    scale : torch.Tensor or float
        Finite, strictly positive scale parameter of the refined normal
        approximation.
    skew : torch.Tensor or float
        Finite skewness parameter controlling the third-moment correction
        term.
    validate_args : bool, optional
        Whether to validate arguments through
        ``torch.distributions.Distribution``. Explicitly setting this to
        ``False`` bypasses parameter validation.
    """

    arg_constraints = {
        "dim": constraints.nonnegative_integer,
        "loc": constraints.real,
        "scale": constraints.positive,
        "skew": constraints.real,
    }
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
        validation_enabled = self._validate_args if validate_args is None else validate_args
        if validation_enabled:
            for name, value in (("dim", dim), ("loc", loc), ("scale", scale), ("skew", skew)):
                _validate_real_parameter_type(name, value)
        self.dim, self.loc, self.scale, self.skew = broadcast_all(dim, loc, scale, skew)
        batch_shape = self.loc.size()
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)
        if self._validate_args:
            for name in ("loc", "scale", "skew"):
                if not bool(torch.isfinite(getattr(self, name)).all()):
                    raise ValueError(f"{name} must contain only finite values")

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
        prob = norm.cdf(x) + skew * (1 - x**2) * norm.log_prob(x).exp() / 6
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
        g_value = pdf_value + skew / 6 * pdf_value * (x**3 - 3 * x)
        return torch.clip(g_value / scale, min=0.0)

    def pmf(self, value):
        # P(X = value) = F(value) - F(value-1)
        pmf_tmp = self.cdf(value) - self.cdf(value - 1)
        return torch.clip(pmf_tmp, min=0.0, max=1.0)

    # def log_prob(self, x):
    #     return torch.log(self.pdf(x))

    @torch.no_grad()
    def icdf(self, p, max_iter=1000, tol=1e-6):
        ## To be optimized: Brent’s method is better for root finding.
        """Inverse CDF (quantile function) using bracketed bisection.

        Parameters
        ----------
        p : torch.Tensor
            Probability values (0 < p < 1)
        max_iter : int, optional
            Positive maximum number of bisection iterations (default: 1000).
        tol : float, optional
            Finite, strictly positive convergence tolerance (default: 1e-6).
            Bisection also stops when finite-precision rounding leaves no
            representable midpoint between the current bounds.

        Returns
        -------
        torch.Tensor
            Quantile values corresponding to probabilities p

        Notes
        -----
        This numerical root-finding operation is not differentiable. Gradient
        recording is disabled internally, even when the probabilities or
        distribution parameters require gradients.
        """
        p = _validate_icdf_probability(p)
        max_iter = validate_integral("max_iter", max_iter)
        if max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        tol = validate_finite_real("tol", tol)
        if tol <= 0:
            raise ValueError("tol must be greater than 0")

        loc = self.loc
        scale = self.scale
        while loc.ndim < p.ndim:
            loc = loc.unsqueeze(-1)
            scale = scale.unsqueeze(-1)

        low = loc - 0.5 - 8.0 * scale
        high = loc - 0.5 + 8.0 * scale

        for _ in range(min(32, max_iter)):
            cdf_low = self.cdf(low)
            cdf_high = self.cdf(high)
            need_lower = cdf_low > p
            need_upper = cdf_high < p
            if not bool(need_lower.any() or need_upper.any()):
                break
            width = high - low
            low = torch.where(need_lower, low - width, low)
            high = torch.where(need_upper, high + width, high)

        for _ in range(max_iter):
            mid = (low + high) / 2
            cdf_mid = self.cdf(mid)
            go_right = cdf_mid < p
            next_low = torch.where(go_right, mid, low)
            next_high = torch.where(go_right, high, mid)
            made_progress = (next_low != low) | (next_high != high)
            low, high = next_low, next_high
            converged = (torch.max(torch.abs(high - low)) < tol) | ~made_progress.any()
            if bool(converged):
                break

        return (low + high) / 2

    @torch.no_grad()
    def interval(self, p):
        """Compute an inclusive confidence interval with retained mass ``1 - p``.

        ``p`` may be a real scalar or an array-like/Tensor of real values in
        ``[0, 1]``. Every value must be finite.

        SciPy evaluates the refined-normal quantiles on the CPU. Tensor inputs
        are detached for this non-differentiable calculation, and the integer
        endpoints are returned on the same device as the distribution
        parameters.
        """
        scipy_p = _validate_interval_probability(p)
        scipy_skew = self.skew.detach().to(device="cpu", dtype=torch.float64).numpy()
        scipy_refined_normal = RefinedNormal()
        lq, uq = scipy_refined_normal.interval(1 - scipy_p, skew=scipy_skew)
        lq = torch.as_tensor(lq, device=self.loc.device)
        uq = torch.as_tensor(uq, device=self.loc.device)
        loc = self.loc.detach()
        scale = self.scale.detach()
        dim = self.dim.detach()
        lq = torch.clamp(torch.floor(scale * lq + loc) - 1, min=0)
        uq = torch.clamp(torch.ceil(scale * uq + loc), min=0)
        lq = torch.minimum(lq, dim)
        uq = torch.minimum(uq, dim)
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
        prob = scipy.stats.norm.cdf(x) + skew * (1 - x**2) * scipy.stats.norm.pdf(x) / 6
        return np.clip(prob, 0, 1)

    # def _pdf(self, x, skew):
    #     return scipy.stats.norm.pdf(x) + skew/6*scipy.stats.norm.pdf(x)*(3*x - x**3)
