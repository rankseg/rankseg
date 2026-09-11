import math
from numbers import Integral, Real

import torch

SUPPORTED_PROB_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def validate_probability_tensor(probs, *, check_values=True):
    if not isinstance(probs, torch.Tensor):
        raise TypeError("probs must be a torch.Tensor")
    if probs.dtype not in SUPPORTED_PROB_DTYPES:
        raise TypeError("probs must have a real floating-point dtype")
    if probs.ndim < 3:
        raise ValueError("probs must have shape (batch_size, num_class, *image_shape)")
    if probs.shape[1] == 0:
        raise ValueError("probs must contain at least one class")
    if any(size == 0 for size in probs.shape[2:]):
        raise ValueError("probs spatial dimensions must be non-empty")
    if check_values and probs.numel() > 0:
        minimum, maximum = torch.aminmax(probs)
        minimum, maximum = torch.stack((minimum, maximum)).detach().cpu().tolist()
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError("probs must contain only finite values")
        if minimum < 0 or maximum > 1:
            raise ValueError("probs must be in the range [0, 1]")


def validate_finite_real(name, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def validate_integral(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an int")
    return int(value)
