import pytest
import torch

from rankseg._validation import validate_probability_tensor

_DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")),
]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_probability_validation_accepts_valid_noncontiguous_tensors(dtype, device):
    probs = torch.rand((2, 3, 4, 5), dtype=dtype, device=device).transpose(-1, -2)

    validate_probability_tensor(probs)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_probability_validation_rejects_nonfinite_values(dtype, device, value):
    probs = torch.tensor([[[0.0, value, 1.0]]], dtype=dtype, device=device)

    with pytest.raises(ValueError, match="probs must contain only finite values"):
        validate_probability_tensor(probs)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_probability_validation_rejects_out_of_range_values(dtype, device, value):
    probs = torch.tensor([[[0.0, value, 1.0]]], dtype=dtype, device=device)

    with pytest.raises(ValueError, match=r"probs must be in the range \[0, 1\]"):
        validate_probability_tensor(probs)


@pytest.mark.parametrize("device", _DEVICES)
def test_probability_validation_reports_nonfinite_values_before_range_errors(device):
    probs = torch.tensor([[[-1.0, float("nan"), 2.0]]], device=device)

    with pytest.raises(ValueError, match="probs must contain only finite values"):
        validate_probability_tensor(probs)


@pytest.mark.parametrize("check_values", [True, False])
def test_probability_validation_does_not_reduce_empty_batches(monkeypatch, check_values):
    probs = torch.empty((0, 2, 4), dtype=torch.float32)

    def fail_aminmax(*args, **kwargs):
        raise AssertionError("aminmax should not be called")

    monkeypatch.setattr(torch, "aminmax", fail_aminmax)

    validate_probability_tensor(probs, check_values=check_values)


def test_probability_validation_skips_value_reduction_when_requested(monkeypatch):
    probs = torch.tensor([[[float("nan"), -1.0, 2.0]]])

    def fail_aminmax(*args, **kwargs):
        raise AssertionError("aminmax should not be called")

    monkeypatch.setattr(torch, "aminmax", fail_aminmax)

    validate_probability_tensor(probs, check_values=False)
