import numpy as np
import pytest
import torch
from torch.distributions.normal import Normal

from rankseg.distribution import RefinedNormal, RefinedNormalPB


def test_refined_normal_pb_supports_expand_and_probability_methods():
    rv = RefinedNormalPB(
        dim=torch.tensor([8, 8]),
        loc=torch.tensor([3.0, 4.0]),
        scale=torch.tensor([1.2, 1.5]),
        skew=torch.tensor([0.1, -0.2]),
    )
    expanded = rv.expand((3, 2))
    values = torch.arange(5, dtype=torch.float32).view(1, 1, 5).expand(1, 2, 5)

    cdf = rv.cdf(values)
    pdf = rv.pdf(values)
    pmf = rv.pmf(values)

    assert expanded.batch_shape == torch.Size([3, 2])
    assert expanded.loc.shape == (3, 2)
    assert cdf.shape == (2, 2, 5)
    assert pdf.shape == (2, 2, 5)
    assert pmf.shape == (2, 2, 5)
    assert bool(torch.all((0.0 <= cdf) & (cdf <= 1.0)))
    assert bool(torch.all(pdf >= 0.0))
    assert bool(torch.all((0.0 <= pmf) & (pmf <= 1.0)))
    assert torch.allclose(pmf, cdf - rv.cdf(values - 1), atol=1e-6)


def test_refined_normal_pb_batch_shape_uses_all_broadcast_parameters():
    vector_skew_rv = RefinedNormalPB(
        dim=10,
        loc=3.0,
        scale=1.5,
        skew=torch.tensor([0.1, -0.2]),
    )
    vector_dim_rv = RefinedNormalPB(
        dim=torch.tensor([8, 10]),
        loc=3.0,
        scale=1.5,
        skew=0.0,
    )

    assert vector_skew_rv.batch_shape == torch.Size([2])
    assert vector_dim_rv.batch_shape == torch.Size([2])
    assert vector_skew_rv.expand((3, 2)).batch_shape == torch.Size([3, 2])
    assert vector_dim_rv.expand((3, 2)).batch_shape == torch.Size([3, 2])


@pytest.mark.parametrize(
    "dim",
    [
        -1,
        3.5,
        float("nan"),
        float("inf"),
        torch.tensor([10.0, -1.0]),
        torch.tensor([10.0, 3.5]),
    ],
)
def test_refined_normal_pb_rejects_invalid_dimensions(dim):
    with pytest.raises(ValueError, match="dim"):
        RefinedNormalPB(dim=dim, loc=3.0, scale=1.5, skew=0.0)


@pytest.mark.parametrize("dim", [True, 1 + 0j, torch.tensor(True), torch.tensor(1 + 0j)])
def test_refined_normal_pb_rejects_non_real_dimension_types(dim):
    with pytest.raises(TypeError, match="dim must contain real numbers"):
        RefinedNormalPB(dim=dim, loc=3.0, scale=1.5, skew=0.0)


@pytest.mark.parametrize("name", ["loc", "scale", "skew"])
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), torch.tensor([0.0, float("nan")])])
def test_refined_normal_pb_rejects_nonfinite_parameters(name, value):
    parameters = {"dim": 10, "loc": 3.0, "scale": 1.5, "skew": 0.0}
    parameters[name] = value

    with pytest.raises(ValueError, match=name):
        RefinedNormalPB(**parameters)


@pytest.mark.parametrize("name", ["loc", "scale", "skew"])
@pytest.mark.parametrize("value", [True, 1 + 0j, torch.tensor(True), torch.tensor(1 + 0j)])
def test_refined_normal_pb_rejects_non_real_parameter_types(name, value):
    parameters = {"dim": 10, "loc": 3.0, "scale": 1.5, "skew": 0.0}
    parameters[name] = value

    with pytest.raises(TypeError, match=rf"{name} must contain real numbers"):
        RefinedNormalPB(**parameters)


@pytest.mark.parametrize("scale", [0.0, -1.0, torch.tensor([1.0, 0.0])])
def test_refined_normal_pb_rejects_nonpositive_scale(scale):
    with pytest.raises(ValueError, match="scale"):
        RefinedNormalPB(dim=10, loc=3.0, scale=scale, skew=0.0)


def test_refined_normal_pb_validate_args_false_bypasses_parameter_value_validation():
    rv = RefinedNormalPB(
        dim=-1.5,
        loc=float("inf"),
        scale=-1.0,
        skew=float("inf"),
        validate_args=False,
    )

    assert not rv._validate_args


def test_refined_normal_interval_and_clipping_behavior():
    scalar_rv = RefinedNormalPB(dim=10, loc=3.0, scale=1.5, skew=0.0)
    lower, upper = scalar_rv.interval(0.1)
    refined_normal = RefinedNormal()
    clipped = refined_normal._cdf(np.array([0.0]), skew=100.0)

    assert scalar_rv.batch_shape == torch.Size()
    assert lower.dtype == torch.int32
    assert upper.dtype == torch.int32
    assert int(lower.item()) <= int(upper.item())
    assert 0 <= int(lower.item()) <= 10
    assert 0 <= int(upper.item()) <= 10
    assert refined_normal._argcheck(0.5)
    assert not refined_normal._argcheck(float("inf"))
    assert np.all((0.0 <= clipped) & (clipped <= 1.0))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_refined_normal_interval_clips_both_endpoints_to_support(dtype, device):
    if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bfloat16 is not supported")

    rv = RefinedNormalPB(
        dim=torch.tensor([8, 10], device=device),
        loc=torch.tensor([-100.0, 100.0], dtype=dtype, device=device),
        scale=torch.tensor([1.0, 1.0], dtype=dtype, device=device),
        skew=torch.tensor([0.0, 0.0], dtype=dtype, device=device),
    )

    lower, upper = rv.interval(0.1)

    expected = torch.tensor([0, 10], dtype=torch.int32, device=device)
    assert torch.equal(lower, expected)
    assert torch.equal(upper, expected)


@pytest.mark.parametrize(
    "tail_probability",
    [
        0.0,
        1.0,
        np.float32(0.1),
        np.array([0.1, 0.2]),
        [0.1, 0.2],
        torch.tensor(0.1),
        torch.tensor([0.1, 0.2]),
    ],
)
def test_refined_normal_interval_accepts_valid_probability_inputs(tail_probability):
    rv = RefinedNormalPB(dim=10, loc=3.0, scale=1.5, skew=0.0)

    lower, upper = rv.interval(tail_probability)

    assert bool(torch.all(lower >= 0))
    assert bool(torch.all(upper <= rv.dim))
    assert bool(torch.all(lower <= upper))


@pytest.mark.parametrize(
    "tail_probability",
    [True, 0.1 + 0.2j, "0.1", np.array([0.1 + 0.2j]), torch.tensor(True), torch.tensor(0.1 + 0.2j)],
)
def test_refined_normal_interval_rejects_non_real_probability_inputs(tail_probability):
    rv = RefinedNormalPB(dim=10, loc=3.0, scale=1.5, skew=0.0)

    with pytest.raises(TypeError, match="p must contain real numbers"):
        rv.interval(tail_probability)


@pytest.mark.parametrize(
    "tail_probability",
    [float("nan"), float("inf"), -float("inf"), np.array([0.1, np.nan]), torch.tensor([0.1, float("inf")])],
)
def test_refined_normal_interval_rejects_nonfinite_probability_inputs(tail_probability):
    rv = RefinedNormalPB(dim=10, loc=3.0, scale=1.5, skew=0.0)

    with pytest.raises(ValueError, match="p must contain only finite values"):
        rv.interval(tail_probability)


@pytest.mark.parametrize(
    "tail_probability",
    [-0.1, 1.1, np.array([0.1, -0.1]), torch.tensor([0.1, 1.1])],
)
def test_refined_normal_interval_rejects_out_of_range_probability_inputs(tail_probability):
    rv = RefinedNormalPB(dim=10, loc=3.0, scale=1.5, skew=0.0)

    with pytest.raises(ValueError, match=r"p must be in the range \[0, 1\]"):
        rv.interval(tail_probability)


@pytest.mark.parametrize("tail_probability", [[], np.array([]), torch.tensor([])])
def test_refined_normal_interval_rejects_empty_probability_inputs(tail_probability):
    rv = RefinedNormalPB(dim=10, loc=3.0, scale=1.5, skew=0.0)

    with pytest.raises(ValueError, match="p must not be empty"):
        rv.interval(tail_probability)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_refined_normal_interval_detaches_tensor_inputs(dtype):
    rv = RefinedNormalPB(
        dim=torch.tensor([8, 10]),
        loc=torch.tensor([3.0, 4.0], dtype=dtype, requires_grad=True),
        scale=torch.tensor([1.2, 1.5], dtype=dtype, requires_grad=True),
        skew=torch.tensor([0.1, -0.2], dtype=dtype, requires_grad=True),
    )
    tail_probability = torch.tensor(0.1, dtype=dtype, requires_grad=True)

    lower, upper = rv.interval(tail_probability)

    assert lower.shape == rv.batch_shape
    assert upper.shape == rv.batch_shape
    assert lower.device == rv.loc.device
    assert upper.device == rv.loc.device
    assert lower.dtype == torch.int32
    assert upper.dtype == torch.int32
    assert not lower.requires_grad
    assert not upper.requires_grad
    assert bool(torch.all(lower >= 0))
    assert bool(torch.all(upper <= rv.dim))
    assert bool(torch.all(lower <= upper))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_refined_normal_interval_cuda_matches_cpu(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bfloat16 is not supported")
    kwargs = {
        "dim": torch.tensor([8, 10]),
        "loc": torch.tensor([3.0, 4.0], dtype=dtype),
        "scale": torch.tensor([1.2, 1.5], dtype=dtype),
        "skew": torch.tensor([0.1, -0.2], dtype=dtype),
    }
    expected = RefinedNormalPB(**kwargs).interval(torch.tensor(0.1, dtype=dtype))
    cuda_kwargs = {name: value.cuda().requires_grad_(name != "dim") for name, value in kwargs.items()}

    actual = RefinedNormalPB(**cuda_kwargs).interval(torch.tensor(0.1, dtype=dtype, device="cuda", requires_grad=True))

    assert actual[0].is_cuda
    assert actual[1].is_cuda
    assert torch.equal(actual[0].cpu(), expected[0])
    assert torch.equal(actual[1].cpu(), expected[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_refined_normal_interval_validates_cuda_probability_inputs():
    rv = RefinedNormalPB(dim=10, loc=torch.tensor(3.0, device="cuda"), scale=1.5, skew=0.0)

    with pytest.raises(ValueError, match="p must contain only finite values"):
        rv.interval(torch.tensor([0.1, float("nan")], device="cuda"))


def test_refined_normal_pb_icdf_matches_closed_form_for_zero_skew():
    probs = torch.tensor([0.2, 0.5, 0.8])
    rv = RefinedNormalPB(dim=10, loc=3.0, scale=1.5, skew=0.0)
    expected = 3.0 - 0.5 + 1.5 * Normal(0, 1).icdf(probs)

    icdf = rv.icdf(probs)

    assert torch.allclose(icdf, expected, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_refined_normal_pb_icdf_accepts_supported_floating_dtypes(dtype):
    probs = torch.tensor([0.2, 0.5, 0.8], dtype=dtype)
    rv = RefinedNormalPB(
        dim=10,
        loc=torch.tensor(3.0, dtype=dtype),
        scale=torch.tensor(1.5, dtype=dtype),
        skew=torch.tensor(0.0, dtype=dtype),
    )

    quantiles = rv.icdf(probs)
    recovered = rv.cdf(quantiles)

    assert quantiles.dtype == dtype
    assert bool(torch.isfinite(quantiles).all())
    assert torch.allclose(recovered, probs, atol=max(2 * torch.finfo(dtype).eps, 1e-6), rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_refined_normal_pb_icdf_stops_when_dtype_precision_prevents_progress(dtype, monkeypatch):
    probs = torch.tensor([0.2, 0.5, 0.8], dtype=dtype)
    rv = RefinedNormalPB(
        dim=10,
        loc=torch.tensor(3.0, dtype=dtype),
        scale=torch.tensor(1.5, dtype=dtype),
        skew=torch.tensor(0.1, dtype=dtype),
    )
    cdf_calls = 0
    original_cdf = rv.cdf

    def count_cdf_calls(value):
        nonlocal cdf_calls
        cdf_calls += 1
        return original_cdf(value)

    monkeypatch.setattr(rv, "cdf", count_cdf_calls)
    quantiles = rv.icdf(probs)

    assert cdf_calls < 100
    assert bool(torch.isfinite(quantiles).all())
    assert torch.allclose(
        original_cdf(quantiles),
        probs,
        atol=max(2 * torch.finfo(dtype).eps, 1e-6),
        rtol=0,
    )


def test_refined_normal_pb_icdf_round_trips_for_zero_skew_batch():
    probs = torch.tensor([[0.2, 0.8], [0.35, 0.65]])
    rv = RefinedNormalPB(
        dim=torch.tensor([[10, 10], [10, 10]]),
        loc=torch.tensor([[3.0, 4.0], [2.5, 5.0]]),
        scale=torch.tensor([[1.5, 1.0], [0.8, 1.2]]),
        skew=torch.zeros(2, 2),
    )

    recovered = rv.cdf(rv.icdf(probs))

    assert torch.allclose(recovered, probs, atol=1e-5)


def test_refined_normal_pb_icdf_round_trips_for_nonzero_skew():
    probs = torch.tensor([0.2, 0.8])
    rv = RefinedNormalPB(dim=10, loc=3.0, scale=1.5, skew=0.1)

    recovered = rv.cdf(rv.icdf(probs))

    assert torch.allclose(recovered, probs, atol=1e-5)


def test_refined_normal_pb_icdf_handles_extreme_skew():
    probs = torch.tensor([1e-4, 0.2, 0.8, 1 - 1e-4])
    rv = RefinedNormalPB(dim=100, loc=0.0, scale=0.1, skew=-2.0)

    x = rv.icdf(probs)
    recovered = rv.cdf(x)

    assert bool(torch.isfinite(x).all())
    assert torch.allclose(recovered, probs, atol=1e-4)


def test_refined_normal_pb_icdf_broadcasts_probabilities_over_batch_parameters():
    probs = torch.tensor([[0.2, 0.5, 0.8]])
    loc = torch.tensor([[3.0], [4.0]])
    scale = torch.tensor([[1.5], [0.8]])
    rv = RefinedNormalPB(dim=10, loc=loc, scale=scale, skew=0.0)
    expected = loc - 0.5 + scale * Normal(0, 1).icdf(probs)

    quantiles = rv.icdf(probs)

    assert quantiles.shape == (2, 3)
    assert torch.allclose(quantiles, expected, atol=1e-4)
    assert torch.allclose(rv.cdf(quantiles), probs.expand_as(quantiles), atol=1e-5)


def test_refined_normal_pb_icdf_disables_autograd(monkeypatch):
    probs = torch.tensor([0.2, 0.5, 0.8], requires_grad=True)
    rv = RefinedNormalPB(
        dim=10,
        loc=torch.tensor(3.0, requires_grad=True),
        scale=torch.tensor(1.5, requires_grad=True),
        skew=torch.tensor(0.1, requires_grad=True),
    )
    grad_states = []
    original_cdf = rv.cdf

    def record_grad_state(value):
        grad_states.append(torch.is_grad_enabled())
        return original_cdf(value)

    monkeypatch.setattr(rv, "cdf", record_grad_state)
    with torch.enable_grad():
        quantiles = rv.icdf(probs)

    assert grad_states and not any(grad_states)
    assert not quantiles.requires_grad
    assert quantiles.grad_fn is None


@pytest.mark.parametrize("probability", [0.2, [0.2], np.array([0.2])])
def test_refined_normal_pb_icdf_requires_tensor_probabilities(probability):
    rv = RefinedNormalPB(dim=10, loc=3.0, scale=1.5, skew=0.0)

    with pytest.raises(TypeError, match="p must be a torch.Tensor"):
        rv.icdf(probability)


@pytest.mark.parametrize(
    "probability",
    [torch.tensor(True), torch.tensor(1), torch.tensor(0.2 + 0.1j)],
)
def test_refined_normal_pb_icdf_rejects_non_floating_probability_dtypes(probability):
    rv = RefinedNormalPB(dim=10, loc=3.0, scale=1.5, skew=0.0)

    with pytest.raises(TypeError, match="real floating-point dtype"):
        rv.icdf(probability)


@pytest.mark.parametrize(
    "probability",
    [torch.tensor(float("nan")), torch.tensor(float("inf")), torch.tensor(float("-inf"))],
)
def test_refined_normal_pb_icdf_rejects_nonfinite_probabilities(probability):
    rv = RefinedNormalPB(dim=10, loc=3.0, scale=1.5, skew=0.0)

    with pytest.raises(ValueError, match="p must contain only finite values"):
        rv.icdf(probability)


@pytest.mark.parametrize("probability", [torch.tensor(-0.1), torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.1)])
def test_refined_normal_pb_icdf_rejects_probabilities_outside_open_unit_interval(probability):
    rv = RefinedNormalPB(dim=10, loc=3.0, scale=1.5, skew=0.0)

    with pytest.raises(ValueError, match=r"p must be in the range \(0, 1\)"):
        rv.icdf(probability)


def test_refined_normal_pb_icdf_rejects_empty_probability_tensors():
    rv = RefinedNormalPB(dim=10, loc=3.0, scale=1.5, skew=0.0)

    with pytest.raises(ValueError, match="p must not be empty"):
        rv.icdf(torch.tensor([]))


@pytest.mark.parametrize("max_iter", [True, 1.5, "10"])
def test_refined_normal_pb_icdf_requires_integer_max_iter(max_iter):
    rv = RefinedNormalPB(dim=10, loc=3.0, scale=1.5, skew=0.0)

    with pytest.raises(TypeError, match="max_iter must be an int"):
        rv.icdf(torch.tensor(0.5), max_iter=max_iter)


@pytest.mark.parametrize("max_iter", [0, -1])
def test_refined_normal_pb_icdf_requires_positive_max_iter(max_iter):
    rv = RefinedNormalPB(dim=10, loc=3.0, scale=1.5, skew=0.0)

    with pytest.raises(ValueError, match="max_iter must be greater than 0"):
        rv.icdf(torch.tensor(0.5), max_iter=max_iter)


@pytest.mark.parametrize("tol", [True, "1e-6"])
def test_refined_normal_pb_icdf_requires_real_tolerance(tol):
    rv = RefinedNormalPB(dim=10, loc=3.0, scale=1.5, skew=0.0)

    with pytest.raises(TypeError, match="tol must be a real number"):
        rv.icdf(torch.tensor(0.5), tol=tol)


@pytest.mark.parametrize("tol", [float("nan"), float("inf"), float("-inf")])
def test_refined_normal_pb_icdf_requires_finite_tolerance(tol):
    rv = RefinedNormalPB(dim=10, loc=3.0, scale=1.5, skew=0.0)

    with pytest.raises(ValueError, match="tol must be finite"):
        rv.icdf(torch.tensor(0.5), tol=tol)


@pytest.mark.parametrize("tol", [0.0, -1e-6])
def test_refined_normal_pb_icdf_requires_positive_tolerance(tol):
    rv = RefinedNormalPB(dim=10, loc=3.0, scale=1.5, skew=0.0)

    with pytest.raises(ValueError, match="tol must be greater than 0"):
        rv.icdf(torch.tensor(0.5), tol=tol)
