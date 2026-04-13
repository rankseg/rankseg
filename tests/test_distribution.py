import numpy as np
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


def test_refined_normal_pb_icdf_matches_closed_form_for_zero_skew():
    probs = torch.tensor([0.2, 0.5, 0.8])
    rv = RefinedNormalPB(dim=10, loc=3.0, scale=1.5, skew=0.0)
    expected = 3.0 - 0.5 + 1.5 * Normal(0, 1).icdf(probs)

    icdf = rv.icdf(probs)

    assert torch.allclose(icdf, expected, atol=1e-4)


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
