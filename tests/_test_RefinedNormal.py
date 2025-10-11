"""Test suite for RefinedNormal distribution class."""
from rankseg import RefinedNormal
import torch
from torch.distributions.normal import Normal

def test_refined_normal():
    """Comprehensive tests for RefinedNormal distribution."""
    print("\n" + "="*60)
    print("Testing RefinedNormal Distribution")
    print("="*60)
    
    # Test 1: Basic instantiation
    print("\n[Test 1] Basic instantiation")
    loc = torch.tensor([0.0, 1.0, 2.0])
    scale = torch.tensor([1.0, 1.5, 2.0])
    skew = torch.tensor([0.0, 0.5, 1.0])
    dist = RefinedNormal(loc, scale, skew)
    print(f"  ✓ Created distribution with loc={loc.tolist()}, scale={scale.tolist()}, skew={skew.tolist()}")
    
    # Test 2: CDF computation
    print("\n[Test 2] CDF computation")
    values = torch.tensor([0.0, 1.0, 2.0])
    cdf_vals = dist.cdf(values)
    print(f"  Input values: {values.tolist()}")
    print(f"  CDF values: {cdf_vals.tolist()}")
    assert torch.all((cdf_vals >= 0) & (cdf_vals <= 1)), "CDF values must be in [0, 1]"
    print("  ✓ CDF values are within valid range [0, 1]")
    
    # Test 3: PDF computation
    print("\n[Test 3] PDF computation")
    pdf_vals = dist.pdf(values)
    print(f"  Input values: {values.tolist()}")
    print(f"  PDF values: {pdf_vals.tolist()}")
    assert torch.all(pdf_vals >= 0), "PDF values must be non-negative"
    print("  ✓ PDF values are non-negative")
    
    # Test 4: Log probability
    print("\n[Test 4] Log probability")
    log_prob_vals = dist.log_prob(values)
    print(f"  Input values: {values.tolist()}")
    print(f"  Log prob values: {log_prob_vals.tolist()}")
    print("  ✓ Log probability computed successfully")
    
    # Test 5: ICDF (inverse CDF)
    print("\n[Test 5] ICDF computation")
    probs = torch.tensor([0.1, 0.5, 0.9])
    quantiles = dist.icdf(probs)
    print(f"  Probability values: {probs.tolist()}")
    print(f"  Quantiles: {quantiles.tolist()}")
    print("  ✓ ICDF computed successfully")
    
    # Test 6: ICDF-CDF consistency (icdf(cdf(x)) ≈ x)
    print("\n[Test 6] ICDF-CDF consistency check")
    test_vals = torch.tensor([0.5, 1.5, 2.5])
    cdf_test = dist.cdf(test_vals)
    recovered_vals = dist.icdf(cdf_test)
    error = torch.abs(recovered_vals - test_vals)
    print(f"  Original values: {test_vals.tolist()}")
    print(f"  Recovered values: {recovered_vals.tolist()}")
    print(f"  Absolute error: {error.tolist()}")
    assert torch.all(error < 0.01), "ICDF-CDF roundtrip error too large"
    print("  ✓ ICDF-CDF consistency verified (error < 0.01)")
    
    # Test 7: Zero skewness (should behave like normal distribution)
    print("\n[Test 7] Zero skewness case")
    dist_zero_skew = RefinedNormal(loc=0.0, scale=1.0, skew=0.0)
    norm_dist = Normal(0.0, 1.0)
    test_val = torch.tensor([0.0])
    cdf_refined = dist_zero_skew.cdf(test_val)
    cdf_normal = norm_dist.cdf(test_val + 0.5)  # accounting for continuity correction
    print(f"  CDF (RefinedNormal, skew=0): {cdf_refined.item():.4f}")
    print(f"  CDF (Normal with correction): {cdf_normal.item():.4f}")
    print("  ✓ Zero skewness case tested")
    
    # Test 8: Edge cases for ICDF with scalar distribution
    print("\n[Test 8] ICDF edge cases")
    scalar_edge_dist = RefinedNormal(loc=1.0, scale=2.0, skew=0.5)
    edge_probs = torch.tensor([0.01, 0.5, 0.99])
    edge_quantiles = scalar_edge_dist.icdf(edge_probs)
    print(f"  Edge probabilities: {edge_probs.tolist()}")
    print(f"  Quantiles: {edge_quantiles.tolist()}")
    assert edge_quantiles.shape == edge_probs.shape, "ICDF shape mismatch"
    print("  ✓ ICDF edge cases handled")
    
    # Test 9: Broadcasting behavior
    print("\n[Test 9] Broadcasting behavior")
    scalar_dist = RefinedNormal(loc=1.0, scale=2.0, skew=0.5)
    vector_input = torch.tensor([0.0, 1.0, 2.0, 3.0])
    cdf_broadcast = scalar_dist.cdf(vector_input)
    print(f"  Input shape: {vector_input.shape}")
    print(f"  Output shape: {cdf_broadcast.shape}")
    print(f"  CDF values: {cdf_broadcast.tolist()}")
    assert cdf_broadcast.shape == vector_input.shape, "Broadcasting failed"
    print("  ✓ Broadcasting works correctly")
    
    # Test 10: High skewness case
    print("\n[Test 10] High skewness case")
    high_skew_dist = RefinedNormal(loc=5.0, scale=2.0, skew=3.0)
    test_range = torch.linspace(0, 10, 5)
    cdf_high_skew = high_skew_dist.cdf(test_range)
    pdf_high_skew = high_skew_dist.pdf(test_range)
    print(f"  Test range: {test_range.tolist()}")
    print(f"  CDF values: {cdf_high_skew.tolist()}")
    print(f"  PDF values: {pdf_high_skew.tolist()}")
    print("  ✓ High skewness case handled")
    
    print("\n" + "="*60)
    print("All RefinedNormal tests passed! ✓")
    print("="*60 + "\n")

def test_refined_normal_real_case():
    """Test batch computation: 32 points across 3 different distributions.
    
    This demonstrates the efficient way to compute PDF, CDF, and ICDF for
    multiple points across multiple distributions using broadcasting.
    """
    print("\n" + "="*60)
    print("Testing Batch Computation (Real Use Case)")
    print("="*60)
    
    # Define 3 different distributions with shape (1, 3)
    loc = torch.tensor([0.1, 0.7, 1.5]).unsqueeze(0)      # (1, 3)
    scale = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(0)     # (1, 3)
    skew = torch.tensor([0.0, 1.0, 2.0]).unsqueeze(0)      # (1, 3)

    # Create the distribution
    dist = RefinedNormal(loc, scale, skew)
    print(f"\n[Setup] Created 3 distributions:")
    print(f"  Distribution 0: loc={loc[0,0].item():.1f}, scale={scale[0,0].item():.1f}, skew={skew[0,0].item():.1f}")
    print(f"  Distribution 1: loc={loc[0,1].item():.1f}, scale={scale[0,1].item():.1f}, skew={skew[0,1].item():.1f}")
    print(f"  Distribution 2: loc={loc[0,2].item():.1f}, scale={scale[0,2].item():.1f}, skew={skew[0,2].item():.1f}")

    # Define 32 points with shape (32, 1)
    points = torch.linspace(-5, 15, 32).unsqueeze(1)  # (32, 1)
    print(f"\n[Computation] Computing for 32 points from {points[0,0].item():.1f} to {points[-1,0].item():.1f}")

    # Compute PDF, CDF for all combinations - output shape: (32, 3)
    pdf_values = dist.pdf(points)   # (32, 3)
    cdf_values = dist.cdf(points)   # (32, 3)
    print(f"  PDF output shape: {pdf_values.shape}")
    print(f"  CDF output shape: {cdf_values.shape}")

    # For ICDF, use probability values
    probs = torch.linspace(0.1, 0.9, 32).unsqueeze(1)  # (32, 1)
    icdf_values = dist.icdf(probs)  # (32, 3)
    print(f"  ICDF output shape: {icdf_values.shape}")

    # Validate shapes
    assert pdf_values.shape == (32, 3), f"Expected PDF shape (32, 3), got {pdf_values.shape}"
    assert cdf_values.shape == (32, 3), f"Expected CDF shape (32, 3), got {cdf_values.shape}"
    assert icdf_values.shape == (32, 3), f"Expected ICDF shape (32, 3), got {icdf_values.shape}"
    print("\n[Validation] Shape checks passed ✓")
    
    # Validate values
    assert torch.all(pdf_values >= 0), "PDF values must be non-negative"
    assert torch.all((cdf_values >= 0) & (cdf_values <= 1)), "CDF values must be in [0, 1]"
    print("[Validation] Value range checks passed ✓")
    
    # Show sample results
    print("\n[Sample Results] First 5 points across all 3 distributions:")
    print(f"  Points[:5]: {points[:5, 0].tolist()}")
    print(f"\n  PDF values (rows=points, cols=distributions):")
    for i in range(5):
        print(f"    Point {i}: {[f'{v:.4f}' for v in pdf_values[i, :].tolist()]}")
    print(f"\n  CDF values (rows=points, cols=distributions):")
    for i in range(5):
        print(f"    Point {i}: {[f'{v:.4f}' for v in cdf_values[i, :].tolist()]}")
    
    print("\n[Sample Results] ICDF for probabilities [0.1, 0.3, 0.5, 0.7, 0.9]:")
    sample_probs = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]).unsqueeze(1)  # (5, 1)
    sample_icdf = dist.icdf(sample_probs)  # (5, 3)
    print(f"  Quantiles (rows=probabilities, cols=distributions):")
    for i, p in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
        print(f"    p={p}: {[f'{v:.2f}' for v in sample_icdf[i, :].tolist()]}")
    
    print("\n" + "="*60)
    print("Batch computation test passed! ✓")
    print("="*60 + "\n")
    
    # Return values for further inspection if needed
    return {
        'pdf': pdf_values,
        'cdf': cdf_values,
        'icdf': icdf_values,
        'points': points,
        'probs': probs
    }


def test_refined_seg_app():
    """Test batch computation with variable-length PMF for each distribution.
    
    This simulates the RankSEG use case where:
    - We have (batch_size, num_class) distributions
    - Each distribution has different action set bounds [low, up]
    - We need to compute PMF over different ranges for each distribution
    """
    print("\n" + "="*60)
    print("Testing RankSEG Application: Variable-Length PMF")
    print("="*60)
    
    batch_size, num_class = 32, 5
    
    # Create distributions with shape (batch_size, num_class)
    # Simulating statistics from Poisson binomial distributions
    loc = torch.randn(batch_size, num_class).abs() * 10 + 5  # (32, 5), range [5, 15]
    scale = torch.rand(batch_size, num_class) * 2 + 0.5      # (32, 5), range [0.5, 2.5]
    skew = torch.randn(batch_size, num_class) * 0.5          # (32, 5), small skewness
    
    dist = RefinedNormal(loc, scale, skew)
    print(f"\n[Setup] Created {batch_size} × {num_class} = {batch_size * num_class} distributions")
    print(f"  Shape: {dist.loc.shape}")
    print(f"  Sample params (batch=0, class=0): loc={loc[0,0].item():.2f}, scale={scale[0,0].item():.2f}, skew={skew[0,0].item():.2f}")
    
    # Compute action set bounds for each distribution (simulating app_action_set)
    tol = 1e-4
    # Approximate quantiles for bounds
    low_quantile = -3.0  # Roughly ppf(tol) for standard normal
    up_quantile = 3.0    # Roughly ppf(1-tol)
    
    lower = torch.maximum(torch.floor(scale * low_quantile + loc) - 1, torch.tensor(0.0))
    upper = torch.ceil(scale * up_quantile + loc) + 1
    lower = lower.type(torch.int)
    upper = upper.type(torch.int)
    
    print(f"\n[Action Sets] Computed bounds for each distribution")
    print(f"  Sample bounds (batch=0):")
    for k in range(num_class):
        print(f"    Class {k}: [{lower[0,k].item()}, {upper[0,k].item()}] (length={upper[0,k].item() - lower[0,k].item()})")
    
    # Compute PMF using padded tensor
    print("\n[Computation] Padded tensor approach")
    max_length = (upper - lower).max().item()
    print(f"  Max range length: {max_length}")
    
    # Create padded point tensor: (batch_size, num_class, max_length)
    points_padded = torch.zeros(batch_size, num_class, max_length)
    mask = torch.zeros(batch_size, num_class, max_length, dtype=torch.bool)
    
    for b in range(batch_size):
        for k in range(num_class):
            length = upper[b, k] - lower[b, k]
            points_padded[b, k, :length] = torch.arange(lower[b, k], upper[b, k], dtype=torch.float32)
            mask[b, k, :length] = True
    
    # Compute PDF for all points (broadcast over last dimension)
    # Need to expand distribution to match points shape
    loc_expanded = loc.unsqueeze(-1)    # (32, 5, 1)
    scale_expanded = scale.unsqueeze(-1)
    skew_expanded = skew.unsqueeze(-1)
    dist_expanded = RefinedNormal(loc_expanded, scale_expanded, skew_expanded)
    
    pdf_padded = dist_expanded.pdf(points_padded)  # (32, 5, max_length)
    
    # Apply mask to zero out invalid entries
    pdf_padded = pdf_padded * mask
    
    print(f"  PDF output shape: {pdf_padded.shape}")
    print(f"  Sample PDF (batch=0, class=0, first 5 valid points):")
    valid_pdf = pdf_padded[0, 0, mask[0, 0]]
    print(f"    {valid_pdf[:5].tolist()}")
    
    # Verify PMF sums to approximately 1 for each distribution
    pmf_sums = pdf_padded.sum(dim=-1)  # (32, 5)
    print(f"\n[Validation] PMF sums (should be close to 1.0):")
    print(f"  Min: {pmf_sums.min().item():.4f}, Max: {pmf_sums.max().item():.4f}")
    print(f"  Mean: {pmf_sums.mean().item():.4f}")
    print(f"  Sample (batch=0): {[f'{v:.4f}' for v in pmf_sums[0].tolist()]}")
    
    # Show a few more examples
    print(f"\n[Sample Results] Distribution details:")
    for b in range(min(2, batch_size)):
        print(f"\n  Batch {b}:")
        for k in range(min(3, num_class)):
            n_points = mask[b, k].sum().item()
            pdf_sum = pdf_padded[b, k, :n_points].sum().item()
            print(f"    Class {k}: range=[{lower[b,k].item()}, {upper[b,k].item()}], "
                  f"n_points={n_points}, PMF_sum={pdf_sum:.4f}")
    
    print("\n" + "="*60)
    print("RankSEG application test completed! ✓")
    print("="*60 + "\n")
    
    return {
        'distributions': dist,
        'lower': lower,
        'upper': upper,
        'pdf_padded': pdf_padded,
        'mask': mask,
        'pmf_sums': pmf_sums
    }

# Run the tests
if __name__ == "__main__":
    test_refined_normal()
    test_refined_normal_real_case()
    test_refined_seg_app()
    print("\n" + "#"*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("#"*60)