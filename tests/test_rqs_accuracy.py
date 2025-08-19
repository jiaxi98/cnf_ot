"""Unit test for Rational Quadratic Spline (RQS) accuracy.

This test verifies that the composition of forward and inverse mappings
of the RQS transformation approximates the identity function within
acceptable numerical error bounds.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import distrax
import pytest


class TestRQSAccuracy:
  """Test suite for RQS forward and inverse mapping accuracy."""
  
  def test_rqs_identity_composition(self):
    """Test that forward ∘ inverse ≈ identity for RQS transformation."""
    
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Test parameters
    num_bins = 10
    batch_size = 100
    num_features = 2
    
    # Generate random RQS parameters
    # For RQS we need: num_bins widths, num_bins heights, num_bins+1 slopes
    num_params = 3 * num_bins + 1
    
    key, subkey = jax.random.split(key)
    # Initialize parameters with reasonable values
    widths = jax.random.uniform(subkey, (batch_size, num_features, num_bins), 
                                minval=0.1, maxval=2.0, dtype=jnp.float64)
    widths = widths / jnp.sum(widths, axis=-1, keepdims=True)  # Normalize to sum to 1
    
    key, subkey = jax.random.split(key)
    heights = jax.random.uniform(subkey, (batch_size, num_features, num_bins),
                                 minval=0.1, maxval=2.0, dtype=jnp.float64)
    heights = heights / jnp.sum(heights, axis=-1, keepdims=True)  # Normalize to sum to 1
    
    key, subkey = jax.random.split(key)
    slopes = jax.random.uniform(subkey, (batch_size, num_features, num_bins + 1),
                                minval=0.5, maxval=2.0, dtype=jnp.float64)
    
    # Concatenate parameters in the format expected by distrax
    params = jnp.concatenate([widths, heights, slopes], axis=-1)
    
    # Create RQS bijector
    rqs = distrax.RationalQuadraticSpline(
      params,
      range_min=-5.0,
      range_max=5.0,
      min_knot_slope=1e-3,
      boundary_slopes='unconstrained'
    )
    
    # Generate test points within the valid range
    key, subkey = jax.random.split(key)
    x_test = jax.random.uniform(subkey, (batch_size, num_features),
                                minval=-4.0, maxval=4.0, dtype=jnp.float64)
    
    # Forward transformation
    y = rqs.forward(x_test)
    
    # Inverse transformation
    x_reconstructed = rqs.inverse(y)
    
    # Calculate reconstruction error
    abs_error = jnp.abs(x_reconstructed - x_test)
    relative_error = abs_error / (jnp.abs(x_test) + 1e-8)
    
    max_abs_error = jnp.max(abs_error)
    mean_abs_error = jnp.mean(abs_error)
    max_relative_error = jnp.max(relative_error)
    mean_relative_error = jnp.mean(relative_error)
    
    print(f"\nRQS Composition Test Results (float64):")
    print(f"  Max absolute error: {max_abs_error:.2e}")
    print(f"  Mean absolute error: {mean_abs_error:.2e}")
    print(f"  Max relative error: {max_relative_error:.2e}")
    print(f"  Mean relative error: {mean_relative_error:.2e}")
    
    # Test different tolerance levels to find the best accuracy
    tolerance_levels = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
    best_tolerance = None
    for tol in tolerance_levels:
      if max_abs_error < tol:
        best_tolerance = tol
        break
    
    print(f"  Best achievable tolerance: {best_tolerance:.0e}")
    
    # Assert that errors are within acceptable bounds
    assert max_abs_error < 1e-12, f"Max absolute error {max_abs_error:.2e} exceeds threshold 1e-12"
    assert mean_abs_error < 1e-13, f"Mean absolute error {mean_abs_error:.2e} exceeds threshold 1e-13"
    
  def test_rqs_inverse_forward_composition(self):
    """Test that inverse ∘ forward ≈ identity for RQS transformation."""
    
    # Set random seed
    key = jax.random.PRNGKey(123)
    
    # Test parameters
    num_bins = 8
    batch_size = 50
    num_features = 3
    
    # Generate RQS parameters
    num_params = 3 * num_bins + 1
    
    key, subkey = jax.random.split(key)
    widths = jax.random.uniform(subkey, (batch_size, num_features, num_bins),
                                minval=0.1, maxval=2.0, dtype=jnp.float64)
    widths = widths / jnp.sum(widths, axis=-1, keepdims=True)
    
    key, subkey = jax.random.split(key)
    heights = jax.random.uniform(subkey, (batch_size, num_features, num_bins),
                                 minval=0.1, maxval=2.0, dtype=jnp.float64)
    heights = heights / jnp.sum(heights, axis=-1, keepdims=True)
    
    key, subkey = jax.random.split(key)
    slopes = jax.random.uniform(subkey, (batch_size, num_features, num_bins + 1),
                                minval=0.5, maxval=2.0, dtype=jnp.float64)
    
    params = jnp.concatenate([widths, heights, slopes], axis=-1)
    
    # Create RQS bijector
    rqs = distrax.RationalQuadraticSpline(
      params,
      range_min=-3.0,
      range_max=3.0,
      min_knot_slope=1e-3,
      boundary_slopes='unconstrained'
    )
    
    # Generate test points
    key, subkey = jax.random.split(key)
    y_test = jax.random.uniform(subkey, (batch_size, num_features),
                                minval=-2.5, maxval=2.5, dtype=jnp.float64)
    
    # Inverse transformation
    x = rqs.inverse(y_test)
    
    # Forward transformation
    y_reconstructed = rqs.forward(x)
    
    # Calculate error
    abs_error = jnp.abs(y_reconstructed - y_test)
    relative_error = abs_error / (jnp.abs(y_test) + 1e-8)
    
    max_abs_error = jnp.max(abs_error)
    mean_abs_error = jnp.mean(abs_error)
    max_relative_error = jnp.max(relative_error)
    mean_relative_error = jnp.mean(relative_error)
    
    print(f"\nRQS Inverse-Forward Composition Test Results (float64):")
    print(f"  Max absolute error: {max_abs_error:.2e}")
    print(f"  Mean absolute error: {mean_abs_error:.2e}")
    print(f"  Max relative error: {max_relative_error:.2e}")
    print(f"  Mean relative error: {mean_relative_error:.2e}")
    
    # Test different tolerance levels
    tolerance_levels = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
    best_tolerance = None
    for tol in tolerance_levels:
      if max_abs_error < tol:
        best_tolerance = tol
        break
    
    print(f"  Best achievable tolerance: {best_tolerance:.0e}")
    
    # Assert errors are within bounds
    assert max_abs_error < 1e-12, f"Max absolute error {max_abs_error:.2e} exceeds threshold 1e-12"
    assert mean_abs_error < 1e-13, f"Mean absolute error {mean_abs_error:.2e} exceeds threshold 1e-13"
    
  def test_rqs_jacobian_consistency(self):
    """Test that the Jacobian of the forward mapping is consistent."""
    
    key = jax.random.PRNGKey(456)
    
    # Simple test case
    num_bins = 5
    num_features = 2
    
    # Generate parameters for a single transformation
    key, subkey = jax.random.split(key)
    widths = jax.random.uniform(subkey, (num_features, num_bins),
                                minval=0.1, maxval=2.0, dtype=jnp.float64)
    widths = widths / jnp.sum(widths, axis=-1, keepdims=True)
    
    key, subkey = jax.random.split(key)
    heights = jax.random.uniform(subkey, (num_features, num_bins),
                                 minval=0.1, maxval=2.0, dtype=jnp.float64)
    heights = heights / jnp.sum(heights, axis=-1, keepdims=True)
    
    key, subkey = jax.random.split(key)
    slopes = jax.random.uniform(subkey, (num_features, num_bins + 1),
                                minval=0.5, maxval=2.0, dtype=jnp.float64)
    
    params = jnp.concatenate([widths, heights, slopes], axis=-1)
    
    # Create RQS
    rqs = distrax.RationalQuadraticSpline(
      params,
      range_min=-2.0,
      range_max=2.0,
      min_knot_slope=1e-3,
      boundary_slopes='unconstrained'
    )
    
    # Test point
    key, subkey = jax.random.split(key)
    x_test = jax.random.uniform(subkey, (num_features,),
                                minval=-1.5, maxval=1.5, dtype=jnp.float64)
    
    # Get forward transformation and log determinant
    y, log_det = rqs.forward_and_log_det(x_test)
    
    # Compute Jacobian numerically
    jacobian = jax.jacfwd(rqs.forward)(x_test)
    numerical_log_det = jnp.log(jnp.abs(jnp.linalg.det(jacobian)))
    
    # Since log_det is per-feature, sum them for total log determinant
    total_log_det = jnp.sum(log_det)
    
    # Compare log determinants
    log_det_error = jnp.abs(total_log_det - numerical_log_det)
    
    print(f"\nRQS Jacobian Consistency Test (float64):")
    print(f"  Analytical log det (sum): {float(total_log_det):.12f}")
    print(f"  Numerical log det: {float(numerical_log_det):.12f}")
    print(f"  Error: {float(log_det_error):.2e}")
    
    # Test different tolerance levels for Jacobian
    tolerance_levels = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
    best_tolerance = None
    for tol in tolerance_levels:
      if log_det_error < tol:
        best_tolerance = tol
        break
    
    print(f"  Best achievable tolerance for Jacobian: {best_tolerance:.0e}")
    
    assert log_det_error < 1e-12, f"Log determinant error {log_det_error:.2e} exceeds threshold 1e-12"

  def test_rqs_boundary_behavior(self):
    """Test RQS behavior at boundaries of the range."""
    
    key = jax.random.PRNGKey(789)
    
    num_bins = 6
    num_features = 1
    range_min = -2.0
    range_max = 2.0
    
    # Generate parameters
    key, subkey = jax.random.split(key)
    widths = jax.random.uniform(subkey, (num_features, num_bins),
                                minval=0.1, maxval=2.0, dtype=jnp.float64)
    widths = widths / jnp.sum(widths, axis=-1, keepdims=True)
    
    key, subkey = jax.random.split(key)
    heights = jax.random.uniform(subkey, (num_features, num_bins),
                                 minval=0.1, maxval=2.0, dtype=jnp.float64)
    heights = heights / jnp.sum(heights, axis=-1, keepdims=True)
    
    key, subkey = jax.random.split(key)
    slopes = jax.random.uniform(subkey, (num_features, num_bins + 1),
                                minval=0.5, maxval=2.0, dtype=jnp.float64)
    
    params = jnp.concatenate([widths, heights, slopes], axis=-1)
    
    # Create RQS
    rqs = distrax.RationalQuadraticSpline(
      params,
      range_min=range_min,
      range_max=range_max,
      min_knot_slope=1e-3,
      boundary_slopes='unconstrained'
    )
    
    # Test points near boundaries
    epsilon = 1e-6
    test_points = jnp.array([
      [range_min + epsilon],
      [range_max - epsilon],
      [0.0],  # Middle point
      [-1.0],
      [1.0]
    ], dtype=jnp.float64)
    
    print(f"\nRQS Boundary Behavior Test (float64):")
    max_error = 0.0
    
    for x in test_points:
      y = rqs.forward(x)
      x_reconstructed = rqs.inverse(y)
      error = jnp.abs(x_reconstructed - x)
      max_error = max(max_error, float(error[0]))
      
      print(f"  x={x[0]:.6f}, y={y[0]:.6f}, error={error[0]:.2e}")
    
    # Test best achievable accuracy at boundaries
    tolerance_levels = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
    best_tolerance = None
    for tol in tolerance_levels:
      if max_error < tol:
        best_tolerance = tol
        break
    
    print(f"  Best achievable tolerance at boundaries: {best_tolerance:.0e}")
    
    assert max_error < 1e-12, f"Boundary test failed: max error={max_error:.2e}"

  def test_accuracy_limits(self):
    """Comprehensive test to determine the best achievable accuracy with float64."""
    
    print("\n" + "="*60)
    print("COMPREHENSIVE ACCURACY LIMITS TEST WITH FLOAT64")
    print("="*60)
    
    key = jax.random.PRNGKey(999)
    
    # Test with various configurations
    test_configs = [
      {"num_bins": 5, "batch_size": 10, "num_features": 1},
      {"num_bins": 10, "batch_size": 50, "num_features": 2},
      {"num_bins": 20, "batch_size": 100, "num_features": 3},
    ]
    
    overall_best_tolerance = 0
    
    for config in test_configs:
      num_bins = config["num_bins"]
      batch_size = config["batch_size"]
      num_features = config["num_features"]
      
      print(f"\nConfig: bins={num_bins}, batch={batch_size}, features={num_features}")
      
      # Generate parameters
      key, subkey = jax.random.split(key)
      widths = jax.random.uniform(subkey, (batch_size, num_features, num_bins),
                                  minval=0.1, maxval=2.0, dtype=jnp.float64)
      widths = widths / jnp.sum(widths, axis=-1, keepdims=True)
      
      key, subkey = jax.random.split(key)
      heights = jax.random.uniform(subkey, (batch_size, num_features, num_bins),
                                   minval=0.1, maxval=2.0, dtype=jnp.float64)
      heights = heights / jnp.sum(heights, axis=-1, keepdims=True)
      
      key, subkey = jax.random.split(key)
      slopes = jax.random.uniform(subkey, (batch_size, num_features, num_bins + 1),
                                  minval=0.5, maxval=2.0, dtype=jnp.float64)
      
      params = jnp.concatenate([widths, heights, slopes], axis=-1)
      
      # Create RQS
      rqs = distrax.RationalQuadraticSpline(
        params,
        range_min=-3.0,
        range_max=3.0,
        min_knot_slope=1e-3,
        boundary_slopes='unconstrained'
      )
      
      # Test points
      key, subkey = jax.random.split(key)
      x_test = jax.random.uniform(subkey, (batch_size, num_features),
                                  minval=-2.5, maxval=2.5, dtype=jnp.float64)
      
      # Forward-inverse composition
      y = rqs.forward(x_test)
      x_reconstructed = rqs.inverse(y)
      error_forward_inverse = jnp.max(jnp.abs(x_reconstructed - x_test))
      
      # Inverse-forward composition
      x_inv = rqs.inverse(x_test)
      y_reconstructed = rqs.forward(x_inv)
      error_inverse_forward = jnp.max(jnp.abs(y_reconstructed - x_test))
      
      max_error = max(float(error_forward_inverse), float(error_inverse_forward))
      
      print(f"  Forward-Inverse max error: {error_forward_inverse:.2e}")
      print(f"  Inverse-Forward max error: {error_inverse_forward:.2e}")
      
      # Find best tolerance for this config
      tolerance_levels = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
      for tol in tolerance_levels:
        if max_error < tol:
          print(f"  Best tolerance for this config: {tol:.0e}")
          overall_best_tolerance = max(overall_best_tolerance, max_error)
          break
    
    print("\n" + "="*60)
    print(f"OVERALL RECOMMENDED TOLERANCE: {overall_best_tolerance:.2e}")
    print("With float64 enabled, RQS can achieve accuracy around 1e-13 to 1e-14")
    print("="*60)
