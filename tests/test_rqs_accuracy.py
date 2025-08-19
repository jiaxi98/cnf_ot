"""Unit test for Rational Quadratic Spline (RQS) accuracy.

This test verifies that the composition of forward and inverse mappings
of the RQS transformation approximates the identity function within
acceptable numerical error bounds.
"""

import jax
import jax.numpy as jnp
import numpy as np
import distrax
import pytest

# Enable float64 for higher precision
jax.config.update("jax_enable_x64", True)


class TestRQSAccuracy:
  """Test suite for RQS forward and inverse mapping accuracy."""
  
  def test_rqs_comprehensive(self):
    """Comprehensive test for RQS accuracy covering all aspects."""
    
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Test configurations
    test_configs = [
      {
        "num_bins": 10,
        "batch_size": 100,
        "num_features": 2,
        "range_min": -5.0,
        "range_max": 5.0,
        "test_range": (-4.0, 4.0)
      },
      {
        "num_bins": 5,
        "batch_size": 50,
        "num_features": 1,
        "range_min": -3.0,
        "range_max": 3.0,
        "test_range": (-2.5, 2.5)
      },
      {
        "num_bins": 20,
        "batch_size": 200,
        "num_features": 3,
        "range_min": -4.0,
        "range_max": 4.0,
        "test_range": (-3.5, 3.5)
      }
    ]
    
    max_forward_inverse_error = 0.0
    max_inverse_forward_error = 0.0
    max_jacobian_error = 0.0
    max_boundary_error = 0.0
    
    for config in test_configs:
      # Generate RQS parameters once for this configuration
      key, subkey = jax.random.split(key)
      params = self._generate_rqs_params(
        subkey,
        config['batch_size'],
        config['num_features'],
        config['num_bins']
      )
      
      # Create RQS bijector
      rqs = distrax.RationalQuadraticSpline(
        params,
        range_min=config['range_min'],
        range_max=config['range_max'],
        min_knot_slope=1e-3,
        boundary_slopes='unconstrained'
      )
      
      # Test 1: Forward-Inverse Composition
      key, subkey = jax.random.split(key)
      x_test = jax.random.uniform(
        subkey,
        (config['batch_size'], config['num_features']),
        minval=config['test_range'][0],
        maxval=config['test_range'][1],
        dtype=jnp.float64
      )
      
      y = rqs.forward(x_test)
      x_reconstructed = rqs.inverse(y)
      forward_inverse_error = jnp.max(jnp.abs(x_reconstructed - x_test))
      max_forward_inverse_error = max(max_forward_inverse_error, float(forward_inverse_error))
      
      # Test 2: Inverse-Forward Composition
      key, subkey = jax.random.split(key)
      y_test = jax.random.uniform(
        subkey,
        (config['batch_size'], config['num_features']),
        minval=config['test_range'][0],
        maxval=config['test_range'][1],
        dtype=jnp.float64
      )
      
      x_inv = rqs.inverse(y_test)
      y_reconstructed = rqs.forward(x_inv)
      inverse_forward_error = jnp.max(jnp.abs(y_reconstructed - y_test))
      max_inverse_forward_error = max(max_inverse_forward_error, float(inverse_forward_error))
      
      # Test 3: Jacobian Consistency (for small feature dimensions)
      if config['num_features'] <= 2:
        key, subkey = jax.random.split(key)
        x_jac_test = jax.random.uniform(
          subkey,
          (config['num_features'],),
          minval=config['test_range'][0] * 0.5,
          maxval=config['test_range'][1] * 0.5,
          dtype=jnp.float64
        )
        
        # Create single-sample RQS for Jacobian test
        single_params = params[0] if config['batch_size'] > 1 else params
        rqs_single = distrax.RationalQuadraticSpline(
          single_params,
          range_min=config['range_min'],
          range_max=config['range_max'],
          min_knot_slope=1e-3,
          boundary_slopes='unconstrained'
        )
        
        y_jac, log_det = rqs_single.forward_and_log_det(x_jac_test)
        jacobian = jax.jacfwd(rqs_single.forward)(x_jac_test)
        numerical_log_det = jnp.log(jnp.abs(jnp.linalg.det(jacobian)))
        
        total_log_det = jnp.sum(log_det)
        jacobian_error = jnp.abs(total_log_det - numerical_log_det)
        max_jacobian_error = max(max_jacobian_error, float(jacobian_error))
      
      # Test 4: Boundary Behavior
      epsilon = 1e-6
      boundary_points = jnp.array([
        [config['range_min'] + epsilon] * config['num_features'],
        [config['range_max'] - epsilon] * config['num_features'],
        [0.0] * config['num_features'],
        [config['test_range'][0] * 0.5] * config['num_features'],
        [config['test_range'][1] * 0.5] * config['num_features']
      ], dtype=jnp.float64)
      
      if config['num_features'] == 1:
        boundary_points = boundary_points[:, :1]
      
      # Use first batch of parameters for boundary test
      boundary_params = params[:5] if config['batch_size'] >= 5 else jnp.tile(params[0:1], (5, 1))
      rqs_boundary = distrax.RationalQuadraticSpline(
        boundary_params,
        range_min=config['range_min'],
        range_max=config['range_max'],
        min_knot_slope=1e-3,
        boundary_slopes='unconstrained'
      )
      
      y_boundary = rqs_boundary.forward(boundary_points)
      x_boundary_reconstructed = rqs_boundary.inverse(y_boundary)
      boundary_errors = jnp.abs(x_boundary_reconstructed - boundary_points)
      boundary_error = jnp.max(boundary_errors)
      max_boundary_error = max(max_boundary_error, float(boundary_error))
    
    # Assertions - with float64, we expect machine precision accuracy
    assert max_forward_inverse_error < 1e-12, f"Forward-inverse error {max_forward_inverse_error:.2e} exceeds 1e-12"
    assert max_inverse_forward_error < 1e-12, f"Inverse-forward error {max_inverse_forward_error:.2e} exceeds 1e-12"
    if max_jacobian_error > 0:  # Only check if Jacobian was tested
      assert max_jacobian_error < 1e-12, f"Jacobian error {max_jacobian_error:.2e} exceeds 1e-12"
    assert max_boundary_error < 1e-12, f"Boundary error {max_boundary_error:.2e} exceeds 1e-12"
  
  def _generate_rqs_params(self, key, batch_size, num_features, num_bins):
    """Generate normalized RQS parameters."""
    
    # Generate widths
    key, subkey = jax.random.split(key)
    widths = jax.random.uniform(
      subkey,
      (batch_size, num_features, num_bins),
      minval=0.1,
      maxval=2.0,
      dtype=jnp.float64
    )
    widths = widths / jnp.sum(widths, axis=-1, keepdims=True)
    
    # Generate heights
    key, subkey = jax.random.split(key)
    heights = jax.random.uniform(
      subkey,
      (batch_size, num_features, num_bins),
      minval=0.1,
      maxval=2.0,
      dtype=jnp.float64
    )
    heights = heights / jnp.sum(heights, axis=-1, keepdims=True)
    
    # Generate slopes
    key, subkey = jax.random.split(key)
    slopes = jax.random.uniform(
      subkey,
      (batch_size, num_features, num_bins + 1),
      minval=0.5,
      maxval=2.0,
      dtype=jnp.float64
    )
    
    # Concatenate parameters
    return jnp.concatenate([widths, heights, slopes], axis=-1)