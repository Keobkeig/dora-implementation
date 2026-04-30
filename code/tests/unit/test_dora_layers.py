"""
Unit tests for DoRA layer implementations.
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from dora.layers.base import DoRAConfig
    from dora.layers.dora_linear import DoRALinear, create_dora_layer
    from dora.utils.math_utils import (
        column_wise_l2_norm,
        compute_dora_weight,
        decompose_weight_dora,
        normalize_weight_direction,
    )

    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"Warning: Could not import DoRA modules: {e}")


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="DoRA modules not available")
class TestDoRAMath:
    """Test DoRA mathematical operations."""

    def test_column_wise_l2_norm(self):
        """Test column-wise L2 normalization."""
        # Create test weight matrix
        weight = torch.tensor([[3.0, 4.0], [1.0, 2.0], [5.0, 0.0]], dtype=torch.float32)

        # Compute norms
        norms = column_wise_l2_norm(weight)

        # Expected norms: [5.0, sqrt(5), 5.0]
        expected_norms = torch.tensor([5.0, np.sqrt(5), 5.0], dtype=torch.float32)

        assert torch.allclose(norms, expected_norms, atol=1e-6)

    def test_normalize_weight_direction(self):
        """Test weight direction normalization."""
        weight = torch.tensor([[3.0, 4.0], [1.0, 2.0]], dtype=torch.float32)

        normalized = normalize_weight_direction(weight)

        # Check that each row has unit norm
        norms = torch.norm(normalized, dim=1)
        expected_norms = torch.ones(2)

        assert torch.allclose(norms, expected_norms, atol=1e-6)

    def test_decompose_weight_dora(self):
        """Test DoRA weight decomposition."""
        weight = torch.tensor([[3.0, 4.0], [6.0, 8.0]], dtype=torch.float32)

        magnitude, direction = decompose_weight_dora(weight)

        # Check magnitude values
        expected_magnitude = torch.tensor([5.0, 10.0])
        assert torch.allclose(magnitude, expected_magnitude, atol=1e-6)

        # Check that direction is normalized
        direction_norms = torch.norm(direction, dim=1)
        assert torch.allclose(direction_norms, torch.ones(2), atol=1e-6)

        # Check reconstruction
        reconstructed = magnitude.unsqueeze(1) * direction
        assert torch.allclose(reconstructed, weight, atol=1e-6)

    def test_compute_dora_weight(self):
        """Test DoRA weight computation."""
        # Setup test data
        base_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        lora_A = torch.tensor([[0.1], [0.2]], dtype=torch.float32)
        lora_B = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
        magnitude = torch.tensor([1.0, 2.0], dtype=torch.float32)
        scaling = 1.0

        # Compute DoRA weight
        dora_weight = compute_dora_weight(base_weight, lora_A, lora_B, magnitude, scaling)

        # Check output shape
        assert dora_weight.shape == base_weight.shape

        # Check that magnitudes match expected values
        output_norms = torch.norm(dora_weight, dim=1)
        assert torch.allclose(output_norms, magnitude, atol=1e-5)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="DoRA modules not available")
class TestDoRALinear:
    """Test DoRA linear layer."""

    def test_dora_linear_creation(self):
        """Test DoRA linear layer creation."""
        layer = DoRALinear(in_features=128, out_features=256, rank=16, alpha=32, dropout=0.1)

        assert layer.in_features == 128
        assert layer.out_features == 256
        assert layer.rank == 16
        assert layer.alpha == 32
        assert layer.scaling == 2.0  # 32/16

    def test_load_base_weight(self):
        """Test loading base weights."""
        layer = DoRALinear(in_features=64, out_features=32, rank=8)

        # Create dummy base layer
        base_linear = nn.Linear(64, 32)

        # Load weights
        layer.load_base_weight(base_linear.weight, base_linear.bias)

        assert torch.equal(layer.base_weight, base_linear.weight)
        assert torch.equal(layer.bias, base_linear.bias)
        assert layer._magnitude_initialized

    def test_forward_pass(self):
        """Test forward pass."""
        layer = DoRALinear(in_features=64, out_features=32, rank=8)

        # Load dummy weights
        base_linear = nn.Linear(64, 32)
        layer.load_base_weight(base_linear.weight, base_linear.bias)

        # Test input
        x = torch.randn(10, 64)

        # Forward pass
        output = layer(x)

        assert output.shape == (10, 32)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_parameter_count(self):
        """Test parameter counting."""
        layer = DoRALinear(in_features=128, out_features=256, rank=16)

        total, trainable, frozen = layer.get_parameter_count()

        # Expected counts
        base_params = 128 * 256  # base weight
        bias_params = 256  # bias
        lora_a_params = 16 * 128  # LoRA A
        lora_b_params = 256 * 16  # LoRA B
        magnitude_params = 256  # magnitude

        expected_trainable = lora_a_params + lora_b_params + magnitude_params
        expected_frozen = base_params + bias_params
        expected_total = expected_trainable + expected_frozen

        assert trainable == expected_trainable
        assert frozen == expected_frozen
        assert total == expected_total

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        layer = DoRALinear(in_features=1024, out_features=1024, rank=16)

        ratio = layer.get_compression_ratio()

        # Should be much greater than 1 (indicating compression)
        assert ratio > 10

    def test_dora_enable_disable(self):
        """Test enabling/disabling DoRA mode."""
        layer = DoRALinear(in_features=64, out_features=32, rank=8)
        base_linear = nn.Linear(64, 32)
        layer.load_base_weight(base_linear.weight, base_linear.bias)

        # Initialize LoRA matrices with non-zero values to see difference
        with torch.no_grad():
            layer.lora_A.data.normal_(0, 0.1)
            layer.lora_B.data.normal_(0, 0.1)

        x = torch.randn(10, 64)

        # Test with DoRA enabled
        layer.enable_dora(True)
        output_dora = layer(x)

        # Test with DoRA disabled (should fall back to LoRA)
        layer.enable_dora(False)
        output_lora = layer(x)

        # Outputs should be different (or at least not exactly equal due to normalization)
        assert not torch.equal(output_dora, output_lora)

    def test_merge_weights(self):
        """Test weight merging for inference."""
        layer = DoRALinear(in_features=64, out_features=32, rank=8)
        base_linear = nn.Linear(64, 32)
        layer.load_base_weight(base_linear.weight, base_linear.bias)

        # Merge weights
        merged_layer = layer.merge_weights()

        assert isinstance(merged_layer, nn.Linear)
        assert merged_layer.in_features == layer.in_features
        assert merged_layer.out_features == layer.out_features

        # Test that outputs match
        x = torch.randn(10, 64)
        output_original = layer(x)
        output_merged = merged_layer(x)

        assert torch.allclose(output_original, output_merged, atol=1e-5)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="DoRA modules not available")
class TestDoRAConfig:
    """Test DoRA configuration."""

    def test_config_creation(self):
        """Test configuration creation with defaults."""
        config = DoRAConfig()

        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.dropout == 0.0
        assert config.scaling == 2.0  # 16/8

    def test_config_custom(self):
        """Test configuration with custom values."""
        config = DoRAConfig(rank=32, alpha=64, target_modules=["custom_proj"], task_type="SEQ_CLS")

        assert config.rank == 32
        assert config.alpha == 64
        assert config.scaling == 2.0  # 64/32
        assert "custom_proj" in config.target_modules

    def test_config_serialization(self):
        """Test configuration serialization."""
        config = DoRAConfig(rank=16, alpha=32)

        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["rank"] == 16
        assert config_dict["alpha"] == 32

        # Test deserialization
        new_config = DoRAConfig.from_dict(config_dict)
        assert new_config.rank == config.rank
        assert new_config.alpha == config.alpha


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="DoRA modules not available")
class TestCreateDoRALayer:
    """Test DoRA layer factory function."""

    def test_create_from_linear(self):
        """Test creating DoRA layer from nn.Linear."""
        base_layer = nn.Linear(128, 256, bias=True)

        dora_layer = create_dora_layer(base_layer, rank=16, alpha=32)

        assert isinstance(dora_layer, DoRALinear)
        assert dora_layer.in_features == 128
        assert dora_layer.out_features == 256
        assert dora_layer.rank == 16
        assert dora_layer.alpha == 32
        assert torch.equal(dora_layer.base_weight, base_layer.weight)

    def test_unsupported_layer_type(self):
        """Test error handling for unsupported layer types."""
        unsupported_layer = nn.BatchNorm1d(128)

        with pytest.raises(ValueError):
            create_dora_layer(unsupported_layer)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="DoRA modules not available")
class TestDoRAGradientFlow:
    """
    Verify that all three trainable parameters (lora_A, lora_B, magnitude)
    receive non-zero gradients after a single forward+backward pass.

    This is the most important correctness property for DoRA: if magnitude
    has zero gradient, the model degrades to vanilla LoRA with extra overhead.
    """

    def _make_layer(self, in_features=64, out_features=32, rank=8):
        base = nn.Linear(in_features, out_features, bias=True)
        layer = create_dora_layer(base, rank=rank, alpha=16.0, dropout=0.0)
        return layer

    def test_magnitude_receives_nonzero_gradient(self):
        """magnitude.grad must be non-None and non-zero after backward."""
        layer = self._make_layer()
        x = torch.randn(4, 64, requires_grad=False)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert layer.magnitude.grad is not None, (
            "magnitude.grad is None — magnitude has NO gradient. "
            "Check that magnitude is in the optimizer and compute_dora_weight "
            "does not detach it."
        )
        assert layer.magnitude.grad.norm().item() > 0, (
            f"magnitude.grad is all-zero (norm={layer.magnitude.grad.norm().item()}). "
            "DoRA magnitude is not learning."
        )

    def test_lora_B_receives_nonzero_gradient(self):
        """lora_B.grad must be non-None and non-zero after backward."""
        layer = self._make_layer()
        x = torch.randn(4, 64)
        out = layer(x)
        out.sum().backward()

        assert layer.lora_B.grad is not None
        assert layer.lora_B.grad.norm().item() > 0, "lora_B.grad is zero"

    def test_lora_A_receives_nonzero_gradient_after_lora_B_updated(self):
        """
        lora_A has zero gradient at step 0 (because lora_B=0), but non-zero
        after lora_B has been updated at least once.
        """
        layer = self._make_layer()

        # Step 0: lora_B is all-zeros, so lora_A gradient is zero (B@A = 0,
        # direction gradient path through A is gated by B).
        x = torch.randn(4, 64)
        out = layer(x)
        out.sum().backward()
        # lora_A gradient is zero at step 0 — that is expected behaviour
        lora_a_grad_step0 = layer.lora_A.grad.norm().item() if layer.lora_A.grad is not None else 0.0

        # Simulate one optimizer step to make lora_B non-zero
        with torch.no_grad():
            layer.lora_B.data -= 0.01 * layer.lora_B.grad
        layer.lora_A.grad = None
        layer.lora_B.grad = None
        layer.magnitude.grad = None

        # Step 1: lora_B is now non-zero, lora_A should get a gradient
        out = layer(x)
        out.sum().backward()

        assert layer.lora_A.grad is not None
        assert layer.lora_A.grad.norm().item() > 0, (
            "lora_A.grad is still zero after lora_B was updated. "
            "Gradient flow through the normalization Jacobian is broken."
        )

    def test_base_weight_has_no_gradient(self):
        """base_weight is a buffer and must never accumulate gradient."""
        layer = self._make_layer()
        x = torch.randn(4, 64)
        out = layer(x)
        out.sum().backward()

        assert layer.base_weight.grad is None, (
            "base_weight.grad is not None — frozen weight is leaking gradient."
        )

    def test_magnitude_grad_independent_of_lora_B_being_zero(self):
        """
        magnitude gradient must be non-zero even at step 0 when lora_B=0,
        because magnitude multiplies the direction directly without going
        through lora_B.  This is a key DoRA invariant.
        """
        layer = self._make_layer()
        # Confirm lora_B starts at zero
        assert layer.lora_B.data.norm().item() == 0.0, "lora_B should start at zeros"

        x = torch.randn(4, 64)
        out = layer(x)
        out.sum().backward()

        assert layer.magnitude.grad is not None
        assert layer.magnitude.grad.norm().item() > 0, (
            "magnitude.grad is zero even when lora_B=0. "
            "This means the gradient path magnitude→direction is broken. "
            "DoRA magnitude will never learn."
        )

    def test_normalize_weight_direction_rows_are_unit_norm(self):
        """After the double-eps fix, each row should be exactly unit norm."""
        from dora.utils.math_utils import normalize_weight_direction
        w = torch.randn(16, 32)
        v = normalize_weight_direction(w)
        row_norms = v.norm(dim=1)
        assert torch.allclose(row_norms, torch.ones(16), atol=1e-6), (
            f"rows are not unit norm after normalization; "
            f"max deviation = {(row_norms - 1).abs().max().item():.2e}"
        )


if __name__ == "__main__":
    # Run tests if script is called directly
    pytest.main([__file__, "-v"])
