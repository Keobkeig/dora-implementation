"""
Unit tests for LoRA layer implementation.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from dora.layers.lora_linear import LoRALinear, apply_lora_to_model, create_lora_layer

    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"Warning: Could not import LoRA modules: {e}")


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="LoRA modules not available")
class TestLoRALinear:
    """Test standard LoRA linear layer."""

    def test_creation(self):
        """Test LoRA layer creation with correct attributes."""
        layer = LoRALinear(in_features=128, out_features=256, rank=16, alpha=32.0)

        assert layer.in_features == 128
        assert layer.out_features == 256
        assert layer.rank == 16
        assert layer.alpha == 32.0
        assert layer.scaling == 2.0  # 32/16

    def test_load_base_weight(self):
        """Test loading pretrained weights."""
        layer = LoRALinear(in_features=64, out_features=32, rank=8)
        base = nn.Linear(64, 32)

        layer.load_base_weight(base.weight, base.bias)

        assert torch.equal(layer.base_weight, base.weight)
        assert torch.equal(layer.bias, base.bias)

    def test_forward_shape(self):
        """Test that forward pass produces the correct output shape."""
        layer = LoRALinear(in_features=64, out_features=32, rank=8)
        base = nn.Linear(64, 32)
        layer.load_base_weight(base.weight, base.bias)

        x = torch.randn(10, 64)
        output = layer(x)

        assert output.shape == (10, 32)

    def test_forward_no_nan_inf(self):
        """Test that forward pass produces finite outputs."""
        layer = LoRALinear(in_features=64, out_features=32, rank=8)
        base = nn.Linear(64, 32)
        layer.load_base_weight(base.weight, base.bias)

        x = torch.randn(10, 64)
        output = layer(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_initial_output_matches_base(self):
        """At init (B=0), LoRA output should match the base linear layer."""
        base = nn.Linear(64, 32)
        layer = create_lora_layer(base, rank=8, alpha=16.0)

        x = torch.randn(5, 64)
        with torch.no_grad():
            base_out = base(x)
            lora_out = layer(x)

        assert torch.allclose(base_out, lora_out, atol=1e-5)

    def test_gradient_flow(self):
        """Test that gradients flow through LoRA parameters."""
        layer = LoRALinear(in_features=64, out_features=32, rank=8)
        base = nn.Linear(64, 32)
        layer.load_base_weight(base.weight, base.bias)

        x = torch.randn(5, 64)
        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert layer.lora_A.grad is not None
        assert layer.lora_B.grad is not None
        # base_weight is a buffer, so no grad
        assert not layer.base_weight.requires_grad

    def test_parameter_count(self):
        """Test parameter counting."""
        layer = LoRALinear(in_features=128, out_features=256, rank=16)

        total, trainable, frozen = layer.get_parameter_count()

        base_params = 128 * 256  # base weight
        bias_params = 256  # bias
        lora_a = 16 * 128
        lora_b = 256 * 16

        assert frozen == base_params + bias_params
        assert trainable == lora_a + lora_b
        assert total == frozen + trainable

    def test_no_magnitude_parameter(self):
        """LoRA should NOT have a magnitude parameter (unlike DoRA)."""
        layer = LoRALinear(in_features=64, out_features=32, rank=8)

        param_names = [name for name, _ in layer.named_parameters()]
        assert "magnitude" not in param_names
        assert "lora_A" in param_names
        assert "lora_B" in param_names

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        layer = LoRALinear(in_features=1024, out_features=1024, rank=16)
        ratio = layer.get_compression_ratio()

        # Should be significantly greater than 1
        assert ratio > 10

    def test_merge_weights(self):
        """Test merging LoRA weights into a standard nn.Linear."""
        base = nn.Linear(64, 32)
        layer = create_lora_layer(base, rank=8, alpha=16.0)

        # Set non-zero LoRA weights to make sure merge works
        with torch.no_grad():
            layer.lora_A.data.normal_(0, 0.1)
            layer.lora_B.data.normal_(0, 0.1)

        merged = layer.merge_weights()

        assert isinstance(merged, nn.Linear)
        assert merged.in_features == 64
        assert merged.out_features == 32

        x = torch.randn(5, 64)
        with torch.no_grad():
            original_out = layer(x)
            merged_out = merged(x)

        assert torch.allclose(original_out, merged_out, atol=1e-5)

    def test_differs_after_training(self):
        """After updating LoRA params, output should differ from base."""
        base = nn.Linear(64, 32)
        layer = create_lora_layer(base, rank=8, alpha=16.0)

        x = torch.randn(5, 64)

        with torch.no_grad():
            before = layer(x).clone()
            layer.lora_A.data.normal_(0, 0.5)
            layer.lora_B.data.normal_(0, 0.5)
            after = layer(x)

        assert not torch.allclose(before, after, atol=1e-3)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="LoRA modules not available")
class TestCreateLoRALayer:
    """Test the LoRA factory function."""

    def test_from_linear(self):
        """Test creating LoRA layer from nn.Linear."""
        base = nn.Linear(128, 256, bias=True)
        lora = create_lora_layer(base, rank=16, alpha=32.0)

        assert isinstance(lora, LoRALinear)
        assert lora.in_features == 128
        assert lora.out_features == 256
        assert lora.rank == 16
        assert torch.equal(lora.base_weight, base.weight)

    def test_unsupported_type(self):
        """Test error for unsupported layer types."""
        unsupported = nn.BatchNorm1d(128)
        with pytest.raises(ValueError):
            create_lora_layer(unsupported)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="LoRA modules not available")
class TestApplyLoRAToModel:
    """Test the model-level LoRA injection helper."""

    def test_apply_to_simple_model(self):
        """Test injecting LoRA into a simple model."""

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)
                self.mlp = nn.Linear(64, 32)  # not in target_modules

            def forward(self, x):
                return self.mlp(self.q_proj(x) + self.v_proj(x))

        model = TinyModel()
        apply_lora_to_model(model, target_modules=["q_proj", "v_proj"], rank=4, alpha=8.0)

        assert isinstance(model.q_proj, LoRALinear)
        assert isinstance(model.v_proj, LoRALinear)
        assert isinstance(model.mlp, nn.Linear)  # untouched

    def test_forward_after_injection(self):
        """Model should still produce valid output after LoRA injection."""

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(32, 32)
                self.out = nn.Linear(32, 8)

            def forward(self, x):
                return self.out(self.q_proj(x))

        model = TinyModel()
        apply_lora_to_model(model, target_modules=["q_proj"], rank=4, alpha=8.0)

        x = torch.randn(3, 32)
        output = model(x)

        assert output.shape == (3, 8)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
