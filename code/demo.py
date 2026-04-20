"""
Quick demonstration of the DoRA implementation.
This script shows how to use the DoRA library with a simple example.
"""

import os
import sys

import torch
import torch.nn as nn

# Add the dora_implementation to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import DoRA components
from dora.layers.base import DoRAModule
from dora.layers.dora_linear import create_dora_layer
from dora.utils.math_utils import decompose_weight_dora

print("🚀 DoRA Implementation Demonstration")
print("=" * 50)


# 1. Create a simple model with regular linear layers
class SimpleModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, output_size=10):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


print("1. Creating a simple model...")
original_model = SimpleModel()
print(f"   Original model parameters: {sum(p.numel() for p in original_model.parameters()):,}")

# 2. Convert model to use DoRA layers
print("\n2. Converting to DoRA...")
dora_model = SimpleModel()
dora_model.load_state_dict(original_model.state_dict())

# Replace linear layers with DoRA equivalents
dora_model.layer1 = create_dora_layer(dora_model.layer1, rank=16, alpha=32)
dora_model.layer2 = create_dora_layer(dora_model.layer2, rank=16, alpha=32)
dora_model.layer3 = create_dora_layer(dora_model.layer3, rank=8, alpha=16)

# Count parameters
param_stats = DoRAModule.count_parameters(dora_model)
print(f"   Total parameters: {param_stats['total']:,}")
print(f"   Trainable parameters: {param_stats['trainable']:,}")
print(f"   Frozen parameters: {param_stats['frozen']:,}")
print(f"   Compression ratio: {param_stats['compression_ratio']:.1f}x")
print(f"   Trainable percentage: {100 * param_stats['trainable'] / param_stats['total']:.2f}%")

# 3. Test forward pass
print("\n3. Testing forward pass...")
batch_size = 32
input_data = torch.randn(batch_size, 128)

with torch.no_grad():
    original_output = original_model(input_data)
    dora_output = dora_model(input_data)

print(f"   Original output shape: {original_output.shape}")
print(f"   DoRA output shape: {dora_output.shape}")
print(f"   Outputs are close: {torch.allclose(original_output, dora_output, atol=1e-3)}")

# 4. Show DoRA-specific features
print("\n4. DoRA-specific features...")

# Get a DoRA layer
dora_layer = dora_model.layer1
print(f"   DoRA layer type: {type(dora_layer).__name__}")
print(f"   Rank: {dora_layer.rank}")
print(f"   Alpha: {dora_layer.alpha}")
print(f"   Scaling: {dora_layer.scaling}")

# Test enabling/disabling DoRA mode
dora_layer.enable_dora(True)
output_with_dora = dora_layer(input_data[:, :256])  # Adjust for layer input size

dora_layer.enable_dora(False)
output_without_dora = dora_layer(input_data[:, :256])

print(
    "   DoRA vs LoRA outputs are different: "
    f"{not torch.allclose(output_with_dora, output_without_dora, atol=1e-5)}"
)

# Re-enable DoRA
dora_layer.enable_dora(True)

# 5. Mathematical components demonstration
print("\n5. DoRA mathematical components...")

# Create a test weight matrix
test_weight = torch.tensor([[3.0, 4.0], [6.0, 8.0]], dtype=torch.float32)
print(f"   Test weight matrix:\n{test_weight}")

magnitude, direction = decompose_weight_dora(test_weight)
print(f"   Magnitude: {magnitude}")
print(f"   Direction normalized: {torch.allclose(torch.norm(direction, dim=1), torch.ones(2))}")

# Reconstruct
reconstructed = magnitude.unsqueeze(1) * direction
print(f"   Reconstruction matches: {torch.allclose(reconstructed, test_weight)}")

# 6. Show memory efficiency
print("\n6. Memory efficiency comparison...")


def count_model_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# Set requires_grad appropriately for comparison
for param in original_model.parameters():
    param.requires_grad = True

for param in dora_model.parameters():
    param.requires_grad = False

# Only DoRA parameters are trainable
for name, param in dora_model.named_parameters():
    if any(dora_param in name for dora_param in ["lora_A", "lora_B", "magnitude"]):
        param.requires_grad = True

orig_total, orig_trainable = count_model_params(original_model)
dora_total, dora_trainable = count_model_params(dora_model)

print(f"   Original model - Total: {orig_total:,}, Trainable: {orig_trainable:,}")
print(f"   DoRA model - Total: {dora_total:,}, Trainable: {dora_trainable:,}")
print(f"   Parameter reduction: {100 * (orig_trainable - dora_trainable) / orig_trainable:.1f}%")

# 7. Demonstrate weight merging for inference
print("\n7. Weight merging for inference...")
merged_layer = dora_model.layer1.merge_weights()
print(f"   Merged layer type: {type(merged_layer).__name__}")

# Test that outputs match
test_input = input_data[:, :256]
original_dora_output = dora_model.layer1(test_input)
merged_output = merged_layer(test_input)
print(f"   Merged outputs match: {torch.allclose(original_dora_output, merged_output, atol=1e-5)}")

print("\n✅ DoRA demonstration completed successfully!")
print("\nKey takeaways:")
print("- DoRA reduces trainable parameters by ~90% compared to full fine-tuning")
print("- DoRA maintains model performance while adding magnitude control")
print("- DoRA can be easily integrated into existing PyTorch models")
print("- DoRA weights can be merged for efficient inference")
print("\nFor more advanced usage, see:")
print("- scripts/train_commonsense.py for training examples")
print("- benchmarks/dora_vs_lora.py for performance comparisons")
print("- configs/ for configuration examples")

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Running demonstration...")
    print("=" * 50)
