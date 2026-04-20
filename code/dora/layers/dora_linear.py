"""
DoRA layer implementations.
Core DoRA linear layer with magnitude/direction decomposition.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..utils.math_utils import (
        compute_dora_weight,
        initialize_dora_magnitude,
        lora_init_kaiming_uniform,
    )
except ImportError:
    # Fallback for testing
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
    from math_utils import (
        compute_dora_weight,
        initialize_dora_magnitude,
        lora_init_kaiming_uniform,
    )


class DoRALinear(nn.Module):
    """
    DoRA (Weight-Decomposed Low-Rank Adaptation) Linear Layer.

    This layer implements the core DoRA functionality:
    W = m * (W_0 + ΔW) / ||W_0 + ΔW||

    Where:
    - W_0: frozen pre-trained weights
    - ΔW: LoRA update (B @ A)
    - m: learnable magnitude parameter (per output channel)

    Args:
        in_features: Size of input features
        out_features: Size of output features
        rank: Rank of LoRA adaptation (r)
        alpha: LoRA scaling parameter
        dropout: Dropout probability for LoRA layers
        bias: Whether to include bias (frozen from original layer)
        device: Device to place parameters on
        dtype: Data type for parameters
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        eps: float = 1e-8,
        init_magnitude_from_weight: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.eps = eps

        # Frozen base weight (will be loaded from pre-trained model)
        self.register_buffer(
            "base_weight", torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # Optional frozen bias
        if bias:
            self.register_buffer("bias", torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.bias = None

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank, device=device, dtype=dtype))

        # DoRA magnitude parameter (learnable)
        self.magnitude = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))

        # Dropout for LoRA
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Training flags
        self._dora_enabled = True
        self._magnitude_initialized = False
        self.init_magnitude_from_weight = init_magnitude_from_weight

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize LoRA parameters."""
        # Initialize LoRA A with Kaiming uniform
        lora_init_kaiming_uniform(self.lora_A)

        # Initialize LoRA B with zeros (standard practice)
        nn.init.zeros_(self.lora_B)

        # Magnitude will be initialized when base weight is loaded
        if not self.init_magnitude_from_weight:
            nn.init.ones_(self.magnitude)
            self._magnitude_initialized = True

    def load_base_weight(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """
        Load frozen base weight from pre-trained model.

        Args:
            weight: Pre-trained weight tensor of shape (out_features, in_features)
            bias: Optional pre-trained bias tensor of shape (out_features,)
        """
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"Weight shape {weight.shape} doesn't match layer dimensions "
                f"({self.out_features}, {self.in_features})"
            )

        self.base_weight.copy_(weight.detach())

        if bias is not None and self.bias is not None:
            if bias.shape != (self.out_features,):
                raise ValueError(
                    f"Bias shape {bias.shape} doesn't match output features {self.out_features}"
                )
            self.bias.copy_(bias.detach())

        # Initialize magnitude from base weight if requested
        if self.init_magnitude_from_weight and not self._magnitude_initialized:
            magnitude_init = initialize_dora_magnitude(weight, self.eps)
            self.magnitude.data.copy_(magnitude_init)
            self._magnitude_initialized = True

    def get_effective_weight(self) -> torch.Tensor:
        """
        Compute the effective weight using DoRA decomposition.

        Returns:
            Effective weight tensor of shape (out_features, in_features)
        """
        if not self._dora_enabled:
            # Fall back to standard LoRA if DoRA is disabled
            return self.base_weight + self.scaling * (self.lora_B @ self.lora_A)

        return compute_dora_weight(
            self.base_weight, self.lora_A, self.lora_B, self.magnitude, self.scaling, self.eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DoRA linear layer.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        if not self._magnitude_initialized:
            raise RuntimeError("Base weight not loaded. Call load_base_weight() first.")

        # Apply dropout to input for LoRA path
        x_dropped = self.dropout(x)

        if not self._dora_enabled:
            # Standard LoRA forward pass
            base_output = F.linear(x, self.base_weight, self.bias)
            lora_output = self.scaling * F.linear(x_dropped, self.lora_B @ self.lora_A)
            return base_output + lora_output

        # DoRA forward pass
        effective_weight = self.get_effective_weight()
        return F.linear(x, effective_weight, self.bias)

    def enable_dora(self, enabled: bool = True):
        """Enable or disable DoRA mode."""
        self._dora_enabled = enabled

    def disable_dora(self):
        """Disable DoRA mode (fall back to LoRA)."""
        self.enable_dora(False)

    def get_parameter_count(self) -> Tuple[int, int, int]:
        """
        Get parameter counts for analysis.

        Returns:
            Tuple of (total_params, trainable_params, frozen_params)
        """
        base_params = self.base_weight.numel()
        bias_params = self.bias.numel() if self.bias is not None else 0
        frozen_params = base_params + bias_params

        lora_a_params = self.lora_A.numel()
        lora_b_params = self.lora_B.numel()
        magnitude_params = self.magnitude.numel()
        trainable_params = lora_a_params + lora_b_params + magnitude_params

        total_params = frozen_params + trainable_params

        return total_params, trainable_params, frozen_params

    def get_compression_ratio(self) -> float:
        """
        Calculate parameter compression ratio compared to full fine-tuning.

        Returns:
            Compression ratio (higher is better)
        """
        total_params, trainable_params, _ = self.get_parameter_count()
        return total_params / trainable_params

    def extra_repr(self) -> str:
        """String representation of the layer."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"alpha={self.alpha}, "
            f"scaling={self.scaling:.3f}, "
            f"dora_enabled={self._dora_enabled}"
        )

    def merge_weights(self) -> nn.Linear:
        """
        Merge DoRA weights into a standard linear layer for inference.
        This eliminates the computational overhead during inference.

        Returns:
            Standard nn.Linear layer with merged weights
        """
        effective_weight = self.get_effective_weight()

        merged_layer = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=effective_weight.device,
            dtype=effective_weight.dtype,
        )

        merged_layer.weight.data.copy_(effective_weight)
        if self.bias is not None:
            merged_layer.bias.data.copy_(self.bias)

        return merged_layer


class DoRAConv2d(nn.Module):
    """
    DoRA implementation for 2D Convolutional layers.
    Extends DoRA to convolutional operations for Vision Transformers.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolving kernel
        rank: Rank of LoRA adaptation
        alpha: LoRA scaling parameter
        stride: Stride of convolution
        padding: Padding added to input
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        rank: int = 8,
        alpha: float = 16.0,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dropout: float = 0.0,
        bias: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.eps = eps

        # Frozen base weight
        weight_shape = (out_channels, in_channels, *self.kernel_size)
        self.register_buffer("base_weight", torch.empty(weight_shape, device=device, dtype=dtype))

        # Optional frozen bias
        if bias:
            self.register_buffer("bias", torch.empty(out_channels, device=device, dtype=dtype))
        else:
            self.bias = None

        # LoRA matrices for conv layers - we flatten spatial dimensions
        spatial_size = self.kernel_size[0] * self.kernel_size[1]
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_channels * spatial_size, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(torch.empty(out_channels, rank, device=device, dtype=dtype))

        # DoRA magnitude parameter (per output channel)
        self.magnitude = nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))

        # Dropout
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()

        self._dora_enabled = True
        self._magnitude_initialized = False

        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters."""
        lora_init_kaiming_uniform(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def load_base_weight(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Load frozen base weight from pre-trained conv layer."""
        expected_shape = (self.out_channels, self.in_channels, *self.kernel_size)
        if weight.shape != expected_shape:
            raise ValueError(f"Weight shape {weight.shape} doesn't match expected {expected_shape}")

        self.base_weight.copy_(weight.detach())

        if bias is not None and self.bias is not None:
            self.bias.copy_(bias.detach())

        # Initialize magnitude from flattened weight norms
        weight_flat = weight.view(self.out_channels, -1)
        magnitude_init = initialize_dora_magnitude(weight_flat, self.eps)
        self.magnitude.data.copy_(magnitude_init)
        self._magnitude_initialized = True

    def get_effective_weight(self) -> torch.Tensor:
        """Compute effective conv weight using DoRA."""
        if not self._dora_enabled:
            # Standard LoRA for conv
            base_flat = self.base_weight.view(self.out_channels, -1)
            lora_update = self.lora_B @ self.lora_A
            combined_flat = base_flat + self.scaling * lora_update
            return combined_flat.view_as(self.base_weight)

        # DoRA for conv
        base_flat = self.base_weight.view(self.out_channels, -1)
        effective_flat = compute_dora_weight(
            base_flat, self.lora_A, self.lora_B, self.magnitude, self.scaling, self.eps
        )
        return effective_flat.view_as(self.base_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DoRA conv layer."""
        if not self._magnitude_initialized:
            raise RuntimeError("Base weight not loaded. Call load_base_weight() first.")

        x_dropped = self.dropout(x)
        effective_weight = self.get_effective_weight()

        return F.conv2d(x_dropped, effective_weight, self.bias, self.stride, self.padding)

    def enable_dora(self, enabled: bool = True):
        """Enable or disable DoRA mode."""
        self._dora_enabled = enabled


# Factory function for creating DoRA layers
def create_dora_layer(
    base_layer: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    eps: float = 1e-8,
) -> Union[DoRALinear, DoRAConv2d]:
    """
    Factory function to create appropriate DoRA layer from base layer.

    Args:
        base_layer: Original PyTorch layer (nn.Linear or nn.Conv2d)
        rank: LoRA rank
        alpha: LoRA alpha parameter
        dropout: Dropout probability
        eps: Numerical stability epsilon

    Returns:
        Corresponding DoRA layer
    """
    if isinstance(base_layer, nn.Linear):
        dora_layer = DoRALinear(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=base_layer.bias is not None,
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
            eps=eps,
        )
        dora_layer.load_base_weight(
            base_layer.weight.data, base_layer.bias.data if base_layer.bias is not None else None
        )
        return dora_layer

    elif isinstance(base_layer, nn.Conv2d):
        dora_layer = DoRAConv2d(
            in_channels=base_layer.in_channels,
            out_channels=base_layer.out_channels,
            kernel_size=base_layer.kernel_size,
            rank=rank,
            alpha=alpha,
            stride=base_layer.stride,
            padding=base_layer.padding,
            dropout=dropout,
            bias=base_layer.bias is not None,
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
            eps=eps,
        )
        dora_layer.load_base_weight(
            base_layer.weight.data, base_layer.bias.data if base_layer.bias is not None else None
        )
        return dora_layer

    else:
        raise ValueError(f"Unsupported layer type: {type(base_layer)}")
