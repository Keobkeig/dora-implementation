"""
DoRA: Weight-Decomposed Low-Rank Adaptation
Mathematical utilities for DoRA implementation.
"""

import math
from typing import Tuple

import torch


def column_wise_l2_norm(weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute column-wise L2 normalization of a weight matrix.

    Args:
        weight: Weight tensor of shape (out_features, in_features)
        eps: Small epsilon for numerical stability

    Returns:
        Column-wise L2 norms of shape (out_features,)
    """
    # Compute L2 norm for each output channel (row)
    norms = torch.norm(weight, dim=1, keepdim=False)
    return torch.clamp(norms, min=eps)


def normalize_weight_direction(weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize weight matrix row-wise to (approximate) unit vectors.

    Args:
        weight: Weight tensor of shape (out_features, in_features)
        eps: Small epsilon for numerical stability

    Returns:
        Normalized weight tensor of same shape
    """
    # column_wise_l2_norm already clamps norms to min=eps, so dividing by
    # (norms + eps) would double-count eps and produce slightly-sub-unit rows.
    # Divide by norms directly; the clamp guarantees no division by zero.
    norms = column_wise_l2_norm(weight, eps)
    return weight / norms.unsqueeze(1)


def decompose_weight_dora(
    weight: torch.Tensor, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose a weight matrix into magnitude and direction components.

    This is the core mathematical operation of DoRA:
    W = m * V where V is normalized direction and m is magnitude.

    Args:
        weight: Weight tensor of shape (out_features, in_features)
        eps: Small epsilon for numerical stability

    Returns:
        Tuple of (magnitude, direction) where:
        - magnitude: tensor of shape (out_features,)
        - direction: normalized tensor of shape (out_features, in_features)
    """
    magnitude = column_wise_l2_norm(weight, eps)
    direction = normalize_weight_direction(weight, eps)
    return magnitude, direction


def compute_dora_weight(
    base_weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    magnitude: torch.Tensor,
    scaling: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the final DoRA weight using the decomposition formula:
    W = m * (W_0 + scaling * B @ A) / ||W_0 + scaling * B @ A||

    Args:
        base_weight: Frozen pre-trained weight W_0 of shape (out_features, in_features)
        lora_A: LoRA A matrix of shape (rank, in_features)
        lora_B: LoRA B matrix of shape (out_features, rank)
        magnitude: Learnable magnitude vector of shape (out_features,)
        scaling: LoRA scaling factor (alpha/rank)
        eps: Small epsilon for numerical stability

    Returns:
        Final DoRA weight of shape (out_features, in_features)
    """
    # Compute LoRA update: ΔW = B @ A
    lora_update = lora_B @ lora_A

    # Combine with base weight: W_0 + scaling * ΔW
    combined_weight = base_weight + scaling * lora_update

    # Normalize direction component
    direction = normalize_weight_direction(combined_weight, eps)

    # Apply learned magnitude: W = m * direction
    final_weight = magnitude.unsqueeze(1) * direction

    return final_weight


def initialize_dora_magnitude(base_weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Initialize magnitude parameter from base weight.

    Args:
        base_weight: Pre-trained weight tensor
        eps: Small epsilon for numerical stability

    Returns:
        Initialized magnitude parameter
    """
    magnitude, _ = decompose_weight_dora(base_weight, eps)
    return magnitude.clone().detach()


def lora_init_kaiming_uniform(
    tensor: torch.Tensor, a: float = 0, nonlinearity: str = "linear"
) -> torch.Tensor:
    """
    Kaiming uniform initialization for LoRA matrices.

    Args:
        tensor: Tensor to initialize
        a: Negative slope for leaky relu (not used for linear)
        nonlinearity: Type of nonlinearity

    Returns:
        Initialized tensor
    """
    num_input_fmaps = tensor.size(1)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size

    gain = math.sqrt(2.0 / (1 + a * a)) if nonlinearity == "leaky_relu" else 1.0
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std

    with torch.no_grad():
        tensor.uniform_(-bound, bound)
    return tensor


class DoRAMath:
    """
    Helper class containing all DoRA mathematical operations.
    """

    @staticmethod
    def compute_effective_weight(
        base_weight: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        magnitude: torch.Tensor,
        alpha: float,
        rank: int,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        High-level interface for computing DoRA effective weight.
        """
        scaling = alpha / rank
        return compute_dora_weight(base_weight, lora_A, lora_B, magnitude, scaling, eps)

    @staticmethod
    def magnitude_grad_scale(magnitude: torch.Tensor, base_weight: torch.Tensor) -> float:
        """
        Compute appropriate gradient scaling for magnitude parameter.
        This helps with training stability.
        """
        base_norm = torch.norm(base_weight).item()
        mag_norm = torch.norm(magnitude).item()
        return base_norm / (mag_norm + 1e-8)
