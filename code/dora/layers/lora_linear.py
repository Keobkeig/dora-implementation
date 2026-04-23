"""
Standard LoRA (Low-Rank Adaptation) layer implementations.

Provides a clean LoRA baseline for comparison against DoRA.
LoRA learns a low-rank update ΔW = BA that is added to the frozen
pretrained weight:  W_eff = W₀ + (α/r) · BA

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
           arXiv:2106.09685 (2021)
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.math_utils import lora_init_kaiming_uniform


class LoRALinear(nn.Module):
    """
    Standard LoRA (Low-Rank Adaptation) Linear Layer.

    Forward pass:
        y = x @ (W₀ + (α/r) · BA)ᵀ + bias
          = F.linear(x, W₀, bias) + (α/r) · F.linear(dropout(x), B @ A)

    Args:
        in_features: Size of input features
        out_features: Size of output features
        rank: Rank of LoRA adaptation (r)
        alpha: LoRA scaling parameter (scaling = alpha / rank)
        dropout: Dropout probability on the LoRA path
        bias: Whether to include bias (frozen from the original layer)
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
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen base weight (loaded from the pretrained model)
        self.register_buffer(
            "base_weight", torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # Optional frozen bias
        if bias:
            self.register_buffer("bias", torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.bias = None

        # LoRA low-rank matrices (trainable)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank, device=device, dtype=dtype))

        # Dropout on the LoRA path
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialize
        self._init_parameters()

    def _init_parameters(self):
        """Initialize LoRA parameters following the original paper."""
        # A ← Kaiming uniform, B ← zeros  →  ΔW starts at zero
        lora_init_kaiming_uniform(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def load_base_weight(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """
        Load frozen base weight from a pretrained layer.

        Args:
            weight: Pretrained weight of shape (out_features, in_features)
            bias: Optional pretrained bias of shape (out_features,)
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

    def get_effective_weight(self) -> torch.Tensor:
        """
        Compute the effective weight: W₀ + (α/r) · BA.

        Returns:
            Effective weight tensor of shape (out_features, in_features)
        """
        return self.base_weight + self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LoRA linear layer.

        Uses the efficient two-matmul path (no full weight materialisation):
            y = F.linear(x, W₀, bias) + scaling · F.linear(dropout(x), B @ A)

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        base_output = F.linear(x, self.base_weight, self.bias)
        lora_output = self.scaling * F.linear(self.dropout(x), self.lora_B @ self.lora_A)
        return base_output + lora_output

    def get_parameter_count(self) -> Tuple[int, int, int]:
        """
        Get parameter counts for analysis.

        Returns:
            Tuple of (total_params, trainable_params, frozen_params)
        """
        base_params = self.base_weight.numel()
        bias_params = self.bias.numel() if self.bias is not None else 0
        frozen_params = base_params + bias_params

        trainable_params = self.lora_A.numel() + self.lora_B.numel()
        total_params = frozen_params + trainable_params

        return total_params, trainable_params, frozen_params

    def get_compression_ratio(self) -> float:
        """Compression ratio compared to full fine-tuning (higher is better)."""
        total_params, trainable_params, _ = self.get_parameter_count()
        return total_params / trainable_params

    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA weights into a standard nn.Linear for overhead-free inference.

        Returns:
            Standard nn.Linear with merged weights.
        """
        effective_weight = self.get_effective_weight()
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=effective_weight.device,
            dtype=effective_weight.dtype,
        )
        merged.weight.data.copy_(effective_weight)
        if self.bias is not None:
            merged.bias.data.copy_(self.bias)
        return merged

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"alpha={self.alpha}, "
            f"scaling={self.scaling:.3f}"
        )


# ---------------------------------------------------------------------------
# Factory + model-injection helpers
# ---------------------------------------------------------------------------

def create_lora_layer(
    base_layer: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> "LoRALinear":
    """
    Create a LoRALinear layer from an existing nn.Linear.

    Args:
        base_layer: Original nn.Linear layer
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: Dropout probability on the LoRA path

    Returns:
        LoRALinear layer with pretrained weights loaded
    """
    if not isinstance(base_layer, nn.Linear):
        raise ValueError(f"create_lora_layer only supports nn.Linear, got {type(base_layer)}")

    lora = LoRALinear(
        in_features=base_layer.in_features,
        out_features=base_layer.out_features,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        bias=base_layer.bias is not None,
        device=base_layer.weight.device,
        dtype=base_layer.weight.dtype,
    )
    lora.load_base_weight(
        base_layer.weight.data,
        base_layer.bias.data if base_layer.bias is not None else None,
    )
    return lora


def apply_lora_to_model(
    model: nn.Module,
    target_modules: Optional[list] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Apply standard LoRA to matching nn.Linear layers in any model.

    Walks the module tree and replaces each nn.Linear whose attribute name
    (last component of the dotted path) appears in *target_modules*.

    Args:
        model: Any nn.Module (typically a HuggingFace model)
        target_modules: Attribute names to replace, e.g. ["q_proj", "v_proj"]
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: Dropout on the LoRA path

    Returns:
        The same model object with LoRA layers injected in-place.
    """
    import logging

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Collect replacements first to avoid mutating during iteration
    replacements = []
    for name, module in model.named_modules():
        leaf_name = name.split(".")[-1]
        if leaf_name in target_modules and isinstance(module, nn.Linear):
            parent_path = ".".join(name.split(".")[:-1])
            replacements.append((parent_path, leaf_name, module))

    for parent_path, leaf_name, module in replacements:
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, leaf_name, create_lora_layer(module, rank, alpha, dropout))

    logging.info(
        f"Applied LoRA to {len(replacements)} linear layers "
        f"(target_modules={target_modules}, rank={rank}, alpha={alpha})"
    )
    return model
