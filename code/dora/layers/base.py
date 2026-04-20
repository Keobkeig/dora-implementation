"""
Base classes and utilities for DoRA layers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class DoRALayer(ABC, nn.Module):
    """
    Abstract base class for DoRA layers.
    Defines the common interface that all DoRA layers should implement.
    """

    def __init__(self):
        super().__init__()
        self._dora_enabled = True
        self._magnitude_initialized = False

    @abstractmethod
    def load_base_weight(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Load frozen base weight from pre-trained model."""
        pass

    @abstractmethod
    def get_effective_weight(self) -> torch.Tensor:
        """Compute the effective weight using DoRA decomposition."""
        pass

    @abstractmethod
    def enable_dora(self, enabled: bool = True):
        """Enable or disable DoRA mode."""
        pass

    @abstractmethod
    def get_parameter_count(self) -> Tuple[int, int, int]:
        """Get (total_params, trainable_params, frozen_params)."""
        pass

    def disable_dora(self):
        """Disable DoRA mode (fall back to LoRA)."""
        self.enable_dora(False)

    def is_dora_enabled(self) -> bool:
        """Check if DoRA mode is enabled."""
        return self._dora_enabled

    def is_magnitude_initialized(self) -> bool:
        """Check if magnitude parameter is initialized."""
        return self._magnitude_initialized


class DoRAConfig:
    """
    Configuration class for DoRA hyperparameters.
    """

    def __init__(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        eps: float = 1e-8,
        init_magnitude_from_weight: bool = True,
        target_modules: Optional[list] = None,
        bias: str = "none",  # "none", "all", "lora_only"
        task_type: str = "CAUSAL_LM",  # Task type for different model architectures
    ):
        """
        Initialize DoRA configuration.

        Args:
            rank: Rank of LoRA adaptation matrices
            alpha: LoRA scaling parameter (typically alpha/rank is used as scaling)
            dropout: Dropout probability for LoRA layers
            eps: Small epsilon for numerical stability in normalization
            init_magnitude_from_weight: Whether to initialize magnitude from base weights
            target_modules: List of module names to apply DoRA to
            bias: Bias handling strategy
            task_type: Type of task (affects which modules are targeted)
        """
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.eps = eps
        self.init_magnitude_from_weight = init_magnitude_from_weight
        self.target_modules = target_modules or self._get_default_target_modules(task_type)
        self.bias = bias
        self.task_type = task_type

        # Derived parameters
        self.scaling = alpha / rank

    def _get_default_target_modules(self, task_type: str) -> list:
        """Get default target modules based on task type."""
        if task_type == "CAUSAL_LM":
            # For decoder-only models like LLaMA, GPT
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif task_type == "SEQ_CLS" or task_type == "TOKEN_CLS":
            # For encoder models like BERT
            return ["query", "value", "key", "dense"]
        elif task_type == "IMAGE_CLASSIFICATION":
            # For Vision Transformers
            return ["qkv", "proj", "fc1", "fc2"]
        else:
            # Generic attention modules
            return ["q_proj", "v_proj", "k_proj", "o_proj"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "eps": self.eps,
            "init_magnitude_from_weight": self.init_magnitude_from_weight,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
            "scaling": self.scaling,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DoRAConfig":
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k != "scaling"})

    def __repr__(self) -> str:
        return f"DoRAConfig({self.to_dict()})"


class DoRAModule:
    """
    Utility class for identifying and working with DoRA modules.
    """

    @staticmethod
    def is_dora_layer(layer: nn.Module) -> bool:
        """Check if a layer is a DoRA layer."""
        return isinstance(layer, DoRALayer)

    @staticmethod
    def get_dora_layers(model: nn.Module) -> Dict[str, DoRALayer]:
        """Get all DoRA layers in a model."""
        dora_layers = {}
        for name, module in model.named_modules():
            if DoRAModule.is_dora_layer(module):
                dora_layers[name] = module
        return dora_layers

    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """
        Count parameters in a model with DoRA layers.

        Returns:
            Dictionary with parameter counts
        """
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        dora_magnitude_params = 0
        lora_params = 0

        for module in model.modules():
            if DoRAModule.is_dora_layer(module):
                t_params, tr_params, f_params = module.get_parameter_count()
                total_params += t_params
                trainable_params += tr_params
                frozen_params += f_params

                # Count DoRA-specific parameters
                dora_magnitude_params += module.magnitude.numel()
                lora_params += module.lora_A.numel() + module.lora_B.numel()
            else:
                # Regular parameters
                for param in module.parameters(recurse=False):
                    total_params += param.numel()
                    if param.requires_grad:
                        trainable_params += param.numel()
                    else:
                        frozen_params += param.numel()

        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": frozen_params,
            "dora_magnitude": dora_magnitude_params,
            "lora": lora_params,
            "compression_ratio": total_params / trainable_params if trainable_params > 0 else 0,
        }

    @staticmethod
    def enable_dora_layers(model: nn.Module, enabled: bool = True):
        """Enable or disable DoRA mode for all DoRA layers in model."""
        for module in model.modules():
            if DoRAModule.is_dora_layer(module):
                module.enable_dora(enabled)

    @staticmethod
    def merge_dora_weights(model: nn.Module) -> nn.Module:
        """
        Create a new model with DoRA weights merged for inference.
        This eliminates the computational overhead during inference.
        """
        # This is a simplified version - in practice, this would need
        # to handle the specific model architecture
        merged_model = model.__class__(**model.config.__dict__)

        # Copy state dict and merge DoRA layers
        merged_state_dict = {}

        for name, module in model.named_modules():
            if DoRAModule.is_dora_layer(module):
                # Get the effective weight and create standard layer
                effective_weight = module.get_effective_weight()
                merged_state_dict[name + ".weight"] = effective_weight
                if module.bias is not None:
                    merged_state_dict[name + ".bias"] = module.bias
            else:
                # Copy regular parameters
                for param_name, param in module.named_parameters(recurse=False):
                    full_name = f"{name}.{param_name}" if name else param_name
                    merged_state_dict[full_name] = param

        merged_model.load_state_dict(merged_state_dict, strict=False)
        return merged_model


class DoRAStateManager:
    """
    Utility class for saving and loading DoRA model states.
    """

    @staticmethod
    def save_dora_state(model: nn.Module, path: str, include_base_weights: bool = False):
        """
        Save DoRA-specific state (trainable parameters only).

        Args:
            model: Model with DoRA layers
            path: Path to save state
            include_base_weights: Whether to include frozen base weights
        """
        dora_state = {}
        config_dict = {}

        for name, module in model.named_modules():
            if DoRAModule.is_dora_layer(module):
                module_state = {
                    "lora_A": module.lora_A.data,
                    "lora_B": module.lora_B.data,
                    "magnitude": module.magnitude.data,
                }

                if include_base_weights:
                    module_state["base_weight"] = module.base_weight
                    if hasattr(module, "bias") and module.bias is not None:
                        module_state["bias"] = module.bias

                dora_state[name] = module_state

                # Save configuration
                config_dict[name] = {
                    "rank": module.rank,
                    "alpha": module.alpha,
                    "in_features": getattr(module, "in_features", None),
                    "out_features": getattr(module, "out_features", None),
                    "in_channels": getattr(module, "in_channels", None),
                    "out_channels": getattr(module, "out_channels", None),
                }

        torch.save(
            {
                "dora_state": dora_state,
                "config": config_dict,
                "model_type": model.__class__.__name__,
            },
            path,
        )

    @staticmethod
    def load_dora_state(model: nn.Module, path: str, strict: bool = True):
        """
        Load DoRA-specific state into model.

        Args:
            model: Model with DoRA layers to load into
            path: Path to saved state
            strict: Whether to strictly match layer names
        """
        checkpoint = torch.load(path, map_location="cpu")
        dora_state = checkpoint["dora_state"]

        for name, module in model.named_modules():
            if DoRAModule.is_dora_layer(module) and name in dora_state:
                state = dora_state[name]

                module.lora_A.data.copy_(state["lora_A"])
                module.lora_B.data.copy_(state["lora_B"])
                module.magnitude.data.copy_(state["magnitude"])

                if "base_weight" in state:
                    module.base_weight.copy_(state["base_weight"])
                if "bias" in state and module.bias is not None:
                    module.bias.copy_(state["bias"])

                module._magnitude_initialized = True
            elif strict and DoRAModule.is_dora_layer(module):
                raise KeyError(f"DoRA layer {name} not found in checkpoint")
