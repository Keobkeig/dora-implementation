"""
DoRA implementation for Vision Transformer models.
Provides utilities to convert ViT models to use DoRA layers.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import ViTForImageClassification, ViTModel
    from transformers.models.vit.modeling_vit import (
        ViTAttention,
        ViTIntermediate,
        ViTLayer,
        ViTOutput,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. ViT integration disabled.")

from ..layers.base import DoRAModule
from ..layers.dora_linear import create_dora_layer


@dataclass
class ViTDoRAConfig:
    """Configuration for Vision Transformer DoRA adaptation."""

    # DoRA parameters
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    eps: float = 1e-8

    # Target modules for adaptation
    target_modules: List[str] = None

    # Bias handling
    bias: str = "none"  # "none", "all", "lora_only"

    # ViT-specific settings
    adapt_embeddings: bool = False  # Whether to adapt patch embeddings
    adapt_classifier: bool = True  # Whether to adapt final classifier

    # Advanced settings
    use_dora_magnitude_init: bool = True
    init_lora_weights: bool = True

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "query",
                "key",
                "value",  # Attention projections
                "dense",  # Attention output projection
                "intermediate.dense",  # MLP hidden layer
                "output.dense",  # MLP output layer
            ]


class DoRAViTAttention(nn.Module):
    """
    DoRA-enabled Vision Transformer Attention module.
    """

    def __init__(self, original_attention: "ViTAttention", config: ViTDoRAConfig):
        super().__init__()
        self.num_attention_heads = original_attention.num_attention_heads
        self.attention_head_size = original_attention.attention_head_size
        self.all_head_size = original_attention.all_head_size
        self.config = original_attention.config

        # Replace Q, K, V projections with DoRA layers
        if "query" in config.target_modules:
            self.query = create_dora_layer(
                original_attention.query, config.rank, config.alpha, config.dropout, config.eps
            )
        else:
            self.query = original_attention.query

        if "key" in config.target_modules:
            self.key = create_dora_layer(
                original_attention.key, config.rank, config.alpha, config.dropout, config.eps
            )
        else:
            self.key = original_attention.key

        if "value" in config.target_modules:
            self.value = create_dora_layer(
                original_attention.value, config.rank, config.alpha, config.dropout, config.eps
            )
        else:
            self.value = original_attention.value

        # Copy dropout
        self.dropout = original_attention.dropout

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor for attention score computation."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """ViT attention forward pass with DoRA projections."""
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size**0.5)

        # Apply attention mask if provided
        if head_mask is not None:
            attention_scores = attention_scores + head_mask

        # Normalize attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class DoRAViTSelfOutput(nn.Module):
    """
    DoRA-enabled ViT self-attention output projection.
    """

    def __init__(self, original_output: "ViTOutput", config: ViTDoRAConfig):
        super().__init__()

        # Replace dense layer with DoRA if specified
        if "dense" in config.target_modules:
            self.dense = create_dora_layer(
                original_output.dense, config.rank, config.alpha, config.dropout, config.eps
            )
        else:
            self.dense = original_output.dense

        self.dropout = original_output.dropout

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class DoRAViTIntermediate(nn.Module):
    """
    DoRA-enabled ViT intermediate (MLP hidden) layer.
    """

    def __init__(self, original_intermediate: "ViTIntermediate", config: ViTDoRAConfig):
        super().__init__()

        if "intermediate.dense" in config.target_modules:
            self.dense = create_dora_layer(
                original_intermediate.dense, config.rank, config.alpha, config.dropout, config.eps
            )
        else:
            self.dense = original_intermediate.dense

        # Copy activation function
        if hasattr(original_intermediate, "intermediate_act_fn"):
            self.intermediate_act_fn = original_intermediate.intermediate_act_fn
        else:
            self.intermediate_act_fn = F.gelu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class DoRAViTOutput(nn.Module):
    """
    DoRA-enabled ViT output (MLP output) layer.
    """

    def __init__(self, original_output: "ViTOutput", config: ViTDoRAConfig):
        super().__init__()

        if "output.dense" in config.target_modules:
            self.dense = create_dora_layer(
                original_output.dense, config.rank, config.alpha, config.dropout, config.eps
            )
        else:
            self.dense = original_output.dense

        self.dropout = original_output.dropout

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class DoRAViTLayer(nn.Module):
    """
    DoRA-enabled ViT Layer (Transformer block).
    """

    def __init__(self, original_layer: "ViTLayer", config: ViTDoRAConfig):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = original_layer.chunk_size_feed_forward
        self.seq_len_dim = 1

        # Copy layer norm (these are not adapted)
        self.layernorm_before = original_layer.layernorm_before
        self.layernorm_after = original_layer.layernorm_after

        # Replace attention with DoRA version
        self.attention = DoRAViTAttention(original_layer.attention, config)
        self.attention_output = DoRAViTSelfOutput(original_layer.attention.output, config)

        # Replace MLP layers with DoRA versions
        self.intermediate = DoRAViTIntermediate(original_layer.intermediate, config)
        self.output = DoRAViTOutput(original_layer.output, config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """ViT layer forward pass with residual connections."""
        # Self attention
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # Add attention weights if output_attentions=True

        # Apply attention output projection and residual connection
        attention_output = self.attention_output(attention_output, hidden_states)

        # Feed forward
        layer_output = self.layernorm_after(attention_output)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, attention_output)

        outputs = (layer_output,) + outputs
        return outputs


class ViTDoRAModel:
    """
    Utility class for converting Vision Transformer models to use DoRA.
    """

    @staticmethod
    def from_pretrained(
        model_name_or_path: str, dora_config: Optional[ViTDoRAConfig] = None, **kwargs
    ) -> Union["ViTModel", "ViTForImageClassification"]:
        """
        Load a pre-trained ViT model and convert it to use DoRA.

        Args:
            model_name_or_path: Path or model identifier
            dora_config: DoRA configuration
            **kwargs: Additional arguments for model loading

        Returns:
            Model with DoRA layers
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for ViT integration")

        if dora_config is None:
            dora_config = ViTDoRAConfig()

        # Load the original model
        if "ForImageClassification" in kwargs.get("model_class", "ViTForImageClassification"):
            model = ViTForImageClassification.from_pretrained(model_name_or_path, **kwargs)
            base_model = model.vit
        else:
            model = ViTModel.from_pretrained(model_name_or_path, **kwargs)
            base_model = model

        # Convert to DoRA
        ViTDoRAModel._convert_to_dora(base_model, model, dora_config)

        return model

    @staticmethod
    def _convert_to_dora(base_model: "ViTModel", full_model: nn.Module, config: ViTDoRAConfig):
        """
        Convert ViT model layers to use DoRA.

        Args:
            base_model: ViT encoder model
            full_model: Full model (may include classifier)
            config: DoRA configuration
        """
        # Convert encoder layers
        for layer_idx, transformer_layer in enumerate(base_model.encoder.layer):
            base_model.encoder.layer[layer_idx] = DoRAViTLayer(transformer_layer, config)

        # Optionally adapt classifier
        if config.adapt_classifier and hasattr(full_model, "classifier"):
            if isinstance(full_model.classifier, nn.Linear):
                full_model.classifier = create_dora_layer(
                    full_model.classifier, config.rank, config.alpha, config.dropout, config.eps
                )

        # Optionally adapt patch embeddings
        if config.adapt_embeddings and hasattr(base_model.embeddings, "patch_embeddings"):
            patch_emb = base_model.embeddings.patch_embeddings
            if hasattr(patch_emb, "projection") and isinstance(patch_emb.projection, nn.Conv2d):
                patch_emb.projection = create_dora_layer(
                    patch_emb.projection, config.rank, config.alpha, config.dropout, config.eps
                )

        logging.info(f"Converted ViT model to DoRA with config: {config}")

    @staticmethod
    def get_target_modules(model_type: str = "vit") -> List[str]:
        """Get default target modules for different ViT variants."""
        if model_type.lower() in ["vit", "vision_transformer"]:
            return ["query", "key", "value", "dense", "intermediate.dense", "output.dense"]
        elif model_type.lower() in ["deit", "distilled_vit"]:
            # DeiT might have slightly different naming
            return ["query", "key", "value", "dense", "intermediate.dense", "output.dense"]
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def print_trainable_parameters(model: nn.Module) -> Dict[str, int]:
        """Print and return trainable parameter statistics."""
        param_stats = DoRAModule.count_parameters(model)

        print("=" * 50)
        print("ViT DoRA TRAINABLE PARAMETERS SUMMARY")
        print("=" * 50)
        print(f"Total parameters: {param_stats['total']:,}")
        print(f"Trainable parameters: {param_stats['trainable']:,}")
        print(f"Frozen parameters: {param_stats['frozen']:,}")
        print(f"DoRA magnitude parameters: {param_stats['dora_magnitude']:,}")
        print(f"LoRA parameters: {param_stats['lora']:,}")
        print(f"Trainable %: {100 * param_stats['trainable'] / param_stats['total']:.2f}%")
        print(f"Compression ratio: {param_stats['compression_ratio']:.1f}x")
        print("=" * 50)

        return param_stats


# Utility functions for ViT conversion
def convert_vit_to_dora(
    model: Union["ViTModel", "ViTForImageClassification"],
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
    dropout: float = 0.1,
    adapt_classifier: bool = True,
) -> Union["ViTModel", "ViTForImageClassification"]:
    """
    Simple utility function to convert ViT model to DoRA.

    Args:
        model: ViT model to convert
        rank: LoRA rank
        alpha: LoRA alpha
        target_modules: Modules to adapt
        dropout: Dropout probability
        adapt_classifier: Whether to adapt final classifier

    Returns:
        Converted model
    """
    config = ViTDoRAConfig(
        rank=rank,
        alpha=alpha,
        target_modules=target_modules,
        dropout=dropout,
        adapt_classifier=adapt_classifier,
    )

    base_model = model.vit if hasattr(model, "vit") else model
    ViTDoRAModel._convert_to_dora(base_model, model, config)

    return model


def create_dora_config_for_vit(
    model_size: str = "base",
    task_type: str = "image_classification",
    rank: int = 8,
    alpha: float = 16.0,
) -> ViTDoRAConfig:
    """
    Create optimal DoRA configuration for different ViT model sizes.

    Args:
        model_size: Model size (tiny, small, base, large, huge)
        task_type: Type of task
        rank: Base rank
        alpha: Base alpha value

    Returns:
        Optimized DoRA configuration
    """
    # Adjust rank based on model size
    size_to_rank = {
        "tiny": max(rank // 2, 4),
        "small": rank,
        "base": rank,
        "large": rank * 2,
        "huge": rank * 4,
    }

    adjusted_rank = size_to_rank.get(model_size, rank)
    adjusted_alpha = alpha * (adjusted_rank / rank)

    # Different dropout for different tasks
    task_dropout = {
        "image_classification": 0.1,
        "object_detection": 0.05,
        "segmentation": 0.05,
        "fine_grained": 0.15,
    }

    dropout = task_dropout.get(task_type, 0.1)

    return ViTDoRAConfig(
        rank=adjusted_rank,
        alpha=adjusted_alpha,
        target_modules=ViTDoRAModel.get_target_modules("vit"),
        dropout=dropout,
        adapt_classifier=True,
        adapt_embeddings=False,  # Usually not needed
    )
