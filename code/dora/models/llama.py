"""
DoRA implementation for LLaMA/LLaMA2 models.
Provides utilities to convert LLaMA models to use DoRA layers.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

try:
    from transformers import LlamaForCausalLM, LlamaModel
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. LLaMA integration disabled.")

from ..layers.base import DoRAModule
from ..layers.dora_linear import create_dora_layer


@dataclass
class LlamaDoRAConfig:
    """Configuration for LLaMA DoRA adaptation."""

    # DoRA parameters
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    eps: float = 1e-8

    # Target modules for adaptation
    target_modules: List[str] = None

    # Bias handling
    bias: str = "none"  # "none", "all", "lora_only"

    # Model-specific settings
    fan_in_fan_out: bool = False  # Set to True for some models
    init_lora_weights: bool = True

    # Advanced settings
    use_rslora: bool = False  # Rank-stabilized LoRA
    use_dora_magnitude_init: bool = True

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",  # Attention
                "gate_proj",
                "up_proj",
                "down_proj",  # MLP
            ]


class DoRALlamaAttention(nn.Module):
    """
    DoRA-enabled LLaMA Attention module.
    Replaces linear layers with DoRA equivalents.
    """

    def __init__(self, original_attention: "LlamaAttention", config: LlamaDoRAConfig):
        super().__init__()
        self.config = original_attention.config
        self.hidden_size = original_attention.hidden_size
        self.num_heads = original_attention.num_heads
        self.head_dim = original_attention.head_dim
        self.num_key_value_heads = getattr(
            original_attention, "num_key_value_heads", self.num_heads
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = original_attention.max_position_embeddings

        # Copy RoPE embedding if it exists
        if hasattr(original_attention, "rotary_emb"):
            self.rotary_emb = original_attention.rotary_emb

        # Replace linear layers with DoRA layers
        self._replace_linear_layers(original_attention, config)

    def _replace_linear_layers(self, original: "LlamaAttention", config: LlamaDoRAConfig):
        """Replace attention linear layers with DoRA equivalents."""
        # Create DoRA layers for each projection
        if "q_proj" in config.target_modules:
            self.q_proj = create_dora_layer(
                original.q_proj, config.rank, config.alpha, config.dropout, config.eps
            )
        else:
            self.q_proj = original.q_proj

        if "k_proj" in config.target_modules:
            self.k_proj = create_dora_layer(
                original.k_proj, config.rank, config.alpha, config.dropout, config.eps
            )
        else:
            self.k_proj = original.k_proj

        if "v_proj" in config.target_modules:
            self.v_proj = create_dora_layer(
                original.v_proj, config.rank, config.alpha, config.dropout, config.eps
            )
        else:
            self.v_proj = original.v_proj

        if "o_proj" in config.target_modules:
            self.o_proj = create_dora_layer(
                original.o_proj, config.rank, config.alpha, config.dropout, config.eps
            )
        else:
            self.o_proj = original.o_proj

    def forward(self, *args, **kwargs):
        """Forward pass - delegates to the original attention mechanism."""
        # This would need to implement the full LLaMA attention forward pass
        # For brevity, we'll assume the projections work the same way
        # In practice, you'd copy the exact forward logic from transformers
        return self._llama_attention_forward(*args, **kwargs)

    def _llama_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        """
        Simplified LLaMA attention forward pass.
        In practice, this should match the exact implementation from transformers.
        """
        bsz, q_len, _ = hidden_states.size()

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # Apply RoPE if available
        if hasattr(self, "rotary_emb"):
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = self._apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # Repeat K,V heads if needed for grouped-query attention
        if self.num_key_value_groups > 1:
            key_states = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / torch.sqrt(
            torch.tensor(self.head_dim)
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (past_key_value,)

        return outputs

    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value heads for grouped-query attention."""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        """Apply rotary positional embedding."""
        # Simplified implementation - would need full RoPE logic
        return q, k


class DoRALlamaMLP(nn.Module):
    """
    DoRA-enabled LLaMA MLP module.
    """

    def __init__(self, original_mlp: "LlamaMLP", config: LlamaDoRAConfig):
        super().__init__()
        self.config = original_mlp.config
        self.hidden_size = original_mlp.hidden_size
        self.intermediate_size = original_mlp.intermediate_size

        # Replace MLP linear layers with DoRA equivalents
        if "gate_proj" in config.target_modules:
            self.gate_proj = create_dora_layer(
                original_mlp.gate_proj, config.rank, config.alpha, config.dropout, config.eps
            )
        else:
            self.gate_proj = original_mlp.gate_proj

        if "up_proj" in config.target_modules:
            self.up_proj = create_dora_layer(
                original_mlp.up_proj, config.rank, config.alpha, config.dropout, config.eps
            )
        else:
            self.up_proj = original_mlp.up_proj

        if "down_proj" in config.target_modules:
            self.down_proj = create_dora_layer(
                original_mlp.down_proj, config.rank, config.alpha, config.dropout, config.eps
            )
        else:
            self.down_proj = original_mlp.down_proj

        # Copy activation function
        self.act_fn = original_mlp.act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MLP forward pass with SiLU gating."""
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class LlamaDoRAModel:
    """
    Utility class for converting LLaMA models to use DoRA.
    """

    @staticmethod
    def from_pretrained(
        model_name_or_path: str, dora_config: Optional[LlamaDoRAConfig] = None, **kwargs
    ) -> Union["LlamaModel", "LlamaForCausalLM"]:
        """
        Load a pre-trained LLaMA model and convert it to use DoRA.

        Args:
            model_name_or_path: Path or model identifier
            dora_config: DoRA configuration
            **kwargs: Additional arguments for model loading

        Returns:
            Model with DoRA layers
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for LLaMA integration")

        if dora_config is None:
            dora_config = LlamaDoRAConfig()

        # Load the original model
        if "ForCausalLM" in kwargs.get("model_class", "LlamaForCausalLM"):
            model = LlamaForCausalLM.from_pretrained(model_name_or_path, **kwargs)
            base_model = model.model
        else:
            model = LlamaModel.from_pretrained(model_name_or_path, **kwargs)
            base_model = model

        # Convert to DoRA
        LlamaDoRAModel._convert_to_dora(base_model, dora_config)

        return model

    @staticmethod
    def _convert_to_dora(model: "LlamaModel", config: LlamaDoRAConfig):
        """
        Convert LLaMA model layers to use DoRA.

        Args:
            model: LLaMA model to convert
            config: DoRA configuration
        """
        # Convert each decoder layer
        for _layer_idx, decoder_layer in enumerate(model.layers):
            # Convert attention
            if any(
                module in config.target_modules
                for module in ["q_proj", "k_proj", "v_proj", "o_proj"]
            ):
                original_attention = decoder_layer.self_attn
                decoder_layer.self_attn = DoRALlamaAttention(original_attention, config)

            # Convert MLP
            if any(
                module in config.target_modules for module in ["gate_proj", "up_proj", "down_proj"]
            ):
                original_mlp = decoder_layer.mlp
                decoder_layer.mlp = DoRALlamaMLP(original_mlp, config)

        logging.info(f"Converted LLaMA model to DoRA with config: {config}")

    @staticmethod
    def get_target_modules(model_type: str = "llama") -> List[str]:
        """Get default target modules for different LLaMA variants."""
        if model_type.lower() in ["llama", "llama2"]:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def print_trainable_parameters(model: nn.Module) -> Dict[str, int]:
        """Print and return trainable parameter statistics."""
        param_stats = DoRAModule.count_parameters(model)

        print("=" * 50)
        print("TRAINABLE PARAMETERS SUMMARY")
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

    @staticmethod
    def save_dora_adapter(model: nn.Module, save_path: str):
        """Save only the DoRA adapter weights."""
        from ..layers.base import DoRAStateManager

        DoRAStateManager.save_dora_state(model, save_path, include_base_weights=False)
        logging.info(f"DoRA adapter saved to {save_path}")

    @staticmethod
    def load_dora_adapter(model: nn.Module, adapter_path: str):
        """Load DoRA adapter weights into model."""
        from ..layers.base import DoRAStateManager

        DoRAStateManager.load_dora_state(model, adapter_path, strict=True)
        logging.info(f"DoRA adapter loaded from {adapter_path}")


# Utility functions for model conversion
def convert_llama_to_dora(
    model: Union["LlamaModel", "LlamaForCausalLM"],
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
    dropout: float = 0.0,
) -> Union["LlamaModel", "LlamaForCausalLM"]:
    """
    Simple utility function to convert LLaMA model to DoRA.

    Args:
        model: LLaMA model to convert
        rank: LoRA rank
        alpha: LoRA alpha
        target_modules: Modules to adapt
        dropout: Dropout probability

    Returns:
        Converted model
    """
    config = LlamaDoRAConfig(rank=rank, alpha=alpha, target_modules=target_modules, dropout=dropout)

    base_model = model.model if hasattr(model, "model") else model
    LlamaDoRAModel._convert_to_dora(base_model, config)

    return model


def apply_dora_to_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    eps: float = 1e-8,
) -> nn.Module:
    """
    Apply DoRA to matching nn.Linear layers in any HuggingFace model.

    Works with AutoModelForSequenceClassification, AutoModelForCausalLM, etc.
    Walks the module tree and replaces each nn.Linear whose attribute name
    (last component of the dotted path) appears in target_modules.

    Args:
        model: Any nn.Module (typically a HuggingFace model)
        target_modules: Attribute names to replace, e.g. ["q_proj", "v_proj"]
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: Dropout on the LoRA path
        eps: Epsilon for magnitude normalisation

    Returns:
        The same model object with DoRA layers injected in-place.
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    # Collect replacements first to avoid mutating the module tree mid-iteration
    replacements = []
    for name, module in model.named_modules():
        leaf_name = name.split(".")[-1]
        if leaf_name in target_modules and isinstance(module, nn.Linear):
            parent_path = ".".join(name.split(".")[:-1])
            replacements.append((parent_path, leaf_name, module))

    for parent_path, leaf_name, module in replacements:
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, leaf_name, create_dora_layer(module, rank, alpha, dropout, eps))

    logging.info(
        f"Applied DoRA to {len(replacements)} linear layers "
        f"(target_modules={target_modules}, rank={rank}, alpha={alpha})"
    )
    return model


def create_dora_config_for_llama(
    model_size: str = "7B", task_type: str = "causal_lm", rank: int = 8, alpha: float = 16.0
) -> LlamaDoRAConfig:
    """
    Create optimal DoRA configuration for different LLaMA model sizes.

    Args:
        model_size: Model size (7B, 13B, 30B, 65B)
        task_type: Type of task
        rank: Base rank (may be adjusted based on model size)
        alpha: Base alpha value

    Returns:
        Optimized DoRA configuration
    """
    # Adjust rank based on model size for better performance
    size_to_rank = {
        "7B": rank,
        "13B": max(rank, 16),
        "30B": max(rank, 32),
        "65B": max(rank, 64),
        "70B": max(rank, 64),
    }

    adjusted_rank = size_to_rank.get(model_size, rank)

    # Adjust alpha proportionally
    adjusted_alpha = alpha * (adjusted_rank / rank)

    return LlamaDoRAConfig(
        rank=adjusted_rank,
        alpha=adjusted_alpha,
        target_modules=LlamaDoRAModel.get_target_modules("llama"),
        dropout=0.05 if "chat" in task_type.lower() else 0.0,
    )
