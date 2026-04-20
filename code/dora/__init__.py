"""
DoRA: Weight-Decomposed Low-Rank Adaptation

A PyTorch implementation of DoRA from scratch.
"""

__version__ = "0.1.0"
__author__ = "DoRA Implementation Team"

from .layers.base import DoRAConfig, DoRAModule, DoRAStateManager
from .layers.dora_linear import DoRAConv2d, DoRALinear, create_dora_layer
from .utils.math_utils import DoRAMath

__all__ = [
    "DoRALinear",
    "DoRAConv2d",
    "create_dora_layer",
    "DoRAConfig",
    "DoRAModule",
    "DoRAStateManager",
    "DoRAMath",
]
