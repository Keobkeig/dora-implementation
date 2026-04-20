"""
DoRA utils package.
"""

from ..layers.base import DoRAModule
from ..layers.dora_linear import create_dora_layer
from .math_utils import DoRAMath


def count_parameters(model):
    """Compatibility helper used by legacy scripts."""
    return DoRAModule.count_parameters(model)["total"]


__all__ = ["DoRAMath", "create_dora_layer", "count_parameters"]
