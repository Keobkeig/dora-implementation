"""Compatibility utilities for model adaptation and parameter counting."""

from ..layers.base import DoRAModule
from ..layers.dora_linear import create_dora_layer


def count_parameters(model):
    """Return total parameter count for backward compatibility."""
    return DoRAModule.count_parameters(model)["total"]


__all__ = ["create_dora_layer", "count_parameters"]
