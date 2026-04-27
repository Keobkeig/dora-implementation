"""
DoRA integration for Vision-Language-Action (VLA) models — specifically OpenVLA.

Architecture overview (OpenVLA-7B):
  ┌─────────────────────────────────────────────────────────────────┐
  │  Visual Encoder   SigLIP-SO400M (~400M)                         │
  │    ViT-like blocks: attention.{query, key, value} + mlp.dense   │
  ├─────────────────────────────────────────────────────────────────┤
  │  Projector        Linear (visual tokens → LLM hidden dim)       │
  ├─────────────────────────────────────────────────────────────────┤
  │  LLM Backbone     LLaMA-2-7B (~7B)                              │
  │    attention: {q_proj, k_proj, v_proj, o_proj}                  │
  │    MLP:       {gate_proj, up_proj, down_proj}                   │
  ├─────────────────────────────────────────────────────────────────┤
  │  Action Head      Linear (LLM hidden → 7-DoF discrete tokens)   │
  └─────────────────────────────────────────────────────────────────┘

DoRA strategy:
  - visual_only (cheapest, ~0.4% trainable):
      Adapt SigLIP ViT attention {query, key, value} only.
      Best for grasping tasks where the visual representation needs updating
      but the language reasoning stays frozen.
  - full (recommended for general tasks, ~0.5% trainable):
      Adapt both visual encoder AND LLM backbone attention projections.
      Matches the OpenVLA-OFT paper which shows DoRA/LoRA on both components.

Memory estimates at rank=8, bfloat16:
  ┌──────────────────────────────┬──────────────┬────────────────────┐
  │ Strategy                     │ VRAM (train) │ Trainable params   │
  ├──────────────────────────────┼──────────────┼────────────────────┤
  │ visual_only (no quantisation)│ ~18 GB       │ ~1.8M (0.02%)      │
  │ full (no quantisation)       │ ~22 GB       │ ~40M  (0.50%)      │
  │ full (4-bit QLoRA/QDoRA)     │ ~10-12 GB ✓  │ ~40M  (0.50%)      │
  └──────────────────────────────┴──────────────┴────────────────────┘

  → With a 12 GB GPU, use 4-bit quantisation (bitsandbytes) for full strategy.
  → visual_only without quantisation still needs ~18 GB and will OOM at 12 GB.

Reference:
  Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model" (2024)
  OpenVLA-OFT: https://github.com/moojink/openvla-oft
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target module name sets per VLA component
# ---------------------------------------------------------------------------

# SigLIP visual encoder (ViT-style attention)
SIGLIP_TARGET_MODULES = ["query", "key", "value"]

# LLaMA-2 backbone (same as train_glue.py for 7B)
LLAMA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def apply_dora_to_vla(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    visual_only: bool = False,
    visual_target_modules: Optional[List[str]] = None,
    llm_target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Apply DoRA adapters to an OpenVLA model.

    The function is model-agnostic: it walks the full module tree and replaces
    every nn.Linear whose leaf name is in the target lists.

    Args:
        model:                 The loaded OpenVLA model (nn.Module).
        rank:                  LoRA rank for both components.
        alpha:                 LoRA alpha scaling.
        dropout:               Dropout on the adapter path.
        visual_only:           If True, adapt only the visual encoder
                               (cheaper, good for grasping fine-tuning).
        visual_target_modules: Override for visual encoder target names.
        llm_target_modules:    Override for LLM backbone target names.

    Returns:
        The same model with DoRA layers injected in-place.

    Example::

        from transformers import AutoModelForVision2Seq
        from dora.models.vla import apply_dora_to_vla

        model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,          # bitsandbytes 4-bit quant → fits 12 GB
            device_map="auto",
        )
        apply_dora_to_vla(model, rank=8, visual_only=False)
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from dora.models.llama import apply_dora_to_model as _apply_dora

    vis_mods = visual_target_modules or SIGLIP_TARGET_MODULES
    llm_mods = llm_target_modules  or LLAMA_TARGET_MODULES

    if visual_only:
        target = vis_mods
        logger.info(f"DoRA (visual-only): target={target}")
    else:
        target = list(set(vis_mods + llm_mods))
        logger.info(f"DoRA (full): visual={vis_mods} + llm={llm_mods}")

    _apply_dora(model, target_modules=target, rank=rank, alpha=alpha, dropout=dropout)
    return model


def freeze_vla_base_weights(model: nn.Module):
    """
    Freeze everything except DoRA adapter params and the action head.

    Assumes the action head is accessible at model.action_head (OpenVLA naming).
    Adjust the head_check predicate for other VLA architectures.
    """
    for name, param in model.named_parameters():
        leaf = name.split(".")[-1]
        is_adapter = leaf in ("lora_A", "lora_B", "magnitude")
        is_head = "action_head" in name
        param.requires_grad = is_adapter or is_head

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"VLA trainable: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")


# ---------------------------------------------------------------------------
# Usage notes (printed when module is run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(__doc__)
    print()
    print("To fine-tune OpenVLA with DoRA on a grasping task:")
    print()
    print("  1. Install bitsandbytes for 4-bit quantisation:")
    print("     uv add bitsandbytes")
    print()
    print("  2. Load model with quantisation:")
    print("     model = AutoModelForVision2Seq.from_pretrained(")
    print("         'openvla/openvla-7b',")
    print("         torch_dtype=torch.bfloat16,")
    print("         load_in_4bit=True,")
    print("         device_map='auto',")
    print("     )")
    print()
    print("  3. Apply DoRA:")
    print("     from dora.models.vla import apply_dora_to_vla, freeze_vla_base_weights")
    print("     apply_dora_to_vla(model, rank=8, visual_only=False)")
    print("     freeze_vla_base_weights(model)")
    print()
    print("  4. Fine-tune on a robot action dataset (e.g. BridgeData V2, DROID,")
    print("     or a custom Cornell Grasp → action-label converted dataset).")
    print()
    print("  Cornell Grasp Dataset note:")
    print("  Cornell provides grasp *rectangles*, not robot joint actions.")
    print("  To use it with OpenVLA you need to convert rectangle (cx,cy,θ,w,h)")
    print("  to end-effector delta poses using your robot's forward kinematics,")
    print("  or use a simulation (e.g. ManiSkill, RoboSuite) to generate")
    print("  paired (image, instruction, action) trajectories.")
    print()
    print("  For the ViT-Base grasp pose experiment (fully runnable at 12 GB),")
    print("  use scripts/train_grasp.py instead.")
