#!/usr/bin/env python3
"""
OpenVLA-7B + DoRA architecture demo.

Loads openvla/openvla-7b in 4-bit (NF4) quantisation, identifies DoRA
target layers across the visual encoder and LLM backbone, computes
theoretical trainable-parameter counts for visual_only vs full strategies,
then runs a single forward pass to confirm the architecture works end-to-end.

No training is performed. Runtime: ~5-10 min (includes 14 GB model download).

Usage:
    uv run scripts/openvla_demo.py
"""

import logging
import os
import sys

# Compatibility shim: newer transformers moved these types out of
# tokenization_utils; OpenVLA's custom processor still imports from there.
import transformers.tokenization_utils as _tu
if not hasattr(_tu, "PaddingStrategy"):
    from transformers.tokenization_utils_base import (
        PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy,
    )
    _tu.PaddingStrategy       = PaddingStrategy
    _tu.PreTokenizedInput     = PreTokenizedInput
    _tu.TextInput             = TextInput
    _tu.TruncationStrategy    = TruncationStrategy

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "openvla/openvla-7b"

# Attention projection names per sub-architecture
VISUAL_TARGETS = {"q_proj", "k_proj", "v_proj"}          # SigLIP visual encoder
LLM_TARGETS    = {"q_proj", "k_proj", "v_proj", "o_proj", # LLaMA-2 backbone
                  "gate_proj", "up_proj", "down_proj"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_adapter_params(in_f: int, out_f: int, rank: int = 8) -> int:
    """Parameters added by a single DoRA layer: lora_A + lora_B + magnitude."""
    return rank * in_f + rank * out_f + out_f


def audit_dora_targets(model, rank: int = 8):
    """
    Walk the module tree and categorise every nn.Linear (or 4-bit equivalent)
    whose leaf name is in our target sets.  Returns a summary dict.
    """
    try:
        import bitsandbytes as bnb
        linear_types = (torch.nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except ImportError:
        linear_types = (torch.nn.Linear,)

    visual_layers, llm_layers, other_layers = [], [], []

    for name, module in model.named_modules():
        if not isinstance(module, linear_types):
            continue
        leaf = name.split(".")[-1]
        # Determine which component this layer belongs to
        in_visual = ("vision" in name or "visual" in name or "image" in name)
        in_llm    = ("language" in name or "lm" in name or "llm" in name
                     or "model.layers" in name)

        if leaf in VISUAL_TARGETS and in_visual:
            in_f  = module.in_features  if hasattr(module, "in_features")  else "?"
            out_f = module.out_features if hasattr(module, "out_features") else "?"
            visual_layers.append((name, in_f, out_f))
        elif leaf in LLM_TARGETS and in_llm:
            in_f  = module.in_features  if hasattr(module, "in_features")  else "?"
            out_f = module.out_features if hasattr(module, "out_features") else "?"
            llm_layers.append((name, in_f, out_f))

    total_params = sum(p.numel() for p in model.parameters())

    # Theoretical DoRA adapter params
    def _adapter_total(layers):
        return sum(_count_adapter_params(in_f, out_f, rank)
                   for _, in_f, out_f in layers
                   if isinstance(in_f, int) and isinstance(out_f, int))

    vis_adapter = _adapter_total(visual_layers)
    llm_adapter = _adapter_total(llm_layers)
    full_adapter = vis_adapter + llm_adapter

    return {
        "total_params":    total_params,
        "visual_layers":   visual_layers,
        "llm_layers":      llm_layers,
        "vis_adapter":     vis_adapter,
        "llm_adapter":     llm_adapter,
        "full_adapter":    full_adapter,
    }


def print_summary(stats: dict, rank: int = 8):
    total = stats["total_params"]
    vis   = stats["vis_adapter"]
    full  = stats["full_adapter"]

    print()
    print("=" * 62)
    print("  OpenVLA-7B  ×  DoRA  —  Architecture Report")
    print("=" * 62)
    print(f"  Total parameters       : {total/1e9:.2f} B")
    print(f"  DoRA rank              : {rank}")
    print()
    print(f"  Strategy: visual_only  ({len(stats['visual_layers'])} layers)")
    print(f"    Adapter params       : {vis:,}")
    print(f"    Trainable %%          : {100*vis/total:.4f}%%")
    print()
    print(f"  Strategy: full         ({len(stats['visual_layers'])+len(stats['llm_layers'])} layers)")
    print(f"    Adapter params       : {full:,}")
    print(f"    Trainable %%          : {100*full/total:.4f}%%")
    print()
    print("  Visual encoder target modules  :", sorted(VISUAL_TARGETS))
    print("  LLM backbone target modules    :", sorted(LLM_TARGETS))
    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _dora_params_from_config(config, rank: int = 8) -> dict:
    """
    Compute DoRA adapter parameter counts analytically from the model config,
    without loading any weights.  Works for OpenVLA's PrismaticConfig.
    """
    # LLaMA-2 backbone dimensions
    llm_cfg = getattr(config, "text_config", None)
    if llm_cfg is None:
        # Fallback: LLaMA-2 7B known values
        llm_hidden = 4096
        llm_layers = 32
    else:
        llm_hidden = getattr(llm_cfg, "hidden_size", 4096)
        llm_layers = getattr(llm_cfg, "num_hidden_layers", 32)

    # SigLIP visual encoder dimensions (from OpenVLA paper: SigLIP-SO400M 14/384)
    # ~27 transformer blocks, hidden_size=1152
    vis_hidden = 1152
    vis_layers = 27

    total_params = 7_000_000_000  # 7B approximate

    def _adapter(in_f, out_f):
        return rank * in_f + rank * out_f + out_f  # lora_A + lora_B + magnitude

    # Visual: q_proj, k_proj, v_proj  per layer
    vis_adapter = vis_layers * 3 * _adapter(vis_hidden, vis_hidden)

    # LLM: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj per layer
    intermediate = llm_hidden * 4  # LLaMA-2 MLP ratio
    llm_qkvo = 4 * _adapter(llm_hidden, llm_hidden)
    llm_mlp  = 2 * _adapter(llm_hidden, intermediate) + _adapter(intermediate, llm_hidden)
    llm_adapter = llm_layers * (llm_qkvo + llm_mlp)

    return {
        "total_params":  total_params,
        "visual_layers": vis_layers * 3,   # q/k/v per layer
        "llm_layers":    llm_layers * 7,   # q/k/v/o/gate/up/down per layer
        "vis_adapter":   vis_adapter,
        "llm_adapter":   llm_adapter,
        "full_adapter":  vis_adapter + llm_adapter,
    }


def main():
    # 1. Load processor
    logger.info(f"Loading processor from {MODEL_ID} ...")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    logger.info("Processor loaded.")

    # 2. Load model
    # bitsandbytes 4-bit quantisation segfaults on RTX 5000-series (Blackwell)
    # due to incomplete bnb support for that arch.  Load to CPU in bf16 instead —
    # fully stable, correct weights, adequate for architecture audit + forward pass.
    logger.info("Loading model to CPU in bf16 (weights already cached, ~30-60 s) ...")
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    OpenVLAForActionPrediction = get_class_from_dynamic_module(
        "modeling_prismatic.OpenVLAForActionPrediction",
        MODEL_ID,
    )

    model = OpenVLAForActionPrediction.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded on CPU.")

    # 3. Audit DoRA targets across the real loaded model
    logger.info("Auditing DoRA target layers ...")
    stats = audit_dora_targets(model, rank=8)
    print_summary(stats, rank=8)

    # 4. Forward pass
    logger.info("Running forward pass on CPU ...")
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    instruction = "In: What action should the robot take to grasp the object?\nOut:"
    inputs = processor(instruction, dummy_image, return_tensors="pt")
    # Cast float tensors to match model dtype (bf16)
    inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
              for k, v in inputs.items()}

    with torch.no_grad():
        if hasattr(model, "generate"):
            output_ids = model.generate(**inputs, max_new_tokens=5)
            decoded = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        else:
            _ = model(**inputs)
            decoded = "<forward pass OK>"
    logger.info(f"Forward pass OK. Output: {decoded!r}")

    # 5. Save report
    os.makedirs("../results", exist_ok=True)
    report_path = "../results/openvla_dora_report.txt"
    with open(report_path, "w") as f:
        total = stats["total_params"]
        f.write(f"model: {MODEL_ID}\n")
        f.write(f"total_params: {total}\n")
        f.write(f"visual_layers: {len(stats['visual_layers'])}\n")
        f.write(f"llm_layers: {len(stats['llm_layers'])}\n")
        f.write(f"dora_rank8_visual_only_params: {stats['vis_adapter']}\n")
        f.write(f"dora_rank8_visual_only_pct: {100*stats['vis_adapter']/total:.4f}\n")
        f.write(f"dora_rank8_full_params: {stats['full_adapter']}\n")
        f.write(f"dora_rank8_full_pct: {100*stats['full_adapter']/total:.4f}\n")
        f.write(f"forward_pass: OK\n")
        f.write(f"sample_output: {decoded!r}\n")
    logger.info(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()
