#!/usr/bin/env python3
"""
Export visual samples for Push-T (SmolVLM VLA).

Generates images with both GT and predicted actions overlaid.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dora.data.lerobot_dataset import PUSHT_INSTRUCTION, load_pusht
from dora.layers.base import DoRAStateManager
from dora.layers.lora_linear import LoRALinear, apply_lora_to_model
from dora.models.llama import apply_dora_to_model

from scripts.train_vla import SMOLVLM_TARGET_MODULES, SmolVLMActionModel


def _load_lora_weights(model: torch.nn.Module, adapter_path: str) -> None:
    payload = torch.load(adapter_path, map_location="cpu")
    lora_state = payload.get("lora_state", {})
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and name in lora_state:
            state = lora_state[name]
            module.lora_A.data.copy_(state["lora_A"])
            module.lora_B.data.copy_(state["lora_B"])


def _draw_action_overlay(img: Image.Image, gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cx, cy = w // 2, h // 2

    # Scale actions for arrow visualization (assumes normalized action space)
    scale = 20
    gt_dx, gt_dy = int(gt[0] * scale), int(gt[1] * scale)
    pr_dx, pr_dy = int(pred[0] * scale), int(pred[1] * scale)

    # GT arrow (green)
    draw.line([(cx, cy), (cx + gt_dx, cy + gt_dy)], fill=(0, 200, 0), width=2)
    draw.ellipse([(cx + gt_dx - 2, cy + gt_dy - 2), (cx + gt_dx + 2, cy + gt_dy + 2)], fill=(0, 200, 0))

    # Pred arrow (red)
    draw.line([(cx, cy), (cx + pr_dx, cy + pr_dy)], fill=(220, 50, 50), width=2)
    draw.ellipse([(cx + pr_dx - 2, cy + pr_dy - 2), (cx + pr_dx + 2, cy + pr_dy + 2)], fill=(220, 50, 50))

    text = f"GT: [{gt[0]:+.3f}, {gt[1]:+.3f}]  Pred: [{pred[0]:+.3f}, {pred[1]:+.3f}]"
    draw.rectangle([(0, h - 16), (w, h)], fill=(0, 0, 0))
    draw.text((4, h - 14), text, fill=(255, 255, 255))
    return img


def main():
    ap = argparse.ArgumentParser(description="Export Push-T sample visualizations")
    ap.add_argument("--method", choices=["dora", "lora", "full"], default="dora")
    ap.add_argument("--adapter_path", default=None)
    ap.add_argument("--head_path", default=None)
    ap.add_argument("--output_dir", default="../results/pusht_samples")
    ap.add_argument("--num_samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=["train", "val"], default="val")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    processor = None
    train_ds, val_ds = load_pusht(processor=processor, val_split=0.2, seed=args.seed)
    ds = val_ds if args.split == "val" else train_ds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmolVLMActionModel(dtype=torch.float32).to(device)

    if args.method == "dora":
        apply_dora_to_model(model.vlm, target_modules=SMOLVLM_TARGET_MODULES, rank=8, alpha=16.0, dropout=0.0)
        if args.adapter_path:
            DoRAStateManager.load_dora_state(model.vlm, args.adapter_path, strict=True)
    elif args.method == "lora":
        apply_lora_to_model(model.vlm, target_modules=SMOLVLM_TARGET_MODULES, rank=8, alpha=16.0, dropout=0.0)
        if args.adapter_path:
            _load_lora_weights(model.vlm, args.adapter_path)

    if args.head_path:
        model.action_head.load_state_dict(torch.load(args.head_path, map_location="cpu"))

    model.eval()

    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[: args.num_samples]

    for i, idx in enumerate(indices):
        sample = ds[idx]
        with torch.no_grad():
            out = model(
                pixel_values=sample["pixel_values"].unsqueeze(0).to(device),
                input_ids=sample["input_ids"].unsqueeze(0).to(device),
                attention_mask=sample["attention_mask"].unsqueeze(0).to(device),
            )
        pred = out["logits"].squeeze(0).float().cpu().numpy()
        gt = sample["labels"].float().cpu().numpy()

        # Recreate the original 96x96 frame for display
        img = ds.ds[ds.indices[idx]]["observation.image"]
        img_np = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img_np)
        pil = _draw_action_overlay(pil, gt, pred)

        out_path = os.path.join(args.output_dir, f"pusht_{i:02d}.png")
        pil.save(out_path)

    print(f"Saved {len(indices)} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
