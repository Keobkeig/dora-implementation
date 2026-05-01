#!/usr/bin/env python3
"""
Export visual samples for Cornell Grasp (ViT/SigLIP).

Generates images with both GT and predicted grasp rectangles overlaid.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dora.data.cornell_grasp import _build_corners, load_cornell_grasp
from dora.layers.base import DoRAStateManager
from dora.layers.lora_linear import LoRALinear, apply_lora_to_model
from dora.models.llama import apply_dora_to_model

from scripts.train_grasp import MODEL_PRESETS, VisionGraspModel, _get_target_modules


def _load_lora_weights(model: torch.nn.Module, adapter_path: str) -> None:
    payload = torch.load(adapter_path, map_location="cpu")
    lora_state = payload.get("lora_state", {})
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and name in lora_state:
            state = lora_state[name]
            module.lora_A.data.copy_(state["lora_A"])
            module.lora_B.data.copy_(state["lora_B"])


def _pose_to_corners(target: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    cx = float(np.clip(target[0], 0.0, 1.0)) * img_w
    cy = float(np.clip(target[1], 0.0, 1.0)) * img_h
    angle = 0.5 * float(np.arctan2(target[2], target[3]))
    width = float(abs(target[4])) * img_w
    height = float(abs(target[5])) * img_h
    return _build_corners(cx, cy, angle, width, height)


def _draw_grasp(img: Image.Image, corners: np.ndarray, color: tuple) -> None:
    draw = ImageDraw.Draw(img)
    if not np.isfinite(corners).all():
        return
    pts = [tuple(p) for p in corners.tolist()]
    draw.line(pts + [pts[0]], fill=color, width=2)


def main():
    ap = argparse.ArgumentParser(description="Export Cornell Grasp sample visualizations")
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), default="vit")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--method", choices=["dora", "lora", "full"], default="dora")
    ap.add_argument("--adapter_path", default=None)
    ap.add_argument("--head_path", default=None)
    ap.add_argument("--output_dir", default="../results/grasp_samples")
    ap.add_argument("--num_samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=["train", "val"], default="val")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    model_name = MODEL_PRESETS[args.model] if args.model else args.model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionGraspModel(model_name=model_name, dtype=torch.float32).to(device)

    target_modules = _get_target_modules(model.backbone)
    if args.method == "dora":
        apply_dora_to_model(model.backbone, target_modules=target_modules, rank=8, alpha=16.0, dropout=0.0)
        if args.adapter_path:
            DoRAStateManager.load_dora_state(model.backbone, args.adapter_path, strict=True)
    elif args.method == "lora":
        apply_lora_to_model(model.backbone, target_modules=target_modules, rank=8, alpha=16.0, dropout=0.0)
        if args.adapter_path:
            _load_lora_weights(model.backbone, args.adapter_path)

    if args.head_path:
        model.grasp_head.load_state_dict(torch.load(args.head_path, map_location="cpu"))

    processor = AutoImageProcessor.from_pretrained(model_name)
    train_ds, val_ds = load_cornell_grasp(args.data_dir, processor=processor, val_split=0.2, seed=args.seed)
    ds = val_ds if args.split == "val" else train_ds

    model.eval()
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[: args.num_samples]

    for i, idx in enumerate(indices):
        sample = ds[idx]
        with torch.no_grad():
            out = model(pixel_values=sample["pixel_values"].unsqueeze(0).to(device))
        pred = out["logits"].squeeze(0).float().cpu().numpy()
        gt = sample["labels"].float().cpu().numpy()

        img = Image.open(ds.image_paths[idx]).convert("RGB")
        gt_corners = _pose_to_corners(gt, img.width, img.height)
        pr_corners = _pose_to_corners(pred, img.width, img.height)

        _draw_grasp(img, gt_corners, (0, 200, 0))
        _draw_grasp(img, pr_corners, (220, 50, 50))

        out_path = os.path.join(args.output_dir, f"grasp_{args.model}_{i:02d}.png")
        img.save(out_path)

    print(f"Saved {len(indices)} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
