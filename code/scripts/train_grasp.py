#!/usr/bin/env python3
"""
Fine-tune a vision backbone on Cornell Grasp Dataset for grasp pose estimation.

Compares DoRA vs LoRA vs full fine-tuning using the standard Cornell
success metric: IoU ≥ 0.25 AND |Δangle| ≤ 30°.

Supported backbones:
  --model vit      google/vit-base-patch16-224          (86M  — pure vision)
  --model siglip   google/siglip-base-patch16-224       (400M — OpenVLA visual encoder)
  --model_name <any HuggingFace ViT-compatible model>

Usage:
    uv run scripts/train_grasp.py --data_dir ../data/cornell_grasps --model vit   --method dora --bf16
    uv run scripts/train_grasp.py --data_dir ../data/cornell_grasps --model siglip --method dora --bf16
"""

from typing import Optional
import argparse
import logging
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoImageProcessor,
    AutoModel,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dora.models.llama import apply_dora_to_model
from dora.layers.lora_linear import LoRALinear, apply_lora_to_model
from dora.data.cornell_grasp import load_cornell_grasp, grasp_success_from_arrays

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PRESETS = {
    "vit":    "google/vit-base-patch16-224",
    "siglip": "google/siglip-base-patch16-224",   # OpenVLA's visual encoder
}

# Per-architecture attention target modules (leaf names used by apply_dora/lora_to_model)
_TARGET_MODULES = {
    "vit":                 ["query", "key", "value"],   # BERT-style
    "siglip_vision_model": ["q_proj", "k_proj", "v_proj"],  # LLaMA-style
}
_DEFAULT_TARGET_MODULES = ["query", "key", "value"]


def _get_target_modules(model: nn.Module) -> list:
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    return _TARGET_MODULES.get(model_type, _DEFAULT_TARGET_MODULES)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class VisionGraspModel(nn.Module):
    """
    Vision encoder + lightweight regression head for grasp pose prediction.

    Supports both ViT (CLS-token output) and SigLIP (pooler_output / mean-pool).
    Head predicts 6-D normalised grasp: (x, y, sin2θ, cos2θ, w, h).
    """

    def __init__(self, model_name: str, dtype: torch.dtype = torch.float32):
        super().__init__()
        _full = AutoModel.from_pretrained(
            model_name, torch_dtype=dtype, ignore_mismatched_sizes=True
        )
        # SiglipModel bundles text + vision; we only need the vision encoder
        self.backbone = getattr(_full, "vision_model", _full)
        hidden = self.backbone.config.hidden_size
        self.grasp_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 6),
        )
        self.loss_fn = nn.SmoothL1Loss()

    def _pool(self, outputs) -> torch.Tensor:
        # SigLIP uses pooler_output (mean pool); ViT/BERT use CLS token
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]

    def forward(self, pixel_values: torch.Tensor,
                labels: Optional[torch.Tensor] = None, **kwargs) -> dict:
        features = self._pool(self.backbone(pixel_values=pixel_values))
        logits = self.grasp_head(features)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------------
# Adapter stats callback (per-epoch weight tracking)
# ---------------------------------------------------------------------------

class AdapterStatsCallback(TrainerCallback):
    def __init__(self, model, output_dir):
        self.model = model
        self.output_dir = output_dir
        self.records = []
        self._last_train_loss = float("nan")
        self._init_magnitudes = {}

    def on_train_begin(self, args, state, control, **kwargs):
        for name, m in self.model.named_modules():
            if hasattr(m, "magnitude") and hasattr(m, "lora_A"):
                with torch.no_grad():
                    self._init_magnitudes[name] = m.magnitude.float().cpu().clone()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self._last_train_loss = float(logs["loss"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        import json as _json
        epoch = state.epoch or 0.0
        eval_metric = float(metrics.get("eval_grasp_success_rate", float("nan")))
        eval_loss   = float(metrics.get("eval_loss", float("nan")))

        lora_norms, mag_drifts, mag_means, mag_grad_norms = [], [], [], []
        for name, m in self.model.named_modules():
            if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
                with torch.no_grad():
                    lora_norms.append((m.lora_B.float() @ m.lora_A.float()).norm().item())
                    if hasattr(m, "magnitude"):
                        mag = m.magnitude.float().cpu()
                        mag_means.append(mag.mean().item())
                        m0 = self._init_magnitudes.get(name)
                        if m0 is not None and m0.norm().item() > 0:
                            mag_drifts.append((mag - m0).norm().item() / m0.norm().item())
                if hasattr(m, "magnitude") and m.magnitude.grad is not None:
                    mag_grad_norms.append(m.magnitude.grad.float().norm().item())

        self.records.append({
            "epoch":               round(epoch, 2),
            "eval_metric":         round(eval_metric, 6),
            "eval_loss":           round(eval_loss, 6),
            "train_loss":          round(self._last_train_loss, 6),
            "lora_norm_mean":      round(float(np.mean(lora_norms)),     6) if lora_norms     else None,
            "lora_norm_std":       round(float(np.std(lora_norms)),      6) if lora_norms     else None,
            "magnitude_rel_drift": round(float(np.mean(mag_drifts)),     6) if mag_drifts     else None,
            "magnitude_mean":      round(float(np.mean(mag_means)),      6) if mag_means      else None,
            "magnitude_grad_norm": round(float(np.mean(mag_grad_norms)), 6) if mag_grad_norms else None,
            "n_adapter_layers":    len(lora_norms),
        })

    def on_train_end(self, args, state, control, **kwargs):
        import json as _json
        out = os.path.join(self.output_dir, "adapter_stats.json")
        with open(out, "w") as f:
            _json.dump(self.records, f, indent=2)
        logger.info(f"Adapter stats → {out}")

        layer_rows = self._per_layer_stats()
        if layer_rows:
            layer_path = os.path.join(self.output_dir, "adapter_layer_stats.json")
            with open(layer_path, "w") as f:
                _json.dump(layer_rows, f, indent=2)
            logger.info(f"Per-layer adapter stats → {layer_path}  ({len(layer_rows)} layers)")

    def _per_layer_stats(self) -> list:
        """One row per adapted layer: method, layer, angle_deg, relative_update_norm."""
        rows = []
        for name, module in self.model.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")
                    and hasattr(module, "base_weight")
                    and hasattr(module, "get_effective_weight")):
                continue
            method_name = "dora" if hasattr(module, "magnitude") else "lora"
            with torch.no_grad():
                W0 = module.base_weight.detach().float()
                W_eff = module.get_effective_weight().detach().float()
                dW = W_eff - W0
                w0 = W0.flatten()
                dw = dW.flatten()
                w0_norm = float(torch.linalg.norm(w0).item())
                dw_norm = float(torch.linalg.norm(dw).item())
                if w0_norm > 0 and dw_norm > 0:
                    cos = float((torch.dot(w0, dw) / (w0_norm * dw_norm)).clamp(-1.0, 1.0).item())
                    angle_deg = math.degrees(math.acos(cos))
                else:
                    angle_deg = float("nan")
                rel = dw_norm / w0_norm if w0_norm > 0 else float("nan")
            rows.append({
                "method": method_name,
                "layer": name,
                "angle_deg": round(angle_deg, 4),
                "relative_update_norm": round(rel, 6),
            })
        return rows


class AdapterCheckpointCallback(TrainerCallback):
    """
    Save adapter weights after each evaluation (overwrites latest).
    Captures LoRA A/B and DoRA A/B/magnitude updates during training.
    """

    def __init__(self, model: torch.nn.Module, output_dir: str, method: str):
        self.model = model
        self.output_dir = output_dir
        self.method = method

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.method == "dora":
            from dora.layers.base import DoRAStateManager

            out = os.path.join(self.output_dir, "dora_adapter_latest.pt")
            DoRAStateManager.save_dora_state(self.model, out, include_base_weights=False)
        elif self.method == "lora":
            lora_state = {}
            for name, module in self.model.named_modules():
                if isinstance(module, LoRALinear):
                    lora_state[name] = {
                        "lora_A": module.lora_A.data,
                        "lora_B": module.lora_B.data,
                    }
            out = os.path.join(self.output_dir, "lora_adapter_latest.pt")
            torch.save({"lora_state": lora_state}, out)


# ---------------------------------------------------------------------------
# Freeze / metrics helpers
# ---------------------------------------------------------------------------

def freeze_base_weights(model: VisionGraspModel):
    for name, param in model.named_parameters():
        leaf = name.split(".")[-1]
        is_adapter = leaf in ("lora_A", "lora_B", "magnitude")
        is_head = name.startswith("grasp_head")
        param.requires_grad = is_adapter or is_head
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")


def build_compute_metrics():
    def compute_metrics(p: EvalPrediction) -> dict:
        preds  = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        sr = grasp_success_from_arrays(np.array(preds), np.array(p.label_ids))
        return {"grasp_success_rate": sr}
    return compute_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_grasp_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vision backbone DoRA/LoRA/FFT on Cornell Grasp")
    p.add_argument("--config", default=None, metavar="FILE",
                   help="YAML config file; CLI flags override any YAML value")

    # Not required — validated after YAML merge
    model_grp = p.add_mutually_exclusive_group(required=False)
    model_grp.add_argument("--model", choices=list(MODEL_PRESETS.keys()))
    model_grp.add_argument("--model_name")

    p.add_argument("--data_dir", default=None)
    p.add_argument("--method", choices=["dora", "lora", "full"], default="dora")

    p.add_argument("--rank",    type=int,   default=8)
    p.add_argument("--alpha",   type=float, default=16.0)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--target_modules", nargs="+", default=None)

    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--val_split",    type=float, default=0.2)
    p.add_argument("--output_dir",   default=None)

    p.add_argument("--fp16",  action="store_true")
    p.add_argument("--bf16",  action="store_true")
    p.add_argument("--wandb", action="store_true")
    return p


def parse_args():
    import yaml as _yaml

    p = _build_grasp_parser()
    pre, _ = p.parse_known_args()
    if pre.config:
        with open(pre.config) as _f:
            _cfg = _yaml.safe_load(_f) or {}
        p.set_defaults(**{k: v for k, v in _cfg.items() if v is not None})

    args = p.parse_args()

    if args.model is None and args.model_name is None:
        p.error("--model or --model_name is required (or set 'model:' in the config)")
    if args.data_dir is None:
        p.error("--data_dir is required (or set 'data_dir:' in the config)")

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    model_name = MODEL_PRESETS[args.model] if args.model else args.model_name
    size_tag   = args.model or model_name.split("/")[-1]

    if args.output_dir is None:
        run_tag = f"{args.method}_r{args.rank}" if args.method != "full" else "full"
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "results", f"grasp_{size_tag}_{run_tag}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Model: {model_name} | Method: {args.method} | rank={args.rank}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    processor = AutoImageProcessor.from_pretrained(model_name)
    train_ds, eval_ds = load_cornell_grasp(
        args.data_dir, processor, val_split=args.val_split, seed=args.seed
    )
    logger.info(f"Train: {len(train_ds)}  Val: {len(eval_ds)}")

    # ------------------------------------------------------------------
    # Model + auto-detect target modules
    # ------------------------------------------------------------------
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    model = VisionGraspModel(model_name=model_name, dtype=dtype)

    if args.target_modules is None:
        args.target_modules = _get_target_modules(model.backbone)
        logger.info(f"Auto-detected target_modules={args.target_modules} "
                    f"for {model.backbone.config.model_type}")

    # ------------------------------------------------------------------
    # Apply adapter
    # ------------------------------------------------------------------
    # Resolve LR defaults per method
    if args.lr is None:
        args.lr = 2e-5 if args.method == "full" else 1e-4
        logger.info(f"Auto-selected lr={args.lr} for method={args.method}")

    if args.method == "dora":
        apply_dora_to_model(model.backbone, target_modules=args.target_modules,
                            rank=args.rank, alpha=args.alpha, dropout=args.dropout)
        freeze_base_weights(model)
    elif args.method == "lora":
        apply_lora_to_model(model.backbone, target_modules=args.target_modules,
                            rank=args.rank, alpha=args.alpha, dropout=args.dropout)
        freeze_base_weights(model)
    else:
        for param in model.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"[{args.method.upper()}] Trainable: {trainable:,} / {total:,} "
                f"({100 * trainable / total:.3f}%)")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    warmup_steps    = max(1, int(args.warmup_ratio * steps_per_epoch * args.epochs))

    try:
        import wandb as _w; _wandb_ok = True  # noqa: E702
    except ImportError:
        _wandb_ok = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="grasp_success_rate",
        greater_is_better=True,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="wandb" if (args.wandb and _wandb_ok) else "none",
        run_name=f"grasp_{size_tag}_{args.method}_r{args.rank}",
        logging_steps=20,
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # Adapter runs: custom optimizer so lora_A/lora_B/magnitude get weight_decay=0
    if args.method in ("dora", "lora"):
        _ADAPTER_LEAVES = {"lora_A", "lora_B", "magnitude"}
        _adapter_p, _head_p = [], []
        for _n, _p in model.named_parameters():
            if not _p.requires_grad:
                continue
            (_adapter_p if _n.split(".")[-1] in _ADAPTER_LEAVES else _head_p).append(_p)
        _grasp_optimizer = torch.optim.AdamW(
            [{"params": _adapter_p, "weight_decay": 0.0},
             {"params": _head_p,    "weight_decay": args.weight_decay}],
            lr=args.lr,
        )
        optimizers = (_grasp_optimizer, None)
    else:
        optimizers = (None, None)

    callbacks = []
    if args.method in ("dora", "lora"):
        callbacks.append(AdapterStatsCallback(model, args.output_dir))
        callbacks.append(AdapterCheckpointCallback(model, args.output_dir, args.method))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=build_compute_metrics(),
        callbacks=callbacks,
        optimizers=optimizers,
    )

    logger.info("Training...")
    trainer.train()

    results = trainer.evaluate()
    logger.info(f"Final results: {results}")

    # Save weights
    if args.method == "dora":
        from dora.layers.base import DoRAStateManager
        DoRAStateManager.save_dora_state(
            model.backbone, os.path.join(args.output_dir, "dora_adapter.pt"),
            include_base_weights=False,
        )
    elif args.method == "lora":
        from dora.layers.lora_linear import LoRALinear
        lora_state = {n: {"lora_A": m.lora_A.data, "lora_B": m.lora_B.data}
                      for n, m in model.named_modules() if isinstance(m, LoRALinear)}
        torch.save({"lora_state": lora_state},
                   os.path.join(args.output_dir, "lora_adapter.pt"))
    else:
        model.backbone.save_pretrained(os.path.join(args.output_dir, "backbone_finetuned"))

    torch.save(model.grasp_head.state_dict(),
               os.path.join(args.output_dir, "grasp_head.pt"))
    return results


if __name__ == "__main__":
    main()
