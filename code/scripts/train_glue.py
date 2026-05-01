#!/usr/bin/env python3
"""
Fine-tune LLaMA models on GLUE benchmark tasks.

Supports three methods:
  - dora  : DoRA (Weight-Decomposed Low-Rank Adaptation) — default
  - lora  : Standard LoRA baseline
  - full  : Full fine-tuning (all parameters trainable)

Usage:
    # DoRA (default)
    uv run scripts/train_glue.py --model 1b --task sst2
    uv run scripts/train_glue.py --model 3b --task mrpc --rank 16 --alpha 32

    # LoRA baseline (same rank/alpha for apples-to-apples comparison)
    uv run scripts/train_glue.py --model 1b --task sst2 --method lora

    # Full fine-tuning
    uv run scripts/train_glue.py --model 1b --task sst2 --method full

    # Explicit model name
    uv run scripts/train_glue.py --model_name meta-llama/Llama-3.2-1B --task rte
"""

import argparse
import json
import logging
import math
import os
import sys

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

# Allow importing from the code/ root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dora.models.llama import LlamaDoRAModel, apply_dora_to_model
from dora.layers.lora_linear import LoRALinear, apply_lora_to_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GLUE task metadata
# ---------------------------------------------------------------------------

GLUE_TASKS = {
    "cola": {
        "num_labels": 2,
        "keys": ("sentence", None),
        "primary_metric": "mcc",
        "eval_split": "validation",
    },
    "sst2": {
        "num_labels": 2,
        "keys": ("sentence", None),
        "primary_metric": "accuracy",
        "eval_split": "validation",
    },
    "mrpc": {
        "num_labels": 2,
        "keys": ("sentence1", "sentence2"),
        "primary_metric": "combined",
        "eval_split": "validation",
    },
    "qqp": {
        "num_labels": 2,
        "keys": ("question1", "question2"),
        "primary_metric": "combined",
        "eval_split": "validation",
    },
    "mnli": {
        "num_labels": 3,
        "keys": ("premise", "hypothesis"),
        "primary_metric": "accuracy",
        "eval_split": "validation_matched",
    },
    "qnli": {
        "num_labels": 2,
        "keys": ("question", "sentence"),
        "primary_metric": "accuracy",
        "eval_split": "validation",
    },
    "rte": {
        "num_labels": 2,
        "keys": ("sentence1", "sentence2"),
        "primary_metric": "accuracy",
        "eval_split": "validation",
    },
    "wnli": {
        "num_labels": 2,
        "keys": ("sentence1", "sentence2"),
        "primary_metric": "accuracy",
        "eval_split": "validation",
    },
    "stsb": {
        "num_labels": 1,
        "keys": ("sentence1", "sentence2"),
        "primary_metric": "combined",
        "eval_split": "validation",
    },
}

# ---------------------------------------------------------------------------
# Model size presets
# ---------------------------------------------------------------------------

MODEL_PRESETS = {
    "1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",   # LLaMA arch, fully open
    "3b": "openlm-research/open_llama_3b",          # LLaMA arch, fully open
    "7b": "huggyllama/llama-7b",                    # LLaMA arch, fully open
    "roberta":       "FacebookAI/roberta-base",     # RoBERTa-base  (~125M params)
    "roberta_large": "FacebookAI/roberta-large",    # RoBERTa-large (~355M params)
}

# Architecture-specific target modules (attention projections only)
_TARGET_MODULES_BY_ARCH = {
    # LLaMA / Mistral / Falcon style
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
    # BERT / RoBERTa / ViT style — "dense" excluded: also matches classifier.dense
    "roberta": ["query", "key", "value"],
    "bert":    ["query", "key", "value"],
    "vit":                ["query", "key", "value"],
    # SigLIP (OpenVLA's visual encoder) uses LLaMA-style projection names
    "siglip_vision_model": ["q_proj", "k_proj", "v_proj"],
}
_DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def get_default_target_modules(model: torch.nn.Module) -> list:
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    return _TARGET_MODULES_BY_ARCH.get(model_type, _DEFAULT_TARGET_MODULES)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def build_compute_metrics(task: str):
    """Return a task-specific compute_metrics function for HF Trainer."""
    is_regression = task == "stsb"

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        labels = p.label_ids

        if is_regression:
            preds = preds.squeeze()
            labels = labels.squeeze()
            r, _ = pearsonr(preds, labels)
            rho, _ = spearmanr(preds, labels)
            return {
                "pearson": float(r),
                "spearmanr": float(rho),
                "combined": float((r + rho) / 2),
            }

        preds = np.argmax(preds, axis=1)
        accuracy = float(np.mean(preds == labels))

        if task == "cola":
            return {"mcc": float(matthews_corrcoef(labels, preds))}

        if task in ("mrpc", "qqp"):
            f1 = float(f1_score(labels, preds, average="binary"))
            return {"accuracy": accuracy, "f1": f1, "combined": (accuracy + f1) / 2}

        return {"accuracy": accuracy}

    return compute_metrics


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def tokenize_glue(dataset, tokenizer, task: str, max_length: int):
    key1, key2 = GLUE_TASKS[task]["keys"]

    def tokenize_fn(examples):
        args = (
            (examples[key1],)
            if key2 is None
            else (examples[key1], examples[key2])
        )
        return tokenizer(*args, padding=False, truncation=True, max_length=max_length)

    remove_cols = [c for c in dataset.column_names if c != "label"]
    ds = dataset.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    if "label" in ds.column_names:
        ds = ds.rename_column("label", "labels")
    return ds


# ---------------------------------------------------------------------------
# Per-epoch adapter statistics callback
# ---------------------------------------------------------------------------

class AdapterStatsCallback(TrainerCallback):
    """
    After every evaluation, snapshot per-layer adapter statistics and write
    them to `<output_dir>/adapter_stats.json` when training ends.

    Captured per epoch:
      - epoch, eval metric (accuracy / F1 / MCC), train loss
      - mean & std of DoRA magnitude vectors  (DoRA only)
      - Frobenius norm of the low-rank update  ΔW = (α/r) · lora_B @ lora_A
        (both DoRA and LoRA)
    """

    def __init__(self, model: torch.nn.Module, output_dir: str, primary_metric: str):
        self.model = model
        self.output_dir = output_dir
        self.primary_metric = primary_metric
        self.records: list = []
        self._last_train_loss: float = float("nan")
        self._init_magnitudes: dict = {}   # layer_name → initial magnitude vector (cpu fp32)

    def on_train_begin(self, args, state, control, **kwargs):
        """Snapshot initial magnitude vectors and verify gradient flow."""
        for name, module in self.model.named_modules():
            if hasattr(module, "magnitude") and hasattr(module, "lora_A"):
                with torch.no_grad():
                    # Store full vector (not just norm) so we can compute true drift
                    self._init_magnitudes[name] = module.magnitude.float().cpu().clone()

        # Sanity-check gradient flow for magnitude parameters
        n_mag_params = sum(
            1 for _, m in self.model.named_modules()
            if hasattr(m, "magnitude") and m.magnitude.requires_grad
        )
        n_frozen = sum(
            1 for _, m in self.model.named_modules()
            if hasattr(m, "magnitude") and not m.magnitude.requires_grad
        )
        import logging as _logging
        _log = _logging.getLogger(__name__)
        _log.info(
            f"[AdapterStats] DoRA magnitude params: {n_mag_params} trainable, "
            f"{n_frozen} frozen"
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self._last_train_loss = float(logs["loss"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = state.epoch or 0.0
        eval_metric = float(metrics.get(f"eval_{self.primary_metric}", float("nan")))
        eval_loss   = float(metrics.get("eval_loss", float("nan")))

        mag_means, mag_stds, lora_norms = [], [], []

        # True element-wise relative drift: ||m_t - m_0||_2 / ||m_0||_2
        # (NOT |norm(m_t) - norm(m_0)| which masks sign-cancellation)
        mag_rel_drifts = []
        mag_grad_norms = []   # gradient norms for magnitude params (non-None if grad exists)

        for name, module in self.model.named_modules():
            # DoRA layer
            if hasattr(module, "magnitude") and hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                with torch.no_grad():
                    mag = module.magnitude.float().cpu()
                    mag_means.append(mag.mean().item())
                    mag_stds.append(mag.std().item())
                    # Correct drift: Frobenius distance from init, normalised by init norm
                    m0 = self._init_magnitudes.get(name)
                    if m0 is not None and m0.norm().item() > 0:
                        drift = (mag - m0).norm().item() / m0.norm().item()
                        mag_rel_drifts.append(drift)
                    delta = module.lora_B.float() @ module.lora_A.float()
                    lora_norms.append(delta.norm().item())
                # Track whether magnitude is receiving gradients
                if module.magnitude.grad is not None:
                    mag_grad_norms.append(module.magnitude.grad.float().norm().item())

            # LoRA-only layer (no magnitude)
            elif hasattr(module, "lora_A") and hasattr(module, "lora_B") and not hasattr(module, "magnitude"):
                with torch.no_grad():
                    delta = module.lora_B.float() @ module.lora_A.float()
                    lora_norms.append(delta.norm().item())

        record = {
            "epoch":                  round(epoch, 2),
            "eval_metric":            round(eval_metric, 6),
            "eval_loss":              round(eval_loss, 6),
            "train_loss":             round(self._last_train_loss, 6),
            "lora_norm_mean":         round(float(np.mean(lora_norms)),       6) if lora_norms      else None,
            "lora_norm_std":          round(float(np.std(lora_norms)),         6) if lora_norms      else None,
            # Correct: ||m_t - m_0||_F / ||m_0||_F (element-wise Frobenius drift)
            "magnitude_rel_drift":    round(float(np.mean(mag_rel_drifts)),   6) if mag_rel_drifts  else None,
            "magnitude_mean":         round(float(np.mean(mag_means)),         6) if mag_means       else None,
            # If this is 0.0 or None, magnitude params have no/zero gradient → frozen
            "magnitude_grad_norm":    round(float(np.mean(mag_grad_norms)),   6) if mag_grad_norms  else None,
            "n_adapter_layers":       len(lora_norms),
        }
        self.records.append(record)

    def _per_layer_stats(self) -> list:
        """
        One row per adapted layer with the values needed for a polar scatter:
            method, layer, angle_deg, relative_update_norm

        angle_deg               = arccos(<W0, ΔW>_F / (||W0||_F · ||ΔW||_F))
        relative_update_norm    = ||ΔW||_F / ||W0||_F
        ΔW                      = W_eff - W0   (uses the layer's own
                                                get_effective_weight, so DoRA
                                                magnitude + scaling are baked in)
        """
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

    def on_train_end(self, args, state, control, **kwargs):
        out_path = os.path.join(self.output_dir, "adapter_stats.json")
        with open(out_path, "w") as f:
            json.dump(self.records, f, indent=2)
        logger.info(f"Adapter stats saved → {out_path}  ({len(self.records)} epochs)")

        layer_rows = self._per_layer_stats()
        if layer_rows:
            layer_path = os.path.join(self.output_dir, "adapter_layer_stats.json")
            with open(layer_path, "w") as f:
                json.dump(layer_rows, f, indent=2)
            logger.info(f"Per-layer adapter stats → {layer_path}  ({len(layer_rows)} layers)")


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
# DoRA / LoRA freeze and optimizer helpers
# ---------------------------------------------------------------------------

def freeze_base_weights(model: torch.nn.Module):
    """Keep only adapter params and the classification head trainable."""
    for name, param in model.named_parameters():
        leaf = name.split(".")[-1]
        is_adapter = leaf in ("lora_A", "lora_B", "magnitude")
        # LlamaForSequenceClassification head: model.score
        # RobertaForSequenceClassification head: classifier.dense / classifier.out_proj
        is_head = "score" in name or "classifier" in name
        param.requires_grad = is_adapter or is_head

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")


def build_adapter_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """
    AdamW with weight_decay=0 for adapter matrices and weight_decay for the
    classification head.

    The HuggingFace Trainer default applies weight_decay to *all* trainable
    parameters whose names do not end in "bias" or contain "LayerNorm.weight".
    That incorrectly penalises lora_A, lora_B and magnitude toward zero,
    counteracting the task-adaptation gradient and degrading both LoRA and DoRA.

    Head parameters (score / classifier) benefit from mild regularisation, so
    weight_decay is applied there and nowhere else.
    """
    _ADAPTER_LEAVES = {"lora_A", "lora_B", "magnitude"}
    adapter_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        leaf = name.split(".")[-1]
        if leaf in _ADAPTER_LEAVES:
            adapter_params.append(param)
        else:
            head_params.append(param)   # score.weight, classifier.dense.weight, …

    logger.info(
        f"Optimizer groups — adapter (wd=0): {sum(p.numel() for p in adapter_params):,} params  |  "
        f"head (wd={weight_decay}): {sum(p.numel() for p in head_params):,} params"
    )
    return torch.optim.AdamW(
        [
            {"params": adapter_params, "weight_decay": 0.0},
            {"params": head_params,    "weight_decay": weight_decay},
        ],
        lr=lr,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_glue_parser() -> argparse.ArgumentParser:
    """Return the full argument parser (used in two-pass YAML loading)."""
    parser = argparse.ArgumentParser(description="Train on GLUE (DoRA / LoRA / full FT)")
    parser.add_argument("--config", default=None, metavar="FILE",
                        help="YAML config file; CLI flags override any YAML value")

    # Model — not required here; validated after YAML merge
    model_group = parser.add_mutually_exclusive_group(required=False)
    model_group.add_argument(
        "--model", choices=list(MODEL_PRESETS.keys()), help="Size preset"
    )
    model_group.add_argument("--model_name", type=str, help="HuggingFace model ID or local path")

    parser.add_argument("--task", default=None, choices=list(GLUE_TASKS.keys()))
    parser.add_argument("--method", choices=["dora", "lora", "full"], default="dora")

    # Adapter
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+", default=None)

    # Training
    parser.add_argument("--lr", type=float, default=None,
                        help="Defaults to 2e-4 (adapter) / 2e-5 (full FT)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)

    # Hardware
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no_mps", action="store_true")

    # Logging
    parser.add_argument("--wandb", action="store_true")
    return parser


def parse_args():
    import yaml as _yaml

    parser = _build_glue_parser()

    # Pass 1 — discover --config path without failing on unknown args
    pre, _ = parser.parse_known_args()
    if pre.config:
        with open(pre.config) as _f:
            _cfg = _yaml.safe_load(_f) or {}
        # YAML values become new defaults; explicit CLI flags still win
        parser.set_defaults(**{k: v for k, v in _cfg.items() if v is not None})

    # Pass 2 — full parse with YAML-enriched defaults
    args = parser.parse_args()

    if args.model is None and args.model_name is None:
        parser.error("--model or --model_name is required (or set 'model:' in the config)")
    if args.task is None:
        parser.error("--task is required (or set 'task:' in the config)")

    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    model_name = MODEL_PRESETS[args.model] if args.model else args.model_name
    size_tag = args.model or model_name.split("/")[-1]
    task_cfg = GLUE_TASKS[args.task]

    if args.output_dir is None:
        run_tag = f"{args.method}_r{args.rank}" if args.method != "full" else "full"
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..",
            "results",
            f"glue_{args.task}_{size_tag}_{run_tag}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    if args.method == "full":
        logger.info(f"Model: {model_name} | Task: {args.task} | Method: full fine-tuning")
    else:
        target_label = args.target_modules if args.target_modules is not None else "auto"
        logger.info(
            f"Model: {model_name} | Task: {args.task} | Method: {args.method} | "
            f"rank={args.rank} alpha={args.alpha} target={target_label}"
        )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # LLaMA has no pad token; reuse EOS
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # right-pad for classification

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    dtype = (
        torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=task_cfg["num_labels"],
        torch_dtype=dtype,
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Resolve target modules (auto-detect if not explicitly passed)
    if args.target_modules is None:
        args.target_modules = get_default_target_modules(model)
        logger.info(f"Auto-detected target_modules={args.target_modules} for {model.config.model_type}")

    # ------------------------------------------------------------------
    # Apply adapter or prepare for full fine-tuning
    # ------------------------------------------------------------------
    # Resolve LR defaults per method (None sentinel avoids fragile float comparison)
    if args.lr is None:
        args.lr = 2e-5 if args.method == "full" else 2e-4
        logger.info(f"Auto-selected lr={args.lr} for method={args.method}")

    if args.method == "dora":
        apply_dora_to_model(
            model,
            target_modules=args.target_modules,
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
        )
        freeze_base_weights(model)
    elif args.method == "lora":
        apply_lora_to_model(
            model,
            target_modules=args.target_modules,
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
        )
        freeze_base_weights(model)
    else:
        # Full fine-tuning: all parameters trainable
        for param in model.parameters():
            param.requires_grad = True

    # Print trainable-parameter summary
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"[{args.method.upper()}] Trainable: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.3f}%)"
    )

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    logger.info(f"Loading GLUE/{args.task}...")
    raw = load_dataset("glue", args.task)
    train_ds = tokenize_glue(raw["train"], tokenizer, args.task, args.max_length)
    eval_ds = tokenize_glue(raw[task_cfg["eval_split"]], tokenizer, args.task, args.max_length)

    logger.info(f"Train: {len(train_ds):,}  Eval: {len(eval_ds):,}")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))

    # Enable MPS (Apple GPU) if available and not explicitly disabled
    use_mps = (
        not args.no_mps
        and not torch.cuda.is_available()
        and torch.backends.mps.is_available()
    )
    if use_mps:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        logger.info("Using MPS (Apple GPU)")

    try:
        import wandb as _wandb  # noqa: F401
        _wandb_available = True
    except ImportError:
        _wandb_available = False

    report_to = "wandb" if (args.wandb and _wandb_available) else "none"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=task_cfg["primary_metric"],
        greater_is_better=True,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=report_to,
        run_name=f"{args.method}_{args.task}_{size_tag}" + (f"_r{args.rank}" if args.method != "full" else ""),
        logging_steps=50,
        seed=args.seed,
        dataloader_num_workers=0 if use_mps else 2,
        dataloader_pin_memory=not use_mps,
    )

    stats_cb = AdapterStatsCallback(
        model=model,
        output_dir=args.output_dir,
        primary_metric=task_cfg["primary_metric"],
    )

    # For adapter runs, pass a custom optimizer so adapter matrices (lora_A,
    # lora_B, magnitude) get weight_decay=0. The HF Trainer default would apply
    # weight_decay to every non-bias, non-LayerNorm parameter, which incorrectly
    # penalises the adapter toward zero and suppresses DoRA magnitude learning.
    # For full fine-tuning, let the Trainer build its own optimizer (standard).
    if args.method in ("dora", "lora"):
        adapter_optimizer = build_adapter_optimizer(model, args.lr, args.weight_decay)
        optimizers = (adapter_optimizer, None)   # None → Trainer builds scheduler
    else:
        optimizers = (None, None)

    callbacks = []
    if args.method in ("dora", "lora"):
        callbacks.append(AdapterStatsCallback(model, args.output_dir, task_cfg["primary_metric"]))
        callbacks.append(AdapterCheckpointCallback(model, args.output_dir, args.method))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=build_compute_metrics(args.task),
        callbacks=callbacks,
        optimizers=optimizers,
    )

    logger.info("Training...")
    trainer.train()

    logger.info("Final evaluation...")
    results = trainer.evaluate()
    logger.info(f"Results on {args.task}: {results}")

    # Save model / adapter weights
    if args.method == "full":
        # Save the full model
        model.save_pretrained(os.path.join(args.output_dir, "full_model"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "full_model"))
        logger.info(f"Full model saved -> {args.output_dir}/full_model")
    elif args.method == "dora":
        adapter_path = os.path.join(args.output_dir, "dora_adapter.pt")
        LlamaDoRAModel.save_dora_adapter(model, adapter_path)
        logger.info(f"DoRA adapter saved -> {adapter_path}")
    else:
        adapter_path = os.path.join(args.output_dir, "lora_adapter.pt")
        lora_state = {}
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                lora_state[name] = {
                    "lora_A": module.lora_A.data,
                    "lora_B": module.lora_B.data,
                }
        torch.save({"lora_state": lora_state}, adapter_path)
        logger.info(f"LoRA adapter saved -> {adapter_path}")

    return results


if __name__ == "__main__":
    main()
