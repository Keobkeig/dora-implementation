#!/usr/bin/env python3
"""
Fine-tune LLaMA models with DoRA or LoRA on GLUE benchmark tasks.

Usage:
    # DoRA (default)
    python scripts/train_glue.py --model 1b --task sst2
    python scripts/train_glue.py --model 3b --task mrpc --rank 16 --alpha 32

    # LoRA baseline
    python scripts/train_glue.py --model 1b --task sst2 --method lora
    python scripts/train_glue.py --model 1b --task sst2 --method lora --rank 8

    # Explicit model name
    python scripts/train_glue.py --model_name meta-llama/Llama-3.2-1B --task rte
"""

import argparse
import logging
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
}

# Default DoRA target modules for sequence classification (attention only;
# adding MLP projections increases trainable params but often helps)
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


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
# DoRA freeze helper
# ---------------------------------------------------------------------------

def freeze_base_weights(model: torch.nn.Module):
    """Keep only DoRA adapter params and the classification head trainable."""
    for name, param in model.named_parameters():
        leaf = name.split(".")[-1]
        is_dora = leaf in ("lora_A", "lora_B", "magnitude")
        is_head = "score" in name  # LlamaForSequenceClassification uses model.score
        param.requires_grad = is_dora or is_head

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train LLaMA+DoRA on GLUE")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model", choices=list(MODEL_PRESETS.keys()), help="Size preset (1b / 3b / 7b)"
    )
    model_group.add_argument("--model_name", type=str, help="HuggingFace model ID or local path")

    parser.add_argument("--task", required=True, choices=list(GLUE_TASKS.keys()))
    parser.add_argument(
        "--method",
        choices=["dora", "lora"],
        default="dora",
        help="Adapter method: dora (default) or lora baseline",
    )

    # Adapter hyperparameters
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=DEFAULT_TARGET_MODULES,
        help="Linear layer names to apply adapter to",
    )

    # Training
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--warmup_ratio", type=float, default=0.06, help="Fraction of steps used for LR warmup")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)

    # Hardware
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no_mps", action="store_true", help="Disable MPS (Apple GPU) even if available")

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging (off by default)")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    model_name = MODEL_PRESETS[args.model] if args.model else args.model_name
    size_tag = args.model or model_name.split("/")[-1]
    task_cfg = GLUE_TASKS[args.task]

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..",
            "results",
            f"glue_{args.task}_{size_tag}_{args.method}_r{args.rank}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(
        f"Model: {model_name} | Task: {args.task} | Method: {args.method} | "
        f"rank={args.rank} alpha={args.alpha} target={args.target_modules}"
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

    # ------------------------------------------------------------------
    # Apply adapter (DoRA or LoRA)
    # ------------------------------------------------------------------
    if args.method == "dora":
        apply_dora_to_model(
            model,
            target_modules=args.target_modules,
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
        )
    else:
        apply_lora_to_model(
            model,
            target_modules=args.target_modules,
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
        )
    freeze_base_weights(model)

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
    import math
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
        load_best_model_at_end=True,
        metric_for_best_model=task_cfg["primary_metric"],
        greater_is_better=True,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=report_to,
        run_name=f"{args.method}_{args.task}_{size_tag}_r{args.rank}",
        logging_steps=50,
        seed=args.seed,
        dataloader_num_workers=0 if use_mps else 2,
        dataloader_pin_memory=not use_mps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=build_compute_metrics(args.task),
    )

    logger.info("Training...")
    trainer.train()

    logger.info("Final evaluation...")
    results = trainer.evaluate()
    logger.info(f"Results on {args.task}: {results}")

    # Save adapter weights only (not the full model)
    adapter_path = os.path.join(args.output_dir, f"{args.method}_adapter.pt")
    if args.method == "dora":
        LlamaDoRAModel.save_dora_adapter(model, adapter_path)
    else:
        # For LoRA, save the LoRA-specific state dict
        lora_state = {}
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                lora_state[name] = {
                    "lora_A": module.lora_A.data,
                    "lora_B": module.lora_B.data,
                }
        torch.save({"lora_state": lora_state}, adapter_path)
    logger.info(f"{args.method.upper()} adapter saved → {adapter_path}")

    return results


if __name__ == "__main__":
    main()
