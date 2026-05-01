#!/usr/bin/env python3
"""
Fine-tune Wav2Vec2 on Google Speech Commands for keyword spotting.

Supports three methods:
  - dora  : DoRA adapters on Wav2Vec2 attention projections
  - lora  : LoRA baseline on the same projections
  - full  : Full fine-tuning

Default label mode is the standard 12-class keyword-spotting setup:
yes/no/up/down/left/right/on/off/stop/go/_unknown_/_silence_.

Usage:
    uv run scripts/train_speech_commands.py --method dora --rank 8 --alpha 16

Fast smoke run:
    uv run scripts/train_speech_commands.py --method dora --max_train_samples 2000 \
        --max_eval_samples 500 --max_test_samples 500 --epochs 1 --batch_size 8
"""

import argparse
import io
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from datasets import Audio, DatasetDict, load_dataset
from scipy.io import wavfile
from scipy.signal import resample_poly
from transformers import (
    AutoFeatureExtractor,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2ForSequenceClassification,
    set_seed,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dora.layers.base import DoRAStateManager
from dora.layers.lora_linear import LoRALinear, apply_lora_to_model
from dora.models.llama import apply_dora_to_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_MODEL_NAME = "facebook/wav2vec2-base"
DEFAULT_DATASET_NAME = "google/speech_commands"
DEFAULT_DATASET_CONFIG = "v0.02"
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]
KWS_LABELS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "_unknown_",
    "_silence_",
]


@dataclass
class DataCollatorAudioClassification:
    feature_extractor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)
        inputs = [{"input_values": feature["input_values"]} for feature in features]
        batch = self.feature_extractor.pad(inputs, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch


def parse_args():
    parser = argparse.ArgumentParser(description="Wav2Vec2 + DoRA on Speech Commands KWS")

    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset_config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--method", choices=["dora", "lora", "full"], default="dora")
    parser.add_argument(
        "--label_mode",
        choices=["kws12", "full"],
        default="kws12",
        help="kws12 collapses non-command words to _unknown_; full keeps raw dataset labels",
    )

    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+", default=None)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_duration_seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path to a Trainer checkpoint directory. Set --epochs to the total target epochs.",
    )

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no_mps", action="store_true", help="Disable Apple MPS even if available")
    parser.add_argument("--wandb", action="store_true")

    return parser.parse_args()


def build_label_maps(raw: DatasetDict, label_mode: str) -> tuple[dict[int, int], dict[int, str]]:
    raw_label_feature = raw["train"].features["label"]

    if label_mode == "full":
        id2label = {i: raw_label_feature.int2str(i) for i in range(raw_label_feature.num_classes)}
        return {i: i for i in id2label}, id2label

    target_to_id = {label: i for i, label in enumerate(KWS_LABELS)}
    raw_to_kws: dict[int, int] = {}
    for raw_id in range(raw_label_feature.num_classes):
        label = raw_label_feature.int2str(raw_id)
        if label in target_to_id:
            raw_to_kws[raw_id] = target_to_id[label]
        else:
            raw_to_kws[raw_id] = target_to_id["_unknown_"]

    id2label = {i: label for i, label in enumerate(KWS_LABELS)}
    return raw_to_kws, id2label


def limit_split(dataset, max_samples: Optional[int], seed: int):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def prepare_datasets(args, feature_extractor) -> tuple[DatasetDict, dict[int, str]]:
    logger.info(f"Loading {args.dataset_name}/{args.dataset_config}...")
    raw = load_dataset(
        args.dataset_name,
        args.dataset_config,
        trust_remote_code=True,
    )
    raw = raw.cast_column(
        "audio",
        Audio(sampling_rate=feature_extractor.sampling_rate, decode=False),
    )
    raw_to_label, id2label = build_label_maps(raw, args.label_mode)

    raw["train"] = limit_split(raw["train"], args.max_train_samples, args.seed)
    raw["validation"] = limit_split(raw["validation"], args.max_eval_samples, args.seed)
    raw["test"] = limit_split(raw["test"], args.max_test_samples, args.seed)

    max_length = int(feature_extractor.sampling_rate * args.max_duration_seconds)

    def read_audio(audio):
        if isinstance(audio, dict) and "array" in audio:
            array = np.asarray(audio["array"], dtype=np.float32)
            sampling_rate = int(audio["sampling_rate"])
        else:
            source = io.BytesIO(audio["bytes"]) if audio.get("bytes") is not None else audio["path"]
            sampling_rate, array = wavfile.read(source)

        if array.ndim > 1:
            array = np.mean(array, axis=1)

        if np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float32) / float(np.iinfo(array.dtype).max)
        else:
            array = array.astype(np.float32)

        target_rate = int(feature_extractor.sampling_rate)
        if sampling_rate != target_rate:
            divisor = math.gcd(sampling_rate, target_rate)
            array = resample_poly(array, target_rate // divisor, sampling_rate // divisor)

        return np.asarray(array, dtype=np.float32)

    def preprocess(batch):
        arrays = [read_audio(audio) for audio in batch["audio"]]
        inputs = feature_extractor(
            arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        inputs["labels"] = [raw_to_label[int(label)] for label in batch["label"]]
        return inputs

    remove_columns = raw["train"].column_names
    processed = raw.map(
        preprocess,
        batched=True,
        remove_columns=remove_columns,
        num_proc=args.preprocessing_num_workers,
        desc="Preprocessing audio",
    )
    return processed, id2label


def freeze_base_weights(model: torch.nn.Module):
    for name, param in model.named_parameters():
        leaf = name.split(".")[-1]
        is_adapter = leaf in ("lora_A", "lora_B", "magnitude")
        is_head = name.startswith("projector") or name.startswith("classifier")
        param.requires_grad = is_adapter or is_head

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")


def count_parameters(model: torch.nn.Module) -> dict[str, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_percent": float(100 * trainable / total) if total else 0.0,
    }


def build_compute_metrics():
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        labels = p.label_ids
        pred_ids = np.argmax(preds, axis=1)
        return {"accuracy": float(np.mean(pred_ids == labels))}

    return compute_metrics


def metric(metrics: dict[str, float], key: str) -> Optional[float]:
    value = metrics.get(key)
    return float(value) if value is not None else None


def save_lora_adapter(model: torch.nn.Module, path: str):
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[name] = {
                "lora_A": module.lora_A.detach().cpu(),
                "lora_B": module.lora_B.detach().cpu(),
            }
    torch.save({"lora_state": lora_state}, path)


class AdapterStatsCallback(TrainerCallback):
    def __init__(self, model: torch.nn.Module, output_dir: str):
        self.model = model
        self.output_dir = output_dir
        self.records: list = []
        self._last_train_loss: float = float("nan")
        self._init_magnitudes: dict = {}

    def on_train_begin(self, args, state, control, **kwargs):
        for name, m in self.model.named_modules():
            if hasattr(m, "magnitude") and hasattr(m, "lora_A"):
                with torch.no_grad():
                    self._init_magnitudes[name] = m.magnitude.float().cpu().clone()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self._last_train_loss = float(logs["loss"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = state.epoch or 0.0
        eval_metric = float(metrics.get("eval_accuracy", float("nan")))
        eval_loss = float(metrics.get("eval_loss", float("nan")))

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
        out = os.path.join(self.output_dir, "adapter_stats.json")
        with open(out, "w") as f:
            json.dump(self.records, f, indent=2)
        logger.info(f"Adapter stats → {out}")

        layer_rows = self._per_layer_stats()
        if layer_rows:
            layer_path = os.path.join(self.output_dir, "adapter_layer_stats.json")
            with open(layer_path, "w") as f:
                json.dump(layer_rows, f, indent=2)
            logger.info(f"Per-layer adapter stats → {layer_path}  ({len(layer_rows)} layers)")

    def _per_layer_stats(self) -> list:
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
    def __init__(self, model: torch.nn.Module, output_dir: str, method: str):
        self.model = model
        self.output_dir = output_dir
        self.method = method

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.method == "dora":
            out = os.path.join(self.output_dir, "dora_adapter_latest.pt")
            DoRAStateManager.save_dora_state(self.model, out, include_base_weights=False)
        elif self.method == "lora":
            lora_state = {
                name: {"lora_A": m.lora_A.data, "lora_B": m.lora_B.data}
                for name, m in self.model.named_modules()
                if isinstance(m, LoRALinear)
            }
            out = os.path.join(self.output_dir, "lora_adapter_latest.pt")
            torch.save({"lora_state": lora_state}, out)


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.target_modules is None:
        args.target_modules = DEFAULT_TARGET_MODULES

    size_tag = args.model_name.split("/")[-1]
    if args.output_dir is None:
        run_tag = f"{args.method}_r{args.rank}" if args.method != "full" else "full"
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..",
            "results",
            f"speech_commands_{size_tag}_{run_tag}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    use_mps = (
        not args.no_mps
        and not torch.cuda.is_available()
        and torch.backends.mps.is_available()
    )
    if use_mps:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        logger.info("Using MPS (Apple GPU)")

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    dataset, id2label = prepare_datasets(args, feature_extractor)
    label2id = {label: i for i, label in id2label.items()}

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(id2label),
        label2id=label2id,
        id2label=id2label,
        torch_dtype=dtype,
        ignore_mismatched_sizes=True,
    )

    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()

    logger.info(
        f"Model: {args.model_name} | Method: {args.method} | "
        f"rank={args.rank} alpha={args.alpha} target_modules={args.target_modules}"
    )

    if args.method == "dora":
        apply_dora_to_model(
            model.wav2vec2,
            target_modules=args.target_modules,
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
        )
        freeze_base_weights(model)
    elif args.method == "lora":
        apply_lora_to_model(
            model.wav2vec2,
            target_modules=args.target_modules,
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
        )
        freeze_base_weights(model)
    else:
        for param in model.parameters():
            param.requires_grad = True
        if args.lr == 2e-4:
            args.lr = 2e-5
            logger.info(f"Full fine-tuning: auto-setting lr={args.lr}")

    param_stats = count_parameters(model)
    logger.info(
        f"[{args.method.upper()}] Trainable: {param_stats['trainable_parameters']:,} / "
        f"{param_stats['total_parameters']:,} ({param_stats['trainable_percent']:.3f}%)"
    )

    train_size = len(dataset["train"])
    effective_batch = args.batch_size * args.grad_accum
    total_steps = max(1, int(np.ceil(train_size / effective_batch)) * args.epochs)
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))

    try:
        import wandb as _wandb  # noqa: F401

        wandb_available = True
    except ImportError:
        wandb_available = False

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
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="wandb" if (args.wandb and wandb_available) else "none",
        run_name=f"speech_commands_{size_tag}_{args.method}_r{args.rank}",
        logging_steps=50,
        seed=args.seed,
        dataloader_num_workers=0 if use_mps else 2,
        dataloader_pin_memory=not use_mps,
        remove_unused_columns=False,
    )

    callbacks = []
    if args.method in ("dora", "lora"):
        callbacks.append(AdapterStatsCallback(model, args.output_dir))
        callbacks.append(AdapterCheckpointCallback(model, args.output_dir, args.method))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=feature_extractor,
        data_collator=DataCollatorAudioClassification(feature_extractor),
        compute_metrics=build_compute_metrics(),
        callbacks=callbacks,
    )

    logger.info("Training...")
    start = time.perf_counter()
    train_output = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    training_time_seconds = time.perf_counter() - start

    logger.info("Evaluating validation split...")
    validation_metrics = trainer.evaluate(
        eval_dataset=dataset["validation"], metric_key_prefix="eval"
    )

    logger.info("Evaluating test split...")
    test_metrics = trainer.evaluate(eval_dataset=dataset["test"], metric_key_prefix="test")

    if args.method == "dora":
        DoRAStateManager.save_dora_state(
            model.wav2vec2,
            os.path.join(args.output_dir, "dora_adapter.pt"),
            include_base_weights=False,
        )
    elif args.method == "lora":
        save_lora_adapter(model.wav2vec2, os.path.join(args.output_dir, "lora_adapter.pt"))
    else:
        model.save_pretrained(os.path.join(args.output_dir, "full_model"))
        feature_extractor.save_pretrained(os.path.join(args.output_dir, "full_model"))

    torch.save(
        {
            "projector": model.projector.state_dict() if hasattr(model, "projector") else None,
            "classifier": model.classifier.state_dict() if hasattr(model, "classifier") else None,
        },
        os.path.join(args.output_dir, "classification_head.pt"),
    )

    report = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "label_mode": args.label_mode,
        "labels": id2label,
        "method": args.method,
        "rank": args.rank if args.method != "full" else None,
        "alpha": args.alpha if args.method != "full" else None,
        "target_modules": args.target_modules if args.method != "full" else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "train_size": len(dataset["train"]),
        "validation_size": len(dataset["validation"]),
        "test_size": len(dataset["test"]),
        "training_time_seconds": training_time_seconds,
        "training_time_minutes": training_time_seconds / 60.0,
        "train_loss": metric(train_output.metrics, "train_loss"),
        "validation_accuracy": metric(validation_metrics, "eval_accuracy"),
        "validation_loss": metric(validation_metrics, "eval_loss"),
        "test_accuracy": metric(test_metrics, "test_accuracy"),
        "test_loss": metric(test_metrics, "test_loss"),
        **param_stats,
    }

    report_path = os.path.join(args.output_dir, "metrics.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Metrics report saved -> {report_path}")
    logger.info(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
