#!/usr/bin/env python3
"""
Export audio samples from Speech Commands with DoRA/LoRA/Full predictions.

Saves WAV files + manifest.json to --output_dir.
Loads the model from --checkpoint_dir (HF Trainer checkpoint with saved weights)
or --adapter_path (dora_adapter.pt / lora_adapter.pt).
If neither is provided, exports ground-truth samples only (no predictions).

Usage:
    uv run scripts/export_speech_commands_samples.py \
        --checkpoint_dir ../results/speech_commands_wav2vec2-base_dora_r8/checkpoint-XXXX \
        --method dora --num_samples 20

    uv run scripts/export_speech_commands_samples.py \
        --adapter_path ../results/speech_commands_wav2vec2-base_dora_r8/dora_adapter.pt \
        --method dora --num_samples 20
"""

import argparse
import io
import json
import math
import os
import random
import sys

import numpy as np
import torch
from datasets import Audio, load_dataset
from scipy.io import wavfile
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dora.layers.base import DoRAStateManager
from dora.layers.lora_linear import LoRALinear, apply_lora_to_model
from dora.models.llama import apply_dora_to_model

DEFAULT_MODEL_NAME = "facebook/wav2vec2-base"
DEFAULT_DATASET_NAME = "google/speech_commands"
DEFAULT_DATASET_CONFIG = "v0.02"
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]
KWS_LABELS = [
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go", "_unknown_", "_silence_",
]
SAMPLE_RATE = 16_000


def parse_args():
    ap = argparse.ArgumentParser(description="Export Speech Commands audio samples")
    ap.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    ap.add_argument("--dataset_config", default=DEFAULT_DATASET_CONFIG)
    ap.add_argument("--method", choices=["dora", "lora", "full"], default="dora")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--target_modules", nargs="+", default=None)

    # Model weights — one of these is used if provided
    ap.add_argument("--checkpoint_dir", default=None,
                    help="Path to a HF Trainer checkpoint dir containing model weights")
    ap.add_argument("--adapter_path", default=None,
                    help="Path to dora_adapter.pt or lora_adapter.pt")
    ap.add_argument("--head_path", default=None,
                    help="Path to classification_head.pt (used alongside adapter_path)")

    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--split", choices=["train", "validation", "test"], default="test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label_mode", choices=["kws12", "full"], default="kws12")
    return ap.parse_args()


def read_audio_array(audio: dict, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    if isinstance(audio, dict) and "array" in audio:
        array = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
    else:
        source = io.BytesIO(audio["bytes"]) if audio.get("bytes") is not None else audio["path"]
        sr, array = wavfile.read(source)

    if array.ndim > 1:
        array = np.mean(array, axis=1)
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float32) / float(np.iinfo(array.dtype).max)
    else:
        array = array.astype(np.float32)

    if sr != target_rate:
        divisor = math.gcd(sr, target_rate)
        array = resample_poly(array, target_rate // divisor, sr // divisor)

    return np.asarray(array, dtype=np.float32)


def save_wav(path: str, array: np.ndarray, rate: int = SAMPLE_RATE):
    pcm = (np.clip(array, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(path, rate, pcm)


def _load_lora_weights(model: torch.nn.Module, path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    lora_state = payload.get("lora_state", {})
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and name in lora_state:
            s = lora_state[name]
            module.lora_A.data.copy_(s["lora_A"])
            module.lora_B.data.copy_(s["lora_B"])


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.target_modules is None:
        args.target_modules = DEFAULT_TARGET_MODULES

    size_tag = args.model_name.split("/")[-1]
    if args.output_dir is None:
        run_tag = f"{args.method}_r{args.rank}" if args.method != "full" else "full"
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..",
            "results",
            f"speech_commands_{size_tag}_{run_tag}_samples",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)

    # Build label maps
    print(f"Loading dataset {args.dataset_name}/{args.dataset_config}...")
    raw = load_dataset(args.dataset_name, args.dataset_config, trust_remote_code=True)
    raw_label_feature = raw["train"].features["label"]
    if args.label_mode == "kws12":
        target_to_id = {lbl: i for i, lbl in enumerate(KWS_LABELS)}
        raw_to_label = {}
        for raw_id in range(raw_label_feature.num_classes):
            lbl = raw_label_feature.int2str(raw_id)
            raw_to_label[raw_id] = lbl if lbl in target_to_id else "_unknown_"
        id2label = {i: lbl for i, lbl in enumerate(KWS_LABELS)}
    else:
        id2label = {i: raw_label_feature.int2str(i) for i in range(raw_label_feature.num_classes)}
        raw_to_label = {i: raw_label_feature.int2str(i) for i in id2label}

    # Determine if we have a model to run inference
    has_weights = args.checkpoint_dir is not None or args.adapter_path is not None
    model = None

    if has_weights:
        load_path = args.checkpoint_dir if args.checkpoint_dir else args.model_name
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            load_path,
            num_labels=len(id2label),
            id2label=id2label,
            label2id={v: k for k, v in id2label.items()},
            ignore_mismatched_sizes=True,
        )
        if hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()

        if args.adapter_path and args.method in ("dora", "lora"):
            if args.method == "dora":
                apply_dora_to_model(
                    model.wav2vec2,
                    target_modules=args.target_modules,
                    rank=args.rank,
                    alpha=args.alpha,
                    dropout=0.0,
                )
                DoRAStateManager.load_dora_state(model.wav2vec2, args.adapter_path, strict=True)
            else:
                apply_lora_to_model(
                    model.wav2vec2,
                    target_modules=args.target_modules,
                    rank=args.rank,
                    alpha=args.alpha,
                    dropout=0.0,
                )
                _load_lora_weights(model.wav2vec2, args.adapter_path)

            if args.head_path:
                head_state = torch.load(args.head_path, map_location="cpu", weights_only=False)
                if head_state.get("projector") and hasattr(model, "projector"):
                    model.projector.load_state_dict(head_state["projector"])
                if head_state.get("classifier") and hasattr(model, "classifier"):
                    model.classifier.load_state_dict(head_state["classifier"])

        model.eval()
        print(f"Model loaded. Running inference on {args.num_samples} {args.split} samples.")
    else:
        print(f"No model weights provided — exporting ground-truth samples only.")

    # Sample from split
    ds = raw[args.split].cast_column("audio", Audio(sampling_rate=SAMPLE_RATE, decode=True))
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[: args.num_samples]

    manifest = []
    for i, idx in enumerate(indices):
        sample = ds[idx]
        array = read_audio_array(sample["audio"])
        true_label = raw_to_label[int(sample["label"])]

        wav_name = f"sample_{i:02d}_{true_label}.wav"
        wav_path = os.path.join(args.output_dir, wav_name)
        save_wav(wav_path, array)

        entry = {
            "filename": wav_name,
            "true_label": true_label,
            "predicted_label": None,
            "confidence": None,
        }

        if model is not None:
            inputs = feature_extractor(
                array,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            pred_id = int(probs.argmax().item())
            entry["predicted_label"] = id2label[pred_id]
            entry["confidence"] = round(float(probs[pred_id].item()), 4)

        manifest.append(entry)
        status = (
            f"  ✓ {true_label} → {entry['predicted_label']} ({entry['confidence']:.2%})"
            if model is not None
            else f"  GT: {true_label}"
        )
        print(f"[{i+1:02d}/{len(indices)}] {wav_name}{status}")

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if model is not None:
        correct = sum(1 for e in manifest if e["true_label"] == e["predicted_label"])
        print(f"\nSample accuracy: {correct}/{len(manifest)} = {correct/len(manifest):.1%}")
    print(f"Saved {len(manifest)} samples + manifest → {args.output_dir}")


if __name__ == "__main__":
    main()
