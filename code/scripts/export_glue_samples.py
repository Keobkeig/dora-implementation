#!/usr/bin/env python3
"""
Export a few human-readable GLUE samples for poster visuals.
"""

import argparse
import json
import os
import random
import sys

from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_LABEL_MAP = {
    "sst2": {0: "negative", 1: "positive"},
    "mrpc": {0: "not_equivalent", 1: "equivalent"},
    "rte": {0: "not_entailment", 1: "entailment"},
}


def _get_fields(task: str):
    if task == "sst2":
        return ("sentence", None)
    if task == "mrpc":
        return ("sentence1", "sentence2")
    if task == "rte":
        return ("sentence1", "sentence2")
    raise ValueError(f"Unsupported task: {task}")


def main():
    ap = argparse.ArgumentParser(description="Export GLUE text samples")
    ap.add_argument("--task", choices=["sst2", "mrpc", "rte"], required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--num_samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    ds = load_dataset("glue", args.task)[args.split]
    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[: args.num_samples]

    key1, key2 = _get_fields(args.task)
    label_map = _LABEL_MAP.get(args.task, {})

    samples = []
    for idx in indices:
        row = ds[idx]
        label = row.get("label", None)
        record = {
            "task": args.task,
            "split": args.split,
            "index": idx,
            key1: row.get(key1, ""),
            "label_id": label,
            "label": label_map.get(label, str(label)),
        }
        if key2 is not None:
            record[key2] = row.get(key2, "")
        samples.append(record)

    output = args.output
    if output is None:
        output = os.path.join("..", "results", f"glue_{args.task}_samples.json")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Wrote {len(samples)} samples to {output}")


if __name__ == "__main__":
    main()
