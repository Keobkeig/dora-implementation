#!/usr/bin/env bash
# Runs all 9 RoBERTa comparison experiments sequentially (single GPU).
# DoRA vs LoRA vs Full Fine-Tuning on SST-2, RTE, MRPC.
# Est. total time: ~1-2 hours.

set -e
cd "$(dirname "$0")/.."

run() {
    echo "========================================="
    echo "START: $*"
    echo "========================================="
    uv run scripts/train_glue.py "$@"
    echo "DONE: $*"
    echo ""
}

# --- RTE (~2.5k train, fastest) ---
run --model roberta --task rte --method dora --bf16
run --model roberta --task rte --method lora --bf16
run --model roberta --task rte --method full --bf16

# --- MRPC (~3.7k train) ---
run --model roberta --task mrpc --method dora --bf16
run --model roberta --task mrpc --method lora --bf16
run --model roberta --task mrpc --method full --bf16

# --- SST-2 (~67k train, longest) ---
run --model roberta --task sst2 --method dora --bf16
run --model roberta --task sst2 --method lora --bf16
run --model roberta --task sst2 --method full --bf16

echo "All 9 RoBERTa experiments complete."
