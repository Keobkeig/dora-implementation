#!/usr/bin/env bash
# Run all 3 SmolVLM Push-T VLA experiments sequentially.
# DoRA vs LoRA vs full fine-tuning on lerobot/pusht.

set -e
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

run() {
    echo "========================================="
    echo "START: $*"
    echo "========================================="
    uv run scripts/train_vla.py "$@"
    echo "DONE: $*"
    echo ""
}

run --method dora --bf16
run --method lora --bf16
run --method full --bf16

echo "All 3 SmolVLM Push-T experiments complete."
echo "Results in results/vla_pusht_{dora_r8, lora_r8, full}/"
