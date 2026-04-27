#!/usr/bin/env bash
# Runs all Cornell Grasp experiments sequentially (single GPU).
#
# Experiment groups:
#   ViT-Base (86M)  — pure vision baseline
#   SigLIP (400M)   — OpenVLA's visual encoder
#
# Usage:
#   bash scripts/run_grasp_experiments.sh --data_dir ../data/cornell_grasps
#
# Download dataset first (PowerShell):
#   uv run scripts/download_cornell_grasp.py

set -e
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"   # ensure uv is on bash PATH (Windows)

DATA_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir) DATA_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATA_DIR" ]]; then
    echo "Usage: $0 --data_dir /path/to/cornell_grasps"
    exit 1
fi

run() {
    echo "========================================="
    echo "START: $*"
    echo "========================================="
    uv run scripts/train_grasp.py "$@"
    echo "DONE: $*"
    echo ""
}

# --- ViT-Base (86M) ---
echo "### ViT-Base experiments ###"
run --model vit --data_dir "$DATA_DIR" --method dora --bf16
run --model vit --data_dir "$DATA_DIR" --method lora --bf16
run --model vit --data_dir "$DATA_DIR" --method full --bf16

# --- SigLIP / OpenVLA visual encoder (400M) ---
echo "### SigLIP (OpenVLA visual encoder) experiments ###"
run --model siglip --data_dir "$DATA_DIR" --method dora --bf16
run --model siglip --data_dir "$DATA_DIR" --method lora --bf16
run --model siglip --data_dir "$DATA_DIR" --method full --bf16

echo "All 6 Cornell Grasp experiments complete."
