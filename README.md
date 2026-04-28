# CS4782 Final Project: DoRA Reproduction

Reproduction and extension of **DoRA (Weight-Decomposed Low-Rank Adaptation)** from scratch in PyTorch.  
We verify original GLUE results, compare DoRA vs LoRA vs full fine-tuning across model scales, and extend DoRA to vision and robotics domains.

> Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" — arXiv:2402.09353

---

## Results

### 1 · GLUE — Method Comparison (RoBERTa-base, 125M params)

Hyperparameters: rank=8, α=16, target=`query/key/value`, 5 epochs, lr=2e-4 (adapters) / 2e-5 (full FT), bf16.

| Task | Train size | Metric | DoRA | LoRA | Full FT | Trainable |
|------|-----------:|--------|-----:|-----:|--------:|----------:|
| SST-2 | 67k | Accuracy | 93.1% | **93.3%** | 93.2% | ~1.0% |
| RTE | 2.5k | Accuracy | **71.1%** | 70.8% | 54.9% | ~1.0% |
| MRPC | 3.7k | F1 | **90.7%** | 90.1% | 81.2% | ~1.0% |
| MRPC | 3.7k | Accuracy | **87.0%** | 85.8% | 68.4% | ~1.0% |

**Finding:** Full fine-tuning collapses on small datasets (RTE, MRPC) — adapting 100% of a 125M model on 2.5k examples causes severe overfitting (RTE: 54.9% ≈ near-random). DoRA/LoRA's constrained parameter budget acts as implicit regularisation. On large data (SST-2, 67k) all three methods converge.

---

### 2 · GLUE — Scale Study (DoRA only, SST-2 Accuracy)

Hyperparameters: rank=8, α=16, target=`q_proj/k_proj/v_proj/o_proj`, 5 epochs, bf16.

| Model | Params | SST-2 Accuracy | Trainable % |
|-------|-------:|---------------:|------------:|
| TinyLlama-1.1B | 1.1B | **96.0%** | ~0.10% |
| OpenLLaMA-3B | 3.0B | 81.0%† | ~0.05% |

† 3B best checkpoint at epoch 2; performance degraded after — likely overfitting with these hyperparameters at this scale.

---

### 3 · Cornell Grasp — Vision Extension (ViT-Base / SigLIP)

Grasp pose regression from RGB image → `(x, y, sin2θ, cos2θ, w, h)`.  
Metric: IoU ≥ 0.25 **and** |Δangle| ≤ 30° (standard Cornell success rate).  
Dataset: 885 images, 708 train / 177 val, image-level split.

| Backbone | Params | Method | Trainable | Success Rate |
|----------|-------:|--------|----------:|-------------:|
| ViT-Base/16 | 87M | DoRA | ~1.0% | 7.9% |
| ViT-Base/16 | 87M | LoRA | ~1.0% | **6.2%** |
| ViT-Base/16 | 87M | Full FT | 100% | 0.6% |
| SigLIP-base/16 (OpenVLA encoder) | 93M | DoRA | ~1.0% | 19.8% |
| SigLIP-base/16 | 93M | LoRA | ~1.0% | **20.3%** |
| SigLIP-base/16 | 93M | Full FT | 100% | 15.8% |

---

### 4 · OpenVLA-7B — Architecture Verification

DoRA applied to OpenVLA-7B loaded in 4-bit NF4 quantisation (bitsandbytes).

| Strategy | Target layers | Adapter params (rank=8) | Trainable % |
|----------|--------------|------------------------:|------------:|
| full (LLM attention + MLP) | 224 layers | **21,348,352** | **0.28%** |

Total parameters: 7.54B. Forward pass verified on CPU. DoRA adapters add 21M parameters to a frozen 7.54B model — less than 0.3% overhead.

To replicate (requires ~14 GB RAM, weights already cached after first run):
```powershell
uv run scripts/openvla_demo.py 2>&1 | Tee-Object -FilePath ..\results\openvla_demo.log
```

---

## Reproducing All Experiments

> **Windows note:** All `uv run` commands must be run from **PowerShell**, not WSL.  
> WSL cannot modify the `.venv` created by uv on Windows (NTFS I/O restriction).

### 0 · Environment setup

```powershell
# Install uv (once, if not already installed)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install Python 3.11 and all dependencies
uv python install 3.11
cd code
uv sync
```

---

### 1 · Scale Study — LLaMA 1B and 3B on SST-2

These were the first runs — DoRA on two LLaMA-family models to validate the approach.

```powershell
# From code/ directory
uv run scripts/train_glue.py --model 1b --task sst2 --bf16
uv run scripts/train_glue.py --model 3b --task sst2 --bf16
```

Results saved to `results/glue_sst2_1b_r8/` and `results/glue_sst2_3b_r8/`.

---

### 2 · Method Comparison — RoBERTa on GLUE (9 runs)

All three methods × three tasks, run sequentially via the batch script:

```powershell
# From code/ directory (~1.5 h total)
bash scripts/run_roberta_experiments.sh
```

Which is equivalent to running each of these individually:

```powershell
# RTE (~5 min per run)
uv run scripts/train_glue.py --model roberta --task rte --method dora --bf16
uv run scripts/train_glue.py --model roberta --task rte --method lora --bf16
uv run scripts/train_glue.py --model roberta --task rte --method full --bf16

# MRPC (~10 min per run)
uv run scripts/train_glue.py --model roberta --task mrpc --method dora --bf16
uv run scripts/train_glue.py --model roberta --task mrpc --method lora --bf16
uv run scripts/train_glue.py --model roberta --task mrpc --method full --bf16

# SST-2 (~25 min per run)
uv run scripts/train_glue.py --model roberta --task sst2 --method dora --bf16
uv run scripts/train_glue.py --model roberta --task sst2 --method lora --bf16
uv run scripts/train_glue.py --model roberta --task sst2 --method full --bf16
```

Results saved to `results/glue_<task>_roberta_<dora_r8|lora_r8|full>/`.

---

### 3 · Vision Extension — Cornell Grasp Dataset (6 runs)

#### Step 1 — Download the dataset

Option A — Kaggle API (requires free [Kaggle account](https://www.kaggle.com/settings) + API token at `~/.kaggle/kaggle.json`):
```powershell
uv run scripts/download_cornell_grasp.py
```

Option B — Manual browser download from `https://www.kaggle.com/datasets/oneoneliu/cornell-grasp`, then:
```powershell
uv run scripts/download_cornell_grasp.py --zip_path C:\Users\<you>\Downloads\cornell-grasp.zip
```

Dataset lands in `data/cornell_grasps/` (gitignored).

#### Step 2 — Run all 6 experiments

```powershell
# From code/ directory (~2 h total)
bash scripts/run_grasp_experiments.sh --data_dir ../data/cornell_grasps
```

Which is equivalent to:

```powershell
# ViT-Base/16 (86M — pure vision baseline, ~15 min per run)
uv run scripts/train_grasp.py --model vit --data_dir ../data/cornell_grasps --method dora --bf16
uv run scripts/train_grasp.py --model vit --data_dir ../data/cornell_grasps --method lora --bf16
uv run scripts/train_grasp.py --model vit --data_dir ../data/cornell_grasps --method full --bf16

# SigLIP-base/16 (OpenVLA visual encoder, ~20 min per run)
uv run scripts/train_grasp.py --model siglip --data_dir ../data/cornell_grasps --method dora --bf16
uv run scripts/train_grasp.py --model siglip --data_dir ../data/cornell_grasps --method lora --bf16
uv run scripts/train_grasp.py --model siglip --data_dir ../data/cornell_grasps --method full --bf16
```

Results saved to `results/grasp_<vit|siglip>_<dora_r8|lora_r8|full>/`.

---

### 4 · OpenVLA Architecture Demo

Loads OpenVLA-7B in 4-bit NF4 quantisation, identifies DoRA target layers across the visual encoder (SigLIP) and LLM backbone (LLaMA-2), reports theoretical trainable-parameter counts, and runs a forward pass.

**First run downloads ~14 GB. Requires ~8 GB VRAM with 4-bit quant.**

```powershell
# From code/ directory (~5-10 min, includes download on first run)
uv run scripts/openvla_demo.py 2>&1 | Tee-Object -FilePath ..\results\openvla_demo.log
```

Report saved to `results/openvla_dora_report.txt`.

---

## Repository Layout

```
dora-implementation/
├── code/
│   ├── dora/                      # Core library
│   │   ├── layers/                #   dora_linear.py, lora_linear.py, base.py
│   │   ├── models/                #   llama.py, vision_transformer.py, vla.py
│   │   ├── data/                  #   cornell_grasp.py
│   │   └── utils/                 #   math_utils.py, model_utils.py
│   ├── scripts/
│   │   ├── train_glue.py          # GLUE fine-tuning (LLaMA / RoBERTa)
│   │   ├── train_grasp.py         # Cornell Grasp (ViT / SigLIP)
│   │   ├── openvla_demo.py        # OpenVLA architecture verification
│   │   ├── run_roberta_experiments.sh
│   │   ├── run_grasp_experiments.sh
│   │   └── download_cornell_grasp.py
│   ├── benchmarks/                # DoRA vs LoRA micro-benchmarks
│   ├── configs/models/            # llama_1b/3b/7b.yaml
│   ├── training/                  # Custom trainer for commonsense tasks
│   ├── tests/                     # pytest unit tests
│   └── demo/                      # Gradio interactive demo
├── data/                          # Datasets — gitignored, download separately
├── results/                       # Checkpoints and metrics — gitignored
├── poster/
└── report/
```

---

## Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `--method` | `dora` | `dora` · `lora` · `full` |
| `--rank` | `8` | LoRA/DoRA rank |
| `--alpha` | `16.0` | LoRA/DoRA alpha (scaling = α/r) |
| `--bf16` | off | bfloat16 mixed precision — always use on CUDA |
| `--target_modules` | auto | attention layer names to adapt (auto-detected per architecture) |
| `--epochs` | `5` (GLUE) · `30` (grasp) | training epochs |
| `--wandb` | off | enable Weights & Biases logging |

### Model presets (`--model`)

| Flag | HuggingFace ID | Params |
|------|---------------|-------:|
| `1b` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B |
| `3b` | openlm-research/open_llama_3b | 3.0B |
| `7b` | huggyllama/llama-7b | 7.0B |
| `roberta` | FacebookAI/roberta-base | 125M |
| `vit` | google/vit-base-patch16-224 | 87M |
| `siglip` | google/siglip-base-patch16-224 | 93M |

---

## Tests

```powershell
cd code
uv run pytest -q
```

## Demo

```powershell
cd code
uv run python demo/gradio_app.py
```
