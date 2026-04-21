# CS4782 Final Project: DoRA Reproduction

Team repository for the CS4782 final project on **DoRA (Weight-Decomposed Low-Rank Adaptation)**.

We reproduce DoRA fine-tuning of LLaMA models on the GLUE benchmark and compare against LoRA baselines.

## Repository Layout

```
dora_implementation/
├── code/                    # All source code
│   ├── dora/                # Core DoRA package (layers, models, utils)
│   ├── scripts/             # Training entry points (train_glue.py, etc.)
│   ├── training/            # DoRA trainer and training utilities
│   ├── benchmarks/          # LoRA vs DoRA benchmarking
│   ├── configs/             # YAML configs per model size
│   │   └── models/          # llama_1b.yaml, llama_3b.yaml, llama_7b.yaml
│   ├── tests/               # Unit tests
│   ├── notebooks/           # Analysis notebooks
│   ├── demo/                # Gradio demo app
│   ├── pyproject.toml       # Canonical dependency + tooling config
│   └── uv.lock              # Fully locked environment
├── data/                    # Datasets (see data/README or auto-downloaded)
├── results/                 # Training outputs, checkpoints, figures
├── poster/                  # Poster PDF and LaTeX source
├── report/                  # Final 2-page report PDF
├── LICENSE
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) for dependency management

```bash
uv python install 3.11
```

### Install

```bash
cd code
uv sync
```

**GPU support is automatic:**
- **Mac (Apple Silicon):** installs CPU+MPS wheel; MPS is used automatically
- **Linux / Windows (NVIDIA):** installs CUDA 12.8 wheel (RTX 4000+ / 5000 series)

### Run GLUE fine-tuning

```bash
cd code

# SST-2 with TinyLlama-1.1B (fast smoke test)
uv run scripts/train_glue.py --model 1b --task sst2

# MRPC with OpenLLaMA-3B
uv run scripts/train_glue.py --model 3b --task mrpc --rank 16 --alpha 32

# All attention + MLP projections, with W&B logging
uv run scripts/train_glue.py --model 1b --task rte \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --wandb
```

### Model presets

| Flag | Model | Params |
|------|-------|--------|
| `--model 1b` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B |
| `--model 3b` | openlm-research/open_llama_3b | 3B |
| `--model 7b` | huggyllama/llama-7b | 7B |

Or pass any HuggingFace model ID directly with `--model_name`.

### Supported GLUE tasks

| Task | Type | Metric |
|------|------|--------|
| `sst2` | Sentiment | Accuracy |
| `cola` | Acceptability | MCC |
| `mrpc` | Paraphrase | F1 + Accuracy |
| `qqp` | Duplicate Qs | F1 + Accuracy |
| `mnli` | NLI (3-class) | Accuracy |
| `qnli` | NLI | Accuracy |
| `rte` | NLI | Accuracy |
| `wnli` | NLI | Accuracy |
| `stsb` | Similarity | Pearson + Spearman |

### Key training options

```
--rank INT          LoRA rank (default: 8)
--alpha FLOAT       LoRA alpha (default: 16.0)
--dropout FLOAT     Dropout on LoRA path (default: 0.05)
--target_modules    Linear layers to adapt (default: q k v o projections)
--epochs INT        Training epochs (default: 5)
--batch_size INT    Per-device batch size (default: 16)
--lr FLOAT          Learning rate (default: 2e-4)
--bf16              Use bfloat16 (recommended on CUDA)
--wandb             Enable Weights & Biases logging
--output_dir PATH   Override output directory
```

Results are saved to `../results/glue_<task>_<model>_r<rank>/` by default.

## Tests

```bash
cd code
uv run pytest -q
```

## Common Commands

```bash
# Lint + format
cd code
uv run ruff check .
uv run black .

# Run demo app
uv run python demo/gradio_app.py
```

## Project Scope

- DoRA fine-tuning of LLaMA models (1B, 3B, 7B) on GLUE benchmark
- Comparison against LoRA baselines across rank settings
- Parameter efficiency analysis (trainable % vs. performance)

## Collaboration Workflow

- Create feature branches from `main`
- Run lint/tests before pushing (`ruff`, `black`, `pytest`)
- Keep notebooks in `code/notebooks/` and avoid committing large outputs
- Save training results to `results/` with descriptive run names

See `code/CONTRIBUTING.md` for detailed standards.

## Notes

- The lockfile is treated as source of truth for reproducibility
- Python version is pinned via `code/.python-version` and `pyproject.toml`
- DoRA adapter weights (not full model) are saved at end of each run
