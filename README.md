# CS4782 Final Project: DoRA Reproduction

## 1. Introduction

This GitHub repository contains our CS 4782 final project: a from-scratch PyTorch re-implementation and extension of **DoRA (Weight-Decomposed Low-Rank Adaptation)**.

DoRA (Liu et al., 2024) modifies LoRA by decoupling weight magnitude from low-rank directional updates, aiming to improve parameter-efficient fine-tuning at the same adapter rank and parameter budget.

> Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" — ICML 2024, arXiv:2402.09353

## 2. Chosen Result

We targeted DoRA's central empirical claim: at equal rank, DoRA should match or outperform LoRA on GLUE, with the largest gains on low-data tasks where gradient interference is highest.

The primary result corresponds to the DoRA paper's GLUE comparison tables for LoRA vs. DoRA; we reproduce this claim on SST-2, MRPC, and RTE, then extend the same comparison to audio, vision, and robotics tasks.

## 3. GitHub Contents

```
dora-implementation/
├── README.md                      # Project summary and reproduction guide
├── code/                          # Re-implementation code, configs, scripts, tests, demo
├── data/                          # Dataset acquisition notes; raw datasets are not committed
├── results/                       # Metrics, logs, trainer states, generated examples
├── poster/                        # In-class poster PDF and assets
├── report/                        # Final project report PDF and source
├── LICENSE
└── .gitignore
```

Important code paths:

```
code/
├── dora/
│   ├── layers/                    # dora_linear.py, lora_linear.py, base.py
│   ├── models/                    # llama.py, vla.py
│   ├── data/                      # cornell_grasp.py, lerobot_dataset.py
│   └── utils/                     # math_utils.py, model_utils.py
├── scripts/
│   ├── train_glue.py              # GLUE fine-tuning
│   ├── train_grasp.py             # Cornell Grasp
│   ├── train_speech_commands.py   # Wav2Vec2 keyword spotting
│   ├── train_vla.py               # Push-T VLA action prediction
│   ├── openvla_demo.py            # OpenVLA architecture verification
│   ├── run_roberta_experiments.sh
│   ├── run_grasp_experiments.sh
│   └── download_cornell_grasp.py
├── configs/
├── tests/
└── demo/
```

## 4. Re-implementation Details

We implement `DoRALinear`, a drop-in replacement for `nn.Linear` with frozen base weights, low-rank LoRA matrices, and a learnable magnitude vector initialized from the pretrained weight norm. A matched `LoRALinear` baseline lets us compare DoRA vs. LoRA at the same rank and target modules.

Experiments cover four modalities:

| Modality | Models | Dataset | Metric |
|----------|--------|---------|--------|
| NLP | RoBERTa-base/large, TinyLlama-1.1B, OpenLLaMA-3B | GLUE SST-2/MRPC/RTE | Accuracy, F1 |
| Audio | Wav2Vec2-base | Google Speech Commands v0.02 | Validation/test accuracy |
| Vision | ViT-B/16, SigLIP-B/16 | Cornell Grasp | Cornell success rate / IoU |
| Robotics | SmolVLM / OpenVLA-style VLA stack | LeRobot Push-T | Action MSE |

Key modifications from the original paper: we extend DoRA beyond NLP, include full fine-tuning baselines where feasible, track adapter statistics across epochs, and evaluate small-data behavior where full fine-tuning tends to overfit.

Datasets are not committed. GLUE, Speech Commands, and Push-T are downloaded into local Hugging Face/LeRobot caches by the scripts; Cornell Grasp is downloaded separately into `data/cornell_grasps/`. See `data/README.md`.

## 5. Reproduction Steps

### Environment setup

```powershell
# Install uv once, if needed
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install Python 3.11 and all dependencies
uv python install 3.11
cd code
uv sync
```

Windows note: all `uv run` commands should be run from PowerShell. WSL may not be able to modify the `.venv` created by uv on Windows.

Recommended compute: CUDA GPU with bf16 support for the main runs. GLUE RoBERTa runs take roughly 5-25 minutes each; Cornell Grasp takes roughly 2 hours for the 6-run sweep; Wav2Vec2 takes roughly 3 hours; OpenVLA architecture verification downloads about 14 GB and needs roughly 8 GB VRAM with 4-bit quantization or enough CPU RAM for CPU loading.

### Scale study: LLaMA-family models on SST-2

```powershell
# From code/ directory
uv run scripts/train_glue.py --model 1b --task sst2 --bf16
uv run scripts/train_glue.py --model 3b --task sst2 --bf16
```

Results are saved to `results/glue_sst2_1b_r8/` and `results/glue_sst2_3b_r8/`.

### Method comparison: RoBERTa on GLUE

```powershell
# From code/ directory; runs DoRA, LoRA, and full fine-tuning on RTE/MRPC/SST-2
bash scripts/run_roberta_experiments.sh
```

Equivalent individual commands:

```powershell
uv run scripts/train_glue.py --model roberta --task rte --method dora --bf16
uv run scripts/train_glue.py --model roberta --task rte --method lora --bf16
uv run scripts/train_glue.py --model roberta --task rte --method full --bf16

uv run scripts/train_glue.py --model roberta --task mrpc --method dora --bf16
uv run scripts/train_glue.py --model roberta --task mrpc --method lora --bf16
uv run scripts/train_glue.py --model roberta --task mrpc --method full --bf16

uv run scripts/train_glue.py --model roberta --task sst2 --method dora --bf16
uv run scripts/train_glue.py --model roberta --task sst2 --method lora --bf16
uv run scripts/train_glue.py --model roberta --task sst2 --method full --bf16
```

Results are saved to `results/glue_<task>_roberta_<dora_r8|lora_r8|full>/`.

### Rank robustness

```powershell
uv run scripts/train_glue.py --model roberta --task sst2 --method dora --rank 2  --alpha 4  --bf16
uv run scripts/train_glue.py --model roberta --task sst2 --method dora --rank 4  --alpha 8  --bf16
uv run scripts/train_glue.py --model roberta --task sst2 --method dora --rank 8  --alpha 16 --bf16
uv run scripts/train_glue.py --model roberta --task sst2 --method dora --rank 16 --alpha 32 --bf16

uv run scripts/train_glue.py --model roberta --task sst2 --method lora --rank 2  --alpha 4  --bf16
uv run scripts/train_glue.py --model roberta --task sst2 --method lora --rank 4  --alpha 8  --bf16
uv run scripts/train_glue.py --model roberta --task sst2 --method lora --rank 8  --alpha 16 --bf16
uv run scripts/train_glue.py --model roberta --task sst2 --method lora --rank 16 --alpha 32 --bf16

uv run scripts/export_glue_metrics.py --results_dir ../results --output ../results/glue_run_summaries.json
```

### Cornell Grasp

Download the dataset:

```powershell
# From code/ directory; requires Kaggle API token at ~/.kaggle/kaggle.json
uv run scripts/download_cornell_grasp.py
```

Or download the Kaggle archive manually from `https://www.kaggle.com/datasets/oneoneliu/cornell-grasp` and pass it explicitly:

```powershell
uv run scripts/download_cornell_grasp.py --zip_path C:\Users\<you>\Downloads\cornell-grasp.zip
```

Run the experiments:

```powershell
bash scripts/run_grasp_experiments.sh --data_dir ../data/cornell_grasps
```

Equivalent individual commands:

```powershell
uv run scripts/train_grasp.py --model vit --data_dir ../data/cornell_grasps --method dora --bf16
uv run scripts/train_grasp.py --model vit --data_dir ../data/cornell_grasps --method lora --bf16
uv run scripts/train_grasp.py --model vit --data_dir ../data/cornell_grasps --method full --bf16

uv run scripts/train_grasp.py --model siglip --data_dir ../data/cornell_grasps --method dora --bf16
uv run scripts/train_grasp.py --model siglip --data_dir ../data/cornell_grasps --method lora --bf16
uv run scripts/train_grasp.py --model siglip --data_dir ../data/cornell_grasps --method full --bf16
```

Results are saved to `results/grasp_<vit|siglip>_<dora_r8|lora_r8|full>/`.

### Speech Commands: Wav2Vec2 keyword spotting

```powershell
# From code/ directory
uv run scripts/train_speech_commands.py --method dora --rank 8 --alpha 16
```

Fast Apple Silicon smoke run:

```powershell
uv run scripts/train_speech_commands.py `
  --method dora `
  --rank 8 `
  --alpha 16 `
  --epochs 1 `
  --batch_size 8 `
  --max_train_samples 2000 `
  --max_eval_samples 500 `
  --max_test_samples 500
```

Continue from checkpoint:

```powershell
uv run scripts/train_speech_commands.py `
  --method dora `
  --rank 8 `
  --alpha 16 `
  --epochs 2 `
  --resume_from_checkpoint ../results/speech_commands_wav2vec2-base_dora_r8/checkpoint-10606
```

The script reports validation accuracy, validation loss, test accuracy, trainable parameters, and measured training time. Results are saved to `results/speech_commands_wav2vec2-base_dora_r8/metrics.json`; adapter weights are saved to `dora_adapter.pt` and the Wav2Vec2 classification head is saved to `classification_head.pt`.

### OpenVLA architecture verification

```powershell
# From code/ directory; first run downloads ~14 GB
uv run scripts/openvla_demo.py 2>&1 | Tee-Object -FilePath ..\results\openvla_demo.log
```

Report is saved to `results/openvla_dora_report.txt`.

### Export samples for poster/demo

```powershell
uv run scripts/export_vla_samples.py --method dora --adapter_path ../results/vla_pusht_dora_r8/dora_adapter.pt --head_path ../results/vla_pusht_dora_r8/action_head.pt --num_samples 10 --output_dir ../results/pusht_samples_dora
uv run scripts/export_vla_samples.py --method lora --adapter_path ../results/vla_pusht_lora_r8/lora_adapter.pt --head_path ../results/vla_pusht_lora_r8/action_head.pt --num_samples 10 --output_dir ../results/pusht_samples_lora

uv run scripts/export_grasp_samples.py --model vit --data_dir ../data/cornell_grasps --method dora --adapter_path ../results/grasp_vit_dora_r8/dora_adapter.pt --head_path ../results/grasp_vit_dora_r8/grasp_head.pt --num_samples 10 --output_dir ../results/grasp_vit_samples_dora
uv run scripts/export_grasp_samples.py --model vit --data_dir ../data/cornell_grasps --method lora --adapter_path ../results/grasp_vit_lora_r8/lora_adapter.pt --head_path ../results/grasp_vit_lora_r8/grasp_head.pt --num_samples 10 --output_dir ../results/grasp_vit_samples_lora
uv run scripts/export_grasp_samples.py --model siglip --data_dir ../data/cornell_grasps --method dora --adapter_path ../results/grasp_siglip_dora_r8/dora_adapter.pt --head_path ../results/grasp_siglip_dora_r8/grasp_head.pt --num_samples 10 --output_dir ../results/grasp_siglip_samples_dora
uv run scripts/export_grasp_samples.py --model siglip --data_dir ../data/cornell_grasps --method lora --adapter_path ../results/grasp_siglip_lora_r8/lora_adapter.pt --head_path ../results/grasp_siglip_lora_r8/grasp_head.pt --num_samples 10 --output_dir ../results/grasp_siglip_samples_lora
```

### Tests and demo

```powershell
cd code
uv run pytest -q
uv run python demo/gradio_app.py
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--method` | `dora` | `dora` / `lora` / `full` |
| `--rank` | `8` | LoRA/DoRA rank |
| `--alpha` | `16.0` | LoRA/DoRA alpha |
| `--bf16` | off | bfloat16 mixed precision; use on CUDA |
| `--target_modules` | auto | Attention layer names to adapt |
| `--epochs` | `5` GLUE / `30` grasp | Training epochs |
| `--wandb` | off | Enable Weights & Biases logging |

Model presets:

| Flag | Hugging Face ID | Params |
|------|-----------------|-------:|
| `1b` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B |
| `3b` | `openlm-research/open_llama_3b` | 3.0B |
| `7b` | `huggyllama/llama-7b` | 7.0B |
| `roberta` | `FacebookAI/roberta-base` | 125M |
| `vit` | `google/vit-base-patch16-224` | 87M |
| `siglip` | `google/siglip-base-patch16-224` | 93M |
| `wav2vec2-base` | `facebook/wav2vec2-base` | 95M |

## 6. Results/Insights

### NLP: GLUE method comparison

DoRA's advantage is largest on low-data NLP tasks, matching the poster finding that magnitude decoupling helps most when data is scarce.

| Task | Train size | Metric | DoRA | LoRA | Full FT | Trainable |
|------|-----------:|--------|-----:|-----:|--------:|----------:|
| SST-2 | 67k | Accuracy | 93.1% | **93.3%** | 93.2% | ~1.0% |
| RTE | 2.5k | Accuracy | **71.1%** | 70.8% | 54.9% | ~1.0% |
| MRPC | 3.7k | F1 | **90.7%** | 90.1% | 81.2% | ~1.0% |
| MRPC | 3.7k | Accuracy | **87.0%** | 85.8% | 68.4% | ~1.0% |

Full fine-tuning collapses on small datasets: RTE stalls near random-guess accuracy, while DoRA/LoRA's constrained parameter budget acts as useful regularization.

### Scale study: DoRA on SST-2

| Model | Params | SST-2 Accuracy | Trainable % |
|-------|-------:|---------------:|------------:|
| TinyLlama-1.1B | 1.1B | **96.0%** | ~0.10% |
| OpenLLaMA-3B | 3.0B | 81.0%† | ~0.05% |

† 3B best checkpoint was at epoch 2; later performance degraded, likely from overfitting at this scale/hyperparameter setting.

### Audio: Speech Commands with Wav2Vec2-base

Wav2Vec2-base was fine-tuned on Google Speech Commands v0.02 with 84.8k utterances and 12 keyword classes.

| Method | Val Accuracy | Test Accuracy | Train Loss | Test Loss | Trainable | Time |
|--------|-------------:|--------------:|-----------:|----------:|----------:|-----:|
| DoRA (r=8) | **98.6%** | **89.7%** | **0.154** | **0.938** | 826.6k | **183.6 min** |
| LoRA (r=8) | 98.5% | 89.0% | 0.278 | 1.037 | 789.8k | 190.8 min |

DoRA improves test accuracy by +0.7 pp; the magnitude scalar's small overhead is offset by faster directional gradient convergence on the audio encoder.

### Vision: Cornell Grasp

Grasp pose regression predicts `(x, y, sin2θ, cos2θ, w, h)` from RGB images and is scored by Cornell success rate: IoU ≥ 0.25 and |Δangle| ≤ 30°.

| Backbone | Params | Method | Trainable | Success Rate |
|----------|-------:|--------|----------:|-------------:|
| ViT-Base/16 | 87M | DoRA | ~1.0% | 7.9% |
| ViT-Base/16 | 87M | LoRA | ~1.0% | **6.2%** |
| ViT-Base/16 | 87M | Full FT | 100% | 0.6% |
| SigLIP-base/16 | 93M | DoRA | ~1.0% | 19.8% |
| SigLIP-base/16 | 93M | LoRA | ~1.0% | **20.3%** |
| SigLIP-base/16 | 93M | Full FT | 100% | 15.8% |

The poster-level takeaway is that base model quality matters: SigLIP's richer vision-language pretraining improves grasping far more than the adapter choice alone.

### Robotics and OpenVLA verification

For Push-T, DoRA adapts a VLA model to predict 2D actions from an overhead camera and language instruction; action MSE drops from 63.6 to 27.9 over 3 epochs.

For OpenVLA-7B architecture verification, DoRA targets 224 attention/MLP layers with 21,348,352 adapter parameters, adding about 0.28% trainable parameters to a frozen 7.54B model.

## 7. Conclusion

DoRA reproduced the expected low-data NLP behavior: it is most useful on scarce-data tasks such as RTE and MRPC, where full fine-tuning overfits badly and LoRA/DoRA regularize the update.

Across modalities, DoRA's gains are real but task-dependent. Audio showed a small accuracy gain and smoother convergence, vision depended strongly on the backbone, and VLA training suggested that magnitude decoupling may need more updates or larger models to matter.

Future work from the poster: combine DoRA with QLoRA for 4-bit 7B+ training, test DoRA in diffusion/action-distribution heads, and explore SVD-based variants such as EDoRA.

## 8. References

[1] Liu, S., Wang, H., Yin, S., Wu, C., Qiu, X., & Cheng, Y. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation. ICML 2024. arXiv:2402.09353.

[2] Hu, E., Shen, Y., Wallis, P., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. arXiv:2106.09685.

[3] Wang, A., Singh, A., Michael, J., et al. (2019). GLUE: A Multi-Task Benchmark and Analysis Platform. ICLR 2019.

[4] Warden, P. (2018). Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition. arXiv:1804.03209.

[5] Kim, M., et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246.

[6] Wolf, T., et al. (2020). Hugging Face Transformers. EMNLP 2020.

[7] Jiang, C., et al. (2023). SmolVLM. Hugging Face.

[8] Nasiri, M., & Garraghan, P. (2025). EDoRA: Efficient Weight-Decomposed Low-Rank Adaptation via SVD. arXiv:2501.12067.

## 9. Acknowledgements

This project was completed for **CS 4782: Introduction to Deep Learning** at Cornell University in Spring 2025.

Project team: Richie Xue, Shaurya Sen, and Kyle Du. We thank the CS 4782 course staff and poster reviewers for feedback during the final project presentation.
