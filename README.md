# CS4782 Final Project: DoRA Reproduction

## 1. Introduction

This GitHub repository contains our CS 4782 final project: a from-scratch PyTorch re-implementation and extension of **DoRA (Weight-Decomposed Low-Rank Adaptation)**.

DoRA (Liu et al., 2024) modifies LoRA by decoupling weight magnitude from low-rank directional updates, aiming to improve parameter-efficient fine-tuning at the same adapter rank and parameter budget.

> Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" — ICML 2024, arXiv:2402.09353

## 2. Chosen Result

We targeted DoRA's central empirical claim: at comparable rank and parameter budget, DoRA can match or outperform LoRA.

The primary result corresponds to **Table 1** in the DoRA paper, which compares PEFT methods on LLaMA-family commonsense reasoning benchmarks. We evaluate the same LoRA-vs-DoRA claim on GLUE classification tasks (SST-2, MRPC, RTE), then extend the comparison to audio, vision, and robotics tasks.

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

Core implementation lives in `code/dora/`; runnable experiments live in `code/scripts/`; configs are in `code/configs/`; dataset acquisition notes are in `data/README.md`.

## 4. Re-implementation Details

We implement `DoRALinear`, a drop-in replacement for `nn.Linear` with frozen base weights, low-rank LoRA matrices, and a learnable magnitude vector initialized from the pretrained weight norm. A matched `LoRALinear` baseline gives apples-to-apples DoRA vs. LoRA comparisons.

| Modality | Models | Dataset | Metric |
|----------|--------|---------|--------|
| NLP | RoBERTa-base/large, TinyLlama-1.1B, OpenLLaMA-3B | GLUE SST-2/MRPC/RTE | Accuracy, F1 |
| Audio | Wav2Vec2-base | Google Speech Commands v0.02 | Validation/test accuracy |
| Vision | ViT-B/16, SigLIP-B/16 | Cornell Grasp | Cornell success rate / IoU |
| Robotics | SmolVLM-256M for Push-T; OpenVLA-7B architecture audit | LeRobot Push-T | Action MSE / adapter parameter count |

Key modifications: we extend DoRA beyond NLP, include LoRA and full fine-tuning baselines where feasible, and track adapter statistics across epochs. Datasets are not committed; see `data/README.md`.

## 5. Reproduction Steps

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv python install 3.11
cd code
uv sync
```

Run commands from `code/`. Windows users should run `uv` from PowerShell. Most training runs benefit from a CUDA GPU with bf16 support; OpenVLA verification downloads the 7B checkpoint and runs the repository's architecture audit/forward pass on CPU in bf16.

| Task | Command |
|------|---------|
| RoBERTa GLUE comparison | `bash scripts/run_roberta_experiments.sh` |
| LLaMA SST-2 scale study | `uv run scripts/train_glue.py --model 1b --task sst2 --bf16` and `uv run scripts/train_glue.py --model 3b --task sst2 --bf16` |
| Rank sweep | `uv run scripts/run_rank_sweep.py --task sst2 --ranks 2,4,8,16 --methods dora,lora --bf16` |
| Cornell Grasp data | `uv run scripts/download_cornell_grasp.py` |
| Cornell Grasp sweep | `bash scripts/run_grasp_experiments.sh --data_dir ../data/cornell_grasps` |
| Speech Commands | `uv run scripts/train_speech_commands.py --method dora --rank 8 --alpha 16` |
| Push-T VLA | `bash scripts/run_vla_experiments.sh` |
| OpenVLA verification | `uv run scripts/openvla_demo.py` |
| Tests | `uv run pytest -q` |
| Demo | `uv run python demo/gradio_app.py` |

Representative single-run commands:

```powershell
uv run scripts/train_glue.py --model roberta --task rte --method dora --bf16
uv run scripts/train_grasp.py --model vit --data_dir ../data/cornell_grasps --method dora --bf16
uv run scripts/train_speech_commands.py --method dora --rank 8 --alpha 16
```

Outputs are written under `results/`. Common flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--method` | `dora` | `dora` / `lora` / `full` |
| `--rank` | `8` | LoRA/DoRA rank |
| `--alpha` | `16.0` | LoRA/DoRA alpha |
| `--bf16` | off | Mixed precision for CUDA |

## 6. Results/Insights

DoRA's advantage is largest on low-data NLP tasks, while large-data SST-2 shows little separation.

| Task | Train size | Metric | DoRA | LoRA | Full FT | Trainable |
|------|-----------:|--------|-----:|-----:|--------:|----------:|
| SST-2 | 67k | Accuracy | 93.1% | **93.3%** | 93.2% | ~1.0% |
| RTE | 2.5k | Accuracy | **71.1%** | 70.8% | 54.9% | ~1.0% |
| MRPC | 3.7k | F1 | **90.7%** | 90.1% | 81.2% | ~1.0% |
| MRPC | 3.7k | Accuracy | **87.0%** | 85.8% | 68.4% | ~1.0% |

Full fine-tuning collapses on small datasets: RTE stalls near random-guess accuracy, while DoRA/LoRA's constrained parameter budget acts as useful regularization.

Additional cross-modal results:

| Setting | Main result |
|---------|-------------|
| SST-2 scale study | TinyLlama-1.1B DoRA reaches 96.0%; OpenLLaMA-3B peaks at 81.0% before overfitting. |
| Speech Commands | Wav2Vec2 DoRA reaches 98.6% validation / 89.7% test accuracy vs. LoRA 98.5% / 89.0%, with 44.6% lower average train loss. |
| Cornell Grasp | SigLIP features dominate: SigLIP reaches ~22% success vs. ViT around 12-13%; adapter choice matters less than backbone quality. |
| Push-T / VLA | DoRA action MSE drops 63.6 → 27.9 over 3 epochs. |
| OpenVLA-7B | DoRA can target 224 layers with 21.3M adapter parameters, about 0.28% of a frozen 7.54B model. |

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
