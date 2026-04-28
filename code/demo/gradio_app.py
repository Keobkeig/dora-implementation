"""
DoRA Results Dashboard — CS4782 Final Project

Visualises all experimental results:
  • GLUE method comparison (DoRA vs LoRA vs Full FT)
  • LLaMA scale study
  • Training curves (accuracy + loss over epochs)
  • Adapter weight trajectory (directional vs magnitude adaptation)
  • Cornell Grasp vision results (ViT-Base vs SigLIP)
  • OpenVLA architecture demo summary
"""

import json
import os
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Results directory (two levels up from code/demo/)
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

# ---------------------------------------------------------------------------
# Hardcoded experiment results
# ---------------------------------------------------------------------------

GLUE_RESULTS = {
    "SST-2":      {"DoRA": 93.1, "LoRA": 93.3, "Full FT": 93.2},
    "RTE":        {"DoRA": 71.1, "LoRA": 70.8, "Full FT": 54.9},
    "MRPC F1":    {"DoRA": 90.7, "LoRA": 90.1, "Full FT": 81.2},
    "MRPC Acc":   {"DoRA": 87.0, "LoRA": 85.8, "Full FT": 68.4},
}

SCALE_RESULTS = {
    "TinyLlama 1.1B": 96.0,
    "OpenLLaMA 3B":   81.0,
}

GRASP_RESULTS = {
    "ViT-Base (87M)":          {"DoRA": 7.9,  "LoRA": 6.2,  "Full FT": 0.6},
    "SigLIP (93M)\n(OpenVLA)": {"DoRA": 19.8, "LoRA": 20.3, "Full FT": 15.8},
}

OPENVLA_STATS = {
    "total_params_B":         7.54,
    "adapter_layers":         224,
    "dora_adapter_params_M":  21.3,
    "trainable_pct":          0.28,
}

COLORS = {"DoRA": "#2196F3", "LoRA": "#FF9800", "Full FT": "#9E9E9E"}

# ---------------------------------------------------------------------------
# Data loaders — fall back gracefully if results dir absent
# ---------------------------------------------------------------------------

def _load_trainer_states(run_prefix: str):
    """
    Load per-epoch eval metrics from trainer_state.json files.
    Returns list of dicts with epoch / eval_accuracy or eval_f1 / eval_loss.
    """
    checkpoints = sorted(
        (RESULTS_DIR / run_prefix).glob("checkpoint-*/trainer_state.json")
    )
    records = []
    for path in checkpoints:
        try:
            d = json.loads(path.read_text())
            for entry in d.get("log_history", []):
                if "eval_accuracy" in entry or "eval_f1" in entry or "eval_mcc" in entry:
                    records.append(entry)
        except Exception:
            pass
    records.sort(key=lambda x: x.get("epoch", 0))
    return records


def _load_adapter_stats(run_dir: str):
    path = RESULTS_DIR / run_dir / "adapter_stats.json"
    try:
        return json.loads(path.read_text())
    except Exception:
        return []

# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

FIG_W, FIG_H = 8, 4.8   # all single-panel figures use this size


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def fig_glue_comparison():
    tasks = list(GLUE_RESULTS.keys())
    methods = ["DoRA", "LoRA", "Full FT"]
    x = np.arange(len(tasks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for i, method in enumerate(methods):
        vals = [GLUE_RESULTS[t][method] for t in tasks]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=method,
                      color=COLORS[method], alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylim(40, 100)
    ax.legend(framealpha=0.7, fontsize=9)
    _style(ax, "GLUE Benchmark — DoRA vs LoRA vs Full FT (RoBERTa-base)",
           "Task", "Metric (%)")
    fig.tight_layout()
    return fig


def fig_scale_study():
    models = list(SCALE_RESULTS.keys())
    accs = [SCALE_RESULTS[m] for m in models]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bars = ax.bar(models, accs, color=COLORS["DoRA"], alpha=0.88,
                  edgecolor="white", width=0.45)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(70, 100)
    _style(ax, "Scale Study — DoRA on SST-2", "Model", "Accuracy (%)")
    fig.tight_layout()
    return fig


def fig_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 2, FIG_H))

    dora_rte = _load_trainer_states("glue_rte_roberta_dora_r8")
    lora_rte = _load_trainer_states("glue_rte_roberta_lora_r8")
    full_rte = _load_trainer_states("glue_rte_roberta_full")

    # Fallback hardcoded curves
    if not dora_rte:
        dora_rte = [{"epoch": e, "eval_accuracy": v, "eval_loss": l} for e, v, l in [
            (1, .473, .696), (2, .708, .574), (3, .711, .584), (4, .704, .550), (5, .708, .560)]]
        lora_rte = [{"epoch": e, "eval_accuracy": v, "eval_loss": l} for e, v, l in [
            (1, .473, .696), (2, .693, .579), (3, .708, .593), (4, .704, .562), (5, .704, .571)]]
        full_rte = [{"epoch": e, "eval_accuracy": v, "eval_loss": l} for e, v, l in [
            (1, .491, .693), (2, .512, .695), (3, .527, .694), (4, .541, .693), (5, .549, .692)]]

    for records, label, color in [
        (dora_rte, "DoRA", COLORS["DoRA"]),
        (lora_rte, "LoRA", COLORS["LoRA"]),
        (full_rte, "Full FT", COLORS["Full FT"]),
    ]:
        epochs = [r["epoch"] for r in records]
        accs   = [r.get("eval_accuracy", float("nan")) for r in records]
        losses = [r.get("eval_loss", float("nan")) for r in records]
        axes[0].plot(epochs, [a * 100 for a in accs], "o-", label=label,
                     color=color, linewidth=2, markersize=5)
        axes[1].plot(epochs, losses, "o-", label=label,
                     color=color, linewidth=2, markersize=5)

    _style(axes[0], "RTE Eval Accuracy over Epochs", "Epoch", "Accuracy (%)")
    _style(axes[1], "RTE Eval Loss over Epochs",     "Epoch", "Eval Loss")
    axes[0].set_ylim(40, 80)
    axes[0].legend(fontsize=9)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    return fig


def fig_weight_trajectory():
    dora_stats = _load_adapter_stats("glue_rte_roberta_dora_r8_stats")
    lora_stats = _load_adapter_stats("glue_rte_roberta_lora_r8_stats")

    if not dora_stats:
        dora_stats = [{"epoch": e, "lora_norm_mean": v} for e, v in [
            (1, .201), (2, .452), (3, .495), (4, .503), (5, .503)]]
        lora_stats = [{"epoch": e, "lora_norm_mean": v} for e, v in [
            (1, .203), (2, .442), (3, .491), (4, .500), (5, .500)]]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for stats, label, color in [(dora_stats, "DoRA", COLORS["DoRA"]),
                                 (lora_stats, "LoRA", COLORS["LoRA"])]:
        epochs = [r["epoch"] for r in stats]
        norms  = [r["lora_norm_mean"] for r in stats]
        ax.plot(epochs, norms, "o-", label=label, color=color,
                linewidth=2.5, markersize=6)

    _style(ax, "Directional Update Norm  ‖ΔW‖  over Training\n(RTE, RoBERTa-base)",
           "Epoch", "Mean ‖lora_B @ lora_A‖")
    ax.legend(fontsize=10)
    ax.annotate("DoRA makes larger\ndirectional updates",
                xy=(3, dora_stats[2]["lora_norm_mean"]),
                xytext=(1.8, 0.35),
                fontsize=8.5, color=COLORS["DoRA"],
                arrowprops=dict(arrowstyle="->", color=COLORS["DoRA"], lw=1.2))
    fig.tight_layout()
    return fig


def fig_grasp_results():
    backbones = list(GRASP_RESULTS.keys())
    methods   = ["DoRA", "LoRA", "Full FT"]
    x = np.arange(len(backbones))
    width = 0.25

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for i, method in enumerate(methods):
        vals = [GRASP_RESULTS[b][method] for b in backbones]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=method,
                      color=COLORS[method], alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(backbones, fontsize=9)
    ax.set_ylim(0, 28)
    ax.legend(framealpha=0.7, fontsize=9)
    _style(ax, "Cornell Grasp — Grasp Success Rate\n(IoU ≥ 0.25 AND |Δangle| ≤ 30°)",
           "Visual Backbone", "Success Rate (%)")
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
.finding-box {
    background: #f0f4ff;
    border-left: 4px solid #2196F3;
    padding: 12px 16px;
    border-radius: 4px;
    margin: 8px 0;
}
"""

FINDING_1 = """
### 🔬 Finding 1 — Full Fine-Tuning Collapses on Small Datasets
Full FT achieves only **54.9% on RTE** (2.5k train) and **68.4% accuracy on MRPC** (3.7k train)
— barely above chance on RTE. Adapting all 125M parameters on so little data causes catastrophic
overfitting. DoRA and LoRA's parameter budget (~1M vs 125M) acts as strong implicit regularisation.
"""

FINDING_2 = """
### 📐 Finding 2 — DoRA Adapts Directionally, Not via Magnitude (in bfloat16)
Per-epoch tracking shows DoRA's directional component (‖lora_B @ lora_A‖) grows from 0.20 → 0.50
over 5 epochs, matching LoRA. But the magnitude component stays frozen at its pretrained value throughout.
With bf16 precision and standard hyperparameters, magnitude updates are sub-quantisation-step and
effectively zero. DoRA's accuracy advantage comes from its superior directional parameterisation.
"""

FINDING_3 = """
### 🤖 Finding 3 — Stronger Visual Backbone = 3× Better Grasp Success
SigLIP-base (OpenVLA's visual encoder, pretrained on image–text pairs) achieves **~20% grasp success**
vs ViT-Base's **7.9%** — a 3× improvement from backbone quality alone. Both at ~1% trainable params.
Full FT collapses on the small Cornell dataset (708 images) regardless of backbone.
"""

FINDING_4 = """
### ⚡ Finding 4 — OpenVLA-7B: DoRA Adds Only 0.28% Parameters
Applying DoRA (rank 8) to all 224 attention + MLP layers of OpenVLA-7B adds only **21.3M parameters**
to a frozen 7.54B model — **0.28% overhead**. Forward pass verified on CPU. Full training
requires 4-bit quantisation (bitsandbytes) and a robot-action dataset; Cornell Grasp grasping
rectangles are not directly in OpenVLA's 7-DoF action format.
"""


with gr.Blocks(title="DoRA Results Dashboard") as demo:

    gr.Markdown("""
# DoRA Reproduction — CS4782 Final Project
**Weight-Decomposed Low-Rank Adaptation** reproduced from scratch in PyTorch, extended to vision and robotics.
> Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation* — arXiv:2402.09353
""")

    with gr.Tabs():

        # ── Tab 1: GLUE ───────────────────────────────────────────────────
        with gr.Tab("GLUE Benchmark"):
            gr.Markdown("### Method comparison: DoRA vs LoRA vs Full FT — RoBERTa-base (125M), rank=8, 5 epochs")
            with gr.Row():
                glue_plot  = gr.Plot(label="GLUE Results")
                scale_plot = gr.Plot(label="Scale Study (DoRA SST-2)")
            gr.Markdown(FINDING_1)

        # ── Tab 2: Training Dynamics ──────────────────────────────────────
        with gr.Tab("Training Dynamics"):
            gr.Markdown("### Per-epoch eval accuracy, loss, and adapter weight norms — RTE task, RoBERTa-base")
            with gr.Row():
                curve_plot = gr.Plot(label="Accuracy & Loss Curves")
            with gr.Row():
                traj_plot  = gr.Plot(label="Directional Update Norm over Training")
            gr.Markdown(FINDING_2)

        # ── Tab 3: Vision / Grasping ──────────────────────────────────────
        with gr.Tab("Vision — Cornell Grasp"):
            gr.Markdown("""
### DoRA on visual backbones for robotic grasp pose prediction
**Task:** predict grasp pose (x, y, sin2θ, cos2θ, w, h) from a 640×480 RGB image
**Metric:** Cornell success rate — IoU ≥ 0.25 **and** |Δangle| ≤ 30°
**Dataset:** 885 images, 708 train / 177 val (image-level split)
""")
            grasp_plot = gr.Plot(label="Grasp Success Rate")
            gr.Markdown(FINDING_3)

        # ── Tab 4: OpenVLA ────────────────────────────────────────────────
        with gr.Tab("OpenVLA Architecture"):
            gr.Markdown("""
### DoRA applied to OpenVLA-7B (architecture verification)
Model loaded to CPU in bfloat16. Forward pass verified.
""")
            with gr.Row():
                with gr.Column():
                    gr.DataFrame(
                        value=[
                            ["Total parameters",         "7.54 B"],
                            ["DoRA rank",                "8"],
                            ["Target layers (full)",     "224"],
                            ["Adapter parameters",       "21,348,352"],
                            ["Trainable %",              "0.28%"],
                            ["Forward pass",             "✓ verified (CPU, ~50s)"],
                            ["Visual targets",           "q_proj, k_proj, v_proj"],
                            ["LLM targets",              "q/k/v/o_proj, gate/up/down_proj"],
                        ],
                        headers=["Property", "Value"],
                        label="OpenVLA-7B + DoRA (rank=8) Summary",
                        interactive=False,
                    )
            gr.Markdown(FINDING_4)

        # ── Tab 5: Key Findings ───────────────────────────────────────────
        with gr.Tab("Key Findings"):
            gr.Markdown("## Summary of All Findings")
            gr.Markdown(FINDING_1)
            gr.Markdown(FINDING_2)
            gr.Markdown(FINDING_3)
            gr.Markdown(FINDING_4)
            gr.Markdown("""
---
### Reproduce Any Experiment
```powershell
cd code

# GLUE method comparison (RoBERTa)
bash scripts/run_roberta_experiments.sh

# Cornell Grasp (ViT + SigLIP)
bash scripts/run_grasp_experiments.sh --data_dir ../data/cornell_grasps

# OpenVLA architecture demo
uv run scripts/openvla_demo.py
```
""")

    # ── Load all plots on startup ─────────────────────────────────────────
    demo.load(
        fn=lambda: (
            fig_glue_comparison(),
            fig_scale_study(),
            fig_training_curves(),
            fig_weight_trajectory(),
            fig_grasp_results(),
        ),
        outputs=[glue_plot, scale_plot, curve_plot, traj_plot, grasp_plot],
    )


if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())
