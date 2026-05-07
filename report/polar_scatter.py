#!/usr/bin/env python3
"""Generate paper-style adapter Delta M vs. Delta D scatter plots."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dora-matplotlib-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification


RESULTS_ROOT = ROOT.parent / "results"
BASE_MODEL_NAME = "FacebookAI/roberta-base"
ADAPTER_RUNS = {
    "DoRA": {
        "path": ROOT / "dora_adapter.pt",
        "state_key": "dora_state",
    },
    "LoRA": {
        "path": ROOT / "lora_adapter.pt",
        "state_key": "lora_state",
    },
}
FULL_MODEL_PATH = RESULTS_ROOT / "glue_mrpc_roberta_full" / "full_model" / "model.safetensors"
DATA_PATH = ROOT / "polar_scatter_data.csv"
PNG_PATH = ROOT / "polar_scatter.png"
PDF_PATH = ROOT / "polar_scatter_mpl.pdf"
TEX_PATH = ROOT / "polar_scatter.tex"
TARGET_SUFFIXES = ("value",)
EPS = 1e-8
METHOD_STYLES = {
    "Full": {
        "color": "#222222",
        "marker": "*",
        "label": "Full",
        "zorder": 4,
        "tikz": "black",
        "tikz_mark": "star",
        "line_style": "-.",
    },
    "LoRA": {
        "color": "#2f4fd0",
        "marker": "^",
        "label": "LoRA",
        "zorder": 2,
        "tikz": "blue!70!black",
        "tikz_mark": "triangle*",
        "line_style": "--",
    },
    "DoRA": {
        "color": "#c92f2f",
        "marker": "o",
        "label": "DoRA",
        "zorder": 3,
        "tikz": "red!75!black",
        "tikz_mark": "*",
        "line_style": "-",
    },
}


def decompose(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight = weight.detach().float()
    magnitude = torch.linalg.norm(weight, dim=1).clamp_min(EPS)
    direction = weight / magnitude.unsqueeze(1)
    return magnitude, direction


def layer_deltas(base_weight: torch.Tensor, effective_weight: torch.Tensor) -> tuple[float, float]:
    m0, d0 = decompose(base_weight)
    m1, d1 = decompose(effective_weight)
    delta_m = torch.linalg.norm(m1 - m0) / torch.linalg.norm(m0).clamp_min(EPS)
    delta_d = torch.linalg.norm(d1 - d0) / torch.linalg.norm(d0).clamp_min(EPS)
    return float(delta_d.item()), float(delta_m.item())


def load_run_hparams(run: dict, adapter_obj: dict, layer: str) -> tuple[float, float]:
    if "config" in adapter_obj and layer in adapter_obj["config"]:
        config = adapter_obj["config"][layer]
        return float(config["rank"]), float(config["alpha"])

    return 8.0, 16.0


def load_base_model():
    return AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )


def load_adapter_stats(method: str, run: dict, base_model) -> list[dict]:
    adapter_path = run["path"]
    if not adapter_path.exists():
        raise FileNotFoundError(f"missing adapter file for {method}: {adapter_path}")

    adapter_obj = torch.load(adapter_path, map_location="cpu")
    adapter_state = adapter_obj[run["state_key"]]
    rows = []

    for layer, state in adapter_state.items():
        if TARGET_SUFFIXES and not layer.endswith(TARGET_SUFFIXES):
            continue

        base_module = base_model.get_submodule(layer)
        base_weight = base_module.weight.detach().float()
        rank, alpha = load_run_hparams(run, adapter_obj, layer)
        scaling = alpha / rank
        lora_update = state["lora_B"].float() @ state["lora_A"].float()
        direction_weight = base_weight + scaling * lora_update

        if method == "DoRA":
            _, direction = decompose(direction_weight)
            effective_weight = state["magnitude"].float().unsqueeze(1) * direction
        else:
            effective_weight = direction_weight

        delta_d, delta_m = layer_deltas(base_weight, effective_weight)
        rows.append(
            {
                "method": method,
                "layer": layer,
                "delta_d": delta_d,
                "delta_m": delta_m,
            }
        )

    return rows


def load_full_stats(base_model, layers: list[str]) -> list[dict]:
    if not FULL_MODEL_PATH.exists():
        raise FileNotFoundError(
            "missing full fine-tuned model weights: "
            f"{FULL_MODEL_PATH}\n"
            "Download or copy the MRPC full fine-tuned model.safetensors there. "
            f"Do not use {BASE_MODEL_NAME}'s base model.safetensors as a substitute."
        )

    full_state = load_file(str(FULL_MODEL_PATH), device="cpu")
    rows = []
    for layer in layers:
        weight_key = f"{layer}.weight"
        if weight_key not in full_state:
            raise KeyError(f"{FULL_MODEL_PATH} does not contain {weight_key}")

        base_weight = base_model.get_submodule(layer).weight.detach().float()
        full_weight = full_state[weight_key].float()
        delta_d, delta_m = layer_deltas(base_weight, full_weight)
        rows.append(
            {
                "method": "Full",
                "layer": layer,
                "delta_d": delta_d,
                "delta_m": delta_m,
            }
        )
    return rows


def build_dataframe() -> pd.DataFrame:
    base_model = load_base_model()
    rows = []
    # LoRA/DoRA deltas are computed from Hugging Face base weights plus adapter weights.
    for method, run in ADAPTER_RUNS.items():
        rows.extend(load_adapter_stats(method, run, base_model))
    layers = sorted({row["layer"] for row in rows})
    rows.extend(load_full_stats(base_model, layers))
    df = pd.DataFrame(rows)
    df.to_csv(DATA_PATH, index=False)
    return df


def coordinates(df: pd.DataFrame, method: str) -> str:
    pairs = []
    for row in df[df["method"] == method].itertuples(index=False):
        pairs.append(f"({row.delta_d:.6f},{row.delta_m:.6f})")
    return "\n    ".join(pairs)


def fit_line(group: pd.DataFrame) -> tuple[float, float]:
    if len(group) < 2:
        return 0.0, float(group["delta_m"].iloc[0]) if len(group) else 0.0
    slope, intercept = np.polyfit(group["delta_d"], group["delta_m"], 1)
    return float(slope), float(intercept)


def axis_limits(df: pd.DataFrame, y_pad_fraction: float = 0.18) -> tuple[float, float, float, float]:
    x_min = float(df["delta_d"].min())
    x_max = float(df["delta_d"].max())
    y_min = float(df["delta_m"].min())
    y_max = float(df["delta_m"].max())
    x_pad = max((x_max - x_min) * 0.12, 0.001)
    y_pad = max((y_max - y_min) * y_pad_fraction, 0.000001)
    return x_min - x_pad, x_max + x_pad, max(0.0, y_min - y_pad), y_max + y_pad


def write_tikz(df: pd.DataFrame) -> None:
    panels = []
    for method in METHOD_STYLES:
        group = df[df["method"] == method]
        if group.empty:
            continue
        style = METHOD_STYLES[method]
        xmin, xmax, ymin, ymax = axis_limits(group, 0.08 if method == "Full" else 0.18)
        slope, intercept = fit_line(group)
        dash = "dashed" if style["line_style"] == "--" else "dashdotted" if style["line_style"] == "-." else "solid"
        ylabel = r"ylabel={{$\Delta M$}}," if method == "Full" else "ylabel={},"
        panels.append(
            rf"""\nextgroupplot[
    title={{{style['label']}}},
    title style={{font=\Large\bfseries, yshift=4pt}},
    xlabel={{$\Delta D$}},
    {ylabel}
    xmin={xmin:.4f},
    xmax={xmax:.4f},
    ymin={ymin:.4f},
    ymax={ymax:.4f},
]
\addplot[{style['tikz']}, thick, {dash}, forget plot] coordinates {{
    ({xmin:.4f},{slope * xmin + intercept:.6f})
    ({xmax:.4f},{slope * xmax + intercept:.6f})
}};
\addplot[
    only marks,
    mark={style['tikz_mark']},
    mark size=2.7pt,
    {style['tikz']},
    fill={style['tikz']},
    opacity=0.85,
] coordinates {{
    {coordinates(df, method)}
}};
"""
        )
    TEX_PATH.write_text(
        rf"""\documentclass[border=8pt]{{standalone}}

\usepackage{{pgfplots}}
\usepgfplotslibrary{{groupplots}}
\pgfplotsset{{compat=1.18}}

\begin{{document}}
\begin{{tikzpicture}}
\begin{{groupplot}}[
    group style={{group size=3 by 1, horizontal sep=0.42in}},
    width=3.05in,
    height=2.55in,
    xticklabel style={{font=\small}},
    yticklabel style={{font=\small}},
    grid=major,
    major grid style={{draw=black!12}},
    axis line style={{draw=black!55}},
    label style={{font=\large}},
]

{chr(10).join(panels)}

\end{{groupplot}}
\end{{tikzpicture}}
\end{{document}}
""",
        encoding="utf-8",
    )


def main() -> None:
    df = build_dataframe()

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.2), dpi=180)

    for ax, method in zip(axes, METHOD_STYLES):
        group = df[df["method"] == method]
        if group.empty:
            continue
        style = METHOD_STYLES[method]
        xmin, xmax, ymin, ymax = axis_limits(group, 0.08 if method == "Full" else 0.18)
        slope, intercept = fit_line(group)
        ax.plot(
            [xmin, xmax],
            [slope * xmin + intercept, slope * xmax + intercept],
            color=style["color"],
            linestyle=style["line_style"],
            linewidth=2.0,
            zorder=1,
        )
        ax.scatter(
            group["delta_d"].to_numpy(),
            group["delta_m"].to_numpy(),
            s=52,
            alpha=0.9,
            linewidth=0.7,
            edgecolor="white",
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            zorder=style["zorder"],
        )

        ax.set_title(style["label"], fontsize=18, fontweight="bold", pad=8)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r"$\Delta D$", fontsize=13)
        if method == "Full":
            ax.set_ylabel(r"$\Delta M$", fontsize=13)
        ax.grid(True, color="#dddddd", linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(PNG_PATH, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    write_tikz(df)
    print(f"wrote {DATA_PATH}")
    print(f"wrote {PNG_PATH}")
    print(f"wrote {PDF_PATH}")
    print(f"wrote {TEX_PATH}")


if __name__ == "__main__":
    main()
