#!/usr/bin/env python3
"""
Export rank analysis summary JSON for GLUE MRPC.

Collects per-rank, per-method metrics and per-epoch adapter stats,
writing results/rank_analysis_summary.json for the Gradio dashboard.

Usage:
    uv run scripts/export_rank_analysis.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RANKS   = [2, 4, 8, 16, 32]
METHODS = ["dora", "lora"]
TASK    = "mrpc"
MODEL   = "roberta"


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _best_metric(run_dir: Path) -> float | None:
    # Prefer adapter_stats.json — final epoch eval_metric
    stats = _load_json(run_dir / "adapter_stats.json")
    if stats:
        return float(stats[-1]["eval_metric"])

    # Fallback: last checkpoint trainer_state.json
    checkpoints = sorted(
        run_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(p.parent.name.split("-")[-1])
        if p.parent.name.split("-")[-1].isdigit()
        else 0,
    )
    for p in reversed(checkpoints):
        d = _load_json(p)
        if not d:
            continue
        evals = [
            e for e in d.get("log_history", [])
            if "eval_combined" in e or "eval_accuracy" in e
        ]
        if evals:
            e = evals[-1]
            return float(e.get("eval_combined") or e.get("eval_accuracy"))
    return None


def _epoch_curve(run_dir: Path) -> list[dict]:
    stats = _load_json(run_dir / "adapter_stats.json")
    if not stats:
        return []
    return [
        {
            "epoch":               r["epoch"],
            "eval_metric":         r["eval_metric"],
            "eval_loss":           r.get("eval_loss"),
            "train_loss":          r.get("train_loss"),
            "lora_norm_mean":      r.get("lora_norm_mean"),
            "magnitude_rel_drift": r.get("magnitude_rel_drift"),
        }
        for r in stats
    ]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Export rank analysis summary")
    ap.add_argument("--results_dir", default=str(RESULTS_DIR))
    ap.add_argument("--output", default=str(RESULTS_DIR / "rank_analysis_summary.json"))
    ap.add_argument("--ranks",   nargs="+", type=int, default=RANKS)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_path    = Path(args.output).resolve()

    runs = []
    missing = []

    for method in args.methods:
        for rank in args.ranks:
            run_name = f"glue_{TASK}_{MODEL}_{method}_r{rank}"
            run_dir  = results_dir / run_name

            if not run_dir.exists():
                missing.append(run_name)
                continue

            metric = _best_metric(run_dir)
            curve  = _epoch_curve(run_dir)

            runs.append({
                "run":          run_name,
                "task":         TASK,
                "model":        MODEL,
                "method":       method,
                "rank":         rank,
                "alpha":        rank * 2,
                "metric":       metric,
                "metric_pct":   round(metric * 100, 4) if metric is not None else None,
                "epoch_curve":  curve,
                "n_epochs":     len(curve),
            })

    summary = {
        "task":    TASK,
        "model":   MODEL,
        "metric":  "combined (F1 + accuracy) / 2",
        "ranks":   args.ranks,
        "methods": args.methods,
        "runs":    runs,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {len(runs)} runs to {out_path}")

    if missing:
        print(f"Missing ({len(missing)} runs not found):")
        for name in missing:
            print(f"  {name}")

    # Print table
    print(f"\n{'Rank':>6}" + "".join(f"  {m.upper():>10}" for m in args.methods))
    print("-" * (6 + 12 * len(args.methods)))
    by_rank = {(r["method"], r["rank"]): r for r in runs}
    for rank in args.ranks:
        row = f"{rank:>6}"
        for method in args.methods:
            entry = by_rank.get((method, rank))
            val = entry["metric_pct"] if entry else None
            row += f"  {val:>9.2f}%" if val is not None else f"  {'—':>10}"
        print(row)


if __name__ == "__main__":
    main()
