#!/usr/bin/env python3
"""
Rank analysis: DoRA vs LoRA at ranks [2, 4, 8, 16, 32] on GLUE MRPC.

MRPC (3.7K train, combined F1+accuracy metric) is the medium-sized GLUE task —
large enough to show meaningful rank effects, small enough to complete quickly.
r=8 results are reused if they already exist.

Usage (from the code/ directory):
    uv run run_rank_analysis.py
    uv run run_rank_analysis.py --dry-run
    uv run run_rank_analysis.py --ranks 4 8 16        # custom rank sweep
    uv run run_rank_analysis.py --methods dora        # DoRA only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
RANKS   = [2, 4, 8, 16, 32]
METHODS = ["dora", "lora"]
TASK    = "mrpc"
MODEL   = "roberta"


def output_dir(method: str, rank: int) -> Path:
    return RESULTS_DIR / f"glue_{TASK}_{MODEL}_{method}_r{rank}"


def already_done(method: str, rank: int) -> bool:
    d = output_dir(method, rank)
    # Consider done if adapter_stats.json or a checkpoint exists
    return (d / "adapter_stats.json").exists() or any(d.glob("checkpoint-*"))


def run_one(method: str, rank: int, log_dir: Path, dry_run: bool) -> int:
    name    = f"{TASK}_{MODEL}_{method}_r{rank}"
    out_dir = output_dir(method, rank)
    log_path = log_dir / f"{name}.log"

    cmd = [
        "uv", "run", "--frozen", "scripts/train_glue.py",
        "--task",       TASK,
        "--model",      MODEL,          # "roberta" → FacebookAI/roberta-base
        "--method",     method,
        "--rank",       str(rank),
        "--alpha",      str(float(rank * 2)),   # alpha = 2× rank (standard scaling)
        "--output_dir", str(out_dir),
    ]

    print(f"\n{'='*70}")
    print(f"  RUN : {name}")
    print(f"  CMD : {' '.join(cmd)}")
    print(f"  LOG : {log_path}")
    print(f"{'='*70}\n", flush=True)

    if dry_run:
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with open(log_path, "w") as fh:
        fh.write(f"# {name}\n# started: {datetime.now()}\n# cmd: {' '.join(cmd)}\n\n")
        fh.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
            fh.flush()
        proc.wait()

    elapsed = time.time() - t0
    status  = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"\n  {status} — {elapsed/60:.1f} min  [{name}]\n", flush=True)
    return proc.returncode


def load_metric(method: str, rank: int) -> float | None:
    d = output_dir(method, rank)

    # Prefer adapter_stats.json (has eval_metric at each epoch)
    stats_path = d / "adapter_stats.json"
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text())
            if stats:
                return float(stats[-1]["eval_metric"])
        except Exception:
            pass

    # Fallback: trainer_state.json in the last checkpoint
    checkpoints = sorted(
        d.glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(p.parent.name.split("-")[-1]) if p.parent.name.split("-")[-1].isdigit() else 0,
    )
    for p in reversed(checkpoints):
        try:
            d_ts = json.loads(p.read_text())
            evals = [
                e for e in d_ts.get("log_history", [])
                if "eval_combined" in e or "eval_accuracy" in e
            ]
            if evals:
                e = evals[-1]
                return float(e.get("eval_combined") or e.get("eval_accuracy"))
        except Exception:
            pass

    return None


def print_summary(ranks: list[int], methods: list[str]):
    print(f"\n{'='*70}")
    print(f"  RANK ANALYSIS SUMMARY — GLUE {TASK.upper()} ({MODEL})")
    print(f"  Metric: combined (F1 + accuracy) / 2")
    print(f"{'='*70}")
    header = f"{'Rank':>6}" + "".join(f"  {m.upper():>10}" for m in methods)
    print(header)
    print("-" * len(header))
    for r in ranks:
        row = f"{r:>6}"
        for m in methods:
            val = load_metric(m, r)
            row += f"  {val*100:>9.2f}%" if val is not None else f"  {'—':>10}"
        print(row)
    print()


def main():
    ap = argparse.ArgumentParser(description="Rank analysis — GLUE MRPC")
    ap.add_argument("--ranks",   nargs="+", type=int, default=RANKS)
    ap.add_argument("--methods", nargs="+", choices=["dora", "lora"], default=METHODS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force",   action="store_true", help="Re-run even if results exist")
    args = ap.parse_args()

    code_dir = Path(__file__).parent.resolve()
    os.chdir(code_dir)
    log_dir = code_dir / "logs" / "rank_analysis"

    queue = [
        (m, r) for m in args.methods for r in args.ranks
        if args.force or not already_done(m, r)
    ]

    skipped = [
        (m, r) for m in args.methods for r in args.ranks
        if not args.force and already_done(m, r)
    ]

    if skipped:
        print(f"Skipping {len(skipped)} already-completed runs:")
        for m, r in skipped:
            print(f"  {TASK}_{MODEL}_{m}_r{r}")

    if not queue:
        print("\nAll runs already complete.")
        print_summary(args.ranks, args.methods)
        return

    print(f"\nQueued {len(queue)} runs:")
    for m, r in queue:
        print(f"  {TASK}_{MODEL}_{m}_r{r}")

    failures = []
    t_total = time.time()
    for method, rank in queue:
        rc = run_one(method, rank, log_dir, args.dry_run)
        if rc != 0:
            failures.append((method, rank))

    if not args.dry_run:
        print_summary(args.ranks, args.methods)
        elapsed = (time.time() - t_total) / 60
        print(f"Total time: {elapsed:.1f} min")
        if failures:
            print(f"Failed runs: {[(m, r) for m, r in failures]}")
            sys.exit(1)


if __name__ == "__main__":
    main()
