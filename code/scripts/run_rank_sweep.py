#!/usr/bin/env python3
"""
Parallel rank sweep runner for DoRA vs LoRA on GLUE.

Defaults to a small dataset (RTE) and rank 8 for both DoRA/LoRA.
Skips runs whose output directory already exists unless --force is set.
Use --rerun-incomplete to rerun runs missing adapter_stats.json.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def _parse_ranks(text: str):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [int(p) for p in parts]


def _make_cmd(task: str, method: str, rank: int, alpha: int, bf16: bool):
    cmd = [
        "uv",
        "run",
        "--frozen",
        "scripts/train_glue.py",
        "--model",
        "roberta",
        "--task",
        task,
        "--method",
        method,
        "--rank",
        str(rank),
        "--alpha",
        str(alpha),
    ]
    if bf16:
        cmd.append("--bf16")
    return cmd


def _output_dir(task: str, method: str, rank: int) -> Path:
    code_dir = Path(__file__).parent.resolve()
    results = code_dir.parent.parent / "results"
    return results / f"glue_{task}_roberta_{method}_r{rank}"


def _is_complete(out_dir: Path) -> bool:
    return (out_dir / "adapter_stats.json").exists()


def _run_queue(cmds, max_parallel: int):
    running = []
    idx = 0
    total = len(cmds)

    while idx < total or running:
        while idx < total and len(running) < max_parallel:
            name, cmd = cmds[idx]
            print(f"[START] {name}\n  CMD: {' '.join(cmd)}", flush=True)
            proc = subprocess.Popen(cmd)
            running.append((name, proc))
            idx += 1

        time.sleep(1)
        still_running = []
        for name, proc in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((name, proc))
            else:
                status = "OK" if rc == 0 else f"FAILED (exit {rc})"
                print(f"[DONE] {name} — {status}", flush=True)
        running = still_running


def main():
    ap = argparse.ArgumentParser(description="Run DoRA/LoRA rank sweep in parallel")
    ap.add_argument("--task", default="rte", choices=["rte", "mrpc", "sst2"])
    ap.add_argument("--ranks", default="8")
    ap.add_argument("--methods", default="dora,lora")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rerun-incomplete", action="store_true")
    ap.add_argument("--max-parallel", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ranks = _parse_ranks(args.ranks)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    cmds = []
    for method in methods:
        for rank in ranks:
            alpha = rank * 2
            out_dir = _output_dir(args.task, method, rank)
            if out_dir.exists() and not args.force:
                if args.rerun_incomplete and not _is_complete(out_dir):
                    print(f"[RERUN] {out_dir} missing adapter_stats.json")
                else:
                    print(f"[SKIP] {out_dir} exists")
                    continue
            name = f"{args.task}_roberta_{method}_r{rank}"
            cmds.append((name, _make_cmd(args.task, method, rank, alpha, args.bf16)))

    if not cmds:
        print("No runs to execute.")
        return

    max_parallel = args.max_parallel
    if args.dry_run:
        for name, cmd in cmds:
            print(f"[DRY] {name}: {' '.join(cmd)}")
        return

    _run_queue(cmds, max_parallel=max_parallel)


if __name__ == "__main__":
    main()
