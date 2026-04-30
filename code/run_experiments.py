#!/usr/bin/env python3
"""
Sequential experiment runner.

Usage (from the code/ directory):
    uv run run_experiments.py              # run all 20 experiments
    uv run run_experiments.py --dry-run    # print commands only
    uv run run_experiments.py --from rte_roberta_lora_r8   # resume from a named run
    uv run run_experiments.py --only glue  # run only one group (glue | vla | grasp)

Each experiment is run via:
    uv run scripts/<script>.py --config configs/<group>/<name>.yaml

Output is tee-d to logs/<name>.log so you can tail -f while it runs.
Results land in ../results/ as normal (auto-generated output_dir).

Exit codes per run are recorded; a non-zero exit does NOT abort the full queue
— the runner logs the failure and continues with the next experiment.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Experiment queue
# Order: fastest → slowest within each group so early results arrive quickly.
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    # ── GLUE / RoBERTa-base (125 M) ─────────────────────────────────────
    ("scripts/train_glue.py",  "configs/glue/rte_roberta_lora_r8.yaml"),
    ("scripts/train_glue.py",  "configs/glue/rte_roberta_dora_r8.yaml"),
    ("scripts/train_glue.py",  "configs/glue/mrpc_roberta_lora_r8.yaml"),
    ("scripts/train_glue.py",  "configs/glue/mrpc_roberta_dora_r8.yaml"),
    ("scripts/train_glue.py",  "configs/glue/sst2_roberta_lora_r8.yaml"),
    ("scripts/train_glue.py",  "configs/glue/sst2_roberta_dora_r8.yaml"),
    # ── GLUE / RoBERTa-large (355 M) ────────────────────────────────────
    ("scripts/train_glue.py",  "configs/glue/rte_roberta_large_lora_r8.yaml"),
    ("scripts/train_glue.py",  "configs/glue/rte_roberta_large_dora_r8.yaml"),
    ("scripts/train_glue.py",  "configs/glue/mrpc_roberta_large_lora_r8.yaml"),
    ("scripts/train_glue.py",  "configs/glue/mrpc_roberta_large_dora_r8.yaml"),
    ("scripts/train_glue.py",  "configs/glue/sst2_roberta_large_lora_r8.yaml"),
    ("scripts/train_glue.py",  "configs/glue/sst2_roberta_large_dora_r8.yaml"),
    # ── GLUE / LLaMA 1 B & 3 B ──────────────────────────────────────────
    ("scripts/train_glue.py",  "configs/glue/sst2_1b_dora_r8.yaml"),
    ("scripts/train_glue.py",  "configs/glue/sst2_3b_dora_r8.yaml"),
    # ── Grasp / ViT-base (86 M) ──────────────────────────────────────────
    ("scripts/train_grasp.py", "configs/grasp/vit_lora_r8.yaml"),
    ("scripts/train_grasp.py", "configs/grasp/vit_dora_r8.yaml"),
    # ── Grasp / SigLIP-base (400 M) ──────────────────────────────────────
    ("scripts/train_grasp.py", "configs/grasp/siglip_lora_r8.yaml"),
    ("scripts/train_grasp.py", "configs/grasp/siglip_dora_r8.yaml"),
    # ── VLA / SmolVLM Push-T ─────────────────────────────────────────────
    ("scripts/train_vla.py",   "configs/vla/pusht_lora_r8.yaml"),
    ("scripts/train_vla.py",   "configs/vla/pusht_dora_r8.yaml"),
]

_GROUP_MAP = {
    "glue":  lambda s, c: "glue"  in c,
    "grasp": lambda s, c: "grasp" in c,
    "vla":   lambda s, c: "vla"   in c,
}


def run_name(config_path: str) -> str:
    return Path(config_path).stem


def run_one(script: str, config: str, log_dir: Path, dry_run: bool) -> int:
    name   = run_name(config)
    log_path = log_dir / f"{name}.log"
    # --frozen: skip env sync (avoids building optional C-extensions like evdev)
    cmd    = ["uv", "run", "--frozen", script, "--config", config]

    print(f"\n{'='*70}")
    print(f"  RUN : {name}")
    print(f"  CMD : {' '.join(cmd)}")
    print(f"  LOG : {log_path}")
    print(f"{'='*70}\n", flush=True)

    if dry_run:
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with open(log_path, "w") as log_fh:
        # Write header
        log_fh.write(f"# {name}\n# started: {datetime.now()}\n# cmd: {' '.join(cmd)}\n\n")
        log_fh.flush()

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
            log_fh.write(line)
            log_fh.flush()

        proc.wait()

    elapsed = time.time() - t0
    status  = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"\n  {status} — {elapsed/60:.1f} min  [{name}]\n", flush=True)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="Sequential experiment runner")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing them")
    ap.add_argument("--from",  dest="start_from", default=None, metavar="NAME",
                    help="Skip experiments before this config name (resume)")
    ap.add_argument("--only",  dest="only_group", default=None,
                    choices=list(_GROUP_MAP.keys()),
                    help="Run only one experiment group")
    args = ap.parse_args()

    # Resolve to code/ directory so relative paths in configs work
    code_dir = Path(__file__).parent.resolve()
    os.chdir(code_dir)

    log_dir = code_dir / "logs"

    # Filter queue
    queue = EXPERIMENTS
    if args.only_group:
        pred  = _GROUP_MAP[args.only_group]
        queue = [(s, c) for s, c in queue if pred(s, c)]

    if args.start_from:
        names = [run_name(c) for _, c in queue]
        if args.start_from not in names:
            sys.exit(f"Error: --from '{args.start_from}' not found in queue.\n"
                     f"Available: {names}")
        idx   = names.index(args.start_from)
        queue = queue[idx:]
        print(f"Resuming from experiment #{idx+1}: {args.start_from}")

    total   = len(queue)
    failed  = []
    t_start = time.time()

    print(f"\nExperiment runner — {total} run(s) queued")
    print(f"Working directory: {code_dir}")
    if args.dry_run:
        print("DRY RUN — no processes will be started\n")

    for i, (script, config) in enumerate(queue, 1):
        print(f"\n[{i}/{total}]", end="")
        rc = run_one(script, config, log_dir, args.dry_run)
        if rc != 0:
            failed.append((run_name(config), rc))

    # Summary
    wall = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  DONE — {total} experiments in {wall/3600:.1f} h")
    if failed:
        print(f"  FAILED ({len(failed)}):")
        for name, rc in failed:
            print(f"    - {name}  (exit {rc})")
    else:
        print("  All experiments succeeded.")
    print(f"{'='*70}\n")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
