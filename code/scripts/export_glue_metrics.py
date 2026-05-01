#!/usr/bin/env python3
"""
Export GLUE metrics + adapter stats summaries for poster charts.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _load_adapter_stats(results_dir: Path, run_name: str):
    run_dir = results_dir / run_name
    stats = _load_json(run_dir / "adapter_stats.json")
    if stats is not None:
        return stats, str(run_dir / "adapter_stats.json")

    # Fallback: check legacy *_stats directory
    stats_dir = results_dir / f"{run_name}_stats"
    stats = _load_json(stats_dir / "adapter_stats.json")
    if stats is not None:
        return stats, str(stats_dir / "adapter_stats.json")

    return None, None


def main():
    ap = argparse.ArgumentParser(description="Export GLUE run summaries")
    ap.add_argument("--results_dir", default="../results")
    ap.add_argument("--output", default="../results/glue_run_summaries.json")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_path = Path(args.output).resolve()
    summaries = []

    for run_dir in sorted(results_dir.glob("glue_*")):
        if not run_dir.is_dir():
            continue
        # Skip legacy *_stats dirs; use them as fallback for the base run.
        if run_dir.name.endswith("_stats"):
            continue

        adapter_stats, stats_source = _load_adapter_stats(results_dir, run_dir.name)
        last_stats = adapter_stats[-1] if adapter_stats else None

        summary = {
            "run": run_dir.name,
            "adapter_stats_last": last_stats,
            "adapter_stats_source": stats_source,
        }
        summaries.append(summary)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"Wrote {len(summaries)} run summaries to {out_path}")


if __name__ == "__main__":
    main()
