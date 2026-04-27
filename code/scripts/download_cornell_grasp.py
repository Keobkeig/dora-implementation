#!/usr/bin/env python3
"""
Download and extract the Cornell Grasp Dataset via Kaggle.

Setup (one-time):
  1. Create a free account at https://www.kaggle.com
  2. Go to https://www.kaggle.com/settings → "Create New Token"
     → downloads kaggle.json
  3. Place it at:   C:\\Users\\<you>\\.kaggle\\kaggle.json
     (or ~/.kaggle/kaggle.json on Linux/WSL)

Then run:
    uv run scripts/download_cornell_grasp.py

Or manually: download the zip from
    https://www.kaggle.com/datasets/oneoneliu/cornell-grasp
and pass --zip_path to point at it.
"""

import argparse
import os
import shutil
import sys
import zipfile
import tarfile
from pathlib import Path

DEFAULT_OUT = Path(__file__).parent.parent.parent / "data" / "cornell_grasps"
KAGGLE_DATASET = "oneoneliu/cornell-grasp"


def download_via_kaggle(out_dir: Path):
    try:
        import kaggle  # noqa: F401 — triggers auth check
    except ImportError:
        print("Installing kaggle ...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])
        import kaggle  # noqa: F401

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\nkaggle.json not found. One-time setup:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New Token'  →  saves kaggle.json")
        print(f"  3. Move it to:  {kaggle_json}")
        print("\nThen re-run this script.")
        sys.exit(1)

    print(f"Downloading '{KAGGLE_DATASET}' via Kaggle API ...")
    tmp = out_dir.parent / "_kaggle_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    from kaggle.api.kaggle_api_extended import KaggleApiExtended
    api = KaggleApiExtended()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(tmp), unzip=False, quiet=False)

    zip_files = list(tmp.glob("*.zip"))
    if not zip_files:
        raise RuntimeError(f"No zip found in {tmp} after download.")
    return zip_files[0]


def extract_zip(zip_path: Path, out_dir: Path):
    print(f"Extracting {zip_path.name} → {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    _flatten_if_needed(out_dir)


def extract_tar(tar_path: Path, out_dir: Path):
    print(f"Extracting {tar_path.name} → {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out_dir)
    _flatten_if_needed(out_dir)


def _flatten_if_needed(out_dir: Path):
    """If everything extracted into a single subdir, move files up one level."""
    pngs = list(out_dir.glob("pcd*r.png"))
    if pngs:
        return
    subdirs = [p for p in out_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        subdir = subdirs[0]
        for item in subdir.iterdir():
            dest = out_dir / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
        try:
            subdir.rmdir()
        except OSError:
            pass


def verify(out_dir: Path):
    pngs  = list(out_dir.glob("pcd*r.png")) + list(out_dir.glob("**/*r.png"))
    cpos  = list(out_dir.glob("pcd*cpos.txt")) + list(out_dir.glob("**/*cpos.txt"))
    print(f"Verified: {len(pngs)} images, {len(cpos)} positive-grasp files in {out_dir}")
    if not pngs:
        print("ERROR: no pcd*r.png images found — check extraction path.")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--zip_path", default=None,
                   help="Path to a pre-downloaded .zip or .tar.gz — skips API download")
    p.add_argument("--keep_archive", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)

    # Already present?
    existing = list(out_dir.glob("pcd*r.png")) + list(out_dir.glob("**/*r.png"))
    if existing:
        print(f"Dataset already present at {out_dir} ({len(existing)} images).")
        return

    # Use provided archive or download via Kaggle
    if args.zip_path:
        archive = Path(args.zip_path)
    else:
        archive = download_via_kaggle(out_dir)

    suffix = "".join(archive.suffixes)
    if ".tar" in suffix or suffix.endswith(".gz"):
        extract_tar(archive, out_dir)
    else:
        extract_zip(archive, out_dir)

    verify(out_dir)

    if not args.keep_archive and not args.zip_path:
        archive.unlink(missing_ok=True)
        tmp = out_dir.parent / "_kaggle_tmp"
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    print("All done. Dataset ready at:", out_dir)


if __name__ == "__main__":
    main()
