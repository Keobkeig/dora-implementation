"""
Cornell Grasp Dataset loader for grasp pose estimation.

Download instructions:
  Option A (direct):
    wget http://pr.cs.cornell.edu/grasping/rect_data/data.tar.gz
    tar -xzf data.tar.gz -C data/cornell_grasps/

  Option B (Kaggle mirror, more reliable):
    kaggle datasets download -d oneoneliu/cornell-grasp
    unzip cornell-grasp.zip -d data/cornell_grasps/

Expected directory layout (flat or nested — both work):
    data/cornell_grasps/
        pcd0001r.png
        pcd0001cpos.txt
        pcd0001cneg.txt
        ...

Each *cpos.txt file lists positive grasps — 4 lines of "x y" per grasp, blank line between.
Each grasp is an oriented rectangle: corners[0]→corners[1] defines the opening (width) axis.
"""

import glob
import logging
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def _parse_grasp_file(filepath: str) -> List[np.ndarray]:
    """
    Return a list of [4, 2] corner arrays from a Cornell annotation file.

    Handles two common formats:
      A) Blank-line separated groups of 4 points (original Cornell release)
      B) Continuous 4-lines-per-grasp with no blank separator (Kaggle mirror)
    """
    points = []
    try:
        with open(filepath) as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        points.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        pass
    except OSError:
        return []

    # Group every 4 consecutive points into one grasp rectangle
    grasps = []
    for i in range(0, len(points) - 3, 4):
        grasps.append(np.array(points[i:i + 4], dtype=np.float32))
    return grasps


def rect_to_pose(corners: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Convert [4, 2] corner rectangle to (cx, cy, angle_rad, width, height).
    corners[0]→corners[1] defines the width (opening) direction.
    """
    cx = float(corners[:, 0].mean())
    cy = float(corners[:, 1].mean())
    dx, dy = corners[1, 0] - corners[0, 0], corners[1, 1] - corners[0, 1]
    width = float(np.hypot(dx, dy))
    angle = float(np.arctan2(dy, dx))
    dx2, dy2 = corners[2, 0] - corners[1, 0], corners[2, 1] - corners[1, 1]
    height = float(np.hypot(dx2, dy2))
    return cx, cy, angle, width, height


def pose_to_target(cx: float, cy: float, angle: float, width: float, height: float,
                   img_w: float = 1.0, img_h: float = 1.0) -> np.ndarray:
    """
    Encode grasp as 6-D regression target:
      [x_n, y_n, sin(2θ), cos(2θ), w_n, h_n]
    Using 2θ captures the 180° grasp symmetry.
    Normalise coords by image dims (default 1.0 → keep pixel units).
    """
    return np.array([
        cx / img_w,
        cy / img_h,
        np.sin(2.0 * angle),
        np.cos(2.0 * angle),
        width / img_w,
        height / img_h,
    ], dtype=np.float32)


def _build_corners(cx: float, cy: float, angle: float,
                   width: float, height: float) -> np.ndarray:
    """Reconstruct [4, 2] corners from pose."""
    ca, sa = np.cos(angle), np.sin(angle)
    dx1, dy1 = (width / 2) * ca,  (width / 2) * sa
    dx2, dy2 = -(height / 2) * sa, (height / 2) * ca
    return np.array([
        [cx - dx1 - dx2, cy - dy1 - dy2],
        [cx + dx1 - dx2, cy + dy1 - dy2],
        [cx + dx1 + dx2, cy + dy1 + dy2],
        [cx - dx1 + dx2, cy - dy1 + dy2],
    ], dtype=np.float32)


def _iou_oriented(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    """Polygon IoU between two 4-corner oriented bounding boxes."""
    try:
        from shapely.geometry import Polygon
        pa = Polygon(corners_a.tolist())
        pb = Polygon(corners_b.tolist())
        if not pa.is_valid or not pb.is_valid:
            return 0.0
        inter = pa.intersection(pb).area
        union = pa.union(pb).area
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return 0.0


def _is_success(pred: np.ndarray, gt: np.ndarray,
                img_w: float = 1.0, img_h: float = 1.0,
                iou_thresh: float = 0.25, angle_thresh: float = 30.0) -> bool:
    """
    Cornell grasp success criterion: IoU ≥ 0.25 AND |Δangle| ≤ 30°.
    By default works in normalised space (img_w=img_h=1).
    """
    cx_p, cy_p = pred[0] * img_w, pred[1] * img_h
    angle_p = 0.5 * float(np.arctan2(pred[2], pred[3]))
    w_p, h_p = pred[4] * img_w, pred[5] * img_h

    cx_g, cy_g = gt[0] * img_w, gt[1] * img_h
    angle_g = 0.5 * float(np.arctan2(gt[2], gt[3]))
    w_g, h_g = gt[4] * img_w, gt[5] * img_h

    c_p = _build_corners(cx_p, cy_p, angle_p, w_p, h_p)
    c_g = _build_corners(cx_g, cy_g, angle_g, w_g, h_g)

    iou = _iou_oriented(c_p, c_g)
    angle_diff = abs(np.degrees(angle_p) - np.degrees(angle_g))
    angle_diff = min(angle_diff, 180.0 - angle_diff)
    return iou >= iou_thresh and angle_diff <= angle_thresh


def grasp_success_from_arrays(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Cornell grasp success rate from batched numpy arrays.
    Args:
        preds:  [N, 6] predicted targets (normalised)
        labels: [N, 6] ground-truth targets (normalised)
    Returns:
        success_rate in [0, 1]
    """
    successes = [_is_success(p, g) for p, g in zip(preds, labels)]
    return float(np.mean(successes)) if successes else 0.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CornellGraspDataset(Dataset):
    """
    One sample = one (image, best positive grasp) pair.

    Returns dict with keys:
        pixel_values : float32 tensor [3, H, W]   (preprocessed by ViT processor)
        labels       : float32 tensor [6]          (normalised regression target)
    """

    def __init__(self, image_paths: List[str], grasp_corners: List[np.ndarray],
                 processor):
        assert len(image_paths) == len(grasp_corners)
        self.image_paths = image_paths
        self.grasp_corners = grasp_corners  # [4, 2] in pixel coords per sample
        self.processor = processor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        corners = self.grasp_corners[idx]
        cx, cy, angle, width, height = rect_to_pose(corners)
        target = pose_to_target(cx, cy, angle, width, height, img.width, img.height)

        pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(target, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cornell_grasp(
    data_dir: str,
    processor,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple["CornellGraspDataset", "CornellGraspDataset"]:
    """
    Scan *data_dir* for Cornell Grasp files, build train/val datasets.

    The split is done at the image level so no image appears in both sets.
    """
    data_dir = str(Path(data_dir).expanduser())
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Cornell Grasp data directory not found: {data_dir}\n\n"
            "Download instructions:\n"
            "  wget http://pr.cs.cornell.edu/grasping/rect_data/data.tar.gz\n"
            "  tar -xzf data.tar.gz -C data/cornell_grasps/\n"
            "Then pass --data_dir data/cornell_grasps"
        )

    # Find all RGB images (named pcd*r.png or pcd*r.jpg)
    image_files = sorted(
        glob.glob(os.path.join(data_dir, "**", "pcd*r.png"), recursive=True)
        + glob.glob(os.path.join(data_dir, "**", "pcd*r.jpg"), recursive=True)
    )
    if not image_files:
        raise FileNotFoundError(
            f"No 'pcd*r.png' images found in {data_dir}. "
            "Check that the dataset is extracted correctly."
        )

    image_paths, grasp_corners = [], []
    for img_path in image_files:
        # Derive positive grasp file: pcd0001r.png → pcd0001cpos.txt
        stem = os.path.basename(img_path)          # pcd0001r.png
        base = stem.replace("r.png", "").replace("r.jpg", "")  # pcd0001
        cpos_path = os.path.join(os.path.dirname(img_path), base + "cpos.txt")
        grasps = _parse_grasp_file(cpos_path)
        if not grasps:
            continue
        image_paths.append(img_path)
        grasp_corners.append(grasps[0])  # use first positive grasp per image

    if not image_paths:
        raise RuntimeError(f"No valid (image, grasp) pairs found in {data_dir}.")

    logger.info(f"Loaded {len(image_paths)} image/grasp pairs from {data_dir}")

    # Image-level train/val split
    rng = random.Random(seed)
    indices = list(range(len(image_paths)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_split))
    val_idx = set(indices[:n_val])

    def _split(idx_set, complement=False):
        idx_list = [i for i in range(len(image_paths))
                    if (i in idx_set) != complement]
        return (
            [image_paths[i] for i in idx_list],
            [grasp_corners[i] for i in idx_list],
        )

    train_paths, train_grasps = _split(val_idx, complement=True)
    val_paths,   val_grasps   = _split(val_idx, complement=False)

    logger.info(f"Split → train: {len(train_paths)}, val: {len(val_paths)}")

    return (
        CornellGraspDataset(train_paths, train_grasps, processor),
        CornellGraspDataset(val_paths,   val_grasps,   processor),
    )
