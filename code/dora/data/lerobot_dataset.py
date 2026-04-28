"""
Push-T dataset loader for SmolVLM action prediction.

Wraps the LeRobot lerobot/pusht dataset into a PyTorch Dataset compatible
with HuggingFace Trainer.  Each sample is a single timestep:
    pixel_values   : float bf16/f32  [n_patches, 3, 512, 512]  (SmolVLM processed)
    input_ids      : long             [seq_len]
    attention_mask : long             [seq_len]
    labels         : float f32        [2]   (normalised x, y action)

Split is done at the episode level so no episode appears in both train and val.
"""

import random
from typing import Tuple

import torch
from torch.utils.data import Dataset

# Language instruction shared by all pusht frames
PUSHT_INSTRUCTION = "Push the T-shaped block onto the T-shaped target."
SMOLVLM_MODEL_ID  = "HuggingFaceTB/SmolVLM-256M-Instruct"


class PushTVLADataset(Dataset):
    """
    Single-timestep (image, instruction) → action dataset from lerobot/pusht.

    The processor is applied per-sample rather than per-batch to keep the
    collator simple (all outputs are same length after padding inside the proc).
    """

    def __init__(self, indices: list, lerobot_ds, processor, dtype=torch.float32):
        self.indices   = indices
        self.ds        = lerobot_ds
        self.processor = processor
        self.dtype     = dtype

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        sample = self.ds[self.indices[idx]]

        # Convert float32 [3, 96, 96] tensor → PIL image
        from PIL import Image
        import numpy as np
        img_np = (sample["observation.image"].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        # Process image + text together
        encoding = self.processor(
            text=f"<image>{PUSHT_INSTRUCTION}",
            images=pil_img,
            return_tensors="pt",
            padding=False,
        )

        return {
            "pixel_values":   encoding["pixel_values"].squeeze(0).to(self.dtype),  # [n_patches, 3, 512, 512]
            "input_ids":      encoding["input_ids"].squeeze(0),                      # [seq_len]
            "attention_mask": encoding["attention_mask"].squeeze(0),                  # [seq_len]
            "labels":         sample["action"].float(),                               # [2]
        }


def load_pusht(
    processor=None,
    val_split: float = 0.2,
    seed: int = 42,
    dtype=torch.float32,
) -> Tuple["PushTVLADataset", "PushTVLADataset"]:
    """
    Load lerobot/pusht and return (train_dataset, val_dataset).
    Split is episode-level (no leakage between splits).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from transformers import AutoProcessor

    if processor is None:
        # do_image_splitting=False: disable SmolVLM's tile-into-17-patches behavior.
        # pusht images are 96×96 and don't need it; tiling adds 17× compute overhead
        # and hits Blackwell-specific kernel issues at high token counts.
        processor = AutoProcessor.from_pretrained(SMOLVLM_MODEL_ID, do_image_splitting=False)

    ds = LeRobotDataset("lerobot/pusht")

    # Use episode_data_index for O(1) episode boundary lookup (avoids iterating 25k frames)
    edi      = ds.episode_data_index            # {'from': tensor[N], 'to': tensor[N]}
    n_eps    = len(edi["from"])
    episodes = list(range(n_eps))

    rng = random.Random(seed)
    rng.shuffle(episodes)
    n_val = max(1, int(n_eps * val_split))
    val_eps   = sorted(episodes[:n_val])
    train_eps = sorted(episodes[n_val:])

    def _frames_for(ep_list):
        out = []
        for ep in ep_list:
            out.extend(range(edi["from"][ep].item(), edi["to"][ep].item()))
        return out

    train_idx = _frames_for(train_eps)
    val_idx   = _frames_for(val_eps)

    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        f"PushT split → {len(train_eps)} train episodes ({len(train_idx)} frames), "
        f"{len(val_eps)} val episodes ({len(val_idx)} frames)"
    )

    return (
        PushTVLADataset(train_idx, ds, processor, dtype=dtype),
        PushTVLADataset(val_idx,   ds, processor, dtype=dtype),
    )
