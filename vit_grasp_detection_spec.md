# ViT Grasp Detection: DoRA vs LoRA vs FFT — Full Experiment Specification

## Project Goal

Fine-tune a pre-trained Vision Transformer (ViT-Base/16) on the **Cornell Grasp Dataset** for **robotic grasp pose regression**, comparing three fine-tuning strategies:

1. **DoRA** (Weight-Decomposed Low-Rank Adaptation) — our custom implementation
2. **LoRA** (Low-Rank Adaptation) — our custom implementation
3. **FFT** (Full Fine-Tuning) — standard baseline

The experiment measures grasp detection accuracy, parameter efficiency, training speed, and convergence behavior across all three methods, with a rank sweep for DoRA/LoRA.

---

## Existing Codebase Context

> [!IMPORTANT]
> All new code MUST live under the existing project at:
> `/Users/shauryasen/Class/code/dev/4782/dora-implementation/code/`
> The project uses `uv` for dependency management with Python 3.11.

### Repository Structure (relevant files)

```
code/
├── dora/
│   ├── __init__.py               # Exports DoRALinear, LoRALinear, etc.
│   ├── layers/
│   │   ├── base.py               # DoRAModule, DoRAConfig, DoRAStateManager
│   │   ├── dora_linear.py        # DoRALinear, DoRAConv2d, create_dora_layer()
│   │   └── lora_linear.py        # LoRALinear, create_lora_layer(), apply_lora_to_model()
│   ├── models/
│   │   ├── vision_transformer.py # ViTDoRAConfig, ViTDoRAModel, convert_vit_to_dora()
│   │   └── llama.py
│   └── utils/
│       ├── math_utils.py         # compute_dora_weight(), initialize_dora_magnitude()
│       └── model_utils.py
├── training/
│   └── trainer.py                # DoRATrainer, TrainingConfig
├── benchmarks/
│   └── dora_vs_lora.py           # DoRABenchmark, MemoryProfiler, SpeedProfiler
├── scripts/
│   └── train_simple_classification.py  # Example training pattern to follow
├── requirements.txt
└── pyproject.toml                # Python >=3.11,<3.12; uses uv
```

### Key APIs You Must Use

#### DoRA Layer Injection (already implemented)
```python
from dora.models.vision_transformer import convert_vit_to_dora, ViTDoRAConfig
from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
config = ViTDoRAConfig(rank=8, alpha=16.0, dropout=0.1,
                       target_modules=["query", "key", "value", "dense",
                                       "intermediate.dense", "output.dense"])
model = convert_vit_to_dora(model, rank=8, alpha=16.0)
```

#### LoRA Layer Injection (already implemented)
```python
from dora.layers.lora_linear import apply_lora_to_model

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
# For ViT, target_modules should be: ["query", "key", "value", "dense"]
model = apply_lora_to_model(model, target_modules=["query", "key", "value", "dense"],
                            rank=8, alpha=16.0, dropout=0.1)
```

#### Important Implementation Details

1. **DoRALinear** stores `base_weight` as a buffer (frozen), `lora_A`, `lora_B`, and `magnitude` as Parameters (trainable). Forward: `W = m * (W_0 + scaling * BA) / ||W_0 + scaling * BA||`
2. **LoRALinear** stores `base_weight` as a buffer (frozen), `lora_A` and `lora_B` as Parameters. Forward: `y = F.linear(x, W_0, bias) + scaling * F.linear(dropout(x), B @ A)`
3. **convert_vit_to_dora()** replaces all ViT encoder layers with `DoRAViTLayer` wrappers. The classifier head is also replaced.
4. **apply_lora_to_model()** walks the module tree and replaces any `nn.Linear` whose leaf name matches `target_modules`.
5. Both DoRA and LoRA use `create_dora_layer()` / `create_lora_layer()` factory functions that automatically call `load_base_weight()` from the pre-trained layer.
6. The DoRA ViT conversion replaces attention (Q, K, V), attention output (dense), intermediate (intermediate.dense), and output (output.dense) layers.

#### Training Pattern
Follow the pattern in `scripts/train_simple_classification.py`:
- Manual training loop with `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`
- Epoch-level evaluation with accuracy, loss tracking
- Console logging of per-epoch metrics
- Parameter counting via `DoRAModule.count_parameters(model)`

---

## Dataset: Cornell Grasp Dataset

### Overview
- **Source**: http://pr.cs.cornell.edu/grasping/rect_data/data.php
- **Size**: ~885 RGB-D images of 240 objects
- **Annotations**: Oriented grasp rectangles defined by 4 corner points in `cpos.txt` (positive grasps) and `cneg.txt` (negative grasps)
- **We only use positive grasps** (`cpos.txt`) for regression

### Download & Storage
- Download the dataset to `code/data/cornell/` (create this directory)
- The raw data is a tar.gz archive (~77MB) from the Cornell website
- Alternative: use a mirror on Kaggle (`oneoneliu/cornell-grasp`)

### Annotation Format
Each `pcdXXXXcpos.txt` file contains groups of 4 lines, each group defining one grasp rectangle:
```
x1 y1
x2 y2
x3 y3
x4 y4
```
Where (x1,y1)→(x2,y2) is one edge and (x3,y3)→(x4,y4) is the opposite edge.

### Grasp Representation (5-DOF)
Convert each rectangle to a 5-dimensional grasp vector:
```python
# From 4 corner points, compute:
center_x = mean of all 4 x-coordinates
center_y = mean of all 4 y-coordinates
angle = atan2(y2 - y1, x2 - x1)  # orientation of the first edge
width = distance((x1,y1), (x2,y2))  # length of the first edge
height = distance((x2,y2), (x3,y3)) # distance between edges (gripper opening)

grasp = [center_x, center_y, angle, width, height]
```

### Preprocessing Pipeline
1. Load RGB image (ignore depth for simplicity; depth is optional enhancement)
2. Parse the best grasp rectangle (highest-quality or first valid one per image)
3. Crop to a **320×320** region centered on the grasp center (with boundary padding)
4. Resize to **224×224** (ViT input size)
5. Normalize with ImageNet mean/std: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`
6. Normalize grasp coordinates relative to the crop (so they're in [0, 1] range)
7. Normalize angle to [-π, π] range
8. Normalize width and height by dividing by the crop size

### Train/Test Split
Use **image-wise split** (not object-wise): 80% train, 10% validation, 10% test. Seed the random split with `seed=42` for reproducibility.

---

## Model Architecture

### Base Model
- **Model**: `google/vit-base-patch16-224` from HuggingFace
- **Parameters**: ~86M
- **Architecture**: 12 transformer encoder blocks, hidden size 768, 12 attention heads
- **Input**: 224×224 RGB images → 196 patches of 16×16

### Task Head Modification

> [!IMPORTANT]
> The default `ViTForImageClassification` has a classification head. We need to **replace it with a regression head** for grasp prediction.

Replace the classifier with:
```python
model.classifier = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 5)   # Outputs: [cx, cy, angle, width, height]
)
```

This must be done **before** applying DoRA/LoRA so that the DoRA/LoRA injection also adapts the regression head layers.

### Three Experiment Configurations

#### 1. FFT (Full Fine-Tuning)
- Load `ViTForImageClassification`, replace classifier head
- Unfreeze ALL parameters
- Train everything end-to-end
- Loss: `MSELoss` on the 5-DOF grasp vector

#### 2. LoRA
- Load `ViTForImageClassification`, replace classifier head
- Apply `apply_lora_to_model()` with `target_modules=["query", "key", "value", "dense"]`
- Freeze all base weights, only train LoRA matrices (A, B) + regression head
- Loss: `MSELoss` on the 5-DOF grasp vector

#### 3. DoRA
- Load `ViTForImageClassification`, replace classifier head
- Apply `convert_vit_to_dora()` which replaces attention + MLP layers
- Freeze all base weights, only train LoRA matrices (A, B) + magnitude vectors (m) + regression head
- Loss: `MSELoss` on the 5-DOF grasp vector

---

## Hyperparameters

### Shared Across All Methods
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Batch Size | 32 |
| Max Epochs | 30 |
| LR Scheduler | CosineAnnealingLR |
| Warmup Steps | 100 |
| Max Grad Norm | 1.0 |
| Image Size | 224×224 |
| Loss Function | MSELoss |
| Random Seed | 42 |
| Early Stopping Patience | 7 epochs |
| Data Augmentation | RandomHorizontalFlip, RandomRotation(±10°), ColorJitter(0.1) |

### Method-Specific
| Parameter | FFT | LoRA | DoRA |
|-----------|-----|------|------|
| Learning Rate | 1e-5 | 2e-4 | 2e-4 |
| Dropout | 0.1 | 0.1 | 0.1 |
| Trainable Params | ~86M | ~0.3-2M (rank-dependent) | ~0.3-2M + magnitude |

### Rank Sweep (for LoRA & DoRA)
Run each of LoRA and DoRA with: `rank ∈ {4, 8, 16, 32}`
Use `alpha = 2 * rank` for each configuration (standard heuristic).

---

## Evaluation Metrics

### Primary Metric: Grasp Detection Accuracy (Rectangle Metric)
A predicted grasp is **correct** if:
1. **IoU** between predicted and ground-truth grasp rectangles > 0.25
2. **Angle difference** between predicted and ground-truth < 30°

Report: accuracy = (# correct predictions) / (# total predictions)

### Secondary Metrics
1. **MSE Loss** on the 5-DOF grasp vector (train and validation)
2. **Per-component errors**: individual MSE for cx, cy, angle, width, height
3. **Trainable parameter count** for each method and rank
4. **Training wall-clock time** per epoch
5. **Peak GPU memory** usage during training
6. **Convergence speed**: epochs to reach 80% of best accuracy

---

## Files to Create

All new files go under `code/`. Here is the complete file manifest:

### 1. `code/experiments/grasp_detection/` — Main experiment directory

#### `code/experiments/grasp_detection/__init__.py`
Empty init file.

#### `code/experiments/grasp_detection/dataset.py`
Cornell Grasp Dataset implementation.

**Must implement:**
- `class CornellGraspDataset(torch.utils.data.Dataset)`:
  - `__init__(self, data_dir, split, transform, seed, img_size)`: Parse all image/annotation pairs, create train/val/test split
  - `_parse_grasp_rectangles(self, filepath)`: Parse `cpos.txt` files, handle NaN values
  - `_rectangle_to_grasp(self, points)`: Convert 4 corners to 5-DOF `[cx, cy, angle, w, h]`
  - `__getitem__(self, idx)`: Load image, crop around grasp center, resize to 224×224, normalize, return `(image_tensor, grasp_tensor)`
  - `__len__(self)`: Return dataset size
- `def get_cornell_dataloaders(data_dir, batch_size, seed, img_size)`: Factory function returning train/val/test DataLoaders
- `def download_cornell_dataset(target_dir)`: Download and extract the dataset if not present

#### `code/experiments/grasp_detection/model.py`
Model construction for all three methods.

**Must implement:**
- `def create_grasp_model(method, rank, alpha, dropout)`:
  - Loads `google/vit-base-patch16-224`
  - Replaces classifier head with regression head (768→256→5)
  - Applies DoRA, LoRA, or keeps as-is based on `method` parameter
  - Freezes appropriate parameters
  - Returns model
- `def count_trainable_params(model)`: Returns dict with total, trainable, frozen counts
- `def freeze_base_weights(model, method)`: Freeze all non-adapter, non-head parameters

**For LoRA**: Use `apply_lora_to_model()` from `dora.layers.lora_linear`
**For DoRA**: Use `convert_vit_to_dora()` from `dora.models.vision_transformer`
**For FFT**: Keep all parameters trainable

> [!WARNING]
> The regression head must be created **before** applying DoRA/LoRA. For DoRA, `convert_vit_to_dora()` will also adapt the classifier/regression head if `adapt_classifier=True`. For LoRA, the head layers will be matched if "dense" is not too generic — be careful with target module names. You may need to manually keep the regression head trainable after freezing.

#### `code/experiments/grasp_detection/metrics.py`
Grasp evaluation metrics.

**Must implement:**
- `def compute_grasp_iou(pred_rect, gt_rect)`: Compute IoU between two oriented rectangles using Shapely or manual polygon intersection
- `def rectangle_metric(pred_grasps, gt_grasps, iou_threshold=0.25, angle_threshold=30)`: Compute grasp detection accuracy
- `def grasp_to_rectangle(grasp_5dof)`: Convert [cx, cy, angle, w, h] back to 4 corner points
- `def per_component_mse(preds, targets)`: Return dict of per-component MSE values

#### `code/experiments/grasp_detection/train.py`
Main training script — the primary entry point.

**Must implement:**
- `def train_one_epoch(model, dataloader, optimizer, criterion, device)`: Single epoch training, returns avg loss
- `def evaluate(model, dataloader, criterion, device, compute_rectangle_metric)`: Evaluation, returns loss + rectangle accuracy
- `def run_experiment(method, rank, alpha, ...)`: Full training pipeline for one configuration
- `def main()`: Orchestrate all experiments using argparse

**Command-line interface:**
```bash
# Run a single experiment
python -m experiments.grasp_detection.train --method dora --rank 8 --alpha 16

# Run full sweep
python -m experiments.grasp_detection.train --sweep

# Run with specific GPU
CUDA_VISIBLE_DEVICES=0 python -m experiments.grasp_detection.train --sweep
```

**Arguments:**
- `--method`: One of `dora`, `lora`, `fft`
- `--rank`: LoRA/DoRA rank (default 8)
- `--alpha`: LoRA/DoRA alpha (default 16)
- `--data_dir`: Path to Cornell dataset (default `data/cornell`)
- `--batch_size`: Batch size (default 32)
- `--epochs`: Max epochs (default 30)
- `--lr`: Learning rate (default depends on method)
- `--seed`: Random seed (default 42)
- `--output_dir`: Where to save results (default `results/grasp_detection`)
- `--sweep`: Run full rank sweep across all methods
- `--device`: Device to use (default auto-detect)
- `--wandb`: Enable W&B logging (default False)

**Training loop must:**
1. Log per-epoch: train_loss, val_loss, val_rectangle_accuracy, learning_rate
2. Save best model checkpoint based on val_rectangle_accuracy
3. Implement early stopping (patience=7)
4. Save training curves (loss and accuracy per epoch) to JSON
5. Print parameter counts at start of each run
6. Track wall-clock time per epoch

#### `code/experiments/grasp_detection/visualize.py`
Visualization and analysis script.

**Must implement:**
- `def plot_training_curves(results_dir)`: Plot loss and accuracy curves for all methods, save as PNG
- `def plot_parameter_efficiency(results_dir)`: Bar chart of accuracy vs. trainable params
- `def plot_rank_sweep(results_dir)`: Line plot of accuracy vs. rank for DoRA and LoRA
- `def plot_convergence_comparison(results_dir)`: Overlay convergence curves
- `def visualize_grasp_predictions(model, dataset, num_samples)`: Show images with predicted vs. GT grasp rectangles
- `def generate_summary_table(results_dir)`: Print a LaTeX/markdown table summarizing all results
- `def main()`: Generate all plots from saved results

**Output plots (save to `results/grasp_detection/plots/`):**
1. `training_curves.png` — Loss curves for all methods
2. `accuracy_comparison.png` — Bar chart of final rectangle accuracy
3. `rank_sweep.png` — Accuracy vs. rank for DoRA and LoRA
4. `parameter_efficiency.png` — Accuracy vs. trainable parameter count
5. `convergence_speed.png` — Epochs to reach threshold accuracy
6. `grasp_predictions.png` — Qualitative visualization of predictions
7. `memory_comparison.png` — GPU memory usage comparison

#### `code/experiments/grasp_detection/config.py`
Centralized experiment configuration.

**Must implement:**
- `@dataclass class ExperimentConfig`: All hyperparameters in one place
- `def get_sweep_configs()`: Return list of configs for the full sweep
- `def get_default_config(method)`: Return default config for each method

#### `code/experiments/__init__.py`
Empty init file.

### 2. Results Storage

```
results/grasp_detection/
├── dora_r4/          # DoRA rank=4 results
│   ├── metrics.json  # Per-epoch metrics
│   ├── best_model.pt # Best checkpoint
│   └── config.json   # Experiment config
├── dora_r8/
├── dora_r16/
├── dora_r32/
├── lora_r4/
├── lora_r8/
├── lora_r16/
├── lora_r32/
├── fft/
├── plots/            # All visualization outputs
└── summary.md        # Auto-generated results summary
```

---

## Step-by-Step Implementation Checklist

### Phase 0: Environment Setup
- [ ] Verify Python 3.11 and `uv` are working
- [ ] Install additional dependencies: `pip install shapely Pillow` (for polygon IoU and image loading)
- [ ] Add `shapely` and `Pillow` to `requirements.txt`
- [ ] Verify the existing DoRA codebase runs: `cd code && python -c "from dora import DoRALinear, LoRALinear; print('OK')"`
- [ ] Verify GPU access: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"`

### Phase 1: Dataset Pipeline
- [ ] Create `code/experiments/__init__.py`
- [ ] Create `code/experiments/grasp_detection/__init__.py`
- [ ] Implement `code/experiments/grasp_detection/dataset.py`
  - [ ] `download_cornell_dataset()` — download + extract
  - [ ] `CornellGraspDataset` class with parsing, preprocessing, augmentation
  - [ ] `get_cornell_dataloaders()` factory
- [ ] Download the Cornell dataset to `code/data/cornell/`
- [ ] Test the dataset: load a few samples, verify shapes, visualize an image+grasp pair
- [ ] Verify train/val/test split sizes are reasonable (~700/~90/~90)

### Phase 2: Model Construction
- [ ] Implement `code/experiments/grasp_detection/model.py`
  - [ ] `create_grasp_model("fft")` — verify output shape is (batch, 5)
  - [ ] `create_grasp_model("lora", rank=8)` — verify only LoRA params are trainable + head
  - [ ] `create_grasp_model("dora", rank=8)` — verify only DoRA params are trainable + head
- [ ] Print and verify parameter counts for each method and rank
- [ ] Smoke-test forward pass: random input → model → output shape (batch, 5)

### Phase 3: Evaluation Metrics
- [ ] Implement `code/experiments/grasp_detection/metrics.py`
  - [ ] `grasp_to_rectangle()` — convert 5-DOF to 4 corners
  - [ ] `compute_grasp_iou()` — polygon intersection over union
  - [ ] `rectangle_metric()` — accuracy with IoU + angle thresholds
  - [ ] `per_component_mse()` — per-channel breakdown
- [ ] Unit test: create known grasp pairs, verify IoU and metric correctness

### Phase 4: Training Pipeline
- [ ] Implement `code/experiments/grasp_detection/config.py`
- [ ] Implement `code/experiments/grasp_detection/train.py`
  - [ ] `train_one_epoch()` with gradient clipping, loss accumulation
  - [ ] `evaluate()` with rectangle metric computation
  - [ ] `run_experiment()` full pipeline with checkpointing, early stopping
  - [ ] `main()` with argparse CLI
- [ ] Smoke-test: run 2 epochs of FFT on a tiny subset (10 images), verify loss decreases
- [ ] Smoke-test: run 2 epochs of LoRA rank=8, verify loss decreases
- [ ] Smoke-test: run 2 epochs of DoRA rank=8, verify loss decreases

### Phase 5: Full Experiment Run
- [ ] Run FFT experiment (30 epochs, full dataset)
- [ ] Run LoRA sweep: rank ∈ {4, 8, 16, 32} (30 epochs each)
- [ ] Run DoRA sweep: rank ∈ {4, 8, 16, 32} (30 epochs each)
- [ ] Verify all 9 experiment runs saved results to `results/grasp_detection/`
- [ ] Check for any NaN losses or training failures

### Phase 6: Visualization & Analysis
- [ ] Implement `code/experiments/grasp_detection/visualize.py`
- [ ] Generate all plots:
  - [ ] Training curves (loss + accuracy)
  - [ ] Accuracy bar chart comparison
  - [ ] Rank sweep line plot
  - [ ] Parameter efficiency plot
  - [ ] Convergence speed comparison
  - [ ] Qualitative grasp prediction visualizations
  - [ ] Memory usage comparison
- [ ] Generate summary table (markdown + LaTeX)
- [ ] Write auto-generated `results/grasp_detection/summary.md`

### Phase 7: Final Verification
- [ ] All plots saved to `results/grasp_detection/plots/`
- [ ] Summary table shows all 9 configurations with key metrics
- [ ] Verify DoRA shows competitive or better accuracy than LoRA at same rank
- [ ] Verify FFT achieves highest accuracy (expected, as upper bound)
- [ ] Verify parameter counts are consistent with expectations
- [ ] Code is clean, documented, and follows existing project style (black, 100 char lines)
- [ ] Run `python -m experiments.grasp_detection.train --sweep` end-to-end without errors

---

## Expected Results (Ballpark)

| Method | Rank | Trainable Params | Rectangle Accuracy | Notes |
|--------|------|------------------|-------------------|-------|
| FFT | — | ~86M (100%) | 85-92% | Upper bound |
| LoRA | 4 | ~0.3M (<1%) | 70-78% | Lowest rank |
| LoRA | 8 | ~0.6M (<1%) | 75-82% | |
| LoRA | 16 | ~1.2M (~1.4%) | 78-85% | |
| LoRA | 32 | ~2.4M (~2.8%) | 80-87% | |
| DoRA | 4 | ~0.3M + 9K mag | 73-80% | Should beat LoRA-4 |
| DoRA | 8 | ~0.6M + 9K mag | 78-85% | Should beat LoRA-8 |
| DoRA | 16 | ~1.2M + 9K mag | 82-88% | Should beat LoRA-16 |
| DoRA | 32 | ~2.4M + 9K mag | 84-90% | Should approach FFT |

> [!NOTE]
> The key insight we expect to demonstrate: DoRA achieves **LoRA+2-5% accuracy** at the same rank due to its magnitude-direction decomposition, while adding negligible extra parameters (only `out_features` magnitude scalars per adapted layer).

---

## Potential Issues & Mitigations

### 1. Cornell Dataset Download Issues
- **Mitigation**: Implement a fallback to download from Kaggle mirror. Include manual download instructions in the error message.

### 2. NaN Values in Annotations
- **Mitigation**: In `_parse_grasp_rectangles()`, check every coordinate for NaN. If any NaN found in a 4-line group, skip that entire rectangle. If no valid rectangles exist for an image, skip that image.

### 3. Small Dataset Size (~885 images)
- **Mitigation**: Use strong data augmentation (flip, rotation, color jitter). Consider using 5-fold cross-validation if initial results show high variance. At minimum, run 3 seeds and report mean ± std.

### 4. IoU Computation for Oriented Rectangles
- **Mitigation**: Use `shapely.geometry.Polygon` for robust polygon intersection computation. Fallback: use the simplified axis-aligned approximation if Shapely is unavailable.

### 5. LoRA Target Module Naming
- **Mitigation**: The ViT model from HuggingFace uses: `vit.encoder.layer.X.attention.attention.query` (etc.). The `apply_lora_to_model()` function matches on the **leaf name** (last component), so `"query"`, `"key"`, `"value"`, `"dense"` will match correctly. **Verify by printing `model.named_modules()` before and after injection.**

### 6. Regression Head Trainability
- **Mitigation**: After applying DoRA/LoRA and freezing base weights, explicitly ensure the regression head parameters have `requires_grad=True`:
```python
for name, param in model.named_parameters():
    if "classifier" in name or "head" in name:
        param.requires_grad = True
```

---

## Dependencies to Add

Add these to `requirements.txt` (or install via `uv add`):
```
shapely>=2.0.0
Pillow>=9.0.0
```

Both `torch`, `torchvision`, `transformers`, `matplotlib`, `seaborn`, `numpy`, `pandas`, `scikit-learn`, and `tqdm` are already in the existing `requirements.txt`.

---

## Running the Full Experiment (TL;DR)

```bash
cd /Users/shauryasen/Class/code/dev/4782/dora-implementation/code

# 1. Install deps
uv pip install shapely Pillow

# 2. Download dataset
python -m experiments.grasp_detection.dataset --download --target_dir data/cornell

# 3. Run full sweep (all methods, all ranks)
python -m experiments.grasp_detection.train --sweep --output_dir ../results/grasp_detection

# 4. Generate plots and summary
python -m experiments.grasp_detection.visualize --results_dir ../results/grasp_detection
```
