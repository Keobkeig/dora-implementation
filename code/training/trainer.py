"""
DoRA training utilities and trainer implementation.
"""

import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("Weights & Biases not available. Logging disabled.")

from ..layers.base import DoRAModule


@dataclass
class TrainingConfig:
    """Configuration for DoRA training."""

    # Basic training settings
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # DoRA-specific settings
    dora_magnitude_lr_scale: float = 1.0  # Scale magnitude learning rate
    freeze_base_weights: bool = True
    warmup_steps: int = 100

    # Optimizer settings
    optimizer_type: str = "adamw"  # "adamw", "sgd", "adam"
    scheduler_type: str = "cosine"  # "cosine", "linear", "constant"

    # Training dynamics
    early_stopping_patience: int = 5
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10

    # Output settings
    output_dir: str = "./dora_output"
    save_total_limit: int = 3

    # Evaluation settings
    eval_strategy: str = "steps"  # "steps", "epoch", "no"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Hardware settings
    fp16: bool = False
    bf16: bool = False
    dataloader_num_workers: int = 0

    # Logging
    use_wandb: bool = True
    wandb_project: str = "dora_experiments"
    wandb_run_name: Optional[str] = None


class DoRATrainer:
    """
    Trainer class specifically designed for DoRA fine-tuning.
    Handles DoRA-specific optimizations and training dynamics.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[Any] = None,
        compute_metrics: Optional[Callable] = None,
        data_collator: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        self.data_collator = data_collator

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float("inf") if not config.greater_is_better else float("-inf")
        self.early_stopping_counter = 0

        # Setup components
        self._setup_device()
        self._setup_dataloaders()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_logging()

        # Freeze base weights if specified
        if config.freeze_base_weights:
            self._freeze_base_weights()

    def _setup_device(self):
        """Setup device for training."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logging.info(f"Using device: {self.device}")

    def _setup_dataloaders(self):
        """Setup data loaders."""
        if self.train_dataset is not None:
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.dataloader_num_workers,
                collate_fn=self.data_collator,
            )

        if self.eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.dataloader_num_workers,
                collate_fn=self.data_collator,
            )

    def _setup_optimizer(self):
        """Setup optimizer with DoRA-specific parameter grouping."""
        # Separate DoRA parameters from regular parameters
        dora_magnitude_params = []
        dora_lora_params = []
        regular_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "magnitude" in name and DoRAModule.is_dora_layer(
                self.model.get_submodule(".".join(name.split(".")[:-1]))
            ):
                dora_magnitude_params.append(param)
            elif ("lora_A" in name or "lora_B" in name) and DoRAModule.is_dora_layer(
                self.model.get_submodule(".".join(name.split(".")[:-1]))
            ):
                dora_lora_params.append(param)
            else:
                regular_params.append(param)

        # Create parameter groups with different learning rates
        param_groups = []

        if dora_magnitude_params:
            param_groups.append(
                {
                    "params": dora_magnitude_params,
                    "lr": self.config.learning_rate * self.config.dora_magnitude_lr_scale,
                    "weight_decay": 0.0,  # Usually no weight decay for magnitude
                    "name": "dora_magnitude",
                }
            )

        if dora_lora_params:
            param_groups.append(
                {
                    "params": dora_lora_params,
                    "lr": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                    "name": "dora_lora",
                }
            )

        if regular_params:
            param_groups.append(
                {
                    "params": regular_params,
                    "lr": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                    "name": "regular",
                }
            )

        # Create optimizer
        if self.config.optimizer_type.lower() == "adamw":
            self.optimizer = optim.AdamW(param_groups)
        elif self.config.optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(param_groups)
        elif self.config.optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(param_groups, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")

        logging.info(f"Created optimizer with {len(param_groups)} parameter groups")
        for i, group in enumerate(param_groups):
            logging.info(
                f"  Group {i} ({group['name']}): {len(group['params'])} params, lr={group['lr']}"
            )

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if not hasattr(self, "train_dataloader"):
            self.scheduler = None
            return

        num_training_steps = (
            len(self.train_dataloader)
            * self.config.num_epochs
            // self.config.gradient_accumulation_steps
        )

        if self.config.scheduler_type.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_training_steps
            )
        elif self.config.scheduler_type.lower() == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps
            )
        elif self.config.scheduler_type.lower() == "constant":
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")

        # Add warmup if specified
        if self.config.warmup_steps > 0 and self.scheduler is not None:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.config.warmup_steps,
            )
            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, self.scheduler],
                milestones=[self.config.warmup_steps],
            )

    def _setup_logging(self):
        """Setup logging and wandb if available."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Setup wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=asdict(self.config),
                tags=["dora"],
            )

            # Log model architecture
            dora_layers = DoRAModule.get_dora_layers(self.model)
            param_stats = DoRAModule.count_parameters(self.model)

            wandb.config.update(
                {
                    "num_dora_layers": len(dora_layers),
                    "total_params": param_stats["total"],
                    "trainable_params": param_stats["trainable"],
                    "compression_ratio": param_stats["compression_ratio"],
                }
            )

    def _freeze_base_weights(self):
        """Freeze base model weights, keeping only DoRA parameters trainable."""
        for name, param in self.model.named_parameters():
            # Check if this is a DoRA parameter
            module_path = ".".join(name.split(".")[:-1])
            param_name = name.split(".")[-1]

            try:
                module = self.model.get_submodule(module_path)
                if DoRAModule.is_dora_layer(module):
                    # Only keep DoRA-specific parameters trainable
                    if param_name in ["lora_A", "lora_B", "magnitude"]:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    # Regular module - freeze unless it's a classification head
                    if "classifier" in name or "head" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            except Exception:
                # Fallback: freeze by default
                param.requires_grad = False

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logging.info(
            f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)"
        )

    def train(self) -> Dict[str, float]:
        """
        Main training loop.

        Returns:
            Training metrics
        """
        self.model.train()

        if self.train_dataset is None:
            raise ValueError("No training dataset provided")

        logging.info("Starting DoRA training...")
        logging.info(f"  Num examples: {len(self.train_dataset)}")
        logging.info(f"  Num epochs: {self.config.num_epochs}")
        logging.info(f"  Batch size: {self.config.batch_size}")
        logging.info(f"  Gradient accumulation steps: {self.config.gradient_accumulation_steps}")

        # Training loop
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0

            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                disable=False,
            )

            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]

                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

                # Backward pass
                loss.backward()
                epoch_loss += loss.item()

                # Optimization step
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )

                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        lr = self.optimizer.param_groups[0]["lr"]
                        self._log_metrics(
                            {
                                "train_loss": loss.item() * self.config.gradient_accumulation_steps,
                                "learning_rate": lr,
                                "epoch": epoch,
                                "step": self.global_step,
                            }
                        )

                    # Evaluation
                    if (
                        self.config.eval_strategy == "steps"
                        and self.global_step % self.config.eval_steps == 0
                        and self.eval_dataset is not None
                    ):

                        eval_metrics = self.evaluate()
                        self._handle_evaluation(eval_metrics)
                        self.model.train()

                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "loss": loss.item() * self.config.gradient_accumulation_steps,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                )

            # End of epoch evaluation
            if self.config.eval_strategy == "epoch" and self.eval_dataset is not None:
                eval_metrics = self.evaluate()
                self._handle_evaluation(eval_metrics)

            # Early stopping check
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Final save
        self.save_checkpoint(is_final=True)

        logging.info("Training completed!")
        return {"final_loss": epoch_loss / len(self.train_dataloader)}

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluation loop.

        Returns:
            Evaluation metrics
        """
        if self.eval_dataset is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]

                total_loss += loss.item()
                total_samples += len(batch["input_ids"]) if "input_ids" in batch else len(batch)

                # Collect predictions for metrics computation
                if self.compute_metrics is not None:
                    if hasattr(outputs, "logits"):
                        predictions = torch.argmax(outputs.logits, dim=-1)
                        all_predictions.extend(predictions.cpu().numpy())
                        if "labels" in batch:
                            all_labels.extend(batch["labels"].cpu().numpy())

        # Compute metrics
        eval_loss = total_loss / len(self.eval_dataloader)
        metrics = {"eval_loss": eval_loss}

        if self.compute_metrics is not None and all_predictions and all_labels:
            computed_metrics = self.compute_metrics(all_predictions, all_labels)
            metrics.update(computed_metrics)

        self._log_metrics(metrics)
        return metrics

    def _handle_evaluation(self, eval_metrics: Dict[str, float]):
        """Handle evaluation results for early stopping and best model saving."""
        metric_value = eval_metrics.get(self.config.metric_for_best_model, float("inf"))

        is_better = (self.config.greater_is_better and metric_value > self.best_metric) or (
            not self.config.greater_is_better and metric_value < self.best_metric
        )

        if is_better:
            self.best_metric = metric_value
            self.early_stopping_counter = 0
            self.save_checkpoint(is_best=True)
            logging.info(f"New best model: {self.config.metric_for_best_model}={metric_value:.4f}")
        else:
            self.early_stopping_counter += 1

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to wandb and console."""
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log(metrics, step=self.global_step)

        # Console logging
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logging.info(f"Step {self.global_step} | {metric_str}")

    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
        if is_best:
            checkpoint_dir = os.path.join(self.config.output_dir, "best_model")
        elif is_final:
            checkpoint_dir = os.path.join(self.config.output_dir, "final_model")

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save DoRA adapter weights only
        from ..layers.base import DoRAStateManager

        adapter_path = os.path.join(checkpoint_dir, "dora_adapter.pt")
        DoRAStateManager.save_dora_state(self.model, adapter_path)

        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config),
        }

        if self.scheduler is not None:
            training_state["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))

        # Save tokenizer if available
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(checkpoint_dir)

        logging.info(f"Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint."""
        # Load DoRA adapter
        adapter_path = os.path.join(checkpoint_dir, "dora_adapter.pt")
        if os.path.exists(adapter_path):
            from ..layers.base import DoRAStateManager

            DoRAStateManager.load_dora_state(self.model, adapter_path)

        # Load training state
        training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=self.device)

            self.global_step = training_state["global_step"]
            self.epoch = training_state["epoch"]
            self.best_metric = training_state["best_metric"]

            if hasattr(self, "optimizer"):
                self.optimizer.load_state_dict(training_state["optimizer_state_dict"])

            if hasattr(self, "scheduler") and self.scheduler is not None:
                self.scheduler.load_state_dict(training_state["scheduler_state_dict"])

        logging.info(f"Checkpoint loaded from {checkpoint_dir}")


# Utility functions
def create_dora_trainer(
    model: nn.Module,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    **config_kwargs,
) -> DoRATrainer:
    """
    Factory function to create DoRA trainer with sensible defaults.

    Args:
        model: Model with DoRA layers
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        **config_kwargs: Additional config parameters

    Returns:
        DoRA trainer instance
    """
    config = TrainingConfig(**config_kwargs)
    return DoRATrainer(
        model=model, config=config, train_dataset=train_dataset, eval_dataset=eval_dataset
    )
