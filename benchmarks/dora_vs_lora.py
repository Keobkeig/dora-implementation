"""
Benchmarking framework for comparing DoRA vs LoRA performance.
"""

import gc
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn as nn

from training.trainer import DoRATrainer, TrainingConfig

from ..layers.base import DoRAModule


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""

    # Models to compare
    model_types: List[str] = None  # ["dora", "lora", "full_ft"]

    # Rank configurations to test
    ranks: List[int] = None  # [4, 8, 16, 32, 64]

    # Alpha values to test
    alphas: List[float] = None  # [8, 16, 32, 64]

    # Datasets to evaluate on
    datasets: List[str] = None  # ["boolq", "piqa", "siqa"]

    # Number of runs for statistical significance
    num_runs: int = 3

    # Training configuration
    max_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-4

    # Memory profiling
    profile_memory: bool = True
    profile_speed: bool = True

    # Output settings
    output_dir: str = "./benchmark_results"
    save_plots: bool = True
    save_detailed_results: bool = True

    def __post_init__(self):
        if self.model_types is None:
            self.model_types = ["dora", "lora", "full_ft"]
        if self.ranks is None:
            self.ranks = [4, 8, 16, 32]
        if self.alphas is None:
            self.alphas = [8, 16, 32]
        if self.datasets is None:
            self.datasets = ["boolq", "piqa"]


class MemoryProfiler:
    """Utility class for profiling memory usage."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset memory tracking."""
        self.peak_memory_mb = 0
        self.initial_memory_mb = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        gc.collect()

    def start_monitoring(self):
        """Start memory monitoring."""
        self.reset()
        if torch.cuda.is_available():
            self.initial_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            process = psutil.Process(os.getpid())
            self.initial_memory_mb = process.memory_info().rss / 1024 / 1024

    def get_current_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            current_mb = torch.cuda.memory_allocated() / 1024 / 1024
            peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            self.peak_memory_mb = max(self.peak_memory_mb, peak_mb)
            return current_mb - self.initial_memory_mb
        else:
            process = psutil.Process(os.getpid())
            current_mb = process.memory_info().rss / 1024 / 1024
            return current_mb - self.initial_memory_mb

    def get_peak_usage(self) -> float:
        """Get peak memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024 - self.initial_memory_mb
        else:
            return self.peak_memory_mb - self.initial_memory_mb


class SpeedProfiler:
    """Utility class for profiling training and inference speed."""

    def __init__(self):
        self.times = {}

    def start_timer(self, name: str):
        """Start timing an operation."""
        self.times[name + "_start"] = time.time()

    def end_timer(self, name: str) -> float:
        """End timing and return elapsed time."""
        if name + "_start" not in self.times:
            raise ValueError(f"Timer '{name}' was not started")

        elapsed = time.time() - self.times[name + "_start"]
        self.times[name] = elapsed
        return elapsed

    def get_time(self, name: str) -> float:
        """Get elapsed time for an operation."""
        return self.times.get(name, 0.0)

    def clear(self):
        """Clear all recorded times."""
        self.times.clear()


class DoRABenchmark:
    """
    Main benchmarking class for comparing DoRA vs LoRA vs Full Fine-tuning.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
        self.memory_profiler = MemoryProfiler()
        self.speed_profiler = SpeedProfiler()

        # Setup output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(config.output_dir, "benchmark.log")),
                logging.StreamHandler(),
            ],
        )

    def run_parameter_efficiency_benchmark(
        self,
        base_model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset,
    ) -> Dict[str, Any]:
        """
        Benchmark parameter efficiency across different ranks and methods.

        Args:
            base_model: Base model to adapt
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset

        Returns:
            Benchmark results
        """
        logging.info("Starting parameter efficiency benchmark...")

        results = {
            "parameter_counts": {},
            "performance_metrics": {},
            "memory_usage": {},
            "training_time": {},
        }

        for method in self.config.model_types:
            logging.info(f"Benchmarking method: {method}")
            results["parameter_counts"][method] = {}
            results["performance_metrics"][method] = {}
            results["memory_usage"][method] = {}
            results["training_time"][method] = {}

            if method == "full_ft":
                # Full fine-tuning baseline
                model = self._prepare_model(base_model, method, rank=None, alpha=None)
                result = self._run_single_experiment(model, train_dataset, eval_dataset, method)

                results["parameter_counts"][method]["all"] = result["param_count"]
                results["performance_metrics"][method]["all"] = result["metrics"]
                results["memory_usage"][method]["all"] = result["memory"]
                results["training_time"][method]["all"] = result["time"]
            else:
                # LoRA/DoRA with different ranks
                for rank in self.config.ranks:
                    for alpha in self.config.alphas:
                        key = f"r{rank}_a{alpha}"
                        logging.info(f"  Testing {method} with rank={rank}, alpha={alpha}")

                        model = self._prepare_model(base_model, method, rank, alpha)
                        result = self._run_single_experiment(
                            model, train_dataset, eval_dataset, f"{method}_{key}"
                        )

                        results["parameter_counts"][method][key] = result["param_count"]
                        results["performance_metrics"][method][key] = result["metrics"]
                        results["memory_usage"][method][key] = result["memory"]
                        results["training_time"][method][key] = result["time"]

        self.results["parameter_efficiency"] = results
        return results

    def run_accuracy_benchmark(
        self,
        base_model: nn.Module,
        datasets: Dict[str, Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]],
    ) -> Dict[str, Any]:
        """
        Benchmark accuracy across different datasets and methods.

        Args:
            base_model: Base model to adapt
            datasets: Dictionary of dataset_name -> (train_dataset, eval_dataset)

        Returns:
            Accuracy benchmark results
        """
        logging.info("Starting accuracy benchmark...")

        results = {}

        for dataset_name, (train_dataset, eval_dataset) in datasets.items():
            logging.info(f"Benchmarking on dataset: {dataset_name}")
            results[dataset_name] = {}

            for method in self.config.model_types:
                results[dataset_name][method] = {}

                if method == "full_ft":
                    model = self._prepare_model(base_model, method, rank=None, alpha=None)
                    result = self._run_single_experiment(
                        model, train_dataset, eval_dataset, f"{method}_{dataset_name}"
                    )
                    results[dataset_name][method]["all"] = result["metrics"]
                else:
                    # Test best rank/alpha combination for each method
                    best_rank = max(self.config.ranks)
                    best_alpha = max(self.config.alphas)

                    model = self._prepare_model(base_model, method, best_rank, best_alpha)
                    result = self._run_single_experiment(
                        model, train_dataset, eval_dataset, f"{method}_{dataset_name}"
                    )
                    results[dataset_name][method][f"r{best_rank}_a{best_alpha}"] = result["metrics"]

        self.results["accuracy"] = results
        return results

    def run_speed_benchmark(
        self, base_model: nn.Module, sample_input: torch.Tensor, num_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark inference speed for different methods.

        Args:
            base_model: Base model
            sample_input: Sample input tensor
            num_iterations: Number of inference iterations

        Returns:
            Speed benchmark results
        """
        logging.info("Starting speed benchmark...")

        results = {}

        for method in self.config.model_types:
            results[method] = {}

            if method == "full_ft":
                model = self._prepare_model(base_model, method, rank=None, alpha=None)
                speed = self._measure_inference_speed(model, sample_input, num_iterations)
                results[method]["all"] = speed
            else:
                for rank in self.config.ranks:
                    alpha = 16  # Use fixed alpha for speed test
                    key = f"r{rank}_a{alpha}"

                    model = self._prepare_model(base_model, method, rank, alpha)
                    speed = self._measure_inference_speed(model, sample_input, num_iterations)
                    results[method][key] = speed

        self.results["speed"] = results
        return results

    def run_ablation_study(
        self,
        base_model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset,
        parameter_grid: Dict[str, List[Any]],
    ) -> Dict[str, Any]:
        """
        Run ablation study on hyperparameters.

        Args:
            base_model: Base model
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            parameter_grid: Grid of parameters to test

        Returns:
            Ablation study results
        """
        logging.info("Starting ablation study...")

        results = {}

        # Generate all parameter combinations
        import itertools

        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())

        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination, strict=True))
            key = "_".join([f"{k}_{v}" for k, v in param_dict.items()])

            logging.info(f"Testing combination: {param_dict}")

            # Prepare model with current parameters
            model = self._prepare_model(
                base_model,
                "dora",  # Focus on DoRA for ablation
                param_dict.get("rank", 8),
                param_dict.get("alpha", 16),
            )

            # Run experiment
            result = self._run_single_experiment(
                model, train_dataset, eval_dataset, f"ablation_{key}"
            )
            results[key] = {
                "params": param_dict,
                "metrics": result["metrics"],
                "param_count": result["param_count"],
            }

        self.results["ablation"] = results
        return results

    def _prepare_model(
        self, base_model: nn.Module, method: str, rank: Optional[int], alpha: Optional[float]
    ) -> nn.Module:
        """Prepare model for benchmarking based on method."""
        # Create a copy of the base model
        model = type(base_model)(base_model.config)
        model.load_state_dict(base_model.state_dict())

        if method == "dora":
            # Convert to DoRA
            if hasattr(model, "model"):  # For models like LlamaForCausalLM
                from ..models.llama import LlamaDoRAConfig, LlamaDoRAModel

                dora_config = LlamaDoRAConfig(rank=rank, alpha=alpha)
                LlamaDoRAModel._convert_to_dora(model.model, dora_config)
            else:
                # Assume it's a vision model
                from ..models.vision_transformer import ViTDoRAConfig, ViTDoRAModel

                dora_config = ViTDoRAConfig(rank=rank, alpha=alpha)
                ViTDoRAModel._convert_to_dora(model, model, dora_config)

        elif method == "lora":
            # Convert to standard LoRA (disable DoRA mode)
            if hasattr(model, "model"):
                from ..models.llama import LlamaDoRAConfig, LlamaDoRAModel

                dora_config = LlamaDoRAConfig(rank=rank, alpha=alpha)
                LlamaDoRAModel._convert_to_dora(model.model, dora_config)
                # Disable DoRA mode to get standard LoRA
                DoRAModule.enable_dora_layers(model, enabled=False)
            else:
                from ..models.vision_transformer import ViTDoRAConfig, ViTDoRAModel

                dora_config = ViTDoRAConfig(rank=rank, alpha=alpha)
                ViTDoRAModel._convert_to_dora(model, model, dora_config)
                DoRAModule.enable_dora_layers(model, enabled=False)

        elif method == "full_ft":
            # Keep model as-is for full fine-tuning
            pass

        return model

    def _run_single_experiment(
        self,
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset,
        experiment_name: str,
    ) -> Dict[str, Any]:
        """Run a single training/evaluation experiment."""
        logging.info(f"Running experiment: {experiment_name}")

        # Setup training
        training_config = TrainingConfig(
            num_epochs=self.config.max_epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            output_dir=os.path.join(self.config.output_dir, experiment_name),
            use_wandb=False,  # Disable wandb for benchmarking
            eval_strategy="epoch",
        )

        trainer = DoRATrainer(
            model=model,
            config=training_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Profile memory and time
        self.memory_profiler.start_monitoring()
        self.speed_profiler.start_timer("training")

        try:
            # Train model
            train_metrics = trainer.train()

            # Evaluate model
            eval_metrics = trainer.evaluate()

            # Record timing
            training_time = self.speed_profiler.end_timer("training")

            # Record memory usage
            memory_usage = self.memory_profiler.get_peak_usage()

            # Count parameters
            param_stats = DoRAModule.count_parameters(model)

            return {
                "metrics": {**train_metrics, **eval_metrics},
                "memory": memory_usage,
                "time": training_time,
                "param_count": param_stats,
            }

        except Exception as e:
            logging.error(f"Experiment {experiment_name} failed: {str(e)}")
            return {
                "metrics": {"error": str(e)},
                "memory": 0,
                "time": 0,
                "param_count": {"total": 0, "trainable": 0},
            }

    def _measure_inference_speed(
        self, model: nn.Module, sample_input: torch.Tensor, num_iterations: int
    ) -> Dict[str, float]:
        """Measure inference speed."""
        model.eval()
        device = next(model.parameters()).device
        sample_input = sample_input.to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)

        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.time()
                _ = model(sample_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)

        return {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "median_time": np.median(times),
            "throughput": 1.0 / np.mean(times),  # samples per second
        }

    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        report = ["DoRA vs LoRA Benchmark Report", "=" * 40, ""]

        # Parameter efficiency summary
        if "parameter_efficiency" in self.results:
            report.append("PARAMETER EFFICIENCY COMPARISON")
            report.append("-" * 30)

            param_data = self.results["parameter_efficiency"]["parameter_counts"]

            for method in param_data:
                report.append(f"\n{method.upper()}:")
                for config, stats in param_data[method].items():
                    trainable = stats["trainable"]
                    total = stats["total"]
                    ratio = stats["compression_ratio"]
                    report.append(
                        f"  {config}: {trainable:,} trainable "
                        f"({100 * trainable / total:.2f}%), {ratio:.1f}x compression"
                    )

        # Accuracy comparison
        if "accuracy" in self.results:
            report.append("\n\nACCURACY COMPARISON")
            report.append("-" * 20)

            for dataset, methods in self.results["accuracy"].items():
                report.append(f"\nDataset: {dataset}")
                for method, configs in methods.items():
                    for config, metrics in configs.items():
                        eval_loss = metrics.get("eval_loss", "N/A")
                        report.append(f"  {method} ({config}): eval_loss={eval_loss}")

        # Speed comparison
        if "speed" in self.results:
            report.append("\n\nSPEED COMPARISON")
            report.append("-" * 16)

            for method, configs in self.results["speed"].items():
                report.append(f"\n{method.upper()}:")
                for config, speed_metrics in configs.items():
                    mean_time = speed_metrics["mean_time"] * 1000  # Convert to ms
                    throughput = speed_metrics["throughput"]
                    report.append(
                        f"  {config}: {mean_time:.2f}ms per inference, {throughput:.1f} samples/sec"
                    )

        report_text = "\n".join(report)

        # Save report
        with open(os.path.join(self.config.output_dir, "benchmark_report.txt"), "w") as f:
            f.write(report_text)

        return report_text

    def save_results(self):
        """Save detailed benchmark results to JSON."""
        results_file = os.path.join(self.config.output_dir, "benchmark_results.json")

        # Convert to JSON-serializable format
        json_results = {}
        for key, value in self.results.items():
            json_results[key] = self._make_json_serializable(value)

        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=2)

        logging.info(f"Results saved to {results_file}")

    def _make_json_serializable(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

    def plot_results(self):
        """Generate visualization plots for benchmark results."""
        if not self.config.save_plots:
            return

        plt.style.use("seaborn-v0_8")

        # Plot parameter efficiency
        self._plot_parameter_efficiency()

        # Plot speed comparison
        self._plot_speed_comparison()

        # Plot accuracy comparison
        self._plot_accuracy_comparison()

        logging.info("Plots saved to output directory")

    def _plot_parameter_efficiency(self):
        """Plot parameter efficiency comparison."""
        if "parameter_efficiency" not in self.results:
            return

        data = self.results["parameter_efficiency"]["parameter_counts"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Trainable parameters vs accuracy
        methods = []
        trainable_params = []

        for method in ["dora", "lora"]:
            if method in data:
                for config, stats in data[method].items():
                    if config != "all":
                        methods.append(f"{method}_{config}")
                        trainable_params.append(stats["trainable"])

        ax1.bar(range(len(methods)), trainable_params)
        ax1.set_xlabel("Configuration")
        ax1.set_ylabel("Trainable Parameters")
        ax1.set_title("Trainable Parameters by Configuration")
        ax1.tick_params(axis="x", rotation=45)

        # Plot 2: Compression ratio comparison
        compression_ratios = []
        for method in ["dora", "lora"]:
            if method in data:
                for config, stats in data[method].items():
                    if config != "all":
                        compression_ratios.append(stats["compression_ratio"])

        ax2.bar(range(len(methods)), compression_ratios)
        ax2.set_xlabel("Configuration")
        ax2.set_ylabel("Compression Ratio")
        ax2.set_title("Parameter Compression Ratio")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config.output_dir, "parameter_efficiency.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_speed_comparison(self):
        """Plot inference speed comparison."""
        if "speed" not in self.results:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        methods = []
        throughputs = []

        for method, configs in self.results["speed"].items():
            for config, speed_metrics in configs.items():
                methods.append(f"{method}_{config}")
                throughputs.append(speed_metrics["throughput"])

        bars = ax.bar(range(len(methods)), throughputs)
        ax.set_xlabel("Configuration")
        ax.set_ylabel("Throughput (samples/sec)")
        ax.set_title("Inference Speed Comparison")
        ax.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, throughput in zip(bars, throughputs, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{throughput:.1f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config.output_dir, "speed_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_accuracy_comparison(self):
        """Plot accuracy comparison across datasets."""
        if "accuracy" not in self.results:
            return

        fig, axes = plt.subplots(
            1, len(self.results["accuracy"]), figsize=(5 * len(self.results["accuracy"]), 6)
        )
        if len(self.results["accuracy"]) == 1:
            axes = [axes]

        for i, (dataset, methods) in enumerate(self.results["accuracy"].items()):
            method_names = []
            eval_losses = []

            for method, configs in methods.items():
                for config, metrics in configs.items():
                    method_names.append(f"{method}_{config}")
                    eval_losses.append(metrics.get("eval_loss", float("inf")))

            axes[i].bar(range(len(method_names)), eval_losses)
            axes[i].set_xlabel("Method")
            axes[i].set_ylabel("Evaluation Loss")
            axes[i].set_title(f"Accuracy on {dataset}")
            axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config.output_dir, "accuracy_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


# Factory function for easy benchmarking
def run_comprehensive_benchmark(
    base_model: nn.Module,
    datasets: Dict[str, Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]],
    config: Optional[BenchmarkConfig] = None,
    sample_input: Optional[torch.Tensor] = None,
) -> DoRABenchmark:
    """
    Run comprehensive benchmark comparing DoRA, LoRA, and full fine-tuning.

    Args:
        base_model: Base model to benchmark
        datasets: Dictionary of dataset_name -> (train_dataset, eval_dataset)
        config: Benchmark configuration
        sample_input: Sample input for speed testing

    Returns:
        Completed benchmark object with results
    """
    if config is None:
        config = BenchmarkConfig()

    benchmark = DoRABenchmark(config)

    # Run parameter efficiency benchmark
    first_dataset = next(iter(datasets.values()))
    benchmark.run_parameter_efficiency_benchmark(base_model, first_dataset[0], first_dataset[1])

    # Run accuracy benchmark
    benchmark.run_accuracy_benchmark(base_model, datasets)

    # Run speed benchmark if sample input provided
    if sample_input is not None:
        benchmark.run_speed_benchmark(base_model, sample_input)

    # Run ablation study
    ablation_grid = {"rank": [4, 8, 16], "alpha": [8, 16, 32]}
    benchmark.run_ablation_study(base_model, first_dataset[0], first_dataset[1], ablation_grid)

    # Generate outputs
    benchmark.save_results()
    benchmark.plot_results()
    report = benchmark.generate_report()

    print("\nBenchmark completed!")
    print(report)

    return benchmark
