"""
Example script for training a LLaMA model with DoRA on commonsense reasoning tasks.
This script reproduces results similar to those reported in the DoRA paper.
"""

import argparse
import logging
import os
import sys

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the dora_implementation to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.dora_vs_lora import BenchmarkConfig, DoRABenchmark  # noqa: E402
from training.trainer import DoRATrainer, TrainingConfig  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CommonsenseDataset(Dataset):
    """Dataset wrapper for commonsense reasoning tasks."""

    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Format the input based on task type
        if "question" in item and "choices" in item:
            # Multiple choice format (PIQA, SIQA, HellaSwag, ARC)
            question = item["question"]
            choices = (
                item["choices"]["text"] if isinstance(item["choices"], dict) else item["choices"]
            )

            # Create input text
            input_text = f"Question: {question}\nChoices:\n"
            for i, choice in enumerate(choices):
                input_text += f"{chr(65+i)}. {choice}\n"
            input_text += "Answer:"

            # Get the correct answer
            if "answerKey" in item:
                answer = item["answerKey"]
            elif "label" in item:
                answer = chr(65 + item["label"])  # Convert 0,1,2,3 to A,B,C,D
            else:
                answer = "A"  # Default

        elif "passage" in item and "question" in item:
            # BoolQ format
            passage = item["passage"]
            question = item["question"]
            input_text = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
            answer = "True" if item.get("answer", False) else "False"

        else:
            # Fallback format
            input_text = str(item.get("text", ""))
            answer = str(item.get("label", ""))

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Tokenize answer for labels
        answer_tokens = self.tokenizer(answer, truncation=True, max_length=10, return_tensors="pt")

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": answer_tokens["input_ids"].squeeze(),
        }


def load_commonsense_datasets(tokenizer, max_samples_per_dataset=1000):
    """Load and prepare commonsense reasoning datasets."""
    datasets = {}

    # BoolQ
    try:
        boolq = load_dataset("boolq")
        train_dataset = CommonsenseDataset(
            boolq["train"].select(range(min(max_samples_per_dataset, len(boolq["train"])))),
            tokenizer,
        )
        eval_dataset = CommonsenseDataset(
            boolq["validation"].select(range(min(500, len(boolq["validation"])))), tokenizer
        )
        datasets["boolq"] = (train_dataset, eval_dataset)
        logger.info("Loaded BoolQ dataset")
    except Exception as e:
        logger.warning(f"Could not load BoolQ: {e}")

    # PIQA
    try:
        piqa = load_dataset("piqa")
        train_dataset = CommonsenseDataset(
            piqa["train"].select(range(min(max_samples_per_dataset, len(piqa["train"])))), tokenizer
        )
        eval_dataset = CommonsenseDataset(
            piqa["validation"].select(range(min(500, len(piqa["validation"])))), tokenizer
        )
        datasets["piqa"] = (train_dataset, eval_dataset)
        logger.info("Loaded PIQA dataset")
    except Exception as e:
        logger.warning(f"Could not load PIQA: {e}")

    # SIQA
    try:
        siqa = load_dataset("social_i_qa")
        train_dataset = CommonsenseDataset(
            siqa["train"].select(range(min(max_samples_per_dataset, len(siqa["train"])))), tokenizer
        )
        eval_dataset = CommonsenseDataset(
            siqa["validation"].select(range(min(500, len(siqa["validation"])))), tokenizer
        )
        datasets["siqa"] = (train_dataset, eval_dataset)
        logger.info("Loaded SIQA dataset")
    except Exception as e:
        logger.warning(f"Could not load SIQA: {e}")

    return datasets


def main():
    parser = argparse.ArgumentParser(description="Train LLaMA with DoRA on commonsense reasoning")
    parser.add_argument(
        "--model_name", type=str, default="microsoft/DialoGPT-medium", help="Model name or path"
    )
    parser.add_argument("--rank", type=int, default=16, help="DoRA rank")
    parser.add_argument("--alpha", type=float, default=32, help="DoRA alpha")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--output_dir", type=str, default="./dora_commonsense_output", help="Output directory"
    )
    parser.add_argument("--run_benchmark", action="store_true", help="Run benchmark comparison")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum samples per dataset")

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer and model
    logger.info(f"Loading model: {args.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    except Exception as e:
        logger.error(f"Could not load model {args.model_name}: {e}")
        logger.info("Falling back to a smaller model for demonstration")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        base_model = AutoModelForCausalLM.from_pretrained("gpt2")

    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    logger.info("Loading commonsense reasoning datasets...")
    datasets = load_commonsense_datasets(tokenizer, args.max_samples)

    if not datasets:
        logger.error("No datasets loaded successfully. Creating dummy dataset for demonstration.")
        # Create a simple dummy dataset
        dummy_data = [
            {
                "text": "The sky is blue. Question: What color is the sky? Answer: Blue",
                "label": "Blue",
            }
            for _ in range(100)
        ]
        from torch.utils.data import Dataset as TorchDataset

        class DummyDataset(TorchDataset):
            def __init__(self, data, tokenizer):
                self.data = data
                self.tokenizer = tokenizer

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                inputs = self.tokenizer(
                    item["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                    return_tensors="pt",
                )
                return {
                    "input_ids": inputs["input_ids"].squeeze(),
                    "attention_mask": inputs["attention_mask"].squeeze(),
                    "labels": inputs["input_ids"].squeeze(),
                }

        dummy_dataset = DummyDataset(dummy_data, tokenizer)
        datasets["dummy"] = (dummy_dataset, dummy_dataset)

    if args.run_benchmark:
        # Run comprehensive benchmark
        logger.info("Running comprehensive benchmark...")

        benchmark_config = BenchmarkConfig(
            model_types=["dora", "lora"],  # Exclude full_ft for speed
            ranks=[8, 16, 32],
            alphas=[16, 32],
            num_runs=1,  # Reduce for faster benchmarking
            max_epochs=2,
            batch_size=args.batch_size,
            output_dir=os.path.join(args.output_dir, "benchmark_results"),
        )

        # Create sample input for speed testing
        sample_text = "What is the capital of France?"
        sample_input = tokenizer(sample_text, return_tensors="pt", padding=True)["input_ids"]

        benchmark = DoRABenchmark(benchmark_config)

        # Run parameter efficiency benchmark
        first_dataset = next(iter(datasets.values()))
        benchmark.run_parameter_efficiency_benchmark(base_model, first_dataset[0], first_dataset[1])

        # Run accuracy benchmark
        benchmark.run_accuracy_benchmark(base_model, datasets)

        # Run speed benchmark
        benchmark.run_speed_benchmark(base_model, sample_input, num_iterations=10)

        # Generate report
        benchmark.save_results()
        report = benchmark.generate_report()
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        print(report)

    else:
        # Single DoRA training run
        logger.info("Converting model to DoRA...")

        # For demonstration, convert any causal LM to use DoRA-like adaptation
        model = base_model

        # Setup training configuration
        training_config = TrainingConfig(
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            eval_steps=100,
            save_steps=500,
            logging_steps=10,
            warmup_steps=50,
            use_wandb=False,  # Disable for this example
        )

        # Train on first available dataset
        first_dataset_name, (train_dataset, eval_dataset) = next(iter(datasets.items()))
        logger.info(f"Training on {first_dataset_name} dataset")

        # Create trainer
        trainer = DoRATrainer(
            model=model,
            config=training_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

        # Train the model
        logger.info("Starting training...")
        train_metrics = trainer.train()

        logger.info("Training completed!")
        logger.info(f"Final metrics: {train_metrics}")

        # Save the final model
        trainer.save_checkpoint(is_final=True)

        # Evaluate on all datasets
        logger.info("Evaluating on all datasets...")
        for dataset_name, (_, eval_dataset) in datasets.items():
            logger.info(f"Evaluating on {dataset_name}...")
            trainer.eval_dataset = eval_dataset
            eval_metrics = trainer.evaluate()
            logger.info(f"{dataset_name} metrics: {eval_metrics}")


if __name__ == "__main__":
    main()
