#!/usr/bin/env python3
"""
Text Classification Training Script for DoRA Implementation

This script demonstrates how to use DoRA for text classification tasks.
Supports multiple classification datasets and comparison with baseline methods.

Usage:
    python scripts/train_classification.py --dataset imdb --rank 16 --alpha 32

Author: DoRA Implementation Project
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dora.utils.model_utils import count_parameters, create_dora_layer  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TextClassifier(nn.Module):
    """Text classifier using pre-trained transformer with DoRA adaptation."""

    def __init__(
        self, model_name="distilbert-base-uncased", num_classes=2, use_dora=False, rank=16, alpha=32
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size

        # Classifier head
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Apply DoRA to transformer layers if requested
        if use_dora:
            self._apply_dora_to_transformer(rank, alpha)

        self.num_classes = num_classes
        self.use_dora = use_dora

    def _apply_dora_to_transformer(self, rank, alpha):
        """Apply DoRA to attention and feed-forward layers."""
        logger.info("Applying DoRA to transformer layers...")

        # Count original parameters
        original_params = count_parameters(self.transformer)

        # Apply DoRA to attention layers
        for layer in self.transformer.transformer.layer:
            # Self-attention layers
            layer.attention.self.query = create_dora_layer(
                layer.attention.self.query, rank=rank, alpha=alpha
            )
            layer.attention.self.key = create_dora_layer(
                layer.attention.self.key, rank=rank, alpha=alpha
            )
            layer.attention.self.value = create_dora_layer(
                layer.attention.self.value, rank=rank, alpha=alpha
            )

            # Feed-forward layers
            layer.intermediate.dense = create_dora_layer(
                layer.intermediate.dense, rank=rank, alpha=alpha
            )
            layer.output.dense = create_dora_layer(layer.output.dense, rank=rank, alpha=alpha)

        # Count DoRA parameters
        dora_params = count_parameters(self.transformer)
        trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)

        logger.info(f"Original parameters: {original_params:,}")
        logger.info(f"DoRA parameters: {dora_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        reduction = (original_params - trainable_params) / original_params * 100
        logger.info(f"Parameter reduction: {reduction:.1f}%")

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        logits = self.classifier(pooled_output)
        return logits


def create_dummy_dataset(num_samples=1000, max_length=128, num_classes=2, vocab_size=10000):
    """Create a dummy text classification dataset for testing."""
    logger.info(f"Creating dummy dataset with {num_samples} samples")

    # Generate random sequences
    input_ids = torch.randint(1, vocab_size, (num_samples, max_length))
    attention_mask = torch.ones_like(input_ids)

    # Generate random labels
    labels = torch.randint(0, num_classes, (num_samples,))

    return TensorDataset(input_ids, attention_mask, labels)


def load_imdb_dataset(tokenizer, max_length=128, num_samples=1000):
    """Load and preprocess IMDB movie review dataset."""
    logger.info("Loading IMDB dataset...")

    # For demo purposes, create synthetic data
    # In practice, you'd load from Hugging Face datasets or similar
    positive_texts = [
        "This movie was absolutely fantastic! Great acting and storyline.",
        "Loved every minute of it. Highly recommend!",
        "Brilliant cinematography and excellent direction.",
        "One of the best films I've seen this year.",
        "Outstanding performance by the lead actor.",
    ] * (num_samples // 10)

    negative_texts = [
        "Terrible movie, waste of time and money.",
        "Poor acting and confusing plot.",
        "Boring and predictable storyline.",
        "Complete disappointment, don't watch.",
        "One of the worst movies ever made.",
    ] * (num_samples // 10)

    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)

    # Shuffle
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]

    # Tokenize
    encodings = tokenizer(
        texts[:num_samples],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    labels = torch.tensor(labels[:num_samples])

    return TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels)


def evaluate_model(model, dataloader, device):
    """Evaluate model performance."""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average="weighted"
    )

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Text Classification with DoRA")
    parser.add_argument(
        "--dataset", choices=["imdb", "dummy"], default="dummy", help="Dataset to use for training"
    )
    parser.add_argument(
        "--model_name", default="distilbert-base-uncased", help="Pre-trained model to use"
    )
    parser.add_argument("--use_dora", action="store_true", help="Use DoRA adaptation")
    parser.add_argument("--rank", type=int, default=16, help="DoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="DoRA alpha parameter")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    if args.dataset != "dummy":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    if args.dataset == "imdb":
        dataset = load_imdb_dataset(tokenizer, args.max_length, args.num_samples)
    else:
        dataset = create_dummy_dataset(args.num_samples, args.max_length, num_classes=2)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    if args.dataset == "dummy":
        # Use a simple model for dummy data
        class SimpleClassifier(nn.Module):
            def __init__(
                self,
                vocab_size=10000,
                embed_dim=128,
                hidden_dim=256,
                num_classes=2,
                use_dora=False,
                rank=16,
                alpha=32,
            ):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.fc1 = nn.Linear(embed_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, num_classes)
                self.dropout = nn.Dropout(0.1)

                if use_dora:
                    self.fc1 = create_dora_layer(self.fc1, rank=rank, alpha=alpha)
                    self.fc2 = create_dora_layer(self.fc2, rank=rank // 2, alpha=alpha)

            def forward(self, input_ids, attention_mask=None):
                # Simple mean pooling
                embeds = self.embedding(input_ids)  # [batch, seq, embed]
                if attention_mask is not None:
                    embeds = embeds * attention_mask.unsqueeze(-1)
                    pooled = embeds.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                else:
                    pooled = embeds.mean(dim=1)

                x = self.dropout(torch.relu(self.fc1(pooled)))
                return self.fc2(x)

        model = SimpleClassifier(use_dora=args.use_dora, rank=args.rank, alpha=args.alpha)
    else:
        model = TextClassifier(
            model_name=args.model_name,
            num_classes=2,
            use_dora=args.use_dora,
            rank=args.rank,
            alpha=args.alpha,
        )

    model.to(device)

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    logger.info(f"Starting training for {args.num_epochs} epochs...")
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Training loop
    best_val_accuracy = 0
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{args.num_epochs}, "
                    f"Batch {batch_idx+1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        # Validation phase
        val_metrics = evaluate_model(model, val_loader, device)

        logger.info(f"Epoch {epoch+1} Results:")
        logger.info(f"  Train Loss: {total_train_loss/len(train_loader):.4f}")
        logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"  Val F1: {val_metrics['f1']:.4f}")

        # Save best model
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            logger.info(f"New best validation accuracy: {best_val_accuracy:.4f}")

    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")

    # Final parameter count
    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("Final model statistics:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable percentage: {(trainable_params/total_params*100):.1f}%")


if __name__ == "__main__":
    main()
