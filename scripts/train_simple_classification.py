#!/usr/bin/env python3
"""
Simple Text Classification Training Script for DoRA Implementation

This script demonstrates how to use DoRA for text classification tasks
using a simple CNN-based text classifier (no external dependencies required).

Usage:
    python scripts/train_simple_classification.py --use_dora --rank 16

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
from torch.utils.data import DataLoader, TensorDataset

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dora.layers.base import DoRAModule  # noqa: E402
from dora.layers.dora_linear import create_dora_layer  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SimpleCNNClassifier(nn.Module):
    """Simple CNN-based text classifier with optional DoRA adaptation."""

    def __init__(
        self,
        vocab_size=10000,
        embed_dim=128,
        num_filters=100,
        filter_sizes=None,
        num_classes=2,
        dropout=0.5,
        use_dora=False,
        rank=16,
        alpha=32,
    ):
        super().__init__()

        if filter_sizes is None:
            filter_sizes = [3, 4, 5]

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # CNN layers
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=fs) for fs in filter_sizes]
        )

        # Classifier
        classifier_input_dim = len(filter_sizes) * num_filters
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # Apply DoRA if requested
        if use_dora:
            self._apply_dora(rank, alpha)

        self.use_dora = use_dora

    def _apply_dora(self, rank, alpha):
        """Apply DoRA to linear layers."""
        logger.info("Applying DoRA to classifier layers...")

        # Apply DoRA to classifier layers
        for i, layer in enumerate(self.classifier):
            if isinstance(layer, nn.Linear):
                self.classifier[i] = create_dora_layer(layer, rank=rank, alpha=alpha)

        # Count parameters
        total_params = DoRAModule.count_parameters(self)["total"]
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable percentage: {(trainable_params/total_params*100):.1f}%")

    def forward(self, x, attention_mask=None):
        # x: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        x = x.transpose(1, 2)  # [batch_size, embed_dim, seq_len]

        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))  # [batch_size, num_filters, new_seq_len]
            pooled = torch.max(conv_out, dim=2)[0]  # [batch_size, num_filters]
            conv_outputs.append(pooled)

        # Concatenate all conv outputs
        x = torch.cat(conv_outputs, dim=1)  # [batch_size, len(filter_sizes) * num_filters]

        # Apply classifier
        logits = self.classifier(x)
        return logits


def create_text_classification_dataset(
    num_samples=1000, seq_len=50, vocab_size=1000, num_classes=2
):
    """Create a synthetic text classification dataset."""
    logger.info(f"Creating synthetic dataset with {num_samples} samples")

    # Generate random sequences
    input_ids = torch.randint(1, vocab_size, (num_samples, seq_len))

    # Create patterns for different classes
    # Class 0: sequences with more small numbers
    # Class 1: sequences with more large numbers
    labels = torch.zeros(num_samples, dtype=torch.long)

    for i in range(num_samples):
        # Simple rule: if mean of sequence > vocab_size/2, label = 1
        if input_ids[i].float().mean() > vocab_size / 2:
            labels[i] = 1
            # Emphasize pattern by adding some large numbers
            input_ids[i, :10] = torch.randint(vocab_size // 2, vocab_size, (10,))
        else:
            labels[i] = 0
            # Emphasize pattern by adding some small numbers
            input_ids[i, :10] = torch.randint(1, vocab_size // 2, (10,))

    return TensorDataset(input_ids, labels)


def create_sentiment_dataset(num_samples=1000, seq_len=50, vocab_size=1000):
    """Create a more realistic sentiment-like dataset."""
    logger.info(f"Creating sentiment dataset with {num_samples} samples")

    # Define "positive" and "negative" word ranges
    positive_words = list(range(1, vocab_size // 3))  # Lower indices = positive
    negative_words = list(range(2 * vocab_size // 3, vocab_size))  # Higher indices = negative
    neutral_words = list(range(vocab_size // 3, 2 * vocab_size // 3))  # Middle = neutral

    input_ids = []
    labels = []

    for _ in range(num_samples):
        # Randomly choose sentiment
        sentiment = np.random.choice([0, 1])  # 0=negative, 1=positive

        sequence = []

        if sentiment == 1:  # Positive
            # More positive words
            for _ in range(seq_len):
                word_type = np.random.choice(["positive", "neutral", "negative"], p=[0.6, 0.3, 0.1])
                if word_type == "positive":
                    word = np.random.choice(positive_words)
                elif word_type == "negative":
                    word = np.random.choice(negative_words)
                else:
                    word = np.random.choice(neutral_words)
                sequence.append(word)
        else:  # Negative
            # More negative words
            for _ in range(seq_len):
                word_type = np.random.choice(["positive", "neutral", "negative"], p=[0.1, 0.3, 0.6])
                if word_type == "positive":
                    word = np.random.choice(positive_words)
                elif word_type == "negative":
                    word = np.random.choice(negative_words)
                else:
                    word = np.random.choice(neutral_words)
                sequence.append(word)

        input_ids.append(sequence)
        labels.append(sentiment)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return TensorDataset(input_ids, labels)


def calculate_accuracy(predictions, labels):
    """Calculate accuracy score."""
    return (predictions == labels).sum() / len(labels)


def calculate_metrics(predictions, labels):
    """Calculate precision, recall, and F1 score."""
    # Simple binary classification metrics
    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision, recall, f1


def evaluate_model(model, dataloader, device, criterion):
    """Evaluate model performance."""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                input_ids, labels = [x.to(device) for x in batch]
                attention_mask = None
            else:
                input_ids, attention_mask, labels = [x.to(device) for x in batch]

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = calculate_accuracy(np.array(all_predictions), np.array(all_labels))
    precision, recall, f1 = calculate_metrics(np.array(all_predictions), np.array(all_labels))

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Simple Text Classification with DoRA")
    parser.add_argument(
        "--dataset",
        choices=["random", "sentiment"],
        default="sentiment",
        help="Dataset type to use",
    )
    parser.add_argument("--use_dora", action="store_true", help="Use DoRA adaptation")
    parser.add_argument("--rank", type=int, default=16, help="DoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="DoRA alpha parameter")
    parser.add_argument(
        "--num_samples", type=int, default=2000, help="Number of samples to generate"
    )
    parser.add_argument("--seq_len", type=int, default=50, help="Sequence length")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num_filters", type=int, default=100, help="Number of CNN filters")

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create dataset
    if args.dataset == "sentiment":
        dataset = create_sentiment_dataset(args.num_samples, args.seq_len, args.vocab_size)
    else:
        dataset = create_text_classification_dataset(
            args.num_samples, args.seq_len, args.vocab_size
        )

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    logger.info(f"Vocabulary size: {args.vocab_size}, Sequence length: {args.seq_len}")

    # Create model
    model = SimpleCNNClassifier(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_filters=args.num_filters,
        use_dora=args.use_dora,
        rank=args.rank,
        alpha=args.alpha,
    ).to(device)

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    logger.info(f"Model created with DoRA: {args.use_dora}")
    if args.use_dora:
        logger.info(f"DoRA parameters - Rank: {args.rank}, Alpha: {args.alpha}")

    # Training loop
    best_val_accuracy = 0
    train_losses = []
    val_accuracies = []

    logger.info(f"Starting training for {args.num_epochs} epochs...")

    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids, labels = [x.to(device) for x in batch]

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1

            if batch_idx % 20 == 0 and batch_idx > 0:
                logger.info(
                    f"Epoch {epoch+1}/{args.num_epochs}, "
                    f"Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        val_metrics = evaluate_model(model, val_loader, device, criterion)
        val_accuracies.append(val_metrics["accuracy"])

        logger.info(f"Epoch {epoch+1} Results:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"  Val F1: {val_metrics['f1']:.4f}")

        # Save best model info
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            logger.info("  🎉 New best validation accuracy!")

    # Final results
    logger.info("\n" + "=" * 50)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 50)
    logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    logger.info(f"Final validation accuracy: {val_accuracies[-1]:.4f}")
    logger.info(f"Training loss improvement: {train_losses[0]:.4f} → {train_losses[-1]:.4f}")

    # Parameter statistics
    total_params = DoRAModule.count_parameters(model)["total"]
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("\nModel Statistics:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable percentage: {(trainable_params/total_params*100):.1f}%")

    # DoRA comparison info
    if args.use_dora:
        logger.info("\nDoRA Configuration:")
        logger.info(f"  Rank: {args.rank}")
        logger.info(f"  Alpha: {args.alpha}")
        logger.info(
            f"  Parameter efficiency: {(1 - trainable_params/total_params)*100:.1f}% reduction"
        )


if __name__ == "__main__":
    main()
