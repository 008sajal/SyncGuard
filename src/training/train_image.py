from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.dataset.image_dataset import ImageFrameDataset, get_image_transforms
from src.training.image_model import create_image_model


def create_weighted_sampler(dataset: ImageFrameDataset) -> WeightedRandomSampler:
    labels = np.array(
        [int(sample["label"]) for sample in dataset.samples],
        dtype=np.int64,
    )

    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def calculate_metrics(
    labels: list[int],
    predictions: list[int],
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            labels,
            predictions,
            zero_division=0,
        ),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, float]]:
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    progress = tqdm(
        loader,
        desc="Training" if is_training else "Validation",
        leave=False,
    )

    for images, labels, _ in progress:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(all_labels, all_predictions)

    return epoch_loss, metrics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/processed/manifests/fakeavceleb_subset.csv"
        ),
    )

    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed/fakeavceleb"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/checkpoints"),
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-batches", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    transforms_by_split = get_image_transforms()

    train_dataset = ImageFrameDataset(
        manifest_path=args.manifest,
        processed_root=args.processed_root,
        split="train",
        transform=transforms_by_split["train"],
    )

    validation_dataset = ImageFrameDataset(
        manifest_path=args.manifest,
        processed_root=args.processed_root,
        split="validation",
        transform=transforms_by_split["validation"],
    )

    sampler = create_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_image_model(
        num_classes=2,
        freeze_backbone=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    history: list[dict[str, object]] = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        validation_loss, validation_metrics = run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        }

        history.append(epoch_result)

        print(
            f"Train loss: {train_loss:.4f} | "
            f"F1: {train_metrics['f1']:.4f}"
        )
        print(
            f"Validation loss: {validation_loss:.4f} | "
            f"Accuracy: {validation_metrics['accuracy']:.4f} | "
            f"Precision: {validation_metrics['precision']:.4f} | "
            f"Recall: {validation_metrics['recall']:.4f} | "
            f"F1: {validation_metrics['f1']:.4f}"
        )

        if validation_metrics["f1"] > best_f1:
            best_f1 = validation_metrics["f1"]

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "validation_f1": best_f1,
                "class_names": ["real", "fake"],
            }

            torch.save(
                checkpoint,
                args.output_dir / "image_resnet18_best.pth",
            )

            print("Saved new best model.")

    with open(
        args.output_dir / "image_training_history.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(history, file, indent=2)

    print(f"\nBest validation F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()