from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.image_dataset import (
    ImageFrameDataset,
    get_image_transforms,
)
from src.training.image_model import create_image_model


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
        "--checkpoint",
        type=Path,
        default=Path(
            "models/checkpoints/image_resnet18_best.pth"
        ),
    )

    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)

    return parser.parse_args()


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


def main() -> None:
    args = parse_arguments()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    transforms_by_split = get_image_transforms()

    dataset = ImageFrameDataset(
        manifest_path=args.manifest,
        processed_root=args.processed_root,
        split=args.split,
        transform=transforms_by_split[args.split],
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = create_image_model(
        num_classes=2,
        freeze_backbone=True,
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    frame_labels: list[int] = []
    frame_predictions: list[int] = []

    video_probabilities: dict[str, list[float]] = defaultdict(list)
    video_labels: dict[str, int] = {}

    with torch.no_grad():
        for images, labels, video_ids in tqdm(
            loader,
            desc="Evaluating",
        ):
            images = images.to(device)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = logits.argmax(dim=1)

            frame_labels.extend(labels.tolist())
            frame_predictions.extend(predictions.cpu().tolist())

            for video_id, label, probability in zip(
                video_ids,
                labels.tolist(),
                probabilities.cpu().tolist(),
            ):
                video_probabilities[video_id].append(probability)
                video_labels[video_id] = label

    video_true_labels: list[int] = []
    video_predictions: list[int] = []

    for video_id, probabilities in video_probabilities.items():
        mean_fake_probability = float(np.mean(probabilities))
        prediction = int(mean_fake_probability >= 0.5)

        video_true_labels.append(video_labels[video_id])
        video_predictions.append(prediction)

    frame_metrics = calculate_metrics(
        frame_labels,
        frame_predictions,
    )

    video_metrics = calculate_metrics(
        video_true_labels,
        video_predictions,
    )

    print("\nFrame-level metrics:")
    for name, value in frame_metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nVideo-level metrics:")
    for name, value in video_metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nVideo-level confusion matrix:")
    print(confusion_matrix(video_true_labels, video_predictions))

    print("\nVideo-level classification report:")
    print(
        classification_report(
            video_true_labels,
            video_predictions,
            target_names=["real", "fake"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()