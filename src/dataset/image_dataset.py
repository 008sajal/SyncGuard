from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageFrameDataset(Dataset):
    """
    PyTorch dataset for the image-only deepfake baseline.

    Each extracted video frame is treated as one training example.
    Labels:
        0 = real
        1 = fake
    """

    def __init__(
        self,
        manifest_path: str | Path,
        processed_root: str | Path,
        split: str,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.processed_root = Path(processed_root)
        self.split = split
        self.transform = transform

        if split not in {"train", "validation", "test"}:
            raise ValueError(
                "split must be 'train', 'validation', or 'test'."
            )

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        dataframe = pd.read_csv(self.manifest_path)
        dataframe = dataframe[dataframe["split"] == split].copy()

        samples: list[dict[str, object]] = []

        for _, row in dataframe.iterrows():
            sample_name = self._safe_name(row["relative_path"])

            sample_root = (
                self.processed_root
                / split
                / row["binary_class"]
                / sample_name
            )

            frames_dir = sample_root / "frames"

            for frame_path in sorted(frames_dir.glob("*.jpg")):
                samples.append(
                    {
                        "frame_path": frame_path,
                        "label": int(row["binary_label"]),
                        "video_id": sample_name,
                        "category": row["category"],
                    }
                )

        if not samples:
            raise RuntimeError(
                f"No image samples found for split: {split}"
            )

        self.samples = samples

    @staticmethod
    def _safe_name(relative_path: str) -> str:
        path = Path(relative_path)
        return "__".join(path.with_suffix("").parts)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[index]

        image = Image.open(sample["frame_path"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(sample["label"])
        video_id = str(sample["video_id"])

        return image, label, video_id


def get_image_transforms() -> dict[str, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return {
        "train": train_transform,
        "validation": evaluation_transform,
        "test": evaluation_transform,
    }