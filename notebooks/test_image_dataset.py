from pathlib import Path

from torch.utils.data import DataLoader

from src.dataset.image_dataset import (
    ImageFrameDataset,
    get_image_transforms,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "manifests"
    / "fakeavceleb_subset.csv"
)

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fakeavceleb"
)


def main() -> None:
    image_transforms = get_image_transforms()

    for split in ["train", "validation", "test"]:
        dataset = ImageFrameDataset(
            manifest_path=MANIFEST_PATH,
            processed_root=PROCESSED_ROOT,
            split=split,
            transform=image_transforms[split],
        )

        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=split == "train",
            num_workers=0,
        )

        images, labels, video_ids = next(iter(loader))

        print(f"\nSplit: {split}")
        print(f"Dataset size: {len(dataset)} frames")
        print(f"Image batch shape: {images.shape}")
        print(f"Labels: {labels.tolist()}")
        print(f"First video ID: {video_ids[0]}")


if __name__ == "__main__":
    main()