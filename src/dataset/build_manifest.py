from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


CATEGORY_TO_BINARY_LABEL = {
    "RealVideo-RealAudio": 0,
    "FakeVideo-RealAudio": 1,
    "RealVideo-FakeAudio": 1,
    "FakeVideo-FakeAudio": 1,
}


def collect_videos(dataset_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for category, binary_label in CATEGORY_TO_BINARY_LABEL.items():
        category_dir = dataset_root / category

        if not category_dir.exists():
            raise FileNotFoundError(
                f"Expected category folder not found: {category_dir}"
            )

        for video_path in category_dir.rglob("*.mp4"):
            relative_path = video_path.relative_to(dataset_root)

            relative_parts = relative_path.parts

            ethnicity = relative_parts[1] if len(relative_parts) > 1 else "unknown"
            gender = relative_parts[2] if len(relative_parts) > 2 else "unknown"
            identity = relative_parts[3] if len(relative_parts) > 3 else "unknown"

            rows.append(
                {
                    "video_path": str(video_path.resolve()),
                    "relative_path": str(relative_path),
                    "category": category,
                    "binary_label": binary_label,
                    "binary_class": "real" if binary_label == 0 else "fake",
                    "ethnicity": ethnicity,
                    "gender": gender,
                    "identity": identity,
                    "filename": video_path.name,
                }
            )

    dataframe = pd.DataFrame(rows)

    if dataframe.empty:
        raise RuntimeError("No MP4 videos were found.")

    return dataframe


def sample_per_category(
    dataframe: pd.DataFrame,
    samples_per_category: int,
    seed: int,
) -> pd.DataFrame:
    sampled_groups: list[pd.DataFrame] = []

    for category, group in dataframe.groupby("category"):
        number_to_sample = min(samples_per_category, len(group))

        sampled_group = group.sample(
            n=number_to_sample,
            random_state=seed,
        )

        sampled_groups.append(sampled_group)

        print(
            f"{category}: selected {number_to_sample} "
            f"out of {len(group)} videos"
        )

    sampled = pd.concat(sampled_groups, ignore_index=True)

    sampled = sampled.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    return sampled


def assign_identity_splits(
    dataframe: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    random_generator = random.Random(seed)

    unique_identities = dataframe["identity"].dropna().unique().tolist()
    random_generator.shuffle(unique_identities)

    total_identities = len(unique_identities)
    train_end = int(total_identities * train_ratio)
    validation_end = train_end + int(
        total_identities * validation_ratio
    )

    train_identities = set(unique_identities[:train_end])
    validation_identities = set(
        unique_identities[train_end:validation_end]
    )

    def choose_split(identity: str) -> str:
        if identity in train_identities:
            return "train"

        if identity in validation_identities:
            return "validation"

        return "test"

    result = dataframe.copy()
    result["split"] = result["identity"].apply(choose_split)

    return result


def print_summary(dataframe: pd.DataFrame) -> None:
    print("\nCategory counts:")
    print(dataframe["category"].value_counts())

    print("\nBinary class counts:")
    print(dataframe["binary_class"].value_counts())

    print("\nSplit counts:")
    print(dataframe["split"].value_counts())

    print("\nSplit × binary class:")
    print(pd.crosstab(dataframe["split"], dataframe["binary_class"]))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FakeAVCeleb subset manifest."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/manifests/fakeavceleb_subset.csv"),
    )

    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    full_manifest = collect_videos(args.dataset_root)

    print(f"Found {len(full_manifest)} videos in total.")

    subset = sample_per_category(
        dataframe=full_manifest,
        samples_per_category=args.samples_per_category,
        seed=args.seed,
    )

    subset = assign_identity_splits(
        dataframe=subset,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.output, index=False)

    print_summary(subset)
    print(f"\nManifest saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()