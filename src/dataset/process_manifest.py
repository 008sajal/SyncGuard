from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.preprocessing.audio_utils import create_mel_spectrogram
from src.preprocessing.video_utils import extract_audio, extract_frames


def safe_name(relative_path: str) -> str:
    """
    Convert a relative video path into a unique filesystem-safe name.
    """
    path = Path(relative_path)

    stem_parts = list(path.with_suffix("").parts)
    return "__".join(stem_parts)


def process_video(
    row: pd.Series,
    output_root: Path,
    num_frames: int,
    sample_rate: int,
    duration: float,
    overwrite: bool,
) -> dict[str, object]:
    video_path = Path(row["video_path"])
    sample_name = safe_name(str(row["relative_path"]))

    split = str(row["split"])
    binary_class = str(row["binary_class"])

    sample_root = output_root / split / binary_class / sample_name
    frames_dir = sample_root / "frames"
    audio_path = sample_root / "audio.wav"
    spectrogram_path = sample_root / "mel_spectrogram.png"

    expected_frames = [
        frames_dir / f"frame_{index:03d}.jpg"
        for index in range(num_frames)
    ]

    if (
        not overwrite
        and audio_path.exists()
        and spectrogram_path.exists()
        and all(frame.exists() for frame in expected_frames)
    ):
        return {
            "status": "skipped",
            "sample_name": sample_name,
            "processed_root": str(sample_root),
            "error": "",
        }

    sample_root.mkdir(parents=True, exist_ok=True)

    extract_frames(
        video_path=video_path,
        output_dir=frames_dir,
        num_frames=num_frames,
    )

    extract_audio(
        video_path=video_path,
        output_audio_path=audio_path,
        sample_rate=sample_rate,
    )

    create_mel_spectrogram(
        audio_path=audio_path,
        output_path=spectrogram_path,
        sample_rate=sample_rate,
        duration=duration,
    )

    return {
        "status": "success",
        "sample_name": sample_name,
        "processed_root": str(sample_root),
        "error": "",
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess videos listed in a SyncGuard manifest."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/processed/manifests/fakeavceleb_subset.csv"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/fakeavceleb"),
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of videos to process.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not args.manifest.exists():
        raise FileNotFoundError(
            f"Manifest not found: {args.manifest}"
        )

    dataframe = pd.read_csv(args.manifest)

    if args.limit is not None:
        dataframe = dataframe.head(args.limit)

    results: list[dict[str, object]] = []

    for _, row in tqdm(
        dataframe.iterrows(),
        total=len(dataframe),
        desc="Processing videos",
    ):
        try:
            result = process_video(
                row=row,
                output_root=args.output_root,
                num_frames=args.num_frames,
                sample_rate=args.sample_rate,
                duration=args.duration,
                overwrite=args.overwrite,
            )
        except Exception as error:
            result = {
                "status": "failed",
                "sample_name": safe_name(
                    str(row["relative_path"])
                ),
                "processed_root": "",
                "error": str(error),
            }

            print("\nFailed video:")
            print(row["video_path"])
            print(error)

            traceback.print_exc()

        results.append(result)

    results_dataframe = pd.DataFrame(results)

    report_path = (
        args.output_root
        / "processing_report.csv"
    )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dataframe.to_csv(
        report_path,
        index=False,
    )

    print("\nProcessing summary:")
    print(results_dataframe["status"].value_counts())

    print(f"\nReport saved to: {report_path.resolve()}")


if __name__ == "__main__":
    main()