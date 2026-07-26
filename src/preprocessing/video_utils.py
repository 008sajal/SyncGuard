from pathlib import Path
import subprocess

import cv2


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    num_frames: int = 8,
) -> list[Path]:
    """Extract evenly spaced frames from a video."""

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    if num_frames <= 0:
        raise ValueError("num_frames must be greater than zero.")

    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        capture.release()
        raise RuntimeError("No readable frames were found in the video.")

    number_to_extract = min(num_frames, total_frames)

    if number_to_extract == 1:
        frame_indices = [total_frames // 2]
    else:
        frame_indices = [
            round(i * (total_frames - 1) / (number_to_extract - 1))
            for i in range(number_to_extract)
        ]

    saved_paths: list[Path] = []

    for output_index, frame_index in enumerate(frame_indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()

        if not success:
            print(f"Warning: frame {frame_index} could not be read.")
            continue

        frame_path = output_dir / f"frame_{output_index:03d}.jpg"

        if not cv2.imwrite(str(frame_path), frame):
            capture.release()
            raise RuntimeError(f"Could not save frame: {frame_path}")

        saved_paths.append(frame_path)

    capture.release()

    if not saved_paths:
        raise RuntimeError("Frame extraction failed.")

    return saved_paths


def extract_audio(
    video_path: str | Path,
    output_audio_path: str | Path,
    sample_rate: int = 16_000,
) -> Path:
    """Extract mono WAV audio from a video using FFmpeg."""

    video_path = Path(video_path)
    output_audio_path = Path(output_audio_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        str(output_audio_path),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and restart VS Code."
        ) from error

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg could not extract the audio.\n"
            f"{result.stderr.strip()}"
        )

    if not output_audio_path.exists():
        raise RuntimeError("FFmpeg finished, but no audio file was created.")

    return output_audio_path