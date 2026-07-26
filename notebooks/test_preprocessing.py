from pathlib import Path

from src.preprocessing.video_utils import extract_audio, extract_frames


PROJECT_ROOT = Path(__file__).resolve().parents[1]

video_path = PROJECT_ROOT / "data" / "raw" / "test_video.mp4"
frames_dir = PROJECT_ROOT / "data" / "processed" / "test_frames"
audio_path = PROJECT_ROOT / "data" / "processed" / "test_audio.wav"


def main() -> None:
    frames = extract_frames(
        video_path=video_path,
        output_dir=frames_dir,
        num_frames=8,
    )

    extracted_audio = extract_audio(
        video_path=video_path,
        output_audio_path=audio_path,
        sample_rate=16_000,
    )

    print(f"Successfully extracted {len(frames)} frames.")

    for frame in frames:
        print(f"  {frame}")

    print(f"Audio saved to: {extracted_audio}")


if __name__ == "__main__":
    main()