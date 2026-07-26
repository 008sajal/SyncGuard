from pathlib import Path

from src.preprocessing.audio_utils import create_mel_spectrogram


PROJECT_ROOT = Path(__file__).resolve().parents[1]

audio_path = PROJECT_ROOT / "data" / "processed" / "test_audio.wav"
spectrogram_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_mel_spectrogram.png"
)


def main() -> None:
    result = create_mel_spectrogram(
        audio_path=audio_path,
        output_path=spectrogram_path,
        sample_rate=16_000,
        duration=5.0,
    )

    print(f"Mel spectrogram saved to: {result}")


if __name__ == "__main__":
    main()