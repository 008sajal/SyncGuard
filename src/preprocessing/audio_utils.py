from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np


def create_mel_spectrogram(
    audio_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16_000,
    duration: float = 5.0,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> Path:
    """Create and save a mel spectrogram image from an audio file."""

    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if duration <= 0:
        raise ValueError("duration must be greater than zero.")

    audio, _ = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True,
    )

    target_length = int(sample_rate * duration)

    if len(audio) < target_length:
        audio = np.pad(
            audio,
            (0, target_length - len(audio)),
            mode="constant",
        )
    else:
        audio = audio[:target_length]

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )

    mel_db = librosa.power_to_db(
        mel_spectrogram,
        ref=np.max,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path