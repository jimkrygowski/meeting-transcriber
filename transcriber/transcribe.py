"""mlx-whisper transcription on the Apple Silicon GPU."""

WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"


def normalize_segments(result: dict) -> list[dict]:
    """Whisper result dict -> clean [{start, end, text}] list."""
    segments = []
    for s in result.get("segments", []):
        text = s["text"].strip()
        if text:
            segments.append({"start": float(s["start"]), "end": float(s["end"]),
                             "text": text})
    return segments


def transcribe(wav_path) -> list[dict]:
    import mlx_whisper  # deferred: heavy import, and keeps unit tests model-free

    result = mlx_whisper.transcribe(str(wav_path), path_or_hf_repo=WHISPER_MODEL)
    return normalize_segments(result)
