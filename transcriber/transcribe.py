"""mlx-whisper transcription on the Apple Silicon GPU."""

# Full large-v3: ~3x slower than turbo but noticeably more accurate,
# especially on names and domain jargon.
WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"


def normalize_segments(result: dict) -> list[dict]:
    """Whisper result dict -> clean [{start, end, text}] list."""
    segments = []
    for s in result.get("segments", []):
        text = s["text"].strip()
        if text:
            segments.append({"start": float(s["start"]), "end": float(s["end"]),
                             "text": text})
    return segments


def transcribe(wav_path, context: str = "") -> list[dict]:
    """Transcribe audio; `context` biases spelling of names/jargon.

    condition_on_previous_text=False stops the decoder from feeding its own
    output back in, which is what causes repeated-phrase hallucination loops
    around silence and crosstalk. It also makes the initial_prompt apply to
    every 30s chunk instead of only the first one.
    """
    import mlx_whisper  # deferred: heavy import, and keeps unit tests model-free

    kwargs = {"condition_on_previous_text": False}
    if context.strip():
        kwargs["initial_prompt"] = context.strip()
    result = mlx_whisper.transcribe(str(wav_path), path_or_hf_repo=WHISPER_MODEL,
                                    **kwargs)
    return normalize_segments(result)
