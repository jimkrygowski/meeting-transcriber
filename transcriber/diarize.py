"""pyannote speaker diarization: who spoke when."""

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

HF_TOKEN_HELP = (
    "Speaker identification needs a one-time HuggingFace setup: "
    "1) create a free account at huggingface.co, "
    "2) accept the conditions at huggingface.co/pyannote/speaker-diarization-community-1, "
    "3) create a read token at huggingface.co/settings/tokens and put it in .env "
    "as HF_TOKEN=... then restart the server. "
    "After the first model download everything runs offline."
)


class DiarizationSetupError(Exception):
    """Diarization unavailable for setup reasons (HF token / gated model)."""


def turns_from_annotation(annotation) -> list[dict]:
    """pyannote Annotation -> sorted [{start, end, speaker}] list."""
    turns = [
        {"start": float(seg.start), "end": float(seg.end), "speaker": str(label)}
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t["start"])
    return turns


def diarize(wav_path, hf_token: str) -> list[dict]:
    if not hf_token:
        raise DiarizationSetupError(HF_TOKEN_HELP)

    import torch  # deferred: heavy imports, keeps unit tests model-free
    from pyannote.audio import Pipeline

    try:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)
    except Exception as e:
        raise DiarizationSetupError(f"{e}. {HF_TOKEN_HELP}") from e
    if pipeline is None:  # pyannote returns None when the model is gated/unauthorized
        raise DiarizationSetupError(HF_TOKEN_HELP)

    try:
        pipeline.to(torch.device("mps"))
    except Exception:
        pass  # fall back to CPU silently; slower but correct

    return turns_from_annotation(pipeline(str(wav_path)))
