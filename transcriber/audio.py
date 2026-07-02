"""ffmpeg-based audio validation, conversion, and clipping."""

import json
import subprocess
from pathlib import Path


class AudioError(Exception):
    """The input can't be read as audio."""


def probe_duration(path: Path | str) -> float:
    """Duration in seconds. Raises AudioError if the file has no audio stream."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type:format=duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioError(f"Not a readable media file: {Path(path).name}")
    data = json.loads(result.stdout)
    if not any(s.get("codec_type") == "audio" for s in data.get("streams", [])):
        raise AudioError(f"No audio track found in {Path(path).name}")
    try:
        return float(data["format"]["duration"])
    except (KeyError, ValueError) as e:
        raise AudioError(f"Could not determine duration of {Path(path).name}") from e


def _run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(["ffmpeg", "-y", "-v", "error", *args],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioError(f"ffmpeg failed: {result.stderr.strip()[:300]}")


def convert_to_wav(src: Path | str, dst: Path | str) -> None:
    """Convert any input to 16 kHz mono PCM WAV (what whisper and pyannote want)."""
    _run_ffmpeg(["-i", str(src), "-ar", "16000", "-ac", "1", str(dst)])


def extract_clip(src: Path | str, dst: Path | str, start: float, duration: float) -> None:
    _run_ffmpeg(["-ss", str(start), "-t", str(duration), "-i", str(src), str(dst)])
