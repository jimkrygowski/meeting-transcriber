# Meeting Transcriber Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fully local web app that transcribes recorded meeting audio with speaker diarization, manual speaker renaming, Markdown/TXT export, and Ollama-generated summaries.

**Architecture:** FastAPI server at `localhost:8484` serving a plain HTML/JS frontend. Pipeline runs in a background thread per job: ffmpeg convert → mlx-whisper transcribe (Apple Silicon GPU) → pyannote diarize → timestamp-overlap merge. Jobs persist to `./data/<job-id>/` as JSON + audio.

**Tech Stack:** Python 3.12 via `uv`, FastAPI + uvicorn, mlx-whisper (`mlx-community/whisper-large-v3-turbo`), pyannote.audio 3.x (`pyannote/speaker-diarization-3.1`), ffmpeg, Ollama (`qwen2.5:7b-instruct`), httpx, pytest.

## Global Constraints

- Python `>=3.12`, dependencies managed by `uv` (`uv run`, `uv add`).
- Server port: `8484`.
- Everything runs on-device. No cloud calls anywhere. Ollama at `http://localhost:11434`.
- Whisper model: `mlx-community/whisper-large-v3-turbo`. Diarization model: `pyannote/speaker-diarization-3.1` (pin `pyannote.audio>=3.3,<4`). Summary model: `qwen2.5:7b-instruct`.
- Job data lives in `./data/<job-id>/` (gitignored). HF token read from `.env` as `HF_TOKEN`.
- Segments always reference stable speaker IDs (`SPEAKER_00`…); display names live only in `speaker_map`.
- Default pytest run excludes `slow` marker (`addopts = "-m 'not slow'"`).
- Commit after every task. Package name: `transcriber`.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `transcriber/__init__.py`, `tests/__init__.py`, `.env.example`

**Interfaces:**
- Produces: importable `transcriber` package; `uv run pytest` works; ffmpeg on PATH.

- [ ] **Step 1: Install system tools** (skip any already present)

```bash
brew install ffmpeg uv
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "meeting-transcriber"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "python-multipart>=0.0.9",
    "mlx-whisper>=0.4",
    "pyannote.audio>=3.3,<4",
    "torch>=2.3",
    "torchaudio>=2.3",
    "httpx>=0.27",
    "python-dotenv>=1.0",
]

[dependency-groups]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
markers = ["slow: full-pipeline tests with real models"]
addopts = "-m 'not slow'"
testpaths = ["tests"]

[tool.uv]
package = false
```

- [ ] **Step 3: Create package skeleton and `.env.example`**

`transcriber/__init__.py` and `tests/__init__.py`: empty files.

`.env.example`:
```
# HuggingFace token for downloading the gated pyannote diarization model (one-time).
# 1. Create a free account at https://huggingface.co
# 2. Accept conditions at https://huggingface.co/pyannote/speaker-diarization-3.1
#    and https://huggingface.co/pyannote/segmentation-3.0
# 3. Create a read token at https://huggingface.co/settings/tokens
HF_TOKEN=
```

- [ ] **Step 4: Sync and verify**

Run: `uv sync && uv run pytest`
Expected: deps resolve; pytest exits with "no tests ran".

Run: `ffprobe -version | head -1`
Expected: version string.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock transcriber tests .env.example
git commit -m "chore: project scaffold with uv, FastAPI, mlx-whisper, pyannote deps"
```

---

### Task 2: Speaker merge logic (`merge.py`)

**Files:**
- Create: `transcriber/merge.py`
- Test: `tests/test_merge.py`

**Interfaces:**
- Produces:
  - `assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]` — segments `{start,end,text}` + turns `{start,end,speaker}` → `{start,end,speaker,text}`.
  - `default_speaker_map(labeled_segments: list[dict]) -> dict[str,str]` — `{"SPEAKER_00": "Speaker 1", ...}` ordered by first appearance.
  - `best_snippets(turns: list[dict], max_len: float) -> dict[str, tuple[float,float]]` — per speaker `(start, duration)` of their longest turn, capped at `max_len`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_merge.py
from transcriber.merge import assign_speakers, best_snippets, default_speaker_map


def seg(start, end, text="hi"):
    return {"start": start, "end": end, "text": text}


def turn(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def test_segment_inside_turn_gets_that_speaker():
    out = assign_speakers([seg(1, 2)], [turn(0, 5, "SPEAKER_00")])
    assert out[0]["speaker"] == "SPEAKER_00"
    assert out[0]["text"] == "hi"


def test_segment_spanning_two_turns_gets_larger_overlap():
    turns = [turn(0, 3, "SPEAKER_00"), turn(3, 10, "SPEAKER_01")]
    out = assign_speakers([seg(2, 6)], turns)
    assert out[0]["speaker"] == "SPEAKER_01"  # 3s overlap beats 1s


def test_overlap_accumulates_across_multiple_turns_of_same_speaker():
    turns = [
        turn(0, 2, "SPEAKER_00"),
        turn(2, 3, "SPEAKER_01"),
        turn(3, 5, "SPEAKER_00"),
    ]
    out = assign_speakers([seg(0, 5)], turns)
    assert out[0]["speaker"] == "SPEAKER_00"  # 4s total beats 1s


def test_segment_in_gap_gets_nearest_turn_speaker():
    turns = [turn(0, 2, "SPEAKER_00"), turn(10, 12, "SPEAKER_01")]
    out = assign_speakers([seg(3, 4)], turns)
    assert out[0]["speaker"] == "SPEAKER_00"


def test_no_turns_falls_back_to_single_speaker():
    out = assign_speakers([seg(0, 1), seg(1, 2)], [])
    assert all(s["speaker"] == "SPEAKER_00" for s in out)


def test_default_speaker_map_ordered_by_first_appearance():
    labeled = [
        {"start": 0, "end": 1, "speaker": "SPEAKER_01", "text": "a"},
        {"start": 1, "end": 2, "speaker": "SPEAKER_00", "text": "b"},
        {"start": 2, "end": 3, "speaker": "SPEAKER_01", "text": "c"},
    ]
    assert default_speaker_map(labeled) == {
        "SPEAKER_01": "Speaker 1",
        "SPEAKER_00": "Speaker 2",
    }


def test_best_snippets_picks_longest_turn_capped():
    turns = [
        turn(0, 2, "SPEAKER_00"),
        turn(5, 30, "SPEAKER_00"),
        turn(40, 43, "SPEAKER_01"),
    ]
    out = best_snippets(turns, max_len=10.0)
    assert out == {"SPEAKER_00": (5.0, 10.0), "SPEAKER_01": (40.0, 3.0)}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_merge.py -v`
Expected: FAIL — `ModuleNotFoundError: transcriber.merge`.

- [ ] **Step 3: Implement**

```python
# transcriber/merge.py
"""Pure functions combining transcript segments with diarization turns."""

FALLBACK_SPEAKER = "SPEAKER_00"


def assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Label each transcript segment with the speaker whose turns overlap it most.

    Segments with no overlapping turn get the speaker of the nearest turn.
    With no turns at all (diarization unavailable), everything is one speaker.
    """
    if not turns:
        return [{**s, "speaker": FALLBACK_SPEAKER} for s in segments]
    labeled = []
    for s in segments:
        overlaps: dict[str, float] = {}
        for t in turns:
            o = min(s["end"], t["end"]) - max(s["start"], t["start"])
            if o > 0:
                overlaps[t["speaker"]] = overlaps.get(t["speaker"], 0.0) + o
        if overlaps:
            speaker = max(overlaps, key=lambda k: overlaps[k])
        else:
            mid = (s["start"] + s["end"]) / 2
            nearest = min(
                turns,
                key=lambda t: min(abs(mid - t["start"]), abs(mid - t["end"])),
            )
            speaker = nearest["speaker"]
        labeled.append({**s, "speaker": speaker})
    return labeled


def default_speaker_map(labeled_segments: list[dict]) -> dict[str, str]:
    """Human-friendly default names, numbered by order of first appearance."""
    mapping: dict[str, str] = {}
    for s in labeled_segments:
        if s["speaker"] not in mapping:
            mapping[s["speaker"]] = f"Speaker {len(mapping) + 1}"
    return mapping


def best_snippets(turns: list[dict], max_len: float) -> dict[str, tuple[float, float]]:
    """Per speaker, the (start, duration) of their longest turn, capped at max_len."""
    best: dict[str, tuple[float, float]] = {}
    for t in turns:
        dur = t["end"] - t["start"]
        if t["speaker"] not in best or dur > best[t["speaker"]][1]:
            best[t["speaker"]] = (float(t["start"]), float(dur))
    return {k: (start, min(dur, max_len)) for k, (start, dur) in best.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_merge.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add transcriber/merge.py tests/test_merge.py
git commit -m "feat: speaker assignment by timestamp overlap"
```

---

### Task 3: Export formatting (`export.py`)

**Files:**
- Create: `transcriber/export.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: labeled segments (`{start,end,speaker,text}`) and `speaker_map` from Task 2's shapes.
- Produces:
  - `format_timestamp(seconds: float) -> str` — `"MM:SS"` under an hour, `"H:MM:SS"` above.
  - `group_by_speaker(segments: list[dict]) -> list[dict]` — merges consecutive same-speaker segments into blocks `{start,end,speaker,text}`.
  - `to_markdown(segments, speaker_map, title, summary=None) -> str`
  - `to_text(segments, speaker_map, title, summary=None) -> str`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_export.py
from transcriber.export import format_timestamp, group_by_speaker, to_markdown, to_text

SEGMENTS = [
    {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "Hello all."},
    {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_00", "text": "Let's start."},
    {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_01", "text": "Sounds good."},
]
NAMES = {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}


def test_format_timestamp():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(65.4) == "01:05"
    assert format_timestamp(3723) == "1:02:03"


def test_group_by_speaker_merges_consecutive():
    blocks = group_by_speaker(SEGMENTS)
    assert len(blocks) == 2
    assert blocks[0]["text"] == "Hello all. Let's start."
    assert blocks[0]["end"] == 4.0
    assert blocks[1]["speaker"] == "SPEAKER_01"


def test_to_markdown_has_title_names_timestamps():
    md = to_markdown(SEGMENTS, NAMES, "Standup")
    assert md.startswith("# Standup\n")
    assert "**Alice** [00:00]: Hello all. Let's start." in md
    assert "**Bob** [00:04]: Sounds good." in md
    assert "## Transcript" in md


def test_to_markdown_includes_summary_when_present():
    md = to_markdown(SEGMENTS, NAMES, "Standup", summary="## Summary\nShort.")
    assert "## Summary\nShort." in md
    assert md.index("## Summary") < md.index("## Transcript")


def test_to_text_plain():
    txt = to_text(SEGMENTS, NAMES, "Standup")
    assert "Alice [00:00]: Hello all. Let's start." in txt
    assert "**" not in txt


def test_unmapped_speaker_falls_back_to_id():
    md = to_markdown(SEGMENTS, {}, "T")
    assert "**SPEAKER_00**" in md
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_export.py -v`
Expected: FAIL — `ModuleNotFoundError: transcriber.export`.

- [ ] **Step 3: Implement**

```python
# transcriber/export.py
"""Render a speaker-labeled transcript as Markdown or plain text."""


def format_timestamp(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def group_by_speaker(segments: list[dict]) -> list[dict]:
    blocks: list[dict] = []
    for s in segments:
        if blocks and blocks[-1]["speaker"] == s["speaker"]:
            blocks[-1]["text"] += " " + s["text"]
            blocks[-1]["end"] = s["end"]
        else:
            blocks.append(dict(s))
    return blocks


def _lines(segments, speaker_map, title, summary, name_fmt):
    lines = [f"# {title}", ""]
    if summary:
        lines += [summary.strip(), ""]
    lines += ["## Transcript", ""]
    for b in group_by_speaker(segments):
        name = speaker_map.get(b["speaker"], b["speaker"])
        lines.append(f"{name_fmt.format(name)} [{format_timestamp(b['start'])}]: {b['text']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def to_markdown(segments, speaker_map, title, summary=None) -> str:
    return _lines(segments, speaker_map, title, summary, "**{}**")


def to_text(segments, speaker_map, title, summary=None) -> str:
    md = _lines(segments, speaker_map, title, summary, "{}")
    return md.replace("## ", "").replace("# ", "")  # longest heading marker first
```

Note: `to_text` header stripping is naive; if tests show it mangles transcript text containing `# `, replace with a line-by-line `lstrip('#').strip()` on heading lines only. Keep whichever is simplest that passes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_export.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add transcriber/export.py tests/test_export.py
git commit -m "feat: markdown and plain-text transcript export"
```

---

### Task 4: Audio handling (`audio.py`)

**Files:**
- Create: `transcriber/audio.py`, `tests/conftest.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Produces:
  - `AudioError(Exception)` — invalid/unreadable input.
  - `probe_duration(path) -> float` — seconds; raises `AudioError` if no audio stream.
  - `convert_to_wav(src, dst) -> None` — 16 kHz mono WAV.
  - `extract_clip(src, dst, start: float, duration: float) -> None`
  - Test fixture `tone_m4a` (2-second 440 Hz tone) reused by later server tests.

- [ ] **Step 1: Write conftest fixture and failing tests**

```python
# tests/conftest.py
import subprocess

import pytest


@pytest.fixture
def tone_m4a(tmp_path):
    """A 2-second 440 Hz tone in m4a, generated by ffmpeg."""
    path = tmp_path / "tone.m4a"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=2", "-c:a", "aac", str(path)],
        check=True,
    )
    return path
```

```python
# tests/test_audio.py
import json
import subprocess

import pytest

from transcriber.audio import AudioError, convert_to_wav, extract_clip, probe_duration


def test_probe_duration_of_tone(tone_m4a):
    assert probe_duration(tone_m4a) == pytest.approx(2.0, abs=0.2)


def test_probe_rejects_non_audio(tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("not audio")
    with pytest.raises(AudioError):
        probe_duration(junk)


def test_probe_rejects_missing_file(tmp_path):
    with pytest.raises(AudioError):
        probe_duration(tmp_path / "missing.mp3")


def test_convert_to_wav_16k_mono(tone_m4a, tmp_path):
    dst = tmp_path / "out.wav"
    convert_to_wav(tone_m4a, dst)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=sample_rate,channels", "-of", "json", str(dst)],
        capture_output=True, text=True, check=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    assert stream["sample_rate"] == "16000"
    assert stream["channels"] == 1


def test_extract_clip(tone_m4a, tmp_path):
    wav = tmp_path / "full.wav"
    convert_to_wav(tone_m4a, wav)
    clip = tmp_path / "clip.wav"
    extract_clip(wav, clip, start=0.5, duration=1.0)
    assert probe_duration(clip) == pytest.approx(1.0, abs=0.1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_audio.py -v`
Expected: FAIL — `ModuleNotFoundError: transcriber.audio`.

- [ ] **Step 3: Implement**

```python
# transcriber/audio.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_audio.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add transcriber/audio.py tests/conftest.py tests/test_audio.py
git commit -m "feat: ffmpeg audio probe, conversion, clip extraction"
```

---

### Task 5: Transcription wrapper (`transcribe.py`)

**Files:**
- Create: `transcriber/transcribe.py`
- Test: `tests/test_transcribe.py`

**Interfaces:**
- Produces:
  - `WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"`
  - `transcribe(wav_path) -> list[dict]` — `{start,end,text}` segments (calls mlx-whisper; exercised by the slow test).
  - `normalize_segments(result: dict) -> list[dict]` — pure, unit-tested.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transcribe.py
from transcriber.transcribe import normalize_segments


def test_normalize_strips_and_drops_empty():
    raw = {"segments": [
        {"start": 0.0, "end": 1.5, "text": "  Hello there. "},
        {"start": 1.5, "end": 2.0, "text": "   "},
        {"start": 2.0, "end": 3.0, "text": "Bye."},
    ]}
    assert normalize_segments(raw) == [
        {"start": 0.0, "end": 1.5, "text": "Hello there."},
        {"start": 2.0, "end": 3.0, "text": "Bye."},
    ]


def test_normalize_handles_missing_segments():
    assert normalize_segments({}) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# transcriber/transcribe.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add transcriber/transcribe.py tests/test_transcribe.py
git commit -m "feat: mlx-whisper transcription wrapper"
```

---

### Task 6: Diarization wrapper (`diarize.py`)

**Files:**
- Create: `transcriber/diarize.py`
- Test: `tests/test_diarize.py`

**Interfaces:**
- Produces:
  - `DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"`
  - `DiarizationSetupError(Exception)` — missing/invalid HF token; message contains user-facing setup steps.
  - `diarize(wav_path, hf_token: str) -> list[dict]` — `{start,end,speaker}` turns sorted by start.
  - `turns_from_annotation(annotation) -> list[dict]` — pure conversion, unit-tested with a fake.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_diarize.py
import pytest

from transcriber.diarize import DiarizationSetupError, diarize, turns_from_annotation


class FakeSegment:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class FakeAnnotation:
    def itertracks(self, yield_label=False):
        yield FakeSegment(5.0, 7.5), "B", "SPEAKER_01"
        yield FakeSegment(0.0, 4.0), "A", "SPEAKER_00"


def test_turns_from_annotation_sorted_by_start():
    assert turns_from_annotation(FakeAnnotation()) == [
        {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 7.5, "speaker": "SPEAKER_01"},
    ]


def test_diarize_without_token_raises_setup_error():
    with pytest.raises(DiarizationSetupError, match="HF_TOKEN"):
        diarize("whatever.wav", hf_token="")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_diarize.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# transcriber/diarize.py
"""pyannote speaker diarization: who spoke when."""

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

HF_TOKEN_HELP = (
    "Speaker identification needs a one-time HuggingFace setup: "
    "1) create a free account at huggingface.co, "
    "2) accept the conditions at huggingface.co/pyannote/speaker-diarization-3.1 "
    "and huggingface.co/pyannote/segmentation-3.0, "
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
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=hf_token)
    except Exception as e:
        raise DiarizationSetupError(f"{e}. {HF_TOKEN_HELP}") from e
    if pipeline is None:  # pyannote returns None when the model is gated/unauthorized
        raise DiarizationSetupError(HF_TOKEN_HELP)

    try:
        pipeline.to(torch.device("mps"))
    except Exception:
        pass  # fall back to CPU silently; slower but correct

    return turns_from_annotation(pipeline(str(wav_path)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_diarize.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add transcriber/diarize.py tests/test_diarize.py
git commit -m "feat: pyannote diarization wrapper with setup-error handling"
```

---

### Task 7: Local summary via Ollama (`summarize.py`)

**Files:**
- Create: `transcriber/summarize.py`
- Test: `tests/test_summarize.py`

**Interfaces:**
- Produces:
  - `OLLAMA_URL = "http://localhost:11434"`, `SUMMARY_MODEL = "qwen2.5:7b-instruct"`
  - `SummaryError(Exception)` — message contains the exact command to fix.
  - `check_ollama() -> None` — raises `SummaryError` if server down or model missing.
  - `summarize(transcript_text: str) -> str` — Markdown with `## Summary` and `## Action Items`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_summarize.py
import httpx
import pytest

from transcriber import summarize as sz


class FakeResponse:
    def __init__(self, json_data, status=200):
        self._json = json_data
        self.status_code = status

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)


def test_check_ollama_down_raises_with_fix(monkeypatch):
    def fail(*a, **k):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(httpx, "get", fail)
    with pytest.raises(sz.SummaryError, match="ollama serve|Ollama app"):
        sz.check_ollama()


def test_check_ollama_missing_model_raises_pull_command(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse({"models": []}))
    with pytest.raises(sz.SummaryError, match="ollama pull qwen2.5:7b-instruct"):
        sz.check_ollama()


def test_summarize_returns_content(monkeypatch):
    monkeypatch.setattr(
        httpx, "get",
        lambda *a, **k: FakeResponse({"models": [{"name": "qwen2.5:7b-instruct"}]}),
    )
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return FakeResponse({"message": {"content": " ## Summary\nGood meeting.\n"}})

    monkeypatch.setattr(httpx, "post", fake_post)
    out = sz.summarize("Alice [00:00]: Hi.")
    assert out == "## Summary\nGood meeting."
    assert "Alice [00:00]: Hi." in captured["json"]["messages"][0]["content"]
    assert captured["json"]["model"] == "qwen2.5:7b-instruct"
    assert captured["json"]["stream"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_summarize.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# transcriber/summarize.py
"""Meeting summary + action items via a local Ollama model."""

import httpx

OLLAMA_URL = "http://localhost:11434"
SUMMARY_MODEL = "qwen2.5:7b-instruct"

PROMPT = """You are given a meeting transcript with speaker names and timestamps.
Produce exactly two Markdown sections:

## Summary
3-6 sentences covering the key points, decisions, and disagreements.

## Action Items
A bullet list of concrete action items, each with an owner when one is
identifiable from the transcript. If there are none, write "None identified."

Transcript:

{transcript}"""


class SummaryError(Exception):
    """Summary unavailable; message says exactly how to fix it."""


def check_ollama() -> None:
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        r.raise_for_status()
    except Exception as e:
        raise SummaryError(
            "Ollama isn't running. Install it with `brew install ollama`, then "
            "start it with `ollama serve` (or open the Ollama app)."
        ) from e
    names = [m.get("name", "") for m in r.json().get("models", [])]
    if not any(n.startswith(SUMMARY_MODEL) for n in names):
        raise SummaryError(
            f"The summary model isn't downloaded yet. Run: `ollama pull {SUMMARY_MODEL}`"
        )


def summarize(transcript_text: str) -> str:
    check_ollama()
    r = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": SUMMARY_MODEL,
            "messages": [{"role": "user",
                          "content": PROMPT.format(transcript=transcript_text)}],
            "stream": False,
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_summarize.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add transcriber/summarize.py tests/test_summarize.py
git commit -m "feat: local meeting summary via Ollama"
```

---

### Task 8: Job store and pipeline orchestration (`jobs.py`)

**Files:**
- Create: `transcriber/jobs.py`
- Test: `tests/test_jobs.py`

**Interfaces:**
- Consumes: `audio.probe_duration/convert_to_wav/extract_clip`, `merge.assign_speakers/default_speaker_map/best_snippets`, `transcribe.transcribe`, `diarize.diarize/DiarizationSetupError`, `summarize.summarize`, `export.to_text`.
- Produces:
  - `JobStore(data_dir)` with: `job_dir(job_id) -> Path`, `read_job/write_job/update_job`, `read_transcript/write_transcript/update_transcript`, `latest_job_id() -> str | None`, `original_path(job_id) -> Path`.
  - `create_job(store, original_name) -> tuple[str, Path]` — job id + destination path for the upload.
  - `run_pipeline(store, job_id, hf_token, transcribe_fn=..., diarize_fn=...)` — synchronous; test with fakes.
  - `start_pipeline(store, job_id, hf_token)` / `start_summary(store, job_id)` — daemon-thread wrappers.
  - `run_summary(store, job_id, summarize_fn=None)`
  - `job.json` shape: `{id, original_name, created, status: processing|done|error, stage, duration, error, warning}`
  - `transcript.json` shape: `{segments, speaker_map, summary, summary_status: none|running|done|error, summary_error}`
  - `SNIPPET_MAX_SECONDS = 10.0`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_jobs.py
import pytest

from transcriber.diarize import DiarizationSetupError
from transcriber.jobs import JobStore, create_job, run_pipeline, run_summary


@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "data")


def make_job(store, tone_m4a):
    job_id, dest = create_job(store, "standup.m4a")
    dest.write_bytes(tone_m4a.read_bytes())
    return job_id


def fake_transcribe(wav):
    return [
        {"start": 0.0, "end": 1.0, "text": "Hello."},
        {"start": 1.0, "end": 2.0, "text": "Hi there."},
    ]


def fake_diarize(wav, token):
    return [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
    ]


def test_create_job_writes_processing_state(store):
    job_id, dest = create_job(store, "standup.m4a")
    job = store.read_job(job_id)
    assert job["status"] == "processing"
    assert job["original_name"] == "standup.m4a"
    assert dest.name == "original.m4a"


def test_pipeline_happy_path(store, tone_m4a):
    job_id = make_job(store, tone_m4a)
    run_pipeline(store, job_id, "tok",
                 transcribe_fn=fake_transcribe, diarize_fn=fake_diarize)
    job = store.read_job(job_id)
    assert job["status"] == "done"
    assert job["duration"] == pytest.approx(2.0, abs=0.2)
    t = store.read_transcript(job_id)
    assert [s["speaker"] for s in t["segments"]] == ["SPEAKER_00", "SPEAKER_01"]
    assert t["speaker_map"] == {"SPEAKER_00": "Speaker 1", "SPEAKER_01": "Speaker 2"}
    assert (store.job_dir(job_id) / "snippets" / "SPEAKER_00.wav").exists()
    assert store.latest_job_id() == job_id


def test_pipeline_diarization_setup_error_degrades_gracefully(store, tone_m4a):
    job_id = make_job(store, tone_m4a)

    def failing_diarize(wav, token):
        raise DiarizationSetupError("needs HF_TOKEN setup")

    run_pipeline(store, job_id, "",
                 transcribe_fn=fake_transcribe, diarize_fn=failing_diarize)
    job = store.read_job(job_id)
    assert job["status"] == "done"
    assert "HF_TOKEN" in job["warning"]
    t = store.read_transcript(job_id)
    assert {s["speaker"] for s in t["segments"]} == {"SPEAKER_00"}


def test_pipeline_hard_error_records_stage(store, tone_m4a):
    job_id = make_job(store, tone_m4a)

    def boom(wav):
        raise RuntimeError("GPU on fire")

    run_pipeline(store, job_id, "tok",
                 transcribe_fn=boom, diarize_fn=fake_diarize)
    job = store.read_job(job_id)
    assert job["status"] == "error"
    assert "transcribing" in job["error"]
    assert "GPU on fire" in job["error"]


def test_run_summary_success_and_failure(store, tone_m4a):
    job_id = make_job(store, tone_m4a)
    run_pipeline(store, job_id, "tok",
                 transcribe_fn=fake_transcribe, diarize_fn=fake_diarize)

    run_summary(store, job_id, summarize_fn=lambda text: "## Summary\nFine.")
    t = store.read_transcript(job_id)
    assert t["summary_status"] == "done"
    assert t["summary"] == "## Summary\nFine."

    def fail(text):
        raise RuntimeError("ollama down")

    run_summary(store, job_id, summarize_fn=fail)
    t = store.read_transcript(job_id)
    assert t["summary_status"] == "error"
    assert "ollama down" in t["summary_error"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_jobs.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# transcriber/jobs.py
"""Job persistence and pipeline orchestration."""

import json
import threading
import time
import uuid
from pathlib import Path

from . import audio, export, merge
from .diarize import DiarizationSetupError
from .diarize import diarize as _diarize
from .transcribe import transcribe as _transcribe

SNIPPET_MAX_SECONDS = 10.0


class JobStore:
    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)

    def job_dir(self, job_id: str) -> Path:
        return self.data_dir / job_id

    def _read(self, job_id: str, name: str) -> dict | None:
        p = self.job_dir(job_id) / name
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def _write(self, job_id: str, name: str, data: dict) -> None:
        (self.job_dir(job_id) / name).write_text(json.dumps(data, indent=2))

    def read_job(self, job_id):
        return self._read(job_id, "job.json")

    def write_job(self, job_id, job):
        self._write(job_id, "job.json", job)

    def update_job(self, job_id, **fields):
        job = self.read_job(job_id)
        job.update(fields)
        self.write_job(job_id, job)
        return job

    def read_transcript(self, job_id):
        return self._read(job_id, "transcript.json")

    def write_transcript(self, job_id, transcript):
        self._write(job_id, "transcript.json", transcript)

    def update_transcript(self, job_id, **fields):
        t = self.read_transcript(job_id)
        t.update(fields)
        self.write_transcript(job_id, t)
        return t

    def latest_job_id(self) -> str | None:
        if not self.data_dir.exists():
            return None
        candidates = [
            (self.read_job(d.name)["created"], d.name)
            for d in self.data_dir.iterdir()
            if (d / "job.json").exists()
        ]
        return max(candidates)[1] if candidates else None

    def original_path(self, job_id: str) -> Path:
        return next(self.job_dir(job_id).glob("original.*"))


def create_job(store: JobStore, original_name: str) -> tuple[str, Path]:
    """Register a job and return (job_id, path where the upload must be written)."""
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    store.job_dir(job_id).mkdir(parents=True)
    suffix = Path(original_name).suffix.lower() or ".bin"
    store.write_job(job_id, {
        "id": job_id,
        "original_name": original_name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "processing",
        "stage": "uploading",
        "duration": None,
        "error": None,
        "warning": None,
    })
    return job_id, store.job_dir(job_id) / f"original{suffix}"


def run_pipeline(store, job_id, hf_token,
                 transcribe_fn=_transcribe, diarize_fn=_diarize):
    d = store.job_dir(job_id)
    wav = d / "audio.wav"
    try:
        store.update_job(job_id, stage="converting")
        src = store.original_path(job_id)
        store.update_job(job_id, duration=audio.probe_duration(src))
        audio.convert_to_wav(src, wav)

        store.update_job(job_id, stage="transcribing")
        segments = transcribe_fn(wav)

        store.update_job(job_id, stage="diarizing")
        warning = None
        try:
            turns = diarize_fn(wav, hf_token)
        except DiarizationSetupError as e:
            turns, warning = [], str(e)

        store.update_job(job_id, stage="finishing")
        labeled = merge.assign_speakers(segments, turns)
        snip_dir = d / "snippets"
        snip_dir.mkdir(exist_ok=True)
        for spk, (start, dur) in merge.best_snippets(turns, SNIPPET_MAX_SECONDS).items():
            audio.extract_clip(wav, snip_dir / f"{spk}.wav", start, dur)
        store.write_transcript(job_id, {
            "segments": labeled,
            "speaker_map": merge.default_speaker_map(labeled),
            "summary": None,
            "summary_status": "none",
            "summary_error": None,
        })
        store.update_job(job_id, status="done", stage="done", warning=warning)
    except Exception as e:
        job = store.read_job(job_id) or {"stage": "?"}
        store.update_job(job_id, status="error",
                         error=f"{job.get('stage')}: {e}")


def run_summary(store, job_id, summarize_fn=None):
    if summarize_fn is None:
        from .summarize import summarize as summarize_fn
    store.update_transcript(job_id, summary_status="running", summary_error=None)
    t = store.read_transcript(job_id)
    try:
        text = export.to_text(t["segments"], t["speaker_map"], "Transcript")
        store.update_transcript(job_id, summary=summarize_fn(text),
                                summary_status="done")
    except Exception as e:
        store.update_transcript(job_id, summary_status="error",
                                summary_error=str(e))


def start_pipeline(store, job_id, hf_token):
    threading.Thread(target=run_pipeline, args=(store, job_id, hf_token),
                     daemon=True).start()


def start_summary(store, job_id):
    threading.Thread(target=run_summary, args=(store, job_id), daemon=True).start()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_jobs.py -v`
Expected: 5 passed. (These use the `tone_m4a` fixture, so ffmpeg runs for real.)

- [ ] **Step 5: Commit**

```bash
git add transcriber/jobs.py tests/test_jobs.py
git commit -m "feat: job store and background pipeline orchestration"
```

---

### Task 9: FastAPI server (`server.py`, `__main__.py`)

**Files:**
- Create: `transcriber/server.py`, `transcriber/__main__.py`, `transcriber/static/.gitkeep` (placeholder until Task 10)
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: everything from Tasks 3, 4, 8.
- Produces HTTP API used by the frontend:
  - `POST /api/jobs` (multipart `file`) → `{"job_id": ...}`; 400 on non-audio.
  - `GET /api/jobs/latest` and `GET /api/jobs/{id}` → `{"job": {...}, "transcript": {...}|null}`; 404 if absent.
  - `POST /api/jobs/{id}/speakers` body `{"speaker_id","name"}` → `{"speaker_map": {...}}`.
  - `POST /api/jobs/{id}/summarize` → `{"summary_status": "running"}`; 409 if transcript not ready.
  - `GET /api/jobs/{id}/snippets/{speaker_id}` → WAV.
  - `GET /api/jobs/{id}/export?fmt=md|txt` → attachment.
  - `/` serves `transcriber/static/` (html mode).
  - Module-level `store` (tests replace it) and `app`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_server.py
import pytest
from fastapi.testclient import TestClient

from transcriber import jobs, server


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "store", jobs.JobStore(tmp_path / "data"))
    # keep tests fast and model-free: pipeline runs synchronously with fakes
    def fake_start_pipeline(store, job_id, hf_token):
        jobs.run_pipeline(
            store, job_id, hf_token,
            transcribe_fn=lambda wav: [
                {"start": 0.0, "end": 1.0, "text": "Hello."},
                {"start": 1.0, "end": 2.0, "text": "Hi."},
            ],
            diarize_fn=lambda wav, tok: [
                {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
                {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
            ],
        )
    monkeypatch.setattr(server.jobs, "start_pipeline", fake_start_pipeline)
    return TestClient(server.app)


def upload(client, tone_m4a):
    with tone_m4a.open("rb") as f:
        r = client.post("/api/jobs", files={"file": ("standup.m4a", f, "audio/m4a")})
    assert r.status_code == 200
    return r.json()["job_id"]


def test_upload_and_fetch(client, tone_m4a):
    job_id = upload(client, tone_m4a)
    r = client.get(f"/api/jobs/{job_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["job"]["status"] == "done"
    assert len(body["transcript"]["segments"]) == 2
    r = client.get("/api/jobs/latest")
    assert r.json()["job"]["id"] == job_id


def test_upload_rejects_non_audio(client, tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("hello")
    with junk.open("rb") as f:
        r = client.post("/api/jobs", files={"file": ("notes.txt", f, "text/plain")})
    assert r.status_code == 400


def test_missing_job_404(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/latest").status_code == 404


def test_rename_speaker(client, tone_m4a):
    job_id = upload(client, tone_m4a)
    r = client.post(f"/api/jobs/{job_id}/speakers",
                    json={"speaker_id": "SPEAKER_00", "name": "Alice"})
    assert r.json()["speaker_map"]["SPEAKER_00"] == "Alice"
    r = client.post(f"/api/jobs/{job_id}/speakers",
                    json={"speaker_id": "SPEAKER_99", "name": "X"})
    assert r.status_code == 404


def test_snippet_served(client, tone_m4a):
    job_id = upload(client, tone_m4a)
    r = client.get(f"/api/jobs/{job_id}/snippets/SPEAKER_00")
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"


def test_export_markdown(client, tone_m4a):
    job_id = upload(client, tone_m4a)
    client.post(f"/api/jobs/{job_id}/speakers",
                json={"speaker_id": "SPEAKER_00", "name": "Alice"})
    r = client.get(f"/api/jobs/{job_id}/export?fmt=md")
    assert "**Alice** [00:00]: Hello." in r.text
    assert "attachment" in r.headers["content-disposition"]
    assert client.get(f"/api/jobs/{job_id}/export?fmt=bogus").status_code == 400


def test_summarize_endpoint(client, tone_m4a, monkeypatch):
    job_id = upload(client, tone_m4a)
    monkeypatch.setattr(
        server.jobs, "start_summary",
        lambda store, jid: jobs.run_summary(store, jid,
                                            summarize_fn=lambda t: "## Summary\nOK."),
    )
    r = client.post(f"/api/jobs/{job_id}/summarize")
    assert r.status_code == 200
    t = client.get(f"/api/jobs/{job_id}").json()["transcript"]
    assert t["summary_status"] == "done"
    assert client.post("/api/jobs/nope/summarize").status_code == 409
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# transcriber/server.py
"""FastAPI app: upload, job status, rename, summarize, snippets, export."""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import audio, export, jobs

load_dotenv()

app = FastAPI(title="Meeting Transcriber")
store = jobs.JobStore(os.environ.get("TRANSCRIBER_DATA", "data"))


def _payload(job_id: str) -> dict:
    job = store.read_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return {"job": job, "transcript": store.read_transcript(job_id)}


@app.post("/api/jobs")
async def upload(file: UploadFile):
    job_id, dest = jobs.create_job(store, file.filename or "recording")
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    try:
        audio.probe_duration(dest)
    except audio.AudioError as e:
        shutil.rmtree(store.job_dir(job_id))
        raise HTTPException(400, str(e))
    jobs.start_pipeline(store, job_id, os.environ.get("HF_TOKEN", ""))
    return {"job_id": job_id}


@app.get("/api/jobs/latest")
def latest():
    job_id = store.latest_job_id()
    if job_id is None:
        raise HTTPException(404, "no jobs yet")
    return _payload(job_id)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _payload(job_id)


class RenameBody(BaseModel):
    speaker_id: str
    name: str


@app.post("/api/jobs/{job_id}/speakers")
def rename_speaker(job_id: str, body: RenameBody):
    t = store.read_transcript(job_id)
    if t is None:
        raise HTTPException(404, "transcript not ready")
    if body.speaker_id not in t["speaker_map"]:
        raise HTTPException(404, "unknown speaker")
    t["speaker_map"][body.speaker_id] = body.name.strip() or body.speaker_id
    store.write_transcript(job_id, t)
    return {"speaker_map": t["speaker_map"]}


@app.post("/api/jobs/{job_id}/summarize")
def summarize_job(job_id: str):
    t = store.read_transcript(job_id)
    if t is None:
        raise HTTPException(409, "transcript not ready")
    if t["summary_status"] != "running":
        jobs.start_summary(store, job_id)
    return {"summary_status": "running"}


@app.get("/api/jobs/{job_id}/snippets/{speaker_id}")
def snippet(job_id: str, speaker_id: str):
    p = store.job_dir(job_id) / "snippets" / f"{speaker_id}.wav"
    if not p.exists():
        raise HTTPException(404, "no snippet for this speaker")
    return FileResponse(p, media_type="audio/wav")


@app.get("/api/jobs/{job_id}/export")
def export_job(job_id: str, fmt: str = "md"):
    body = _payload(job_id)
    if body["transcript"] is None:
        raise HTTPException(409, "transcript not ready")
    t = body["transcript"]
    title = Path(body["job"]["original_name"]).stem
    if fmt == "md":
        content = export.to_markdown(t["segments"], t["speaker_map"], title,
                                     summary=t["summary"])
        media = "text/markdown"
    elif fmt == "txt":
        content = export.to_text(t["segments"], t["speaker_map"], title,
                                 summary=t["summary"])
        media = "text/plain"
    else:
        raise HTTPException(400, "fmt must be md or txt")
    return Response(content, media_type=media, headers={
        "Content-Disposition": f'attachment; filename="{title}-transcript.{fmt}"',
    })


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True),
          name="static")
```

```python
# transcriber/__main__.py
import uvicorn

uvicorn.run("transcriber.server:app", host="127.0.0.1", port=8484)
```

Create `transcriber/static/` with an empty `.gitkeep` so StaticFiles mounts (replaced in Task 10).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: 7 passed. Then run the whole suite: `uv run pytest` — all pass.

- [ ] **Step 5: Commit**

```bash
git add transcriber/server.py transcriber/__main__.py transcriber/static tests/test_server.py
git commit -m "feat: FastAPI server with upload, rename, summarize, export routes"
```

---

### Task 10: Frontend (`static/`)

**Files:**
- Create: `transcriber/static/index.html`, `transcriber/static/app.js`, `transcriber/static/style.css`
- Delete: `transcriber/static/.gitkeep`

**Interfaces:**
- Consumes: the HTTP API exactly as defined in Task 9.
- Produces: single-page UI — drop zone, stage progress, speaker rename panel with snippet playback, grouped transcript, summary, export links.

- [ ] **Step 1: Write `index.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meeting Transcriber</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<header>
  <h1>Meeting Transcriber</h1>
  <p class="sub">Fully local transcription with speaker identification</p>
</header>
<main>
  <section id="dropzone" tabindex="0">
    <p><strong>Drop a recording here</strong> or click to choose a file</p>
    <p class="hint">m4a · mp3 · wav · aac · flac · ogg · mp4</p>
    <input type="file" id="file-input" hidden
           accept=".m4a,.mp3,.wav,.aac,.flac,.ogg,.mp4,.mov,audio/*">
  </section>

  <section id="progress" hidden>
    <div class="spinner"></div>
    <p id="stage-label"></p>
  </section>

  <section id="error" class="banner error" hidden></section>

  <section id="result" hidden>
    <p id="meta" class="sub"></p>
    <div id="warning" class="banner warn" hidden></div>

    <h2>Speakers</h2>
    <p class="sub">Play a snippet, then type the person's name.</p>
    <div id="speakers"></div>

    <h2>Transcript</h2>
    <div id="transcript"></div>

    <h2>Summary &amp; action items</h2>
    <button id="summarize-btn">Generate summary</button>
    <p id="summary-status" class="sub" hidden></p>
    <div id="summary" class="prose" hidden></div>

    <div class="exports">
      <a id="export-md" class="button">Export Markdown</a>
      <a id="export-txt" class="button">Export TXT</a>
    </div>
  </section>
</main>
<script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `app.js`**

```javascript
const $ = (id) => document.getElementById(id);
const STAGE_LABELS = {
  uploading: "Uploading…",
  converting: "Converting audio…",
  transcribing: "Transcribing — a long meeting takes a few minutes…",
  diarizing: "Identifying speakers…",
  finishing: "Finalizing…",
};
let jobId = null;
let pollTimer = null;

// --- upload ---
const dz = $("dropzone");
dz.addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", (e) => {
  if (e.target.files[0]) uploadFile(e.target.files[0]);
});
["dragover", "dragenter"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => {
  if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
});

async function uploadFile(file) {
  showError(null);
  $("result").hidden = true;
  $("progress").hidden = false;
  $("stage-label").textContent = "Uploading…";
  const form = new FormData();
  form.append("file", file);
  const r = await fetch("/api/jobs", { method: "POST", body: form });
  if (!r.ok) {
    $("progress").hidden = true;
    showError((await r.json()).detail || "Upload failed");
    return;
  }
  jobId = (await r.json()).job_id;
  startPolling();
}

// --- polling ---
function startPolling() {
  clearInterval(pollTimer);
  pollTimer = setInterval(refresh, 1500);
  refresh();
}

async function refresh() {
  const r = await fetch(jobId ? `/api/jobs/${jobId}` : "/api/jobs/latest");
  if (!r.ok) { clearInterval(pollTimer); return; }
  const body = await r.json();
  jobId = body.job.id;
  render(body);
  const busy = body.job.status === "processing" ||
    (body.transcript && body.transcript.summary_status === "running");
  if (!busy) clearInterval(pollTimer);
}

// --- rendering ---
function render({ job, transcript }) {
  if (job.status === "processing") {
    $("progress").hidden = false;
    $("stage-label").textContent = STAGE_LABELS[job.stage] || job.stage;
    return;
  }
  $("progress").hidden = true;
  if (job.status === "error") { showError(job.error); return; }

  $("result").hidden = false;
  $("meta").textContent =
    `${job.original_name} · ${fmtTime(job.duration)} · ` +
    `${Object.keys(transcript.speaker_map).length} speaker(s)`;
  $("warning").hidden = !job.warning;
  $("warning").textContent = job.warning || "";
  renderSpeakers(transcript);
  renderTranscript(transcript);
  renderSummary(transcript);
  $("export-md").href = `/api/jobs/${jobId}/export?fmt=md`;
  $("export-txt").href = `/api/jobs/${jobId}/export?fmt=txt`;
}

function renderSpeakers(t) {
  const el = $("speakers");
  el.innerHTML = "";
  for (const [sid, name] of Object.entries(t.speaker_map)) {
    const row = document.createElement("div");
    row.className = "speaker-row";
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "none";
    audio.src = `/api/jobs/${jobId}/snippets/${sid}`;
    const input = document.createElement("input");
    input.value = name;
    input.addEventListener("change", async () => {
      const r = await fetch(`/api/jobs/${jobId}/speakers`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ speaker_id: sid, name: input.value }),
      });
      if (r.ok) {
        t.speaker_map = (await r.json()).speaker_map;
        renderTranscript(t);
      }
    });
    row.append(audio, input);
    el.append(row);
  }
}

function renderTranscript(t) {
  const el = $("transcript");
  el.innerHTML = "";
  let block = null;
  for (const s of t.segments) {
    if (!block || block.speaker !== s.speaker) {
      block = { speaker: s.speaker, start: s.start, texts: [] };
      const div = document.createElement("div");
      div.className = "block";
      div.innerHTML = `<span class="who"></span> <span class="when"></span><p></p>`;
      div.querySelector(".who").textContent = t.speaker_map[s.speaker] || s.speaker;
      div.querySelector(".when").textContent = fmtTime(s.start);
      block.p = div.querySelector("p");
      el.append(div);
    }
    block.texts.push(s.text);
    block.p.textContent = block.texts.join(" ");
  }
}

function renderSummary(t) {
  const btn = $("summarize-btn");
  const status = $("summary-status");
  const out = $("summary");
  btn.disabled = t.summary_status === "running";
  status.hidden = true;
  if (t.summary_status === "running") {
    status.hidden = false;
    status.textContent = "Generating summary locally — this can take a minute…";
  } else if (t.summary_status === "error") {
    status.hidden = false;
    status.textContent = t.summary_error;
  }
  out.hidden = !t.summary;
  out.textContent = t.summary || "";
}

$("summarize-btn").addEventListener("click", async () => {
  await fetch(`/api/jobs/${jobId}/summarize`, { method: "POST" });
  startPolling();
});

// --- helpers ---
function fmtTime(sec) {
  if (sec == null) return "";
  const s = Math.floor(sec), h = Math.floor(s / 3600),
    m = Math.floor((s % 3600) / 60), r = s % 60;
  const mm = String(m).padStart(2, "0"), ss = String(r).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

function showError(msg) {
  $("error").hidden = !msg;
  $("error").textContent = msg || "";
}

// restore the most recent job on load
refresh();
```

- [ ] **Step 3: Write `style.css`**

```css
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  max-width: 760px; margin: 0 auto; padding: 24px; line-height: 1.5;
}
header h1 { margin-bottom: 0; }
.sub { color: #888; margin-top: 4px; }
#dropzone {
  border: 2px dashed #999; border-radius: 12px; padding: 40px;
  text-align: center; cursor: pointer; margin: 24px 0;
}
#dropzone.drag { border-color: #4a90d9; background: rgba(74, 144, 217, .08); }
.hint { color: #888; font-size: .9em; }
#progress { display: flex; align-items: center; gap: 12px; margin: 24px 0; }
.spinner {
  width: 20px; height: 20px; border: 3px solid #ccc; border-top-color: #4a90d9;
  border-radius: 50%; animation: spin 1s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.banner { padding: 12px 16px; border-radius: 8px; margin: 16px 0; }
.banner.error { background: rgba(220, 60, 60, .12); color: #c33; }
.banner.warn { background: rgba(230, 170, 40, .15); }
.speaker-row { display: flex; gap: 12px; align-items: center; margin: 8px 0; }
.speaker-row audio { height: 36px; }
.speaker-row input { font-size: 1em; padding: 6px 10px; border-radius: 6px;
  border: 1px solid #bbb; }
.block { margin: 14px 0; }
.block .who { font-weight: 600; }
.block .when { color: #888; font-size: .85em; margin-left: 6px; }
.block p { margin: 4px 0 0; }
.prose { white-space: pre-wrap; }
.exports { margin: 24px 0; display: flex; gap: 12px; }
.button, button {
  display: inline-block; padding: 8px 16px; border-radius: 8px;
  border: 1px solid #4a90d9; color: #4a90d9; background: none;
  text-decoration: none; font-size: 1em; cursor: pointer;
}
button:disabled { opacity: .5; cursor: default; }
```

- [ ] **Step 4: Verify in a real browser**

Run: `uv run python -m transcriber` (background), open `http://localhost:8484`.
Check: drop zone renders; uploading a real audio file shows stage progress. (Full end-to-end check happens in Task 12; here confirm the page loads, uploads, and polls without console errors.)

- [ ] **Step 5: Commit**

```bash
git rm transcriber/static/.gitkeep
git add transcriber/static
git commit -m "feat: single-page frontend with upload, rename, summary, export"
```

---

### Task 11: Multi-speaker test fixture with ground truth

**Files:**
- Create: `tests/fixtures/generate_conversation.py`, `tests/fixtures/conversation.m4a` (generated), `tests/fixtures/ground_truth.json` (generated)

**Interfaces:**
- Produces: `conversation.m4a` — ~2–3 min, 3 macOS TTS voices; `ground_truth.json` — `{"turns": [{start, end, speaker}], "voices": {...}}` with speakers `"spk0"/"spk1"/"spk2"`; consumed by Task 12.

- [ ] **Step 1: Write the generation script**

```python
# tests/fixtures/generate_conversation.py
"""Generate a 3-speaker conversation with macOS `say` + ffmpeg, plus ground truth.

Run:  python tests/fixtures/generate_conversation.py
Outputs conversation.m4a and ground_truth.json next to this file.
"""

import json
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
VOICE_PREFS = ["Samantha", "Daniel", "Karen", "Moira", "Fred", "Alex", "Tessa"]
GAP_SECONDS = 0.8

# (speaker index, line). Lines are 1-3 sentences so each turn is a few seconds.
SCRIPT = [
    (0, "Good morning everyone, thanks for joining. Today we need to lock down the quarterly roadmap and decide what happens with the mobile release."),
    (1, "Thanks for setting this up. Before we start, I want to flag that the crash rate on the last beta build went up to two percent, which is double our threshold."),
    (2, "I saw that too. Most of the crashes trace back to the new caching layer, and I think we can have a fix ready by Friday."),
    (0, "Okay, let's make that the first priority. If the fix lands by Friday, can we still hit the release date on the twentieth?"),
    (1, "It will be tight but doable. Quality assurance needs at least three full days with the release candidate, so Friday really is the last possible day."),
    (2, "Agreed. I'll pair with Marcus tomorrow morning to get the caching fix reviewed early, and I'll post daily updates in the release channel."),
    (0, "Perfect. Now, on the quarterly roadmap, the top request from customers is offline support. Sales says we lost two enterprise deals over it last month."),
    (1, "Offline support is a big lift. We estimated six weeks of engineering time, and that assumes we freeze the sync protocol first."),
    (2, "We could split it into phases. Read-only offline mode is maybe two weeks, and full offline editing with conflict resolution comes later."),
    (0, "I like the phased approach. Let's commit to read-only offline mode this quarter and put full editing on the candidate list for next quarter."),
    (1, "Works for me. I'll update the roadmap document and circulate it to the leadership team by Wednesday."),
    (2, "One more thing. The analytics dashboard migration is done, so we can shut down the old reporting service and save about four hundred dollars a month."),
    (0, "Nice win. Please schedule the shutdown for next week and make sure the data export is archived first."),
    (1, "I'll take the archive task. I already have a script that dumps everything to cold storage."),
    (0, "Great. So to recap: caching fix by Friday, release on the twentieth, read-only offline mode this quarter, and the old reporting service shuts down next week."),
    (2, "Sounds right. Thanks everyone."),
    (1, "Thanks all, talk on Friday."),
]


def available_voices() -> set[str]:
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True,
                         check=True).stdout
    return {line.split()[0] for line in out.splitlines() if line.strip()}


def duration_of(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def main() -> None:
    have = available_voices()
    voices = [v for v in VOICE_PREFS if v in have][:3]
    if len(voices) < 3:
        raise SystemExit(f"Need 3 of {VOICE_PREFS}, found only {voices}. "
                         "Install voices in System Settings > Accessibility > Spoken Content.")
    print(f"Voices: {voices}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        silence = tmp / "silence.wav"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
             f"anullsrc=r=22050:cl=mono", "-t", str(GAP_SECONDS), str(silence)],
            check=True)

        concat_entries, turns, t = [], [], 0.0
        for i, (spk, line) in enumerate(SCRIPT):
            aiff = tmp / f"clip{i}.aiff"
            wav = tmp / f"clip{i}.wav"
            subprocess.run(["say", "-v", voices[spk], "-o", str(aiff), line],
                           check=True)
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(aiff),
                            "-ar", "22050", "-ac", "1", str(wav)], check=True)
            dur = duration_of(wav)
            turns.append({"start": round(t, 2), "end": round(t + dur, 2),
                          "speaker": f"spk{spk}"})
            t += dur + GAP_SECONDS
            concat_entries += [f"file '{wav}'", f"file '{silence}'"]

        listfile = tmp / "list.txt"
        listfile.write_text("\n".join(concat_entries) + "\n")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", str(listfile), "-c:a", "aac", str(HERE / "conversation.m4a")],
            check=True)

    (HERE / "ground_truth.json").write_text(json.dumps(
        {"voices": {f"spk{i}": v for i, v in enumerate(voices)}, "turns": turns},
        indent=2))
    total = turns[-1]["end"]
    print(f"Wrote conversation.m4a ({total:.0f}s) and ground_truth.json "
          f"({len(turns)} turns)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the fixture**

Run: `uv run python tests/fixtures/generate_conversation.py`
Expected: prints chosen voices and `Wrote conversation.m4a (...) and ground_truth.json (17 turns)`. Total duration should be roughly 2–3 minutes.

- [ ] **Step 3: Sanity-check the audio**

Run: `ffprobe tests/fixtures/conversation.m4a 2>&1 | grep -E "Duration|Audio"`
Expected: duration ~2–3 min, aac audio. Optionally listen with `afplay` for a few seconds.

- [ ] **Step 4: Commit** (committing the rendered audio is intentional — tests must not depend on the local voice set)

```bash
git add tests/fixtures
git commit -m "test: 3-speaker synthesized conversation fixture with ground truth"
```

---

### Task 12: Slow full-pipeline test + README

**Files:**
- Create: `tests/test_pipeline_slow.py`, `README.md`

**Interfaces:**
- Consumes: fixture from Task 11; `audio`, `transcribe`, `diarize`, `merge` modules. Requires `HF_TOKEN` in the environment/.env; skips (with a clear reason) when absent.

- [ ] **Step 1: Write the slow test**

```python
# tests/test_pipeline_slow.py
"""Full pipeline against the synthesized 3-speaker fixture.

Run: uv run pytest -m slow -v   (needs HF_TOKEN; downloads models on first run)

TTS audio is cleaner than real meetings, so this validates pipeline wiring and
basic diarization quality, not worst-case real-world accuracy.
"""

import json
import os
from pathlib import Path

import pytest

from transcriber import audio, merge
from transcriber.diarize import diarize
from transcriber.transcribe import transcribe

FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.slow


def overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def dominant_detected_speaker(gt_turn, detected_turns):
    """Which detected speaker covers most of this ground-truth turn."""
    totals = {}
    for d in detected_turns:
        o = overlap(gt_turn["start"], gt_turn["end"], d["start"], d["end"])
        if o > 0:
            totals[d["speaker"]] = totals.get(d["speaker"], 0.0) + o
    return max(totals, key=lambda k: totals[k]) if totals else None


@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory):
    from dotenv import load_dotenv
    load_dotenv()
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        pytest.skip("HF_TOKEN not set; needed to download the pyannote model")
    wav = tmp_path_factory.mktemp("slow") / "audio.wav"
    audio.convert_to_wav(FIXTURES / "conversation.m4a", wav)
    segments = transcribe(wav)
    turns = diarize(wav, token)
    return segments, turns


def test_transcription_content(pipeline_output):
    segments, _ = pipeline_output
    text = " ".join(s["text"] for s in segments).lower()
    assert "quarterly roadmap" in text
    assert "caching" in text


def test_speaker_count(pipeline_output):
    _, turns = pipeline_output
    assert len({t["speaker"] for t in turns}) == 3


def test_speaker_attribution_accuracy(pipeline_output):
    _, turns = pipeline_output
    gt = json.loads((FIXTURES / "ground_truth.json").read_text())["turns"]

    # Map each ground-truth speaker to their most common detected speaker.
    votes: dict[str, dict[str, float]] = {}
    for g in gt:
        det = dominant_detected_speaker(g, turns)
        if det:
            dur = g["end"] - g["start"]
            votes.setdefault(g["speaker"], {})
            votes[g["speaker"]][det] = votes[g["speaker"]].get(det, 0.0) + dur
    mapping = {gs: max(d, key=lambda k: d[k]) for gs, d in votes.items()}
    assert len(set(mapping.values())) == 3, f"speakers conflated: {mapping}"

    correct = total = 0.0
    for g in gt:
        dur = g["end"] - g["start"]
        total += dur
        if dominant_detected_speaker(g, turns) == mapping[g["speaker"]]:
            correct += dur
    accuracy = correct / total
    assert accuracy >= 0.9, f"attribution accuracy {accuracy:.1%} below 90%"


def test_merge_produces_three_speaker_transcript(pipeline_output):
    segments, turns = pipeline_output
    labeled = merge.assign_speakers(segments, turns)
    assert len(merge.default_speaker_map(labeled)) == 3
```

- [ ] **Step 2: Verify default run still skips it**

Run: `uv run pytest`
Expected: all unit tests pass; slow tests deselected.

- [ ] **Step 3: Run the slow test for real** (requires `HF_TOKEN` in `.env`; first run downloads ~2 GB of models)

Run: `uv run pytest -m slow -v`
Expected: 4 passed (or a clear skip message if HF_TOKEN is missing — resolve the token before calling this done).

- [ ] **Step 4: Write README.md**

Cover, concretely: what the app is (fully local transcription + diarization + summaries); prerequisites (`brew install ffmpeg uv ollama`, `ollama pull qwen2.5:7b-instruct`, HuggingFace token steps copied from `.env.example`); how to run (`uv run python -m transcriber`, open `http://localhost:8484`); how to use (drop file → wait → rename speakers via snippets → summary → export); testing (`uv run pytest`, `uv run pytest -m slow`); regenerating the fixture (`uv run python tests/fixtures/generate_conversation.py`); data location (`./data/<job-id>/`).

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline_slow.py README.md
git commit -m "test: full-pipeline diarization validation; add README"
```

---

### Task 13: End-to-end verification in the browser

**Files:** none (verification only; fix anything found and commit fixes individually).

- [ ] **Step 1: Start the server** — `uv run python -m transcriber` (background).
- [ ] **Step 2: In a real browser**, upload `tests/fixtures/conversation.m4a`. Watch stages advance: converting → transcribing → diarizing → finishing.
- [ ] **Step 3: Verify results**: 3 speakers in the rename panel, snippets play distinct voices, renaming a speaker updates the transcript immediately, transcript text matches the script content.
- [ ] **Step 4: Summary**: with Ollama running and the model pulled, click Generate summary; verify `## Summary` and `## Action Items` appear and mention the caching fix / release date.
- [ ] **Step 5: Exports**: download both formats; confirm speaker names, timestamps, and summary appear.
- [ ] **Step 6:** Fix any issues found (each with its own commit), re-verify, then run `uv run pytest` one final time.
