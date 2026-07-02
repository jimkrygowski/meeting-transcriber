# Meeting Transcriber — Design

**Date:** 2026-07-02
**Status:** Approved

## Goal

A MacWhisper-like app that transcribes recorded meeting audio fully on-device
(MacBook Pro, M3 Pro, 36 GB RAM). It accepts common audio formats, separates
speakers so the user can assign each a name, and produces a readable transcript
plus an AI-generated summary with action items. Nothing leaves the machine.

## Decisions made during brainstorming

- **App type:** Local web app — FastAPI backend + browser UI at localhost.
- **Speaker recognition:** Diarization with manual rename (Speaker 1/2/3 →
  user types names after listening to snippets). No persistent voice profiles.
- **Outputs:** Readable transcript export (Markdown/TXT) and AI summary /
  action items. No meeting library, no subtitle formats.
- **Summary LLM:** Fully local via Ollama. No cloud calls anywhere.
- **Engine (Approach A):** mlx-whisper for transcription (Apple Silicon GPU via
  MLX) + pyannote.audio for diarization, merged by timestamp overlap. Chosen
  over WhisperX (CPU-only on Mac, ~3–5× slower) and whisper.cpp (weak
  diarization, clunky Python integration).

## Architecture

Python 3.12 project managed with `uv`. A FastAPI server on `localhost:8484`
serves a single-page frontend (plain HTML/JS/CSS, no build step) and runs the
processing pipeline as a background job with per-stage progress.

Pipeline stages:

1. **Convert** — ffmpeg decodes the input (m4a, mp3, wav, aac, flac, ogg, and
   audio tracks of mp4/mov) to 16 kHz mono WAV.
2. **Transcribe** — `mlx-whisper` with the `large-v3-turbo` model produces
   timestamped text segments on the GPU.
3. **Diarize** — `pyannote/speaker-diarization-3.1` (PyTorch on MPS) produces
   speaker turns labeled SPEAKER_00, SPEAKER_01, …
4. **Merge** — each transcript segment is assigned the speaker whose turn
   overlaps it most (by duration of overlap).
5. **Summarize (on demand)** — Ollama running Qwen 2.5 7B Instruct generates a
   Markdown summary and action-item list from the speaker-labeled transcript.

### One-time setup requirements

- `brew install ffmpeg`
- `uv` for Python/dependency management
- Ollama installed with `qwen2.5:7b-instruct` pulled
- A free HuggingFace token to download the gated pyannote model. After the
  first download the model is cached locally and no network access is needed
  again. The token is stored in a local `.env` file.

## Components

| Module | Responsibility |
|---|---|
| `audio.py` | Validate input, convert to 16 kHz mono WAV via ffmpeg, extract per-speaker preview snippets (each speaker's clearest/longest turn) |
| `transcribe.py` | mlx-whisper wrapper → list of `{start, end, text}` segments |
| `diarize.py` | pyannote wrapper → list of `{start, end, speaker}` turns |
| `merge.py` | Pure function: segments + turns → speaker-labeled transcript. No I/O. |
| `summarize.py` | Ollama client; prompts for summary + action items, returns Markdown |
| `jobs.py` | Pipeline orchestration, job state machine, per-stage progress, persistence to `./data/<job-id>/` |
| `server.py` | FastAPI routes: upload, job status, transcript fetch, speaker rename, summarize, export, speaker snippet audio |
| `static/` | Frontend: drag-and-drop upload, progress display, transcript view, speaker rename panel with snippet playback, summary section, export buttons |

### Data model

A job directory `./data/<job-id>/` contains:

- the original uploaded file and the converted WAV
- `job.json` — status, stage, progress, error (if any)
- `transcript.json` — segments `[{start, end, speaker_id, text}]`, speaker map
  `{speaker_id: display_name}`, and summary Markdown once generated
- `snippets/<speaker_id>.wav` — preview clip per detected speaker

Speaker rename only edits the speaker map; segments always reference stable
speaker IDs.

## Data flow & UX

1. User drags an audio file onto the page → upload → pipeline runs in the
   background; the UI polls job status and shows live per-stage progress
   (converting → transcribing → identifying speakers).
2. The finished transcript renders with segments labeled Speaker 1/2/3 and
   timestamps. A rename panel lists each detected speaker with a play button
   on that speaker's preview snippet; the user listens, types a name, and it
   applies across the whole transcript immediately.
3. **Generate summary** runs Ollama and appends a summary + action items
   section. **Export** downloads Markdown or TXT with speaker names,
   timestamps, and the summary if generated.
4. Jobs persist to disk; on page load the most recent job is restored, so a
   refresh or server restart doesn't lose a finished transcription.

## Error handling

- Unsupported or corrupt file → clear message before any processing starts
  (ffprobe validation on upload).
- Missing HuggingFace token or model not yet accepted/downloaded → the UI
  shows one-time setup instructions with a link, not a stack trace.
- Ollama not running or model not pulled → the summary button surfaces the
  exact command to fix it; transcription is unaffected.
- A pipeline stage failure records the stage and error in `job.json` and the
  UI shows it. Partial results are kept: if diarization fails, the transcript
  is still shown with a single "Speaker 1".

## Testing

- **Unit tests for `merge.py`** — the most bug-prone logic: overlapping turns,
  gaps between turns, segments spanning two speakers, diarization that starts
  mid-segment.
- **Unit tests** for export formatting and job-state transitions.
- **Audio conversion** tested against small generated fixture files (e.g.,
  sine-wave m4a/mp3 produced by ffmpeg in a fixture).
- **Full pipeline test** with real models behind a `slow` pytest marker, run
  manually, not in the default test run.
- **Frontend** verified by driving the real app in a browser.

## Out of scope (YAGNI)

- Live/microphone recording — file input only.
- Persistent voice profiles / cross-meeting speaker recognition.
- Meeting library with search.
- Subtitle (SRT/VTT) export.
- Cloud LLM integration.
