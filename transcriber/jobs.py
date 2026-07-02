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
