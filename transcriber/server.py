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
