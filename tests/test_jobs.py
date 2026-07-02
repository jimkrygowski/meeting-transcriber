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
