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
            transcribe_fn=lambda wav, context="": [
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


def test_upload_stores_context(client, tone_m4a):
    with tone_m4a.open("rb") as f:
        r = client.post("/api/jobs",
                        files={"file": ("standup.m4a", f, "audio/m4a")},
                        data={"context": "Attendees: Priya, Marek."})
    assert r.status_code == 200
    job = client.get(f"/api/jobs/{r.json()['job_id']}").json()["job"]
    assert job["context"] == "Attendees: Priya, Marek."


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
