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
