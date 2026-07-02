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
