import sys
import types

from transcriber.transcribe import WHISPER_MODEL, normalize_segments, transcribe


def fake_mlx_whisper(captured):
    mod = types.ModuleType("mlx_whisper")

    def fake_transcribe(path, **kwargs):
        captured.update(kwargs, path=path)
        return {"segments": [{"start": 0.0, "end": 1.0, "text": " ok "}]}

    mod.transcribe = fake_transcribe
    return mod


def test_transcribe_disables_previous_text_conditioning(monkeypatch):
    captured = {}
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mlx_whisper(captured))
    out = transcribe("audio.wav")
    assert captured["condition_on_previous_text"] is False
    assert captured["path_or_hf_repo"] == WHISPER_MODEL
    assert "initial_prompt" not in captured
    assert out == [{"start": 0.0, "end": 1.0, "text": "ok"}]


def test_transcribe_passes_context_as_initial_prompt(monkeypatch):
    captured = {}
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mlx_whisper(captured))
    transcribe("audio.wav", context="  Attendees: Priya, Marek. ")
    assert captured["initial_prompt"] == "Attendees: Priya, Marek."


def test_whisper_model_is_full_large_v3():
    assert WHISPER_MODEL == "mlx-community/whisper-large-v3-mlx"


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
