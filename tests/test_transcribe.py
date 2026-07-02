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
