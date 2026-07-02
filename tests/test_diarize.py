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
