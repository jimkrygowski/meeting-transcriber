from transcriber.export import format_timestamp, group_by_speaker, to_markdown, to_text

SEGMENTS = [
    {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "Hello all."},
    {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_00", "text": "Let's start."},
    {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_01", "text": "Sounds good."},
]
NAMES = {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}


def test_format_timestamp():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(65.4) == "01:05"
    assert format_timestamp(3723) == "1:02:03"


def test_group_by_speaker_merges_consecutive():
    blocks = group_by_speaker(SEGMENTS)
    assert len(blocks) == 2
    assert blocks[0]["text"] == "Hello all. Let's start."
    assert blocks[0]["end"] == 4.0
    assert blocks[1]["speaker"] == "SPEAKER_01"


def test_to_markdown_has_title_names_timestamps():
    md = to_markdown(SEGMENTS, NAMES, "Standup")
    assert md.startswith("# Standup\n")
    assert "**Alice** [00:00]: Hello all. Let's start." in md
    assert "**Bob** [00:04]: Sounds good." in md
    assert "## Transcript" in md


def test_to_markdown_includes_summary_when_present():
    md = to_markdown(SEGMENTS, NAMES, "Standup", summary="## Summary\nShort.")
    assert "## Summary\nShort." in md
    assert md.index("## Summary") < md.index("## Transcript")


def test_to_text_plain():
    txt = to_text(SEGMENTS, NAMES, "Standup")
    assert "Alice [00:00]: Hello all. Let's start." in txt
    assert "**" not in txt


def test_unmapped_speaker_falls_back_to_id():
    md = to_markdown(SEGMENTS, {}, "T")
    assert "**SPEAKER_00**" in md
