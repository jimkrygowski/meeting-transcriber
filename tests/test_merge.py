from transcriber.merge import assign_speakers, best_snippets, default_speaker_map


def seg(start, end, text="hi"):
    return {"start": start, "end": end, "text": text}


def turn(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def test_segment_inside_turn_gets_that_speaker():
    out = assign_speakers([seg(1, 2)], [turn(0, 5, "SPEAKER_00")])
    assert out[0]["speaker"] == "SPEAKER_00"
    assert out[0]["text"] == "hi"


def test_segment_spanning_two_turns_gets_larger_overlap():
    turns = [turn(0, 3, "SPEAKER_00"), turn(3, 10, "SPEAKER_01")]
    out = assign_speakers([seg(2, 6)], turns)
    assert out[0]["speaker"] == "SPEAKER_01"  # 3s overlap beats 1s


def test_overlap_accumulates_across_multiple_turns_of_same_speaker():
    turns = [
        turn(0, 2, "SPEAKER_00"),
        turn(2, 3, "SPEAKER_01"),
        turn(3, 5, "SPEAKER_00"),
    ]
    out = assign_speakers([seg(0, 5)], turns)
    assert out[0]["speaker"] == "SPEAKER_00"  # 4s total beats 1s


def test_segment_in_gap_gets_nearest_turn_speaker():
    turns = [turn(0, 2, "SPEAKER_00"), turn(10, 12, "SPEAKER_01")]
    out = assign_speakers([seg(3, 4)], turns)
    assert out[0]["speaker"] == "SPEAKER_00"


def test_no_turns_falls_back_to_single_speaker():
    out = assign_speakers([seg(0, 1), seg(1, 2)], [])
    assert all(s["speaker"] == "SPEAKER_00" for s in out)


def test_default_speaker_map_ordered_by_first_appearance():
    labeled = [
        {"start": 0, "end": 1, "speaker": "SPEAKER_01", "text": "a"},
        {"start": 1, "end": 2, "speaker": "SPEAKER_00", "text": "b"},
        {"start": 2, "end": 3, "speaker": "SPEAKER_01", "text": "c"},
    ]
    assert default_speaker_map(labeled) == {
        "SPEAKER_01": "Speaker 1",
        "SPEAKER_00": "Speaker 2",
    }


def test_best_snippets_picks_longest_turn_capped():
    turns = [
        turn(0, 2, "SPEAKER_00"),
        turn(5, 30, "SPEAKER_00"),
        turn(40, 43, "SPEAKER_01"),
    ]
    out = best_snippets(turns, max_len=10.0)
    assert out == {"SPEAKER_00": (5.0, 10.0), "SPEAKER_01": (40.0, 3.0)}
