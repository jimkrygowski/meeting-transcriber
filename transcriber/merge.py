"""Pure functions combining transcript segments with diarization turns."""

FALLBACK_SPEAKER = "SPEAKER_00"


def assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Label each transcript segment with the speaker whose turns overlap it most.

    Segments with no overlapping turn get the speaker of the nearest turn.
    With no turns at all (diarization unavailable), everything is one speaker.
    """
    if not turns:
        return [{**s, "speaker": FALLBACK_SPEAKER} for s in segments]
    labeled = []
    for s in segments:
        overlaps: dict[str, float] = {}
        for t in turns:
            o = min(s["end"], t["end"]) - max(s["start"], t["start"])
            if o > 0:
                overlaps[t["speaker"]] = overlaps.get(t["speaker"], 0.0) + o
        if overlaps:
            speaker = max(overlaps, key=lambda k: overlaps[k])
        else:
            mid = (s["start"] + s["end"]) / 2
            nearest = min(
                turns,
                key=lambda t: min(abs(mid - t["start"]), abs(mid - t["end"])),
            )
            speaker = nearest["speaker"]
        labeled.append({**s, "speaker": speaker})
    return labeled


def default_speaker_map(labeled_segments: list[dict]) -> dict[str, str]:
    """Human-friendly default names, numbered by order of first appearance."""
    mapping: dict[str, str] = {}
    for s in labeled_segments:
        if s["speaker"] not in mapping:
            mapping[s["speaker"]] = f"Speaker {len(mapping) + 1}"
    return mapping


def best_snippets(turns: list[dict], max_len: float) -> dict[str, tuple[float, float]]:
    """Per speaker, the (start, duration) of their longest turn, capped at max_len."""
    best: dict[str, tuple[float, float]] = {}
    for t in turns:
        dur = t["end"] - t["start"]
        if t["speaker"] not in best or dur > best[t["speaker"]][1]:
            best[t["speaker"]] = (float(t["start"]), float(dur))
    return {k: (start, min(dur, max_len)) for k, (start, dur) in best.items()}
