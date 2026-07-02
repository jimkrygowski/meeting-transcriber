"""Render a speaker-labeled transcript as Markdown or plain text."""


def format_timestamp(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def group_by_speaker(segments: list[dict]) -> list[dict]:
    blocks: list[dict] = []
    for s in segments:
        if blocks and blocks[-1]["speaker"] == s["speaker"]:
            blocks[-1]["text"] += " " + s["text"]
            blocks[-1]["end"] = s["end"]
        else:
            blocks.append(dict(s))
    return blocks


def _lines(segments, speaker_map, title, summary, name_fmt):
    lines = [f"# {title}", ""]
    if summary:
        lines += [summary.strip(), ""]
    lines += ["## Transcript", ""]
    for b in group_by_speaker(segments):
        name = speaker_map.get(b["speaker"], b["speaker"])
        lines.append(f"{name_fmt.format(name)} [{format_timestamp(b['start'])}]: {b['text']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def to_markdown(segments, speaker_map, title, summary=None) -> str:
    return _lines(segments, speaker_map, title, summary, "**{}**")


def to_text(segments, speaker_map, title, summary=None) -> str:
    md = _lines(segments, speaker_map, title, summary, "{}")
    return md.replace("## ", "").replace("# ", "")  # longest heading marker first
