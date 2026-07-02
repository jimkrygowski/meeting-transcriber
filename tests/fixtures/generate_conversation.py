"""Generate a 3-speaker conversation with macOS `say` + ffmpeg, plus ground truth.

Run:  python tests/fixtures/generate_conversation.py
Outputs conversation.m4a and ground_truth.json next to this file.
"""

import json
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
VOICE_PREFS = ["Samantha", "Daniel", "Karen", "Moira", "Fred", "Alex", "Tessa"]
GAP_SECONDS = 0.8

# (speaker index, line). Lines are 1-3 sentences so each turn is a few seconds.
SCRIPT = [
    (0, "Good morning everyone, thanks for joining. Today we need to lock down the quarterly roadmap and decide what happens with the mobile release."),
    (1, "Thanks for setting this up. Before we start, I want to flag that the crash rate on the last beta build went up to two percent, which is double our threshold."),
    (2, "I saw that too. Most of the crashes trace back to the new caching layer, and I think we can have a fix ready by Friday."),
    (0, "Okay, let's make that the first priority. If the fix lands by Friday, can we still hit the release date on the twentieth?"),
    (1, "It will be tight but doable. Quality assurance needs at least three full days with the release candidate, so Friday really is the last possible day."),
    (2, "Agreed. I'll pair with Marcus tomorrow morning to get the caching fix reviewed early, and I'll post daily updates in the release channel."),
    (0, "Perfect. Now, on the quarterly roadmap, the top request from customers is offline support. Sales says we lost two enterprise deals over it last month."),
    (1, "Offline support is a big lift. We estimated six weeks of engineering time, and that assumes we freeze the sync protocol first."),
    (2, "We could split it into phases. Read-only offline mode is maybe two weeks, and full offline editing with conflict resolution comes later."),
    (0, "I like the phased approach. Let's commit to read-only offline mode this quarter and put full editing on the candidate list for next quarter."),
    (1, "Works for me. I'll update the roadmap document and circulate it to the leadership team by Wednesday."),
    (2, "One more thing. The analytics dashboard migration is done, so we can shut down the old reporting service and save about four hundred dollars a month."),
    (0, "Nice win. Please schedule the shutdown for next week and make sure the data export is archived first."),
    (1, "I'll take the archive task. I already have a script that dumps everything to cold storage."),
    (0, "Great. So to recap: caching fix by Friday, release on the twentieth, read-only offline mode this quarter, and the old reporting service shuts down next week."),
    (2, "Sounds right. Thanks everyone."),
    (1, "Thanks all, talk on Friday."),
]


def available_voices() -> set[str]:
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True,
                         check=True).stdout
    return {line.split()[0] for line in out.splitlines() if line.strip()}


def duration_of(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def main() -> None:
    have = available_voices()
    voices = [v for v in VOICE_PREFS if v in have][:3]
    if len(voices) < 3:
        raise SystemExit(f"Need 3 of {VOICE_PREFS}, found only {voices}. "
                         "Install voices in System Settings > Accessibility > Spoken Content.")
    print(f"Voices: {voices}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        silence = tmp / "silence.wav"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
             "anullsrc=r=22050:cl=mono", "-t", str(GAP_SECONDS), str(silence)],
            check=True)

        concat_entries, turns, t = [], [], 0.0
        for i, (spk, line) in enumerate(SCRIPT):
            aiff = tmp / f"clip{i}.aiff"
            wav = tmp / f"clip{i}.wav"
            subprocess.run(["say", "-v", voices[spk], "-o", str(aiff), line],
                           check=True)
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(aiff),
                            "-ar", "22050", "-ac", "1", str(wav)], check=True)
            dur = duration_of(wav)
            turns.append({"start": round(t, 2), "end": round(t + dur, 2),
                          "speaker": f"spk{spk}"})
            t += dur + GAP_SECONDS
            concat_entries += [f"file '{wav}'", f"file '{silence}'"]

        listfile = tmp / "list.txt"
        listfile.write_text("\n".join(concat_entries) + "\n")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", str(listfile), "-c:a", "aac", str(HERE / "conversation.m4a")],
            check=True)

    (HERE / "ground_truth.json").write_text(json.dumps(
        {"voices": {f"spk{i}": v for i, v in enumerate(voices)}, "turns": turns},
        indent=2))
    total = turns[-1]["end"]
    print(f"Wrote conversation.m4a ({total:.0f}s) and ground_truth.json "
          f"({len(turns)} turns)")


if __name__ == "__main__":
    main()
