"""Meeting summary + action items via a local Ollama model."""

import httpx

OLLAMA_URL = "http://localhost:11434"
SUMMARY_MODEL = "qwen2.5:7b-instruct"

PROMPT = """You are given a meeting transcript with speaker names and timestamps.
Produce exactly two Markdown sections:

## Summary
3-6 sentences covering the key points, decisions, and disagreements.

## Action Items
A bullet list of concrete action items, each with an owner when one is
identifiable from the transcript. If there are none, write "None identified."

Transcript:

{transcript}"""


class SummaryError(Exception):
    """Summary unavailable; message says exactly how to fix it."""


def check_ollama() -> None:
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        r.raise_for_status()
    except Exception as e:
        raise SummaryError(
            "Ollama isn't running. Install it with `brew install ollama`, then "
            "start it with `ollama serve` (or open the Ollama app)."
        ) from e
    names = [m.get("name", "") for m in r.json().get("models", [])]
    if not any(n.startswith(SUMMARY_MODEL) for n in names):
        raise SummaryError(
            f"The summary model isn't downloaded yet. Run: `ollama pull {SUMMARY_MODEL}`"
        )


def summarize(transcript_text: str) -> str:
    check_ollama()
    r = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": SUMMARY_MODEL,
            "messages": [{"role": "user",
                          "content": PROMPT.format(transcript=transcript_text)}],
            "stream": False,
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()
