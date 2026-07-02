import httpx
import pytest

from transcriber import summarize as sz


class FakeResponse:
    def __init__(self, json_data, status=200):
        self._json = json_data
        self.status_code = status

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)


def test_check_ollama_down_raises_with_fix(monkeypatch):
    def fail(*a, **k):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(httpx, "get", fail)
    with pytest.raises(sz.SummaryError, match="ollama serve|Ollama app"):
        sz.check_ollama()


def test_check_ollama_missing_model_raises_pull_command(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse({"models": []}))
    with pytest.raises(sz.SummaryError, match="ollama pull qwen2.5:7b-instruct"):
        sz.check_ollama()


def test_summarize_returns_content(monkeypatch):
    monkeypatch.setattr(
        httpx, "get",
        lambda *a, **k: FakeResponse({"models": [{"name": "qwen2.5:7b-instruct"}]}),
    )
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return FakeResponse({"message": {"content": " ## Summary\nGood meeting.\n"}})

    monkeypatch.setattr(httpx, "post", fake_post)
    out = sz.summarize("Alice [00:00]: Hi.")
    assert out == "## Summary\nGood meeting."
    assert "Alice [00:00]: Hi." in captured["json"]["messages"][0]["content"]
    assert captured["json"]["model"] == "qwen2.5:7b-instruct"
    assert captured["json"]["stream"] is False
