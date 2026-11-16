"""Tests for models.py with fake Ollama / Gemini (no network, no keys)."""
import types

import pytest

import config
import models


def test_parse_confidence_takes_last_value_and_strips_line():
    text, conf = models.parse_confidence("Answer here.\nCONFIDENCE: 3\n\nCONFIDENCE: 8")
    assert text == "Answer here."
    assert conf == 8


def test_parse_confidence_handles_markdown_and_missing():
    assert models.parse_confidence("Ans\n**CONFIDENCE:** 9")[1] == 9
    assert models.parse_confidence("No line at all") == ("No line at all", 5)
    assert models.parse_confidence("x CONFIDENCE: 42")[1] == 10   # clamped


class FakeOllama:
    def __init__(self):
        self.calls = []

    def __call__(self, url, json, timeout):
        self.calls.append(json["model"])
        return types.SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": "hi\nCONFIDENCE: 7"},
                          "prompt_eval_count": 1000, "eval_count": 2000})


@pytest.fixture
def real_mode(monkeypatch):
    monkeypatch.setattr(config, "DEMO", False)
    monkeypatch.setattr(config, "API_SLEEP", 0)
    monkeypatch.setattr(models.time, "sleep", lambda s: None)


def test_small_model_is_free(real_mode, monkeypatch):
    fake = FakeOllama()
    monkeypatch.setattr(models.requests, "post", fake)
    res, conf = models.ask_small_with_confidence("What is a stack?")
    assert (res.text, conf, res.cost_usd) == ("hi", 7, 0.0)
    assert fake.calls == [config.SMALL_MODEL]


def test_large_model_via_local_ollama_is_priced(real_mode, monkeypatch):
    fake = FakeOllama()
    monkeypatch.setattr(models.requests, "post", fake)
    monkeypatch.setattr(config, "LARGE_PROVIDER", "ollama")
    monkeypatch.setattr(config, "LARGE_MODEL", "qwen2.5:14b")
    res = models.ask_large("Prove it")
    assert fake.calls == ["qwen2.5:14b"]
    expected = (1000 * config.LARGE_PRICE_IN + 2000 * config.LARGE_PRICE_OUT) / 1e6
    assert res.cost_usd == pytest.approx(expected)


def test_gemini_retries_on_rate_limit_and_counts_thinking_tokens(real_mode, monkeypatch):
    pytest.importorskip("google.genai")
    calls = {"n": 0}

    def generate_content(model, contents, config):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        usage = types.SimpleNamespace(prompt_token_count=100, candidates_token_count=50,
                                      thoughts_token_count=150)
        return types.SimpleNamespace(text="YES", usage_metadata=usage)

    client = types.SimpleNamespace(models=types.SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(models, "_client", client)
    monkeypatch.setattr(config, "LARGE_PROVIDER", "gemini")
    res = models.ask_large("q")
    assert calls["n"] == 2
    assert res.tokens_out == 200
    assert models.judge("q", "cand", "ref") is True


def test_demo_mode_needs_no_models(monkeypatch):
    monkeypatch.setattr(config, "DEMO", True)
    res, conf = models.ask_small_with_confidence("Why does int *p; *p=1; crash?")
    assert res.text.startswith("[DEMO-SMALL]") and 1 <= conf <= 10
    assert models.ask_large("x").cost_usd > 0
    assert isinstance(models.judge("q", res.text, "ref"), bool)
