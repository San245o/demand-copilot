from __future__ import annotations

from app.agents.chat_agent import AVAILABLE_MODELS, _extract_text, _friendly_error


def test_extract_text_handles_plain_string():
    assert _extract_text("hello") == "hello"


def test_extract_text_handles_gemini_list_parts():
    # Gemini streams content as a list of parts; we must join the text parts.
    content = [{"type": "text", "text": "fore"}, {"type": "text", "text": "cast"}]
    assert _extract_text(content) == "forecast"


def test_extract_text_skips_non_text_parts():
    content = [{"type": "text", "text": "ok"}, {"functionCall": {"name": "x"}}]
    assert _extract_text(content) == "ok"


def test_extract_text_empty_for_garbage():
    assert _extract_text(None) == ""
    assert _extract_text(123) == ""


def test_friendly_error_detects_rate_limit():
    msg = _friendly_error(Exception("429 RESOURCE_EXHAUSTED quota exceeded"))
    assert "rate limit" in msg.lower()


def test_default_model_is_flash_lite():
    assert AVAILABLE_MODELS[0] == "gemini-3.1-flash-lite"
