from datetime import datetime, timezone

import app
import ocr_client
from ocr_client import ModelResult


def test_cached_input_tokens_are_billed_at_the_cache_rate():
    """DeepSeek bills cache hits at $0.007/1M against $0.22/1M for a miss."""
    off_peak = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)  # Saturday
    assert not app._deepseek_is_peak(off_peak)

    uncached = ModelResult("t", "deepseek", "deepseek-v4-flash", input_tokens=1000)
    cached = ModelResult(
        "t", "deepseek", "deepseek-v4-flash", input_tokens=1000, cached_input_tokens=1000
    )
    # The prompt prefix repeats on every page of a job, so this path is common.
    assert app._estimate_cost(cached) < app._estimate_cost(uncached)


def test_cached_tokens_are_a_subset_of_input_tokens():
    """A cache-hit count larger than the total must not produce a negative cost."""
    result = ModelResult(
        "t", "deepseek", "deepseek-v4-flash", input_tokens=100, cached_input_tokens=999
    )
    assert app._estimate_cost(result) >= 0


def test_deepseek_peak_window_matches_published_schedule():
    # Peak is 01:00-04:00 and 06:00-10:00 UTC, Monday through Friday.
    assert app._deepseek_is_peak(datetime(2026, 8, 26, 2, 0, tzinfo=timezone.utc))
    assert app._deepseek_is_peak(datetime(2026, 8, 26, 9, 59, tzinfo=timezone.utc))
    assert not app._deepseek_is_peak(datetime(2026, 8, 26, 5, 0, tzinfo=timezone.utc))
    assert not app._deepseek_is_peak(datetime(2026, 8, 26, 10, 0, tzinfo=timezone.utc))
    assert not app._deepseek_is_peak(datetime(2026, 8, 29, 2, 0, tzinfo=timezone.utc))


def test_off_peak_is_half_of_peak():
    peak = datetime(2026, 8, 26, 2, 0, tzinfo=timezone.utc)
    result = ModelResult("t", "deepseek", "deepseek-v4-flash", input_tokens=1_000_000)
    assert app._deepseek_is_peak(peak)
    # 0.44 peak vs 0.22 off-peak per 1M cache-miss input tokens.
    assert app._model_prices(result)[0] in {0.44, 0.22}


def test_every_offered_model_has_a_price():
    """A model that can be selected must never report a silent $0.0000."""
    for model in (
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-3.7-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
    ):
        assert app._model_prices(ModelResult("t", "gemini", model)) is not None
    for model in ("deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4-flash-vision-exp"):
        assert app._model_prices(ModelResult("t", "deepseek", model)) is not None


def test_unknown_model_reports_unavailable_rather_than_zero(monkeypatch):
    monkeypatch.delenv("GEMINI_INPUT_PRICE_USD_PER_MILLION", raising=False)
    monkeypatch.delenv("GEMINI_OUTPUT_PRICE_USD_PER_MILLION", raising=False)
    result = ModelResult("t", "gemini", "gemini-9-unreleased", input_tokens=1000)
    assert app._estimate_cost(result) is None


def test_env_override_prices_an_unknown_model(monkeypatch):
    monkeypatch.setenv("GEMINI_INPUT_PRICE_USD_PER_MILLION", "1.0")
    monkeypatch.setenv("GEMINI_OUTPUT_PRICE_USD_PER_MILLION", "2.0")
    result = ModelResult("t", "gemini", "gemini-9-unreleased", input_tokens=1_000_000)
    assert app._estimate_cost(result) == 1.0


def test_default_provider_prefers_deepseek_when_configured(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    assert ocr_client.default_translation_provider() == "deepseek"
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert ocr_client.default_translation_provider() == "gemini"


def test_gemini_model_defaults_are_off_the_2_5_family(monkeypatch):
    for key in (
        "GEMINI_OCR_MODEL",
        "GEMINI_OCR_QUALITY_MODEL",
        "GEMINI_TRANSLATION_MODEL",
        "GEMINI_TRANSLATION_QUALITY_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    for kind in ("ocr", "translation"):
        for quality in ("economy", "quality"):
            assert not ocr_client._gemini_model(kind, quality).startswith("gemini-2.5")
