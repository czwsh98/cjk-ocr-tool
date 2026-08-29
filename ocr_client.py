from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image


VISION_PROMPT_OCR = """You are a Chinese and Japanese document OCR assistant. Extract the text from this page exactly as written, preserving Japanese, Traditional Chinese, Simplified Chinese, or mixed text.

Rules:
1. Output plain text only — no explanations, markdown, or labels.
2. Preserve headings, lists, paragraph boundaries, and reading order. Use one blank line between paragraphs.
3. Ignore page numbers and repeated running headers or footers when they are clearly decorative.
4. Do not output ruby, furigana, or bopomofo annotations; keep only the base characters.
5. Preserve the original script faithfully. Do not convert, transliterate, summarize, or translate."""

TRANSLATE_PROMPT_EN = """You are a professional translator specialising in CJK languages. Translate the following text into natural, fluent English.

Rules:
1. Output the translation only — no commentary, labels, or markdown.
2. Preserve paragraph breaks (blank lines) from the original.
3. Translate meaning faithfully; do not summarise or omit anything.

Text to translate:
"""

TRANSLATE_PROMPT_ZH = """你是一位专业的中日翻译。请将以下文字翻译成简体中文。

规则：
1. 只输出译文，不加任何说明、标签或 Markdown。
2. 保留原文的段落分隔（空行）。
3. 忠实翻译原意，不省略任何内容。

待翻译文字：
"""


_SENTENCE_END = set("。！？.!?」』）】…")
_CJK_EDGE_RE = re.compile(r"[\u3000-\u30ff\u3400-\u9fff\uf900-\ufaff\uff00-\uffef]$")
_CJK_START_RE = re.compile(r"^[\u3000-\u30ff\u3400-\u9fff\uf900-\ufaff\uff00-\uffef]")
_LIST_ITEM_RE = re.compile(
    r"^\s*(?:[-*+・•]|\d+[.、)]|[（(]\d+[）)]|[一二三四五六七八九十百千万]+[、.])\s*"
)
_MARKDOWN_HEADING_RE = re.compile(r"^\s*#{1,6}\s+")
_NAMED_HEADING_RE = re.compile(
    r"^\s*(?:第\s*[0-9０-９一二三四五六七八九十百千万]+\s*[章節部篇]|"
    r"序章|導論|緒論|前言|目次|摘要|参考文献|參考文獻|"
    r"Introduction|Conclusion|References|Appendix)(?:\b|$)"
)


@dataclass(slots=True)
class ModelResult:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    # Subset of input_tokens served from the provider's prompt cache. Both
    # providers bill these far below the cache-miss rate (DeepSeek $0.007 vs
    # $0.22 per 1M), so cost estimates must account for them separately.
    cached_input_tokens: int = 0


def postprocess_translation_ready_text(text: str) -> str:
    """
    Remove soft hyphens; unwrap Latin hyphenation at EOL; join hard line breaks
    that are not followed by sentence-final punctuation (Japanese-friendly).
    """
    text = text.replace("\u00ad", "")
    lines = [ln.rstrip() for ln in text.splitlines()]
    merged: list[str] = []
    buf = ""

    def flush_buf() -> None:
        nonlocal buf
        if buf:
            merged.append(buf)
            buf = ""

    for line in lines:
        if not line.strip():
            flush_buf()
            merged.append("")
            continue
        s = line.strip()
        if not buf:
            buf = s
            continue
        # Structural lines must stay separate. In particular, joining a heading
        # or two list items creates invalid Markdown and changes DOCX layout.
        if _is_list_item(buf) or _is_list_item(s) or _is_heading(buf) or _is_heading(s):
            flush_buf()
            buf = s
            continue
        if buf.endswith("-") and not buf.endswith("--"):
            buf = buf[:-1] + s
            continue
        if buf[-1] not in _SENTENCE_END:
            separator = "" if _CJK_EDGE_RE.search(buf) or _CJK_START_RE.search(s) else " "
            buf += separator + s
        else:
            merged.append(buf)
            buf = s
    flush_buf()

    out_lines: list[str] = []
    blank_run = 0
    for ln in merged:
        if ln == "":
            blank_run += 1
            if blank_run <= 2:
                out_lines.append("")
        else:
            blank_run = 0
            out_lines.append(ln)
    text_out = "\n".join(out_lines)
    text_out = re.sub(r"\n{3,}", "\n\n", text_out)
    return text_out.strip() + ("\n" if text_out.strip() else "")


def _is_list_item(line: str) -> bool:
    return bool(_LIST_ITEM_RE.match(line))


def _is_heading(line: str) -> bool:
    """Recognize common headings without treating ordinary CJK soft wraps as headings."""
    if _MARKDOWN_HEADING_RE.match(line) or _NAMED_HEADING_RE.match(line):
        return True
    # Short standalone Latin headings are common in mixed-language documents.
    if len(line) <= 60 and not line.endswith(tuple(_SENTENCE_END)) and not _CJK_EDGE_RE.search(line):
        return bool(re.fullmatch(r"[A-Z][A-Za-z0-9 &'/:,-]{1,59}", line))
    # Short all-kanji titles (e.g. 研究背景) have no punctuation or Latin
    # heading marker, but are still structurally distinct from prose.
    if len(line) <= 16 and re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff\uF900-\uFAFF]{2,16}", line):
        return True
    return False


_RETRYABLE_CODES = {429, 503}
_RETRY_DELAYS = [5, 15, 30]  # seconds between attempts


def _gemini_text(contents: list, *, model: str) -> ModelResult:
    """Call Gemini with retries and retain usage metadata for cost reporting."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) for Gemini vision OCR.")
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=120_000))
    last_exc: Exception | None = None
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            resp = client.models.generate_content(model=model, contents=contents)
            if not resp.candidates:
                raise RuntimeError("Gemini returned no candidates.")
            candidate = resp.candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            reason_name = str(getattr(finish_reason, "name", finish_reason or "")).upper()
            reason_name = reason_name.rsplit(".", 1)[-1]
            try:
                response_text = (resp.text or "").strip()
            except (AttributeError, ValueError):
                response_text = ""
            if reason_name not in {"", "STOP", "FINISH_REASON_UNSPECIFIED", "UNSPECIFIED"}:
                raise RuntimeError(
                    f"Gemini did not finish normally ({reason_name}); no output was exported."
                )
            if not response_text:
                raise RuntimeError("Gemini returned no usable text.")
            usage = getattr(resp, "usage_metadata", None)
            return ModelResult(
                text=response_text,
                provider="gemini",
                model=model,
                input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
                output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
                cached_input_tokens=int(getattr(usage, "cached_content_token_count", 0) or 0),
            )
        except (genai_errors.ServerError, genai_errors.ClientError, httpx.NetworkError, httpx.TimeoutException) as e:
            code = getattr(e, "status_code", None) or getattr(e, "code", None)
            if code in _RETRYABLE_CODES and attempt < len(_RETRY_DELAYS):
                last_exc = e
                continue
            if isinstance(e, (httpx.NetworkError, httpx.TimeoutException)) and attempt < len(_RETRY_DELAYS):
                last_exc = e
                continue
            raise
    raise last_exc  # type: ignore[misc]


# Gemini 2.5 is deliberately not a default here. Benchmarked on scanned CJK
# pages, gemini-2.5-flash-lite substitutes characters (女→站, 购→敛), emits
# halfwidth commas in CJK runs, and simplifies Japanese kanji (書→书) against
# rule 5 of VISION_PROMPT_OCR; on translation it left a chapter heading in
# untranslated Japanese. gemini-3.1-flash-lite is both cheaper and more
# accurate than gemini-2.5-flash, so the old quality tier has no niche left.
_GEMINI_FALLBACKS = {
    ("ocr", "economy"): "gemini-3.1-flash-lite",
    ("ocr", "quality"): "gemini-3.5-flash-lite",
    ("translation", "economy"): "gemini-3.1-flash-lite",
    ("translation", "quality"): "gemini-3.7-flash",
}


def _gemini_model(kind: str, quality: str) -> str:
    if kind == "ocr":
        key = "GEMINI_OCR_QUALITY_MODEL" if quality == "quality" else "GEMINI_OCR_MODEL"
    else:
        key = "GEMINI_TRANSLATION_QUALITY_MODEL" if quality == "quality" else "GEMINI_TRANSLATION_MODEL"
    tier = "quality" if quality == "quality" else "economy"
    return os.environ.get(key) or _GEMINI_FALLBACKS[(kind, tier)]


def ocr_image(image_path: Path, *, quality: str = "economy") -> ModelResult:
    model = _gemini_model("ocr", quality)
    with Image.open(image_path.as_posix()) as img:
        return _gemini_text([VISION_PROMPT_OCR, img], model=model)


def _deepseek_text(prompt: str) -> ModelResult:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Set DEEPSEEK_API_KEY to use DeepSeek translation.")
    model = os.environ.get("DEEPSEEK_TRANSLATION_MODEL", "deepseek-v4-flash")
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "disabled"},
            "max_tokens": 16384,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last_exc: Exception | None = None
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
            usage = data.get("usage") or {}
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("DeepSeek returned no choices.")
            choice = choices[0]
            finish_reason = str(choice.get("finish_reason") or "").lower()
            if finish_reason not in {"", "stop"}:
                raise RuntimeError(
                    f"DeepSeek did not finish normally ({finish_reason}); no output was exported."
                )
            message = choice.get("message") or {}
            response_text = (message.get("content") or "").strip()
            if not response_text:
                raise RuntimeError("DeepSeek returned no usable text.")
            return ModelResult(
                text=response_text,
                provider="deepseek",
                model=model,
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                cached_input_tokens=int(usage.get("prompt_cache_hit_tokens") or 0),
            )
        except urllib.error.HTTPError as exc:
            if exc.code in _RETRYABLE_CODES and attempt < len(_RETRY_DELAYS):
                last_exc = exc
                continue
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"DeepSeek API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            if attempt < len(_RETRY_DELAYS):
                last_exc = exc
                continue
            raise RuntimeError(f"DeepSeek API connection failed: {exc.reason}") from exc
    raise last_exc or RuntimeError("DeepSeek API request failed.")


def default_translation_provider() -> str:
    """
    DeepSeek when configured, else Gemini.

    On a Japanese->Chinese benchmark deepseek-v4-flash was both the cheapest
    option tested ($0.000098 vs $0.000207 for gemini-3.1-flash-lite) and the
    most faithful, so it is preferred whenever a key is present.
    """
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    return "gemini"


def translate_text(
    text: str,
    target: str,
    *,
    provider: str | None = None,
    quality: str = "economy",
) -> ModelResult:
    """Translate Japanese text to English or Simplified Chinese."""
    provider = provider or default_translation_provider()
    if not text.strip():
        return ModelResult("", provider, "")
    if target not in {"en", "zh"}:
        raise ValueError("Translation target must be 'en' or 'zh'.")
    prompt = TRANSLATE_PROMPT_EN if target == "en" else TRANSLATE_PROMPT_ZH
    if provider == "deepseek":
        return _deepseek_text(prompt + text)
    if provider != "gemini":
        raise ValueError(f"Unsupported translation provider: {provider}.")
    return _gemini_text([prompt + text], model=_gemini_model("translation", quality))


# Backward-compatible wrappers for callers outside this app.
def ocr_image_gemini(image_path: Path, *, model: str | None = None) -> str:
    if model:
        with Image.open(image_path.as_posix()) as img:
            return _gemini_text([VISION_PROMPT_OCR, img], model=model).text
    return ocr_image(image_path).text


def translate_text_gemini(text: str, target: str, *, model: str | None = None) -> str:
    if model:
        prompt = TRANSLATE_PROMPT_EN if target == "en" else TRANSLATE_PROMPT_ZH
        return _gemini_text([prompt + text], model=model).text
    return translate_text(text, target).text
