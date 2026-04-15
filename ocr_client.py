from __future__ import annotations

import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from PIL import Image


VISION_PROMPT_OCR = """You are a CJK document OCR assistant. Extract all body text from the image exactly as written, preserving the original language (Japanese, Traditional Chinese, Simplified Chinese, or mixed).

Rules:
1. Output plain text only — no explanations, markdown, or labels.
2. Do not output ruby/furigana/bopomofo annotations; keep only the base characters.
3. Separate clearly distinct paragraphs with a single blank line. Treat other line breaks as soft wraps and output the text naturally so it can be rejoined downstream.
4. Preserve the original script faithfully — do not convert between Traditional and Simplified Chinese, and do not transliterate or translate anything."""

TRANSLATE_PROMPT_EN = """You are a professional translator specialising in CJK languages. Translate the following text into natural, fluent English.

Rules:
1. Output the translation only — no commentary, labels, or markdown.
2. Preserve paragraph breaks (blank lines) from the original.
3. Translate meaning faithfully; do not summarise or omit anything.

Text to translate:
"""

TRANSLATE_PROMPT_ZH = """你是一位專業的中日翻譯。請將以下文字翻譯成繁體中文。

規則：
1. 只輸出譯文，不加任何說明、標籤或 Markdown。
2. 保留原文的段落分隔（空行）。
3. 忠實翻譯原意，不省略任何內容。

待翻譯文字：
"""


_SENTENCE_END = set("。！？.!?」』）】…")


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
        if buf.endswith("-") and not buf.endswith("--"):
            buf = buf[:-1] + s
            continue
        if buf[-1] not in _SENTENCE_END:
            buf += s
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


_RETRYABLE_CODES = {429, 503}
_RETRY_DELAYS = [5, 15, 30]  # seconds between attempts


def _gemini_text(contents: list, *, model: str) -> str:
    """Call Gemini with retry on transient errors. Returns stripped response text."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) for Gemini vision OCR.")
    client = genai.Client(api_key=api_key)
    last_exc: Exception | None = None
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            resp = client.models.generate_content(model=model, contents=contents)
            if not resp.candidates:
                raise RuntimeError("Gemini returned no candidates.")
            return (resp.text or "").strip()
        except (genai_errors.ServerError, genai_errors.ClientError) as e:
            code = getattr(e, "status_code", None) or getattr(e, "code", None)
            if code in _RETRYABLE_CODES and attempt < len(_RETRY_DELAYS):
                last_exc = e
                continue
            raise
    raise last_exc  # type: ignore[misc]


def ocr_image_gemini(image_path: Path, *, model: str | None = None) -> str:
    mname = model or os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash")
    img = Image.open(image_path.as_posix())
    return _gemini_text([VISION_PROMPT_OCR, img], model=mname)


def translate_text_gemini(text: str, target: str, *, model: str | None = None) -> str:
    """Translate *text* to *target* ('en' or 'zh'). Returns translated string."""
    if not text.strip():
        return ""
    mname = model or os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash")
    prompt = TRANSLATE_PROMPT_EN if target == "en" else TRANSLATE_PROMPT_ZH
    return _gemini_text([prompt + text], model=mname)
