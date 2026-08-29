from __future__ import annotations

import os
import atexit
import re
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from exporters import PageResult, write_docx, write_markdown
from ocr_client import (
    default_translation_provider,
    ocr_image,
    postprocess_translation_ready_text,
    translate_text,
)
from pdf_processor import extract_pdf_page_text, pdf_page_count, pdf_page_to_image
from range_utils import parse_page_range


load_dotenv()

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_HOURS", "6")) * 3600
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "2"))
TRANSLATION_CHUNK_CHARS = max(1000, int(os.environ.get("TRANSLATION_CHUNK_CHARS", "12000")))
WORK_PREFIX = "cjkocr_"

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="cjk-job")
_cleanup_stop = threading.Event()


def _job_update(job_id: str, **kwargs: object) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            job.update(kwargs)


def _cleanup_expired_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    expired: list[dict] = []
    with _jobs_lock:
        for job_id, job in list(_jobs.items()):
            if job["created_at"] < cutoff and job["status"] not in {"queued", "running"}:
                expired.append(_jobs.pop(job_id))
    for job in expired:
        shutil.rmtree(job["work_root"], ignore_errors=True)


def _cleanup_orphan_workdirs() -> None:
    """Remove stale app-owned temp directories not represented by a live job."""
    cutoff = time.time() - JOB_TTL_SECONDS
    with _jobs_lock:
        active_roots = {Path(job["work_root"]).resolve() for job in _jobs.values()}
    temp_root = Path(tempfile.gettempdir())
    for path in temp_root.glob(f"{WORK_PREFIX}*"):
        try:
            if not path.is_dir() or path.resolve() in active_roots or path.stat().st_mtime >= cutoff:
                continue
            shutil.rmtree(path, ignore_errors=True)
        except OSError:
            continue


def _cleanup_all_workdirs() -> None:
    with _jobs_lock:
        roots = [Path(job["work_root"]) for job in _jobs.values()]
    for root in roots:
        shutil.rmtree(root, ignore_errors=True)


def _cleanup_loop() -> None:
    while not _cleanup_stop.wait(300):
        _cleanup_expired_jobs()
        _cleanup_orphan_workdirs()


_cleanup_orphan_workdirs()
threading.Thread(target=_cleanup_loop, name="cjk-cleanup", daemon=True).start()
atexit.register(_cleanup_stop.set)
atexit.register(_cleanup_all_workdirs)


def _record_usage(job_id: str, result) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job["input_tokens"] += result.input_tokens
        job["output_tokens"] += result.output_tokens
        estimate = _estimate_cost(result)
        if estimate is None:
            job["estimated_cost_usd"] = None
            job["cost_estimate_available"] = False
        elif job.get("cost_estimate_available", True):
            job["estimated_cost_usd"] += estimate
        if result.model and result.model not in job["models"]:
            job["models"].append(result.model)


# USD per 1M tokens as (cache-miss input, output, cache-hit input).
# Gemini 3.6/3.7 input, output and cache rates are promotional through
# 2026-12-31; from 2027-01-01 the cached rate doubles to 0.15.
_GEMINI_PRICES = {
    "gemini-2.5-flash-lite": (0.10, 0.40, 0.01),
    "gemini-2.5-flash": (0.30, 2.50, 0.03),
    "gemini-3.1-flash-lite": (0.25, 1.50, 0.025),
    "gemini-3.5-flash-lite": (0.30, 2.50, 0.03),
    "gemini-3.5-flash": (1.50, 9.00, 0.15),
    "gemini-3.6-flash": (0.75, 3.75, 0.075),
    "gemini-3.7-flash": (0.75, 3.75, 0.075),
}

# DeepSeek publishes one rate card per model with a 50% off-peak discount, so
# only the peak column is stored and halved outside peak hours.
_DEEPSEEK_PEAK_PRICES = {
    "deepseek-v4-flash": (0.44, 1.32, 0.014),
    "deepseek-v4-flash-vision-exp": (0.44, 1.32, 0.014),
    "deepseek-v4-pro": (1.32, 3.96, 0.044),
}


def _deepseek_is_peak(now: datetime) -> bool:
    """Peak is 01:00-04:00 and 06:00-10:00 UTC, Monday through Friday."""
    return now.weekday() < 5 and (1 <= now.hour < 4 or 6 <= now.hour < 10)


def _model_prices(result) -> tuple[float, float, float] | None:
    model_id = (result.model or "").rsplit("/", 1)[-1]
    if model_id in _GEMINI_PRICES:
        return _GEMINI_PRICES[model_id]
    peak_price = _DEEPSEEK_PEAK_PRICES.get(model_id)
    if peak_price is not None:
        if _deepseek_is_peak(datetime.now(timezone.utc)):
            return peak_price
        return tuple(value / 2 for value in peak_price)  # type: ignore[return-value]

    provider_prefix = "GEMINI" if result.provider == "gemini" else "DEEPSEEK"
    input_override = os.environ.get(f"{provider_prefix}_INPUT_PRICE_USD_PER_MILLION")
    output_override = os.environ.get(f"{provider_prefix}_OUTPUT_PRICE_USD_PER_MILLION")
    if input_override is None or output_override is None:
        return None
    try:
        # Without a published cache rate for an unknown model, bill cached
        # tokens at the full input rate so the estimate never reads low.
        return (float(input_override), float(output_override), float(input_override))
    except ValueError:
        return None


def _estimate_cost(result) -> float | None:
    """Estimate standard token cost; DeepSeek follows the call's UTC rate window."""
    price = _model_prices(result)
    if price is None:
        return None
    input_price, output_price, cached_price = price
    # Providers report cache hits as a subset of the total prompt tokens, and
    # bill them at a small fraction of the cache-miss rate.
    cached_tokens = max(0, min(result.cached_input_tokens, result.input_tokens))
    billed_input_tokens = result.input_tokens - cached_tokens
    return (
        billed_input_tokens * input_price
        + cached_tokens * cached_price
        + result.output_tokens * output_price
    ) / 1_000_000


def _translation_chunks(text: str, max_chars: int = TRANSLATION_CHUNK_CHARS) -> list[str]:
    """Split long text at paragraph/line boundaries before sending it to a model."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current = ""

    def add_piece(piece: str) -> None:
        nonlocal current
        if not piece:
            return
        if current and len(current) + 2 + len(piece) > max_chars:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{piece}" if current else piece

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            add_piece(paragraph)
            continue
        # A single huge paragraph (common in TXT exports) is split by lines,
        # then by character count as a last resort.
        for line in paragraph.splitlines() or [paragraph]:
            if len(line) <= max_chars:
                add_piece(line)
                continue
            for start in range(0, len(line), max_chars):
                add_piece(line[start : start + max_chars])
    if current:
        chunks.append(current)
    return chunks or [text]


def _extract_page(
    job_id: str,
    file_path: Path,
    work_root: Path,
    page_number: int,
    ocr_setting: str,
    quality: str,
) -> str:
    # No character-count heuristic decides this anymore. A PDF's embedded
    # text layer is only trusted when the caller explicitly says "never
    # OCR" — plenty of scanned CJK PDFs (Anna's Archive and similar library
    # scans especially) carry a legacy OCR layer that reads a traditional
    # vertical right-to-left column layout one glyph at a time, producing a
    # text layer that is long enough to pass any length threshold while
    # being unusable: reading order scrambled, most characters wrong. A
    # threshold on character count can't tell that text apart from a good
    # embedded layer, so it isn't a safe way to decide whether to OCR.
    if ocr_setting == "never":
        return extract_pdf_page_text(file_path, page_number)

    page_dir = work_root / "page_image"
    image_path = pdf_page_to_image(file_path, page_dir, page_number)
    try:
        result = ocr_image(image_path, quality=quality)
        _record_usage(job_id, result)
        return result.text
    finally:
        shutil.rmtree(page_dir, ignore_errors=True)


def _run_pipeline(job_id: str) -> None:
    with _jobs_lock:
        job = dict(_jobs[job_id])
    file_path = Path(job["file_path"])
    work_root = Path(job["work_root"])

    try:
        _job_update(job_id, status="running", stage="preparing", error=None)
        if job["file_type"] == "pdf":
            total_document_pages = pdf_page_count(file_path)
            selected_pages = parse_page_range(job["page_range"], total_document_pages)
        else:
            total_document_pages = 1
            selected_pages = [1]
        if not selected_pages:
            raise ValueError("No pages were selected.")

        _job_update(
            job_id,
            document_pages=total_document_pages,
            selected_pages=selected_pages,
            total_pages=len(selected_pages),
            done_pages=0,
            stage="extracting",
        )

        pages: list[PageResult] = []
        text_content = ""
        if job["file_type"] != "pdf":
            text_content = file_path.read_text(encoding="utf-8", errors="replace")

        for index, page_number in enumerate(selected_pages, start=1):
            _job_update(job_id, stage=f"extracting_page_{page_number}", current_page=page_number)
            if job["file_type"] == "pdf":
                source = _extract_page(
                    job_id,
                    file_path,
                    work_root,
                    page_number,
                    job["ocr_setting"],
                    job["quality"],
                )
            else:
                source = text_content
            source = postprocess_translation_ready_text(source)
            pages.append(PageResult(page=page_number, source=source))
            _job_update(
                job_id,
                done_pages=index,
                preview_text="\n\n".join(p.source for p in pages)[-6000:],
            )

        document_translation = ""
        if job["mode"] == "translate":
            # OCR/extraction must finish for every selected page before the
            # translator sees any text. Translating page by page loses
            # document-level context and produces an alternating original /
            # translation export. Keep page boundaries in the source export,
            # but send one complete document to the translator (split only
            # when the provider input limit requires it).
            complete_source = "\n\n".join(page.source.strip() for page in pages if page.source.strip())
            translation_chunks = _translation_chunks(complete_source)
            _job_update(
                job_id,
                stage="translating_document",
                done_pages=0,
                translation_chunks=len(translation_chunks),
            )
            translations: list[str] = []
            for index, chunk in enumerate(translation_chunks, start=1):
                _job_update(
                    job_id,
                    stage=f"translating_part_{index}_of_{len(translation_chunks)}",
                    current_page=pages[-1].page,
                )
                result = translate_text(
                    chunk,
                    job["target"],
                    provider=job["translation_provider"],
                    quality=job["quality"],
                )
                _record_usage(job_id, result)
                translated = postprocess_translation_ready_text(result.text).strip()
                if translated:
                    translations.append(translated)
                _job_update(
                    job_id,
                    preview_text="\n\n".join(translations)[-6000:],
                )
            document_translation = "\n\n".join(translations)

        target_label = "Simplified Chinese" if job["target"] == "zh" else "English"
        include_translation = job["mode"] == "translate"
        include_source = not include_translation or job["translation_output"] == "bilingual"
        stem = Path(job["filename"]).stem or "document"
        outputs: dict[str, str] = {}

        if "md" in job["output_formats"]:
            md_path = work_root / f"{stem}.md"
            write_markdown(
                md_path,
                filename=job["filename"],
                pages=pages,
                include_source=include_source,
                include_translation=include_translation,
                target_label=target_label,
                document_translation=document_translation,
            )
            outputs["md"] = str(md_path)
        if "docx" in job["output_formats"]:
            docx_path = work_root / f"{stem}.docx"
            write_docx(
                docx_path,
                filename=job["filename"],
                pages=pages,
                include_source=include_source,
                include_translation=include_translation,
                target_label=target_label,
                document_translation=document_translation,
            )
            outputs["docx"] = str(docx_path)

        _job_update(job_id, status="done", stage="done", outputs=outputs, done_pages=len(pages))
    except Exception as exc:
        traceback.print_exc()
        _job_update(job_id, status="error", stage="error", error=str(exc))


app = FastAPI(title="Yomu — CJK OCR & Translation")


INDEX_HTML = r"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>読む — CJK OCR & Translation</title>
  <style>
    :root{--paper:#f6f0e7;--card:#fffdf8;--ink:#28231f;--muted:#756d64;--red:#9d2c21;--line:#d9cfc1;--soft:#eee4d8;--green:#45634d;--shadow:0 18px 55px rgba(69,48,29,.09)}
    *{box-sizing:border-box} body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Noto Sans CJK TC","PingFang TC",system-ui,sans-serif;min-height:100vh}
    header{height:72px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 40px;background:rgba(255,253,248,.65);backdrop-filter:blur(10px)}
    .brand{display:flex;align-items:baseline;gap:13px}.brand strong{font:700 25px Georgia,"Noto Serif CJK JP",serif;color:var(--red)}.brand span{font:italic 14px Georgia,serif;color:var(--muted)}
    .kicker{font:600 11px ui-monospace,monospace;letter-spacing:.18em;color:var(--muted)}
    main{max-width:940px;margin:54px auto;padding:0 24px 70px}.intro{display:flex;justify-content:space-between;align-items:end;margin-bottom:26px;gap:24px}
    h1{font:500 clamp(32px,5vw,52px)/1.08 Georgia,"Noto Serif CJK TC",serif;margin:0;max-width:640px}.intro p{margin:0 0 5px;color:var(--muted);max-width:260px;font-size:14px;line-height:1.6}
    .panel{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow);overflow:hidden}.section{padding:25px 28px;border-bottom:1px solid var(--soft)}.section:last-child{border-bottom:0}
    .label{display:block;font-size:12px;font-weight:750;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:11px}
    .modes{display:grid;grid-template-columns:1fr 1fr;gap:12px}.mode{border:1px solid var(--line);border-radius:12px;background:#fff;padding:18px;text-align:left;cursor:pointer;color:var(--ink)}.mode.active{border-color:var(--red);box-shadow:inset 0 0 0 1px var(--red);background:#fffaf5}.mode b{display:block;font:600 20px Georgia,"Noto Serif CJK TC",serif;margin-bottom:5px}.mode small{color:var(--muted);font-size:13px}
    .drop{border:1.5px dashed #b9aa99;border-radius:12px;min-height:142px;display:flex;align-items:center;justify-content:center;text-align:center;cursor:pointer;transition:.2s;background:#fffcf7}.drop:hover,.drop.drag{border-color:var(--red);background:#fff7ef}.drop-icon{font:500 34px Georgia,serif}.drop b{display:block;font:600 16px Georgia,serif;margin:4px 0}.drop span{font-size:13px;color:var(--muted)}input[type=file]{display:none}
    .file-pill{display:none;align-items:center;justify-content:space-between;border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-top:12px;background:#fff}.file-pill.show{display:flex}.file-meta{font-size:13px}.file-meta b{display:block;font-size:14px}.link-btn{border:0;background:none;color:var(--red);cursor:pointer;font-weight:700}
    #settings{display:none}#settings.show{display:block}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.field label{display:block;font-size:13px;font-weight:650;margin-bottom:7px}.field small{display:block;color:var(--muted);margin-top:6px;line-height:1.35}
    select,input[type=text]{width:100%;height:44px;border:1px solid var(--line);border-radius:9px;background:#fff;padding:0 12px;font:inherit;color:var(--ink);outline:none}select:focus,input[type=text]:focus{border-color:var(--red);box-shadow:0 0 0 3px rgba(157,44,33,.08)}
    .choices{display:flex;flex-wrap:wrap;gap:9px}.choice{position:relative}.choice input{position:absolute;opacity:0}.choice span{display:block;border:1px solid var(--line);border-radius:999px;padding:9px 14px;background:#fff;font-size:13px;cursor:pointer}.choice input:checked+span{border-color:var(--red);color:var(--red);background:#fff7ef;font-weight:700}
    .translation-only{display:none}.translation-only.show{display:block}.pdf-only.hidden{display:none}.action-row{display:flex;align-items:center;justify-content:space-between;gap:20px}.primary{border:0;border-radius:10px;background:var(--red);color:white;padding:13px 24px;font-weight:750;font-size:14px;cursor:pointer;min-width:165px}.primary:disabled{opacity:.45;cursor:not-allowed}.privacy{color:var(--muted);font-size:12px;line-height:1.5}
    .progress{display:none}.progress.show{display:block}.progress-head{display:flex;justify-content:space-between;gap:20px;margin-bottom:10px}.progress-head b{font-family:Georgia,serif}.progress-head span{font:12px ui-monospace,monospace;color:var(--muted)}.bar{height:8px;background:var(--soft);border-radius:10px;overflow:hidden}.bar i{display:block;height:100%;width:0;background:var(--red);transition:width .35s}.stats{display:flex;gap:22px;flex-wrap:wrap;margin-top:13px;font-size:12px;color:var(--muted)}
    .preview{white-space:pre-wrap;max-height:260px;overflow:auto;border:1px solid var(--soft);background:#fbf8f3;padding:15px;border-radius:9px;margin-top:16px;font:13px/1.65 Georgia,"Noto Serif CJK JP",serif}.downloads{display:flex;gap:10px;margin-top:16px}.downloads a{display:none;text-decoration:none;border:1px solid var(--red);color:var(--red);padding:9px 14px;border-radius:8px;font-size:13px;font-weight:700}.downloads a.show{display:inline-block}.error{display:none;margin-top:12px;color:#8c2119;background:#fff0ed;border:1px solid #ebc1ba;padding:12px;border-radius:8px;font-size:13px}.error.show{display:block}
    @media(max-width:700px){header{padding:0 20px}.kicker{display:none}main{margin-top:32px}.intro{display:block}.intro p{margin-top:12px}.modes,.grid{grid-template-columns:1fr}.section{padding:21px 19px}.action-row{align-items:stretch;flex-direction:column}.primary{width:100%}}
  </style>
</head>
<body>
<header><div class="brand"><strong>読む</strong><span>yomu — read</span></div><div class="kicker">CHINESE · JAPANESE · OCR · TRANSLATION</div></header>
<main>
  <div class="intro"><h1>Turn documents into<br>clean, usable text.</h1><p>中文与日文的文字提取和翻译。选择范围，导出 Markdown 或 Word。</p></div>
  <div class="panel">
    <section class="section">
      <span class="label">1 · 选择任务</span>
      <div class="modes">
        <button class="mode active" data-mode="extract"><b>提取文字</b><small>OCR 或直接提取 → Markdown / Word</small></button>
        <button class="mode" data-mode="translate"><b>日文翻译</b><small>日文 → 简体中文或英文</small></button>
      </div>
    </section>
    <section class="section">
      <span class="label">2 · 上传文件</span>
      <label class="drop" id="drop"><input id="file" type="file" accept=".pdf,.txt,.md,application/pdf,text/plain"><div><div class="drop-icon">文</div><b>拖入 PDF、TXT 或 Markdown</b><span>最大 50 MB · 文件处理后自动过期</span></div></label>
      <div class="file-pill" id="filePill"><div class="file-meta"><b id="fileName"></b><span id="fileInfo"></span></div><button class="link-btn" id="remove">移除</button></div>
    </section>
    <div id="settings">
      <section class="section">
        <span class="label">3 · 处理范围</span>
        <div class="grid">
          <div class="field pdf-only" id="rangeField"><label for="pageRange">页码范围</label><input id="pageRange" type="text" value="all" placeholder="all 或 1-5, 8, 12-18"><small>只处理指定页面，减少时间与 API 费用。</small></div>
          <div class="field pdf-only" id="ocrField"><label for="ocrSetting">PDF 文字识别</label><select id="ocrSetting"><option value="always" selected>使用 OCR（推荐）</option><option value="never">从不：仅用 PDF 中已有文字</option></select><small>许多扫描版 PDF（尤其是老书）自带的文字层是竖排版损坏后的旧 OCR 结果，看起来字数够但读音乱、字都错，无法用字数判断好坏，因此默认总是用 Vision OCR。仅当你确定文件是原生文字（非扫描）时才选“仅用已有文字”以节省费用。</small></div>
          <div class="field"><label>质量</label><div class="choices"><label class="choice"><input type="radio" name="quality" value="economy" checked><span>经济</span></label><label class="choice"><input type="radio" name="quality" value="quality"><span>高质量</span></label></div><small>OCR：经济使用 Gemini 3.1 Flash-Lite，高质量使用 3.5 Flash-Lite。翻译：已配置 DeepSeek 时始终使用 DeepSeek V4 Flash（质量档不影响），否则经济用 Gemini 3.1 Flash-Lite、高质量用 3.7 Flash。</small></div>
          <div class="field translation-only" id="targetField"><label>翻译为</label><div class="choices"><label class="choice"><input type="radio" name="target" value="zh" checked><span>简体中文</span></label><label class="choice"><input type="radio" name="target" value="en"><span>English</span></label></div></div>
          <div class="field translation-only" id="providerField"><label for="provider">翻译服务</label><select id="provider"><option value="gemini">Gemini</option></select><small id="providerHelp">按服务器已配置的 API 显示。</small></div>
          <div class="field translation-only" id="translationOutputField"><label>翻译输出</label><div class="choices"><label class="choice"><input type="radio" name="translationOutput" value="translation" checked><span>仅译文</span></label><label class="choice"><input type="radio" name="translationOutput" value="bilingual"><span>原文＋译文</span></label></div></div>
          <div class="field"><label>导出格式</label><div class="choices"><label class="choice"><input type="checkbox" name="format" value="md" checked><span>Markdown</span></label><label class="choice"><input type="checkbox" name="format" value="docx" checked><span>Word</span></label></div></div>
        </div>
      </section>
      <section class="section"><div class="action-row"><div class="privacy">应用仅监听服务器本机地址，并通过 Cloudflare Access 保护。<br>生成文件将在任务过期后清理。</div><button class="primary" id="start">开始提取</button></div><div class="error" id="error"></div></section>
    </div>
    <section class="section progress" id="progress"><div class="progress-head"><b id="status">准备中</b><span id="counter">0 / 0</span></div><div class="bar"><i id="bar"></i></div><div class="stats"><span id="pagesStat"></span><span id="tokenStat"></span><span id="modelStat"></span></div><div class="preview" id="preview"></div><div class="downloads"><a id="mdDownload">下载 Markdown</a><a id="docxDownload">下载 Word</a></div></section>
  </div>
</main>
<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
let mode='extract', selectedFile=null, pollTimer=null;
const drop=$('#drop'), fileInput=$('#file'), settings=$('#settings'), progress=$('#progress'), errorBox=$('#error');

function setMode(next){mode=next;$$('.mode').forEach(x=>x.classList.toggle('active',x.dataset.mode===mode));$$('.translation-only').forEach(x=>x.classList.toggle('show',mode==='translate'));$('#start').textContent=mode==='translate'?'开始翻译':'开始提取';}
$$('.mode').forEach(x=>x.onclick=()=>setMode(x.dataset.mode));

function useFile(file){const ext=file.name.toLowerCase().split('.').pop();if(!['pdf','txt','md'].includes(ext)){showError('请选择 PDF、TXT 或 Markdown 文件。');return}selectedFile=file;$('#fileName').textContent=file.name;$('#fileInfo').textContent=`${(file.size/1024/1024).toFixed(2)} MB · ${ext.toUpperCase()}`;$('#filePill').classList.add('show');settings.classList.add('show');$$('.pdf-only').forEach(x=>x.classList.toggle('hidden',ext!=='pdf'));errorBox.classList.remove('show');}
fileInput.onchange=()=>fileInput.files[0]&&useFile(fileInput.files[0]);drop.ondragover=e=>{e.preventDefault();drop.classList.add('drag')};drop.ondragleave=()=>drop.classList.remove('drag');drop.ondrop=e=>{e.preventDefault();drop.classList.remove('drag');e.dataTransfer.files[0]&&useFile(e.dataTransfer.files[0])};$('#remove').onclick=()=>{clearInterval(pollTimer);pollTimer=null;selectedFile=null;fileInput.value='';$('#filePill').classList.remove('show');settings.classList.remove('show');progress.classList.remove('show');$('#start').disabled=false};

function checked(name){return document.querySelector(`input[name="${name}"]:checked`)?.value}
function showError(message){errorBox.textContent=message;errorBox.classList.add('show')}
function stageLabel(stage){if(stage==='queued')return'排队等待中';if(stage==='preparing')return'正在检查文件';if(stage==='extracting')return'开始提取文字';if(stage.startsWith('extracting_page_'))return`正在提取第 ${stage.split('_').pop()} 页`;if(stage==='translating')return'开始翻译';if(stage==='translating_document')return'准备翻译完整原文';if(stage.startsWith('translating_part_'))return`正在翻译完整原文（${stage.slice('translating_part_'.length).replace('_of_',' / ')}）`;if(stage.startsWith('translating_page_'))return`正在翻译第 ${stage.split('_').pop()} 页`;if(stage==='done')return'处理完成';return stage||'等待中'}

async function loadConfig(){try{const r=await fetch('/api/config');const c=await r.json();if(c.deepseek){const o=document.createElement('option');o.value='deepseek';o.textContent='DeepSeek V4 Flash';$('#provider').prepend(o);$('#providerHelp').textContent='DeepSeek 更便宜且译文更忠实，默认使用。'}else{$('#providerHelp').textContent='未配置 DeepSeek key，目前使用 Gemini。'}if(c.default_translation_provider)$('#provider').value=c.default_translation_provider}catch{}}
loadConfig();

$('#start').onclick=async()=>{if(!selectedFile)return;const formats=$$('input[name="format"]:checked').map(x=>x.value);if(!formats.length){showError('至少选择一种导出格式。');return}errorBox.classList.remove('show');$('#start').disabled=true;progress.classList.add('show');$('#preview').textContent='';$$('.downloads a').forEach(x=>x.classList.remove('show'));const fd=new FormData();fd.append('file',selectedFile);fd.append('mode',mode);fd.append('page_range',$('#pageRange').value||'all');fd.append('ocr_setting',$('#ocrSetting').value);fd.append('target',checked('target')||'zh');fd.append('translation_provider',$('#provider').value);fd.append('quality',checked('quality'));fd.append('translation_output',checked('translationOutput')||'translation');fd.append('output_formats',formats.join(','));try{const r=await fetch('/api/jobs',{method:'POST',body:fd});const data=await r.json();if(!r.ok)throw new Error(data.detail||'无法创建任务');poll(data.job_id)}catch(e){showError(e.message);$('#start').disabled=false}};

function poll(jobId){
  clearInterval(pollTimer);
  pollTimer=setInterval(async()=>{
    try{
      const r=await fetch('/api/jobs/'+jobId), j=await r.json();
      if(!r.ok)throw new Error(j.detail||'任务读取失败');
      $('#status').textContent=stageLabel(j.stage);
      $('#counter').textContent=`${j.done_pages} / ${j.total_pages||'—'}`;
      $('#bar').style.width=(j.total_pages?Math.round(j.done_pages/j.total_pages*100):3)+'%';
      $('#pagesStat').textContent=j.document_pages?`文档 ${j.document_pages} 页 · 已选 ${j.selected_pages.length} 页`:'';
      $('#tokenStat').textContent=(j.input_tokens||j.output_tokens)?`Tokens ${j.input_tokens} in / ${j.output_tokens} out · ${j.cost_estimate_available&&j.estimated_cost_usd!=null?`est. $${j.estimated_cost_usd.toFixed(4)}`:'cost estimate unavailable'}`:'';
      $('#modelStat').textContent=(j.models||[]).join(' · ');
      $('#preview').textContent=j.preview_text||'处理中…';
      if(j.status==='error'){clearInterval(pollTimer);showError(j.error||'处理失败');$('#start').disabled=false}
      if(j.status==='done'){
        clearInterval(pollTimer);$('#bar').style.width='100%';$('#start').disabled=false;
        for(const fmt of Object.keys(j.outputs)){const a=$('#'+fmt+'Download');a.href=`/api/jobs/${jobId}/download/${fmt}`;a.classList.add('show')}
      }
    }catch(e){clearInterval(pollTimer);showError(e.message);$('#start').disabled=false}
  },700)
}
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/api/config")
def config() -> dict:
    return {
        "gemini": bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
        "deepseek": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "default_translation_provider": default_translation_provider(),
        "max_upload_mb": MAX_UPLOAD_BYTES // 1024 // 1024,
    }


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    mode: str = Form("extract"),
    page_range: str = Form("all"),
    ocr_setting: str = Form("always"),
    target: str = Form("zh"),
    translation_provider: str = Form(""),
    quality: str = Form("economy"),
    translation_output: str = Form("translation"),
    output_formats: str = Form("md,docx"),
) -> dict:
    _cleanup_expired_jobs()
    filename = Path(file.filename or "document").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(400, "Only PDF, TXT, and Markdown files are supported.")
    if mode not in {"extract", "translate"}:
        raise HTTPException(400, "Invalid task mode.")
    # "auto" is accepted as a synonym for "always": it used to skip OCR when
    # a PDF's embedded text layer looked long enough, a heuristic dropped
    # because that layer is often a scrambled legacy OCR pass rather than
    # real content (see _extract_page).
    if ocr_setting == "auto":
        ocr_setting = "always"
    if ocr_setting not in {"always", "never"}:
        raise HTTPException(400, "Invalid OCR setting.")
    if target not in {"zh", "en"} or quality not in {"economy", "quality"}:
        raise HTTPException(400, "Invalid translation target or quality setting.")
    translation_provider = translation_provider or default_translation_provider()
    if translation_provider not in {"gemini", "deepseek"}:
        raise HTTPException(400, "Invalid translation provider.")
    if translation_provider == "deepseek" and not os.environ.get("DEEPSEEK_API_KEY"):
        raise HTTPException(400, "DeepSeek is not configured on this server.")
    if translation_output not in {"translation", "bilingual"}:
        raise HTTPException(400, "Invalid translation output setting.")
    formats = {item.strip() for item in output_formats.split(",") if item.strip()}
    if not formats or not formats.issubset({"md", "docx"}):
        raise HTTPException(400, "Choose Markdown, Word, or both.")

    job_id = uuid.uuid4().hex
    work_root = Path(tempfile.mkdtemp(prefix=f"cjkocr_{job_id}_"))
    file_path = work_root / f"input{suffix}"
    size = 0
    try:
        with file_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024} MB limit.")
                output.write(chunk)
        if suffix == ".pdf":
            with file_path.open("rb") as uploaded_pdf:
                if uploaded_pdf.read(5) != b"%PDF-":
                    raise HTTPException(400, "The uploaded file is not a valid PDF.")
    except Exception:
        shutil.rmtree(work_root, ignore_errors=True)
        raise

    job = {
        "id": job_id,
        "created_at": time.time(),
        "work_root": str(work_root),
        "file_path": str(file_path),
        "filename": filename,
        "file_type": suffix.lstrip("."),
        "mode": mode,
        "page_range": page_range,
        "ocr_setting": ocr_setting if suffix == ".pdf" else "never",
        "target": target,
        "translation_provider": translation_provider,
        "quality": quality,
        "translation_output": translation_output,
        "output_formats": formats,
        "status": "queued",
        "stage": "queued",
        "error": None,
        "document_pages": 0,
        "selected_pages": [],
        "total_pages": 0,
        "done_pages": 0,
        "current_page": 0,
        "preview_text": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0,
        "cost_estimate_available": True,
        "models": [],
        "outputs": {},
    }
    with _jobs_lock:
        _jobs[job_id] = job
    _executor.submit(_run_pipeline, job_id)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found.")
        response = {
            key: job[key]
            for key in (
                "status", "stage", "error", "document_pages", "selected_pages",
                "total_pages", "done_pages", "current_page", "preview_text",
                "input_tokens", "output_tokens", "estimated_cost_usd", "cost_estimate_available", "models",
            )
        }
        response["outputs"] = {fmt: True for fmt in job["outputs"]}
        return response


@app.get("/api/jobs/{job_id}/download/{fmt}")
def download(job_id: str, fmt: str):
    if fmt not in {"md", "docx"}:
        raise HTTPException(400, "Invalid output format.")
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found.")
        output = job["outputs"].get(fmt)
        filename = job["filename"]
    if job["status"] != "done" or not output:
        raise HTTPException(400, "Output is not ready.")
    path = Path(output)
    if not path.is_file():
        raise HTTPException(404, "Output file has expired.")
    media_type = "text/markdown; charset=utf-8" if fmt == "md" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    stem = re.sub(r'[\x00-\x1f\x7f"\\]', "_", Path(filename).stem).strip() or "document"
    return FileResponse(path, media_type=media_type, filename=f"{stem}.{fmt}")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.environ.get("APP_HOST", "127.0.0.1"),
        port=int(os.environ.get("APP_PORT", "8765")),
        reload=False,
    )


if __name__ == "__main__":
    main()
