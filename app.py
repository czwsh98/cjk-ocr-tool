from __future__ import annotations

import shutil
import tempfile
import threading
import traceback
import uuid
from pathlib import Path

import fitz  # pymupdf — PDF text extraction (no-OCR path)
from dotenv import load_dotenv

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from ocr_client import ocr_image_gemini, postprocess_translation_ready_text, translate_text_gemini
from pdf_processor import pdf_to_png_pages

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _job_update(job_id: str, **kwargs: object) -> None:
    with _jobs_lock:
        j = _jobs.get(job_id)
        if j:
            j.update(kwargs)


def _extract_pdf_text(pdf_path: Path) -> list[str]:
    """Extract text from each PDF page using PyMuPDF (no vision OCR)."""
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return pages


def _run_pipeline(
    job_id: str,
    file_path: Path,
    work_root: Path,
    file_type: str,   # "pdf" | "txt"
    ocr_enabled: bool,
    translate: str,   # "none" | "en" | "zh"
) -> None:
    try:
        _job_update(job_id, status="running", stage="starting", error=None)
        page_texts: list[str] = []

        # ── Text extraction ────────────────────────────────────────────────
        if file_type == "pdf" and ocr_enabled:
            _job_update(job_id, stage="pdf_to_images")
            pages_dir = work_root / "pages"
            page_paths = pdf_to_png_pages(file_path, pages_dir, dpi=300)
            total = len(page_paths)
            _job_update(job_id, total_pages=total, done_pages=0, stage="ocr")
            for i, p in enumerate(page_paths, start=1):
                text = ocr_image_gemini(p)
                page_texts.append(text or "")
                preview = "\n".join(("\n\n".join(page_texts)).splitlines()[:30])
                _job_update(job_id, done_pages=i, preview_text=preview, stage=f"ocr_page_{i}")

        elif file_type == "pdf" and not ocr_enabled:
            _job_update(job_id, stage="extracting_text")
            page_texts = _extract_pdf_text(file_path)
            _job_update(job_id, total_pages=len(page_texts), done_pages=len(page_texts))

        else:  # txt
            _job_update(job_id, stage="reading_file")
            content = file_path.read_text(encoding="utf-8", errors="replace")
            page_texts = [content]
            _job_update(job_id, total_pages=1, done_pages=1)

        raw = "\n\n".join(x for x in page_texts if x.strip())
        extracted_text = postprocess_translation_ready_text(raw)
        _job_update(job_id, preview_text="\n".join(extracted_text.splitlines()[:30]))

        # ── Translation pass ───────────────────────────────────────────────
        translation = ""
        lang_label = ""
        if translate in ("en", "zh"):
            lang_label = "English" if translate == "en" else "Traditional Chinese"
            total_chars = sum(len(t) for t in page_texts)
            done_chars = 0
            _job_update(job_id, stage=f"translating_to_{translate}",
                        total_chars=total_chars, done_chars=0)
            translated_pages: list[str] = []
            for i, page_text in enumerate(page_texts, start=1):
                _job_update(job_id, stage=f"translating_page_{i}")
                translated_pages.append(translate_text_gemini(page_text, translate))
                done_chars += len(page_text)
                _job_update(job_id, done_chars=done_chars)
            raw_translated = "\n\n".join(x for x in translated_pages if x.strip())
            translation = postprocess_translation_ready_text(raw_translated)

        # ── Build output ───────────────────────────────────────────────────
        if translation:
            final_txt = (
                f"{'=' * 40}\n"
                f"ORIGINAL\n"
                f"{'=' * 40}\n\n"
                f"{extracted_text}\n\n"
                f"{'=' * 40}\n"
                f"TRANSLATION ({lang_label})\n"
                f"{'=' * 40}\n\n"
                f"{translation}"
            )
            preview_text = "\n".join(translation.splitlines()[:40])
        else:
            final_txt = extracted_text
            preview_text = "\n".join(extracted_text.splitlines()[:40])

        out_path = work_root / "output.txt"
        out_path.write_text(final_txt, encoding="utf-8")

        _job_update(
            job_id,
            status="done",
            stage="done",
            output_path=out_path.as_posix(),
            preview_text=preview_text,
        )
    except Exception as e:
        tb = traceback.format_exc(limit=30)
        _job_update(job_id, status="error", stage="error", error=f"{e}\n\n{tb}")
    finally:
        shutil.rmtree(work_root / "pages", ignore_errors=True)


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title="CJK Document → Clean Text")
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR.as_posix()), name="static")


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>読む — CJK OCR &amp; Translation</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@300;400&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=JetBrains+Mono:wght@300;400&display=swap" rel="stylesheet" />
  <style>
    :root {
      --ink:          #1a1208;
      --paper:        #f5f0e8;
      --aged:         #e8e0ce;
      --accent:       #8b2a0f;
      --accent-light: #c94a1e;
      --muted:        #7a6f5e;
      --border:       #c8bea8;
      --panel:        #faf7f2;
    }

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Libre Baskerville', Georgia, serif;
      background: var(--paper);
      background-image:
        radial-gradient(ellipse at 15% 10%, rgba(139,42,15,0.05) 0%, transparent 50%),
        radial-gradient(ellipse at 85% 90%, rgba(139,42,15,0.04) 0%, transparent 50%);
      color: var(--ink);
      min-height: 100vh;
    }

    /* ── Header ── */
    header {
      border-bottom: 1px solid var(--border);
      padding: 18px 40px;
      display: flex;
      align-items: baseline;
      gap: 14px;
      background: var(--panel);
    }
    header h1 {
      font-size: 1.6rem;
      font-weight: 700;
      letter-spacing: 0.02em;
      color: var(--accent);
    }
    header .subtitle {
      font-family: 'Noto Serif JP', serif;
      font-size: 0.82rem;
      color: var(--muted);
      font-weight: 300;
    }
    header .tagline {
      margin-left: auto;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.72rem;
      color: var(--muted);
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }

    /* ── Main layout ── */
    main {
      max-width: 720px;
      margin: 0 auto;
      padding: 36px 40px 60px;
    }

    /* ── Drop zone ── */
    #fileInput { display: none; }
    .drop-zone {
      border: 1.5px dashed var(--border);
      border-radius: 4px;
      padding: 52px 32px;
      text-align: center;
      cursor: pointer;
      background: var(--panel);
      transition: border-color 0.2s, background 0.2s;
      margin-bottom: 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
    }
    .drop-zone:hover, .drop-zone.drag {
      border-color: var(--accent);
      background: rgba(139,42,15,0.025);
    }
    .dz-icon  { font-size: 2.6rem; line-height: 1; }
    .dz-label { font-size: 1rem; font-weight: 700; color: var(--ink); margin-top: 4px; }
    .dz-sub   { font-size: 0.82rem; color: var(--muted); font-style: italic; }

    /* ── File pill ── */
    .file-pill {
      display: none;
      align-items: center;
      gap: 10px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 2px;
      padding: 10px 14px;
      margin-bottom: 20px;
    }
    .file-pill.on { display: flex; }
    .fp-icon { font-size: 1.1rem; flex-shrink: 0; }
    .fp-name {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.8rem;
      flex: 1;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--ink);
    }
    .fp-type {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      background: var(--aged);
      border-radius: 2px;
      padding: 2px 8px;
      flex-shrink: 0;
    }
    .fp-clear {
      background: none;
      border: none;
      color: var(--border);
      cursor: pointer;
      font-size: 0.9rem;
      padding: 0 2px;
      line-height: 1;
      transition: color 0.15s;
    }
    .fp-clear:hover { color: var(--accent); }

    /* ── Controls row ── */
    .controls { display: none; gap: 10px; align-items: center; margin-bottom: 24px; flex-wrap: wrap; }
    .controls.on { display: flex; }

    /* OCR toggle chip */
    .ocr-toggle {
      display: flex;
      border: 1px solid var(--border);
      border-radius: 2px;
      overflow: hidden;
      flex-shrink: 0;
    }
    .ocr-toggle-btn {
      padding: 8px 14px;
      border: none;
      background: transparent;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      cursor: pointer;
      transition: background 0.15s, color 0.15s;
      white-space: nowrap;
    }
    .ocr-toggle-btn.active {
      background: var(--accent);
      color: #fff;
    }
    .ocr-toggle-btn:not(.active):hover {
      background: var(--aged);
      color: var(--ink);
    }
    .ocr-toggle.hidden { display: none; }

    /* Translation select */
    .tr-select {
      flex: 1;
      min-width: 0;
      padding: 8px 32px 8px 12px;
      border: 1px solid var(--border);
      border-radius: 2px;
      font-family: 'Libre Baskerville', serif;
      font-size: 0.82rem;
      color: var(--ink);
      background: var(--panel)
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%237a6f5e' stroke-width='1.5' stroke-linecap='round' fill='none'/%3E%3C/svg%3E")
        no-repeat right 12px center;
      appearance: none;
      transition: border-color 0.15s;
    }
    .tr-select:focus { outline: none; border-color: var(--accent); }

    /* Process button */
    .proc-btn {
      padding: 9px 22px;
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: 2px;
      font-family: 'Libre Baskerville', serif;
      font-size: 0.82rem;
      font-weight: 700;
      letter-spacing: 0.02em;
      cursor: pointer;
      white-space: nowrap;
      flex-shrink: 0;
      transition: background 0.15s;
    }
    .proc-btn:hover:not(:disabled) { background: var(--accent-light); }
    .proc-btn:disabled { opacity: 0.4; cursor: not-allowed; }

    /* ── Status bar ── */
    .status-bar {
      display: none;
      align-items: center;
      gap: 12px;
      padding: 11px 18px;
      background: var(--aged);
      border-left: 3px solid var(--accent);
      border-radius: 2px;
      margin-bottom: 16px;
      font-size: 0.84rem;
      color: var(--muted);
      font-style: italic;
    }
    .status-bar.on { display: flex; }
    .spinner {
      width: 15px; height: 15px;
      border: 2px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      flex-shrink: 0;
    }
    .spinner.done { animation: none; border-color: var(--accent); border-top-color: transparent; }
    .spinner.err  { animation: none; border-color: var(--accent-light); border-top-color: transparent; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .prog-lbl   { flex: 1; color: var(--ink); font-style: italic; }
    .prog-count {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.72rem;
      font-style: normal;
      letter-spacing: 0.04em;
      color: var(--muted);
      white-space: nowrap;
    }

    /* ── Progress bar ── */
    .bar-wrap { display: none; margin-bottom: 20px; }
    .bar-wrap.on { display: block; }
    .bar-track {
      height: 3px;
      background: var(--aged);
      border-radius: 0;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      background: var(--accent);
      width: 0%;
      transition: width 0.35s ease;
    }
    .bar-fill.done { background: var(--accent); width: 100%; }
    .bar-fill.err  { background: var(--accent-light); width: 100%; }
    .bar-fill.indeterminate { animation: slide 1.1s ease-in-out infinite; width: 28% !important; }
    @keyframes slide { 0% { margin-left: -28%; } 100% { margin-left: 108%; } }

    /* ── Error box ── */
    .err-box {
      display: none;
      padding: 12px 18px;
      background: #fef0ee;
      border-left: 3px solid var(--accent-light);
      border-radius: 2px;
      font-size: 0.8rem;
      color: var(--accent);
      white-space: pre-wrap;
      max-height: 160px;
      overflow-y: auto;
      margin-bottom: 16px;
      font-style: italic;
    }
    .err-box.on { display: block; }

    /* ── Preview panel ── */
    .prev-panel { display: none; border: 1px solid var(--border); border-radius: 2px; overflow: hidden; margin-bottom: 20px; }
    .prev-panel.on { display: block; }
    .panel-header {
      padding: 10px 18px;
      background: var(--aged);
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .panel-title {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--muted);
    }
    .prev-toggle {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      background: none;
      border: 1px solid var(--border);
      border-radius: 2px;
      padding: 3px 10px;
      cursor: pointer;
      transition: border-color 0.15s, color 0.15s;
    }
    .prev-toggle:hover { border-color: var(--ink); color: var(--ink); }
    .prev-body {
      display: none;
      padding: 20px 22px;
      background: var(--panel);
      font-family: 'Noto Serif JP', serif;
      font-size: 0.92rem;
      font-weight: 300;
      line-height: 2;
      white-space: pre-wrap;
      color: var(--ink);
      max-height: 220px;
      overflow-y: auto;
    }
    .prev-body.on { display: block; }

    /* ── Download button ── */
    .dl-btn {
      display: none;
      width: 100%;
      padding: 11px;
      background: var(--accent);
      color: #fff;
      text-align: center;
      text-decoration: none;
      border-radius: 2px;
      font-family: 'Libre Baskerville', serif;
      font-weight: 700;
      font-size: 0.85rem;
      letter-spacing: 0.02em;
      transition: background 0.15s;
    }
    .dl-btn:hover { background: var(--accent-light); }
    .dl-btn.on { display: block; }

    /* ── Responsive ── */
    @media (max-width: 600px) {
      header { padding: 14px 20px; }
      main { padding: 24px 20px 48px; }
      .drop-zone { padding: 36px 20px; }
      .controls { flex-direction: column; align-items: stretch; }
      .ocr-toggle { justify-content: center; }
    }
  </style>
</head>
<body>

<header>
  <h1>読む</h1>
  <span class="subtitle">yomu — read</span>
  <span class="tagline">CJK OCR &amp; Translation</span>
</header>

<main>
  <!-- Drop zone -->
  <label class="drop-zone" id="dropZone" for="fileInput">
    <span class="dz-icon">文</span>
    <span class="dz-label">Drop a PDF or TXT here</span>
    <span class="dz-sub">Japanese, Chinese — scanned or machine-readable</span>
  </label>
  <input type="file" id="fileInput" accept=".pdf,.txt,text/plain,application/pdf" />

  <!-- File pill -->
  <div class="file-pill" id="filePill">
    <span class="fp-icon" id="fpIcon">📄</span>
    <span class="fp-name" id="fpName">—</span>
    <span class="fp-type" id="fpType">PDF</span>
    <button class="fp-clear" id="clearBtn" title="Remove file">✕</button>
  </div>

  <!-- Controls row -->
  <div class="controls" id="controls">
    <!-- OCR toggle (PDF only) -->
    <div class="ocr-toggle" id="ocrToggle">
      <button class="ocr-toggle-btn active" id="ocrOn"  onclick="setOcr(true)">OCR on</button>
      <button class="ocr-toggle-btn"        id="ocrOff" onclick="setOcr(false)">OCR off</button>
    </div>

    <!-- Translation select -->
    <select class="tr-select" id="trSelect">
      <option value="none">No Translation</option>
      <option value="en">→ English</option>
      <option value="zh">→ Chinese</option>
    </select>

    <!-- Process button -->
    <button class="proc-btn" id="procBtn">Process →</button>
  </div>

  <!-- Status bar -->
  <div class="status-bar" id="statusBar">
    <div class="spinner" id="spinner"></div>
    <span class="prog-lbl"  id="progLbl">Starting…</span>
    <span class="prog-count" id="progCount"></span>
  </div>

  <!-- Progress bar -->
  <div class="bar-wrap" id="barWrap">
    <div class="bar-track"><div class="bar-fill indeterminate" id="barFill"></div></div>
  </div>

  <!-- Error -->
  <div class="err-box" id="errBox"></div>

  <!-- Preview panel -->
  <div class="prev-panel" id="prevPanel">
    <div class="panel-header">
      <span class="panel-title">Preview</span>
      <button class="prev-toggle" id="prevBtn">Show</button>
    </div>
    <div class="prev-body" id="prevBox"></div>
  </div>

  <!-- Download -->
  <a class="dl-btn" id="dlBtn" href="#">↓ Download output.txt</a>
</main>

<script>
  // ── Elements ────────────────────────────────────────────────────────────
  const dropZone   = document.getElementById('dropZone');
  const fileInput  = document.getElementById('fileInput');
  const filePill   = document.getElementById('filePill');
  const fpIcon     = document.getElementById('fpIcon');
  const fpName     = document.getElementById('fpName');
  const fpType     = document.getElementById('fpType');
  const clearBtn   = document.getElementById('clearBtn');
  const controls   = document.getElementById('controls');
  const ocrToggle  = document.getElementById('ocrToggle');
  const ocrOnBtn   = document.getElementById('ocrOn');
  const ocrOffBtn  = document.getElementById('ocrOff');
  const trSelect   = document.getElementById('trSelect');
  const procBtn    = document.getElementById('procBtn');
  const statusBar  = document.getElementById('statusBar');
  const spinner    = document.getElementById('spinner');
  const progLbl    = document.getElementById('progLbl');
  const progCount  = document.getElementById('progCount');
  const barWrap    = document.getElementById('barWrap');
  const barFill    = document.getElementById('barFill');
  const errBox     = document.getElementById('errBox');
  const prevPanel  = document.getElementById('prevPanel');
  const prevBtn    = document.getElementById('prevBtn');
  const prevBox    = document.getElementById('prevBox');
  const dlBtn      = document.getElementById('dlBtn');

  // ── State ───────────────────────────────────────────────────────────────
  let currentFile     = null;
  let currentFileType = null;
  let ocrEnabled      = true;
  let pollTimer       = null;

  // ── OCR toggle ──────────────────────────────────────────────────────────
  function setOcr(on) {
    ocrEnabled = on;
    ocrOnBtn.classList.toggle('active',  on);
    ocrOffBtn.classList.toggle('active', !on);
  }
  // expose for onclick attributes
  window.setOcr = setOcr;

  // ── Drop zone ───────────────────────────────────────────────────────────
  dropZone.addEventListener('dragover',  (e) => { e.preventDefault(); dropZone.classList.add('drag'); });
  dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('drag'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault(); dropZone.classList.remove('drag');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    const lower = file.name.toLowerCase();
    if (!lower.endsWith('.pdf') && !lower.endsWith('.txt')) {
      alert('Please choose a PDF or TXT file.');
      return;
    }
    currentFile     = file;
    currentFileType = lower.endsWith('.pdf') ? 'pdf' : 'txt';

    fpIcon.textContent = currentFileType === 'pdf' ? '📄' : '📝';
    fpName.textContent = file.name;
    fpType.textContent = currentFileType.toUpperCase();

    dropZone.style.display = 'none';
    filePill.classList.add('on');
    ocrToggle.classList.toggle('hidden', currentFileType === 'txt');
    controls.classList.add('on');
    resetProgress();
  }

  clearBtn.addEventListener('click', () => {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    currentFile = null; currentFileType = null; fileInput.value = '';
    dropZone.style.display = '';
    filePill.classList.remove('on');
    controls.classList.remove('on');
    resetProgress();
    procBtn.disabled = false;
  });

  // ── Process ─────────────────────────────────────────────────────────────
  procBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    procBtn.disabled = true;
    resetProgress();
    statusBar.classList.add('on');
    barWrap.classList.add('on');
    progLbl.textContent = 'Uploading…';
    barFill.className = 'bar-fill indeterminate';

    try {
      const fd = new FormData();
      fd.append('file', currentFile);
      fd.append('ocr_enabled', (currentFileType === 'pdf' && ocrEnabled) ? 'true' : 'false');
      fd.append('translate', trSelect.value);
      const res = await fetch('/api/jobs', { method: 'POST', body: fd });
      if (!res.ok) { showError(await res.text()); return; }
      const { job_id } = await res.json();
      progLbl.textContent = 'Starting…';
      startPoll(job_id);
    } catch (err) {
      showError('Upload failed: ' + (err.message || err));
    }
  });

  // ── Helpers ─────────────────────────────────────────────────────────────
  function resetProgress() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    statusBar.classList.remove('on');
    barWrap.classList.remove('on');
    spinner.className = 'spinner';
    barFill.className = 'bar-fill indeterminate';
    barFill.style.width = '';
    progLbl.textContent = '';
    progCount.textContent = '';
    errBox.classList.remove('on');  errBox.textContent = '';
    prevPanel.classList.remove('on');
    prevBox.classList.remove('on'); prevBox.textContent = '';
    prevBtn.textContent = 'Show';
    dlBtn.classList.remove('on');
  }

  function setBar(pct, cls) {
    barFill.className = 'bar-fill' + (cls ? ' ' + cls : '');
    barFill.style.width = pct + '%';
  }

  function showError(msg) {
    spinner.className = 'spinner err';
    progLbl.textContent = 'Error';
    setBar(100, 'err');
    errBox.textContent = msg;
    errBox.classList.add('on');
    procBtn.disabled = false;
  }

  function stageLabel(stage) {
    if (!stage || stage === 'starting' || stage === 'queued') return 'Starting…';
    if (stage === 'pdf_to_images')   return 'Rendering pages…';
    if (stage === 'extracting_text') return 'Extracting text…';
    if (stage === 'reading_file')    return 'Reading file…';
    if (stage === 'ocr')             return 'Starting OCR…';
    if (stage.startsWith('ocr_page_'))          return `OCR — page ${stage.split('_').pop()}`;
    if (stage.startsWith('translating_to_'))    return 'Translating…';
    if (stage.startsWith('translating_page_'))  return `Translating — page ${stage.split('_').pop()}`;
    return 'Processing…';
  }

  // ── Polling ─────────────────────────────────────────────────────────────
  function startPoll(jobId) {
    const withTranslation = trSelect.value !== 'none';
    // OCR phase only runs for PDF files with OCR enabled; determines bar split
    const hasOcrPhase = currentFileType === 'pdf' && ocrEnabled;
    pollTimer = setInterval(async () => {
      try {
        const r = await fetch('/api/jobs/' + jobId);
        if (!r.ok) { clearInterval(pollTimer); showError(await r.text()); return; }
        const j = await r.json();
        const inTranslation = j.stage && j.stage.startsWith('translating');

        // Progress bar
        if (j.status !== 'done') {
          let pct = 0;
          if (inTranslation && j.total_chars > 0) {
            const charPct = j.done_chars / j.total_chars;
            pct = hasOcrPhase
              ? 60 + Math.round(charPct * 40)
              : Math.round(charPct * 100);
            const doneK = (j.done_chars / 1000).toFixed(1);
            const totalK = (j.total_chars / 1000).toFixed(1);
            progCount.textContent = `${doneK}k / ${totalK}k chars`;
          } else if (!inTranslation && j.total_pages > 0) {
            const base = Math.round((j.done_pages / j.total_pages) * 100);
            pct = withTranslation ? Math.round(base * 0.6) : base;
            progCount.textContent = `${j.done_pages} / ${j.total_pages} pages`;
          }
          if (pct > 0) setBar(pct);
        }

        if (j.status === 'running') progLbl.textContent = stageLabel(j.stage);

        // Preview
        if (j.preview_text) {
          prevBox.textContent = j.preview_text;
          prevPanel.classList.add('on');
        }

        if (j.status === 'done') {
          clearInterval(pollTimer);
          spinner.className = 'spinner done';
          progLbl.textContent = 'Complete.';
          setBar(100, 'done');
          progCount.textContent = j.total_pages ? `${j.total_pages} pages` : '';
          dlBtn.href = '/api/jobs/' + jobId + '/download';
          dlBtn.classList.add('on');
          procBtn.disabled = false;
        }
        if (j.status === 'error') {
          clearInterval(pollTimer);
          showError(j.error || 'Unknown error');
        }
      } catch (err) {
        clearInterval(pollTimer);
        showError('Connection lost: ' + (err.message || err));
      }
    }, 800);
  }

  // ── Preview toggle ───────────────────────────────────────────────────────
  prevBtn.addEventListener('click', () => {
    const open = prevBox.classList.toggle('on');
    prevBtn.textContent = open ? 'Hide' : 'Show';
  });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    if (STATIC_DIR / "index.html").is_file():
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))
    return HTMLResponse(INDEX_HTML)


@app.post("/api/jobs")
def create_job(
    file: UploadFile = File(...),
    ocr_enabled: str = Form("true"),
    translate: str = Form("none"),
):
    fname = (file.filename or "").lower()
    if not (fname.endswith(".pdf") or fname.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf or .txt uploads are supported.")
    if translate not in ("none", "en", "zh"):
        raise HTTPException(status_code=400, detail="Invalid translate value.")

    file_type = "pdf" if fname.endswith(".pdf") else "txt"
    ocr_flag = ocr_enabled.lower() == "true" and file_type == "pdf"

    job_id = uuid.uuid4().hex
    work_root = Path(tempfile.mkdtemp(prefix=f"cjkocr_{job_id}_"))
    suffix = ".pdf" if file_type == "pdf" else ".txt"
    file_path = work_root / f"input{suffix}"
    with file_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "stage": "queued",
            "file_type": file_type,
            "ocr_enabled": ocr_flag,
            "translate": translate,
            "filename": file.filename or f"document{suffix}",
            "total_pages": 0,
            "done_pages": 0,
            "total_chars": 0,
            "done_chars": 0,
            "preview_text": "",
            "output_path": None,
            "error": None,
        }

    t = threading.Thread(
        target=_run_pipeline,
        args=(job_id, file_path, work_root, file_type, ocr_flag, translate),
        daemon=True,
    )
    t.start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with _jobs_lock:
        j = _jobs.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": j["id"],
        "status": j["status"],
        "stage": j["stage"],
        "filename": j["filename"],
        "total_pages": j["total_pages"],
        "done_pages": j["done_pages"],
        "total_chars": j["total_chars"],
        "done_chars": j["done_chars"],
        "preview_text": j["preview_text"],
        "error": j["error"],
    }


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str, background_tasks: BackgroundTasks):
    with _jobs_lock:
        j = _jobs.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found.")
    if j["status"] != "done" or not j["output_path"]:
        raise HTTPException(status_code=400, detail="Job not complete.")
    p = Path(j["output_path"])
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Output missing.")
    work_root = p.parent
    stem = Path(j["filename"]).stem or "output"
    background_tasks.add_task(shutil.rmtree, work_root, ignore_errors=True)
    return FileResponse(p.as_posix(), media_type="text/plain; charset=utf-8", filename=f"{stem}.txt")


def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8765, reload=False)


if __name__ == "__main__":
    main()
