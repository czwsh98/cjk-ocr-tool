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
  <title>CJK Document → Clean Text</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #f5f5f7; min-height: 100vh; padding: 2.5rem 1rem; color: #1d1d1f; }
    .app { max-width: 580px; margin: 0 auto; }

    /* Header */
    .hdr { margin-bottom: 1.75rem; }
    .hdr h1 { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; }
    .hdr p  { color: #6e6e73; font-size: 0.875rem; margin-top: 0.3rem; }

    /* Card */
    .card { background: #fff; border-radius: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 6px 20px rgba(0,0,0,0.05);
            padding: 1.5rem; }

    /* ── Drop zone ── */
    #fileInput { display: none; }
    .drop-zone { border: 1.5px dashed #d0d0d5; border-radius: 10px; padding: 2.25rem 1rem;
                 text-align: center; cursor: pointer; transition: border-color 0.15s, background 0.15s;
                 display: flex; flex-direction: column; align-items: center; gap: 0.55rem; }
    .drop-zone:hover, .drop-zone.drag { border-color: #0066cc; background: #f0f6ff; }
    .dz-icon  { font-size: 1.75rem; }
    .dz-label { font-size: 0.9rem; color: #3a3a3c; font-weight: 500; }
    .dz-sub   { font-size: 0.78rem; color: #aeaeb2; }

    /* ── File pill ── */
    .file-pill { display: none; align-items: center; gap: 0.65rem;
                 background: #f0f6ff; border: 1px solid #cce0f5;
                 border-radius: 9px; padding: 0.65rem 0.9rem; }
    .file-pill.on { display: flex; }
    .fp-icon { font-size: 1.1rem; flex-shrink: 0; }
    .fp-name { font-size: 0.875rem; font-weight: 500; flex: 1; min-width: 0;
               overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .fp-type { font-size: 0.72rem; color: #6e6e73; background: #e8f0fb;
               border-radius: 4px; padding: 0.15rem 0.45rem; flex-shrink: 0; }
    .fp-clear { background: none; border: none; color: #aeaeb2; cursor: pointer;
                font-size: 0.95rem; padding: 0 0.1rem; line-height: 1; transition: color 0.12s; }
    .fp-clear:hover { color: #ef4444; }

    /* ── Controls row ── */
    .controls { display: none; margin-top: 0.85rem; gap: 0.65rem; align-items: center; }
    .controls.on { display: flex; }
    .ocr-lbl { display: flex; align-items: center; gap: 0.4rem; cursor: pointer;
               font-size: 0.83rem; color: #3a3a3c; user-select: none; white-space: nowrap;
               padding: 0.45rem 0.75rem; border: 1px solid #ddd; border-radius: 7px;
               background: #fff; transition: border-color 0.12s, background 0.12s; }
    .ocr-lbl:hover { border-color: #aaa; }
    .ocr-lbl input { accent-color: #0066cc; width: 14px; height: 14px; cursor: pointer; }
    .ocr-lbl.hidden { display: none; }
    .tr-select { flex: 1; min-width: 0; padding: 0.46rem 2rem 0.46rem 0.7rem;
                 border: 1px solid #ddd; border-radius: 7px; font-size: 0.83rem;
                 background: #fff url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%23999' stroke-width='1.5' stroke-linecap='round' fill='none'/%3E%3C/svg%3E") no-repeat right 0.7rem center;
                 appearance: none; color: #1d1d1f; }
    .tr-select:focus { outline: 2px solid #0066cc; outline-offset: 1px; border-color: transparent; }
    .proc-btn { padding: 0.48rem 1.1rem; background: #0066cc; color: #fff; border: none;
                border-radius: 7px; font-size: 0.83rem; font-weight: 600; cursor: pointer;
                white-space: nowrap; transition: background 0.15s; flex-shrink: 0; }
    .proc-btn:hover:not(:disabled) { background: #0055bb; }
    .proc-btn:disabled { background: #b0c8e8; cursor: not-allowed; }

    /* ── Progress section ── */
    .prog-section { display: none; margin-top: 1.25rem;
                    border-top: 1px solid #f0f0f2; padding-top: 1.25rem; }
    .prog-section.on { display: block; }

    .prog-head { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.85rem; }
    .spinner { width: 15px; height: 15px; border: 2px solid #e0e0e0; border-top-color: #0066cc;
               border-radius: 50%; animation: spin 0.75s linear infinite; flex-shrink: 0; }
    .spinner.done  { animation: none; border-color: #22c55e; }
    .spinner.err   { animation: none; border-color: #ef4444; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .prog-lbl   { font-size: 0.875rem; font-weight: 600; flex: 1; }
    .prog-count { font-size: 0.78rem; color: #6e6e73; white-space: nowrap; }

    .bar-track { height: 5px; background: #ebebed; border-radius: 999px; overflow: hidden; }
    .bar-fill  { height: 100%; border-radius: 999px; background: #0066cc;
                 width: 0%; transition: width 0.35s ease; }
    .bar-fill.done  { background: #22c55e; }
    .bar-fill.err   { background: #ef4444; }
    .bar-fill.indeterminate { animation: slide 1.1s ease-in-out infinite; width: 30% !important; }
    @keyframes slide { 0% { margin-left: -30%; } 100% { margin-left: 105%; } }

    /* Preview */
    .prev-row { display: none; margin-top: 0.85rem; align-items: center; gap: 0.4rem; }
    .prev-row.on { display: flex; }
    .prev-btn { background: none; border: none; font-size: 0.8rem; color: #6e6e73;
                cursor: pointer; padding: 0; line-height: 1; }
    .prev-btn:hover { color: #1d1d1f; }
    .prev-box { margin-top: 0.5rem; display: none; font-size: 0.79rem; white-space: pre-wrap;
                color: #3a3a3c; background: #f8f8fa; border: 1px solid #eaeaec;
                border-radius: 7px; padding: 0.75rem; max-height: 180px; overflow-y: auto;
                line-height: 1.65; }
    .prev-box.on { display: block; }

    /* Error */
    .err-box { display: none; margin-top: 0.85rem; background: #fff5f5;
               border: 1px solid #fca5a5; border-radius: 7px; padding: 0.75rem;
               font-size: 0.79rem; color: #b91c1c; white-space: pre-wrap;
               max-height: 150px; overflow-y: auto; }
    .err-box.on { display: block; }

    /* Download */
    .dl-btn { display: none; margin-top: 1rem; width: 100%; padding: 0.65rem;
              background: #22c55e; color: #fff; text-align: center; text-decoration: none;
              border-radius: 8px; font-weight: 600; font-size: 0.875rem;
              transition: background 0.15s; }
    .dl-btn:hover { background: #16a34a; }
    .dl-btn.on { display: block; }
  </style>
</head>
<body>
<div class="app">
  <div class="hdr">
    <h1>CJK Document → Clean Text</h1>
    <p>Extract and optionally translate Japanese or Chinese documents.</p>
  </div>

  <div class="card">
    <!-- Drop zone -->
    <label class="drop-zone" id="dropZone" for="fileInput">
      <span class="dz-icon">📂</span>
      <span class="dz-label">Drop a PDF or TXT here</span>
      <span class="dz-sub">or click to browse</span>
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
      <label class="ocr-lbl" id="ocrLabel">
        <input type="checkbox" id="ocrCheck" checked />
        Enable OCR
      </label>
      <select class="tr-select" id="trSelect">
        <option value="none">No Translation</option>
        <option value="en">→ English</option>
        <option value="zh">→ Chinese</option>
      </select>
      <button class="proc-btn" id="procBtn">Process →</button>
    </div>

    <!-- Progress section -->
    <div class="prog-section" id="progSection">
      <div class="prog-head">
        <div class="spinner" id="spinner"></div>
        <span class="prog-lbl" id="progLbl">Starting…</span>
        <span class="prog-count" id="progCount"></span>
      </div>
      <div class="bar-track"><div class="bar-fill indeterminate" id="barFill"></div></div>
      <div class="prev-row" id="prevRow">
        <button class="prev-btn" id="prevBtn">▸ Preview text</button>
      </div>
      <div class="prev-box" id="prevBox"></div>
      <div class="err-box" id="errBox"></div>
      <a class="dl-btn" id="dlBtn" href="#">⬇ Download output.txt</a>
    </div>
  </div>
</div>

<script>
  // ── Elements ────────────────────────────────────────────────────────────
  const dropZone  = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const filePill  = document.getElementById('filePill');
  const fpIcon    = document.getElementById('fpIcon');
  const fpName    = document.getElementById('fpName');
  const fpType    = document.getElementById('fpType');
  const clearBtn  = document.getElementById('clearBtn');
  const controls  = document.getElementById('controls');
  const ocrLabel  = document.getElementById('ocrLabel');
  const ocrCheck  = document.getElementById('ocrCheck');
  const trSelect  = document.getElementById('trSelect');
  const procBtn   = document.getElementById('procBtn');
  const progSection = document.getElementById('progSection');
  const spinner   = document.getElementById('spinner');
  const progLbl   = document.getElementById('progLbl');
  const progCount = document.getElementById('progCount');
  const barFill   = document.getElementById('barFill');
  const prevRow   = document.getElementById('prevRow');
  const prevBtn   = document.getElementById('prevBtn');
  const prevBox   = document.getElementById('prevBox');
  const errBox    = document.getElementById('errBox');
  const dlBtn     = document.getElementById('dlBtn');

  // ── State ───────────────────────────────────────────────────────────────
  let currentFile = null;
  let currentFileType = null;
  let pollTimer = null;

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
    currentFile = file;
    currentFileType = lower.endsWith('.pdf') ? 'pdf' : 'txt';

    fpIcon.textContent = currentFileType === 'pdf' ? '📄' : '📝';
    fpName.textContent = file.name;
    fpType.textContent = currentFileType.toUpperCase();

    dropZone.style.display = 'none';
    filePill.classList.add('on');
    ocrLabel.classList.toggle('hidden', currentFileType === 'txt');
    controls.classList.add('on');
    resetProgress();
  }

  clearBtn.addEventListener('click', () => {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    currentFile = null; currentFileType = null; fileInput.value = '';
    dropZone.style.display = '';
    filePill.classList.remove('on');
    controls.classList.remove('on');
    progSection.classList.remove('on');
    procBtn.disabled = false;
  });

  // ── Process ─────────────────────────────────────────────────────────────
  procBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    procBtn.disabled = true;
    resetProgress();
    progSection.classList.add('on');
    progLbl.textContent = 'Uploading…';
    barFill.className = 'bar-fill indeterminate';

    try {
      const fd = new FormData();
      fd.append('file', currentFile);
      fd.append('ocr_enabled', (currentFileType === 'pdf' && ocrCheck.checked) ? 'true' : 'false');
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
    progSection.classList.remove('on');
    spinner.className = 'spinner';
    barFill.className = 'bar-fill indeterminate';
    barFill.style.width = '';
    progLbl.textContent = '';
    progCount.textContent = '';
    prevRow.classList.remove('on');
    prevBox.classList.remove('on'); prevBox.textContent = '';
    errBox.classList.remove('on');  errBox.textContent = '';
    dlBtn.classList.remove('on');
    prevBtn.textContent = '▸ Preview text';
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
    const hasOcrPhase = currentFileType === 'pdf' && ocrCheck.checked;
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
            // Translation: char-based progress
            // Bar occupies 60–100% if OCR ran first, otherwise 0–100%
            const charPct = j.done_chars / j.total_chars;
            pct = hasOcrPhase
              ? 60 + Math.round(charPct * 40)
              : Math.round(charPct * 100);
            const doneK = (j.done_chars / 1000).toFixed(1);
            const totalK = (j.total_chars / 1000).toFixed(1);
            progCount.textContent = `${doneK}k / ${totalK}k chars`;
          } else if (!inTranslation && j.total_pages > 0) {
            // OCR / extraction: page-based progress
            // Bar occupies 0–60% if translation follows, otherwise 0–100%
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
          prevRow.classList.add('on');
        }

        if (j.status === 'done') {
          clearInterval(pollTimer);
          spinner.className = 'spinner done';
          progLbl.textContent = 'Done!';
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

  // Preview toggle
  prevBtn.addEventListener('click', () => {
    const open = prevBox.classList.toggle('on');
    prevBtn.textContent = open ? '▾ Preview text' : '▸ Preview text';
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
