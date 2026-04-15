# CJK Document → Clean Text

A local web app that extracts and optionally translates Japanese and Chinese documents (PDF or TXT) using Google Gemini Vision.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green) ![Gemini](https://img.shields.io/badge/Gemini-2.5--flash-orange)

---

## Features

- **OCR** scanned Japanese, Korean, and Chinese PDFs via Gemini Vision (handles vertical text, furigana, bopomofo)
- **Direct text extraction** from machine-readable PDFs (no OCR needed)
- **TXT file support** — feed in pre-extracted text and go straight to translation
- **Translation** to English or Traditional Chinese, powered by the same Gemini API
- **Context-aware UI** — the OCR toggle only appears for PDFs; TXT files skip straight to translation
- **Real-time progress** — page counter during OCR, character counter during translation
- **Automatic retries** on transient API errors (429 rate limits, 503 overload)
- Output is a clean `.txt` file; if translation was requested, the original and translation are both included

---

## Requirements

### API key

This tool uses the **Google Gemini API**. You must provide your own key.

1. Go to [Google AI Studio](https://aistudio.google.com/) and create an API key
2. Enable billing on your Google Cloud project (the free tier quota is very limited)
3. Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_key_here
```

> The `.env` file is listed in `.gitignore` and will never be committed. Keep it out of version control.

### System dependencies

**Poppler** is required to rasterize PDF pages for OCR:

```bash
# macOS
brew install poppler

# Ubuntu / Debian
sudo apt install poppler-utils
```

### Python

Python 3.11 or newer is recommended.

---

## Installation

```bash
git clone https://github.com/czwsh98/cjk-ocr-tool.git
cd cjk-ocr-tool

# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your API key
echo 'GOOGLE_API_KEY=your_key_here' > .env
```

---

## Usage

```bash
python app.py
```

Open **http://localhost:8765** in your browser.

### Workflow

1. **Drop a file** — PDF or TXT
2. **Configure:**
   - PDF: toggle **Enable OCR** on (scanned) or off (machine-readable text)
   - Choose a translation target: *No Translation*, *→ English*, or *→ Chinese*
3. Click **Process →**
4. Watch the progress bar — pages during OCR, characters during translation
5. Click **Download output.txt** when done

### Output format

| Mode | Output |
|------|--------|
| OCR only | Extracted text |
| OCR + translate | Original + translation, separated by headers |
| TXT + translate | Original + translation, separated by headers |

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Google Gemini API key |
| `GEMINI_VISION_MODEL` | No | Override model (default: `gemini-2.5-flash`) |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` + `uvicorn` | Web server |
| `pdf2image` | Rasterize PDF pages for OCR (requires Poppler) |
| `pymupdf` | Extract text from machine-readable PDFs |
| `Pillow` | Image handling |
| `google-genai` | Gemini Vision OCR and translation |
| `python-dotenv` | Load `.env` file |
| `python-multipart` | File upload parsing |
