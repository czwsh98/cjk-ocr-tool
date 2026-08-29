# Yomu — Chinese/Japanese OCR & Translation

Yomu is a small private web app for two document workflows:

1. Extract Chinese or Japanese text from PDF, TXT, or Markdown and export Markdown, Word, or both.
2. Translate Japanese documents into Traditional Chinese or English, with translation-only or bilingual output.

## What it does

- Select all PDF pages or ranges such as `1-5, 8, 12-18`. TXT and Markdown files are processed as a whole document.
- For PDFs, automatically use embedded text when available and OCR only scanned pages.
- Choose OCR behavior per job: automatic, always OCR, or never OCR.
- Process PDF pages sequentially to keep memory use bounded on a small VPS.
- Use Gemini 3.1 Flash-Lite for economy mode; quality mode uses 3.5 Flash-Lite for OCR and 3.7 Flash for translation.
- Prefer DeepSeek V4 Flash for translation whenever `DEEPSEEK_API_KEY` is set. It is the default in that case because it benchmarked both cheaper and more faithful than every Gemini tier. OCR always stays on Gemini.
- Report an explicit "cost estimate unavailable" instead of `$0.0000` when a configured model has no published price.
- Report input/output tokens and an estimated API cost.
- Split long translation inputs at paragraph boundaries to avoid silent output truncation.
- Reject blocked, truncated, or empty model responses instead of exporting blank pages.
- Export structured `.md` and styled `.docx` files.
- Limit upload size, worker concurrency, and temporary-file lifetime.

The prompts are tuned for Chinese and Japanese documents. Korean is not currently a target language.

## Local setup

Requirements: Python 3.11+, Poppler, and a Gemini API key.

```bash
brew install poppler
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add GOOGLE_API_KEY to .env
python app.py
```

Open <http://127.0.0.1:8765>.

## Configuration

See [`.env.example`](.env.example). The most important variables are:

| Variable | Purpose |
|---|---|
| `GOOGLE_API_KEY` | Required for Gemini OCR and translation |
| `DEEPSEEK_API_KEY` | Optional; when set, DeepSeek becomes the default translation provider |
| `MAX_UPLOAD_MB` | Upload limit, default 50 MB |
| `MAX_WORKERS` | Concurrent jobs, default 2 |
| `JOB_TTL_HOURS` | Temporary-output lifetime, default 6 hours |
| `TRANSLATION_CHUNK_CHARS` | Maximum source characters per translation request, default 12,000 |

When using a custom model ID, set the optional provider input/output price variables in
[`.env.example`](.env.example) if you want a cost estimate. Otherwise the UI reports that
the estimate is unavailable instead of displaying `$0.0000`.

The app listens on `127.0.0.1` by default. Keep it private behind a reverse proxy or Cloudflare Tunnel rather than binding it directly to a public interface.

## Tests

```bash
python -m pytest -q -p no:rerunfailures
```

The `no:rerunfailures` flag avoids an unrelated globally installed pytest plugin on the original development Mac.

## Deployment

The production service template is in [`deploy/yomu.service`](deploy/yomu.service). It expects the project and virtual environment under `/opt/yomu`, listens on `127.0.0.1:8765`, and can be published through Cloudflare Tunnel without taking over ports 80 or 443.

The current instance runs on a dedicated VPS with:

- `yomu.service` managed by systemd and bound only to localhost.
- A remotely managed Cloudflare Tunnel named `yomu-ocr`.
- Public hostname: <https://ocr.ziwei-chen.com>.
- Cloudflare Access in front of the hostname; access is restricted by an email allow policy.

For a new host, install the system packages and Python dependencies, copy `.env.example` to `/opt/yomu/.env`, install the service template, and connect the host from the Cloudflare Zero Trust dashboard. Keep tunnel tokens and API keys out of Git; the example files intentionally contain placeholders only.

```bash
sudo apt install poppler-utils python3-venv
python3 -m venv /opt/yomu/.venv
/opt/yomu/.venv/bin/pip install -r /opt/yomu/requirements.txt
sudo install -m 0644 deploy/yomu.service /etc/systemd/system/yomu.service
sudo systemctl daemon-reload
sudo systemctl enable --now yomu.service
```

When exposing the service publicly, put Cloudflare Access in front of the tunnel and create an explicit allow policy. Do not publish the app directly on a public interface.
