import shutil
import time

import pymupdf
from fastapi.testclient import TestClient

import app
from ocr_client import ModelResult


client = TestClient(app.app)


def _wait(job_id: str) -> dict:
    for _ in range(100):
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in {"done", "error"}:
            return payload
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def _cleanup(job_id: str) -> None:
    with app._jobs_lock:
        job = app._jobs.pop(job_id, None)
    if job:
        shutil.rmtree(job["work_root"], ignore_errors=True)


def test_text_extract_job_and_downloads():
    response = client.post(
        "/api/jobs",
        files={"file": ("sample.txt", "第一段。\n\n第二段。".encode(), "text/plain")},
        data={"mode": "extract", "output_formats": "md,docx"},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    try:
        result = _wait(job_id)
        assert result["status"] == "done"
        assert result["outputs"] == {"md": True, "docx": True}
        assert client.get(f"/api/jobs/{job_id}/download/md").status_code == 200
        assert client.get(f"/api/jobs/{job_id}/download/docx").content[:2] == b"PK"
    finally:
        _cleanup(job_id)


def test_never_ocr_uses_native_pdf_text_without_api(tmp_path):
    """
    "never" is the only mode that trusts a PDF's embedded text layer. There
    is deliberately no character-count "auto" mode anymore: many scanned
    CJK PDFs carry a legacy OCR layer long enough to pass any length
    threshold while being scrambled and mostly wrong (see _extract_page),
    so length can't safely decide whether to trust it.
    """
    pdf_path = tmp_path / "pages.pdf"
    document = pymupdf.open()
    for text in ("Page one native text " * 10, "Page two native text " * 10):
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(pdf_path)
    document.close()

    response = client.post(
        "/api/jobs",
        files={"file": ("pages.pdf", pdf_path.read_bytes(), "application/pdf")},
        data={
            "mode": "extract",
            "page_range": "2",
            "ocr_setting": "never",
            "output_formats": "md",
        },
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    try:
        result = _wait(job_id)
        assert result["status"] == "done"
        assert result["selected_pages"] == [2]
        markdown = client.get(f"/api/jobs/{job_id}/download/md").text
        assert "## Page 2" in markdown
        assert "Page two native text" in markdown
        assert "Page one native text" not in markdown
    finally:
        _cleanup(job_id)


def test_rejects_fake_pdf():
    response = client.post(
        "/api/jobs",
        files={"file": ("fake.pdf", b"not a pdf", "application/pdf")},
    )
    assert response.status_code == 400


def test_long_translation_is_chunked(monkeypatch):
    calls = []

    def fake_translate(text, target, *, provider, quality):
        calls.append(text)
        return ModelResult(text=f"译：{text}", provider="gemini", model="custom-model", input_tokens=1, output_tokens=1)

    monkeypatch.setattr(app, "translate_text", fake_translate)
    chunks = app._translation_chunks("甲" * 15 + "\n\n" + "乙" * 15, max_chars=20)
    assert len(chunks) == 2
    assert all(len(chunk) <= 20 for chunk in chunks)


def test_default_and_auto_and_always_all_ocr_even_with_long_native_text(monkeypatch, tmp_path):
    """
    A PDF's embedded text layer is never trusted just for being long — a
    scanned CJK PDF's legacy OCR layer routinely exceeds any length
    threshold while every character is wrong (see _extract_page). "always",
    the default, and the "auto" synonym must all call Vision OCR rather
    than skip it based on native text length.
    """
    pdf_path = tmp_path / "long_native.pdf"
    document = pymupdf.open()
    page = document.new_page()
    # Long enough to have defeated the old length-threshold heuristic.
    page.insert_text((72, 72), "native filler text " * 50)
    document.save(pdf_path)
    document.close()

    calls = []

    def fake_ocr_image(image_path, *, quality):
        calls.append(image_path)
        return ModelResult(text="OCR text", provider="gemini", model="custom-model", input_tokens=1, output_tokens=1)

    monkeypatch.setattr(app, "ocr_image", fake_ocr_image)

    for ocr_setting in (None, "auto", "always"):
        calls.clear()
        data = {"mode": "extract", "output_formats": "md"}
        if ocr_setting is not None:
            data["ocr_setting"] = ocr_setting
        response = client.post(
            "/api/jobs",
            files={"file": ("long_native.pdf", pdf_path.read_bytes(), "application/pdf")},
            data=data,
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        try:
            result = _wait(job_id)
            assert result["status"] == "done", (ocr_setting, result.get("error"))
            markdown = client.get(f"/api/jobs/{job_id}/download/md").text
            assert "OCR text" in markdown
            assert "native filler text" not in markdown
            assert len(calls) == 1, f"ocr_setting={ocr_setting!r} should call OCR exactly once"
        finally:
            _cleanup(job_id)


def test_invalid_ocr_setting_is_rejected():
    response = client.post(
        "/api/jobs",
        files={"file": ("sample.txt", b"text", "text/plain")},
        data={"ocr_setting": "sometimes"},
    )
    assert response.status_code == 400


def test_unknown_model_cost_is_unavailable(monkeypatch):
    monkeypatch.delenv("GEMINI_INPUT_PRICE_USD_PER_MILLION", raising=False)
    monkeypatch.delenv("GEMINI_OUTPUT_PRICE_USD_PER_MILLION", raising=False)
    result = ModelResult("text", "gemini", "custom-model", 10, 10)
    assert app._estimate_cost(result) is None
