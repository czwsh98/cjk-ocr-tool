import shutil
import time

import pymupdf
from fastapi.testclient import TestClient

import app


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


def test_native_pdf_selected_page_without_api(tmp_path):
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
            "ocr_setting": "auto",
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
