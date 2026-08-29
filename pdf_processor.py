from __future__ import annotations

from pathlib import Path

import pymupdf
from pdf2image import convert_from_path


def pdf_page_count(pdf_path: Path) -> int:
    with pymupdf.open(str(pdf_path)) as doc:
        return doc.page_count


def extract_pdf_page_text(pdf_path: Path, page_number: int) -> str:
    """Extract a single one-based page with PyMuPDF."""
    with pymupdf.open(str(pdf_path)) as doc:
        if page_number < 1 or page_number > doc.page_count:
            raise ValueError(f"Page {page_number} is outside 1-{doc.page_count}.")
        return doc.load_page(page_number - 1).get_text()


def pdf_page_to_png(
    pdf_path: Path,
    out_dir: Path,
    page_number: int,
    *,
    dpi: int = 240,
) -> Path:
    """Rasterize one page at a time so large PDFs do not exhaust RAM."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = convert_from_path(
        pdf_path.as_posix(),
        dpi=dpi,
        fmt="jpeg",
        first_page=page_number,
        last_page=page_number,
        thread_count=1,
        output_folder=out_dir.as_posix(),
        paths_only=True,
        jpegopt={"quality": 90, "optimize": True},
    )
    if not paths:
        raise RuntimeError(f"Could not rasterize page {page_number}.")
    return Path(paths[0])
