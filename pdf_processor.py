from __future__ import annotations

from pathlib import Path

from pdf2image import convert_from_path


def pdf_to_png_pages(
    pdf_path: Path,
    out_dir: Path,
    *,
    dpi: int = 300,
    fmt: str = "png",
) -> list[Path]:
    """
    Rasterize every PDF page to a high-resolution image.

    Poppler must be installed and discoverable by pdf2image
    (e.g. macOS: brew install poppler).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = convert_from_path(
        pdf_path.as_posix(),
        dpi=dpi,
        fmt=fmt,
        thread_count=4,
        output_folder=out_dir.as_posix(),
        paths_only=True,
    )
    return [Path(p) for p in paths]
