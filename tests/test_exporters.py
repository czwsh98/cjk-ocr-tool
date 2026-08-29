from pathlib import Path

from docx import Document

from exporters import PageResult, write_docx, write_markdown


PAGES = [
    PageResult(page=2, source="日本語の原文。", translation="繁體中文譯文。"),
    PageResult(page=4, source="第二段落。", translation="Second paragraph."),
]


def test_markdown_bilingual(tmp_path: Path):
    output = tmp_path / "sample.md"
    write_markdown(
        output,
        filename="sample.pdf",
        pages=PAGES,
        include_source=True,
        include_translation=True,
        target_label="Traditional Chinese",
    )
    text = output.read_text(encoding="utf-8")
    assert "# sample" in text
    assert "## Page 2" in text
    assert "日本語の原文。" in text
    assert "繁體中文譯文。" in text


def test_docx_translation_only(tmp_path: Path):
    output = tmp_path / "sample.docx"
    write_docx(
        output,
        filename="sample.pdf",
        pages=PAGES,
        include_source=False,
        include_translation=True,
        target_label="English",
    )
    document = Document(output)
    text = "\n".join(p.text for p in document.paragraphs)
    assert "日本語の原文。" not in text
    assert "繁體中文譯文。" in text
    assert "Translation — English" in text
