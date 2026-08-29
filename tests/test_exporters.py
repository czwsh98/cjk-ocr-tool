from pathlib import Path

from docx import Document

from exporters import PageResult, write_docx, write_markdown


PAGES = [
    PageResult(page=2, source="日本語の原文。", translation="简体中文译文。"),
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
        target_label="Simplified Chinese",
        document_translation="全篇合并译文。",
    )
    text = output.read_text(encoding="utf-8")
    assert "# sample" in text
    assert "## Page 2" in text
    assert "日本語の原文。" in text
    assert "全篇合并译文。" in text
    assert text.index("日本語の原文。") < text.index("第二段落。") < text.index("## Translation")


def test_docx_translation_only(tmp_path: Path):
    output = tmp_path / "sample.docx"
    write_docx(
        output,
        filename="sample.pdf",
        pages=PAGES,
        include_source=False,
        include_translation=True,
        target_label="English",
        document_translation="Complete English translation.",
    )
    document = Document(output)
    text = "\n".join(p.text for p in document.paragraphs)
    assert "日本語の原文。" not in text
    assert "简体中文译文。" not in text
    assert "Complete English translation." in text
    assert "Translation — English" in text
