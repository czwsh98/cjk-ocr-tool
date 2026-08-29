from ocr_client import postprocess_translation_ready_text


def test_cjk_soft_wrap_joins_without_space():
    assert postprocess_translation_ready_text("これは\n文章です。") == "これは文章です。\n"


def test_latin_soft_wrap_keeps_space():
    assert postprocess_translation_ready_text("hello\nworld.") == "hello world.\n"


def test_latin_hyphenation_joins_word():
    assert postprocess_translation_ready_text("trans-\nlation.") == "translation.\n"


def test_headings_and_lists_are_not_joined():
    source = "第一章 導論\n本章介紹研究背景。\n- 第一點\n- 第二點"
    assert postprocess_translation_ready_text(source) == source + "\n"


def test_fullwidth_punctuation_is_cjk_boundary():
    assert postprocess_translation_ready_text("這，\n下一行。") == "這，下一行。\n"
