from ocr_client import postprocess_translation_ready_text


def test_cjk_soft_wrap_joins_without_space():
    assert postprocess_translation_ready_text("これは\n文章です。") == "これは文章です。\n"


def test_latin_soft_wrap_keeps_space():
    assert postprocess_translation_ready_text("hello\nworld.") == "hello world.\n"


def test_latin_hyphenation_joins_word():
    assert postprocess_translation_ready_text("trans-\nlation.") == "translation.\n"
