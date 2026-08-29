import pytest

from range_utils import parse_page_range


def test_all_pages():
    assert parse_page_range("all", 4) == [1, 2, 3, 4]


def test_disjoint_and_deduplicated_pages():
    assert parse_page_range("1-3, 3, 7, 10-12", 12) == [1, 2, 3, 7, 10, 11, 12]


@pytest.mark.parametrize("value", ["0", "4-2", "1-a", "1-6"])
def test_invalid_ranges(value):
    with pytest.raises(ValueError):
        parse_page_range(value, 5)
