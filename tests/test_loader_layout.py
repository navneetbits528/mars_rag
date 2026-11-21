from types import SimpleNamespace

from marsrag.ingestion.loader import _layout_aware_text


def b(x0, y0, x1, y1, text):
    return (x0, y0, x1, y1, text, 0, 0)


class FakePage:
    def __init__(self, blocks, width=900, height=540):
        self._blocks = blocks
        self.rect = SimpleNamespace(width=width, height=height)

    def get_text(self, kind, sort=False):
        if kind == "blocks":
            return self._blocks
        return "FALLBACK"


def test_three_columns_are_read_left_to_right_without_interleaving():
    page = FakePage([
        b(35, 100, 190, 120, "LEFT article body paragraph with enough words to be an anchor block."),
        b(35, 130, 190, 150, "LEFT second paragraph with enough words to remain in its column."),
        b(260, 100, 570, 120, "MIDDLE article or summary body paragraph with enough words to anchor."),
        b(260, 130, 570, 150, "MIDDLE second paragraph with enough words to remain in its column."),
        b(630, 100, 790, 120, "RIGHT article body paragraph with enough words to be an anchor block."),
        b(630, 130, 790, 150, "RIGHT second paragraph with enough words to remain in its column."),
    ])
    result = _layout_aware_text(page)
    assert result.index("LEFT article") < result.index("LEFT second")
    assert result.index("LEFT second") < result.index("MIDDLE article")
    assert result.index("MIDDLE second") < result.index("RIGHT article")


def test_footer_is_not_moved_before_page_body():
    page = FakePage([
        b(35, 100, 190, 120, "LEFT article body paragraph with enough words to be an anchor block."),
        b(35, 130, 190, 150, "LEFT second paragraph with enough words to remain in its column."),
        b(630, 100, 790, 120, "RIGHT article body paragraph with enough words to be an anchor block."),
        b(630, 130, 790, 150, "RIGHT second paragraph with enough words to remain in its column."),
        b(150, 480, 630, 500, "Read the White Papers Here: https://example.com"),
    ])
    result = _layout_aware_text(page)
    assert result.index("LEFT article") < result.index("Read the White Papers Here")


def test_single_column_falls_back_to_normal_extraction():
    page = FakePage([
        b(30, 10, 400, 20, "First paragraph"),
        b(30, 30, 400, 40, "Second paragraph"),
        b(30, 50, 400, 60, "Third paragraph"),
        b(30, 70, 400, 80, "Fourth paragraph"),
    ])
    assert _layout_aware_text(page) == "FALLBACK"
