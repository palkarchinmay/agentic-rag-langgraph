"""Region text that only repeats the page text is not indexed twice."""
import pytest

parser = pytest.importorskip("app.ingestion.parser")

PAGE = ("Takeaways. SRI account for about 17% of total dollars under management. "
        "Impact investments have a variety of return expectations and risk tolerances.")


def test_sidebar_copy_is_detected():
    sidebar = "[Image Description]\nSRI account for about 17% of total dollars under management."
    assert parser._repeats_page_text(sidebar, PAGE)


def test_new_information_is_kept():
    chart = "[Image Description]\nPhoto of two analysts presenting a quarterly dashboard in an office."
    assert not parser._repeats_page_text(chart, PAGE)
    assert not parser._repeats_page_text("", PAGE)


def test_containment_is_relative_to_the_region():
    # a short region fully inside a long page counts as a repeat, even though
    # the old _overlap_ratio (divides by the larger set) would call it new
    region = "return expectations and risk tolerances"
    assert parser._containment(region, PAGE * 20) == 1.0
    assert parser._overlap_ratio(PAGE * 20, region) < 0.4


def test_structured_descriptions_are_always_kept():
    page = "Access to Finance Access to Basic Services Green Technology 85 58 21 Government Service providers"
    table = "[Image Description]\n| Theme | Funds |\n| Access to Finance | 85 |\n| Access to Basic Services | 58 |"
    flow = "[Image Description]\nGovernment -> Service providers"
    assert not parser._repeats_page_text(table, page)
    assert not parser._repeats_page_text(flow, page)
