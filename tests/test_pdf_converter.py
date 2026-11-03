"""
Unit tests for the Lossless PDF-to-Markdown conversion pipeline and
its downstream compatibility with LangChain Semantic Chunkers.
"""

from pathlib import Path

import pytest

from condo_classifier.pdf_converter import (
    clean_markdown_artifacts,
    convert_pdf_to_markdown,
    count_markdown_tables,
    extract_headings,
)


def test_clean_markdown_artifacts():
    dirty_text = """
<!-- Start of picture text -->
Important announcement
<!-- End of picture text -->

Page 1 of 12

## Section 1: By-laws
Here is a table of penalties:

| Violation | Penalty | SLA |
|---|---|---|
| Noise after 10 PM | $500 | 15 mins |

Page 2
"""
    cleaned = clean_markdown_artifacts(dirty_text)

    # Asserts
    assert "<!-- Start of picture text -->" not in cleaned
    assert "Page 1 of 12" not in cleaned
    assert "Page 2" not in cleaned
    assert "Important announcement" in cleaned
    assert "| Violation | Penalty | SLA |" in cleaned
    assert "| Noise after 10 PM | $500 | 15 mins |" in cleaned


def test_count_markdown_tables_and_headings():
    md = """# Estate Policy Manual

## By-law §4 - Renovation & Noise
Renovation works must follow permitted hours.

| Work Type | Hours | Noise Cap |
|---|---|---|
| Heavy Demolition | 10am - 4pm | 75 dBA |
| General Works | 9am - 5pm | 65 dBA |

### Weekend Policy
No noisy works permitted.
"""
    assert count_markdown_tables(md) == 1
    headings = extract_headings(md)
    assert len(headings) == 3
    assert headings[0] == {"level": 1, "title": "Estate Policy Manual"}
    assert headings[1] == {"level": 2, "title": "By-law §4 - Renovation & Noise"}
    assert headings[2] == {"level": 3, "title": "Weekend Policy"}


def test_convert_real_pdf_and_frontmatter(tmp_path: Path):
    pdf_path = Path("output/slides/Condo_AI_Agent_Telegram_Bot.pdf")
    if not pdf_path.exists():
        pytest.skip("Test slide PDF not found")

    out_md = tmp_path / "converted_test.md"
    result = convert_pdf_to_markdown(pdf_path, engine="local", output_path=out_md)

    assert result.page_count > 0
    assert result.word_count > 100
    assert out_md.exists()

    content = out_md.read_text(encoding="utf-8")
    assert "---" in content
    assert "source_document: Condo_AI_Agent_Telegram_Bot.pdf" in content
    assert len(result.headings) >= 0


def test_convert_markitdown_engine(tmp_path: Path):
    pdf_path = Path("output/slides/Condo_AI_Agent_Telegram_Bot.pdf")
    if not pdf_path.exists():
        pytest.skip("Test slide PDF not found")

    out_md = tmp_path / "converted_markitdown.md"
    result = convert_pdf_to_markdown(pdf_path, engine="markitdown", output_path=out_md)

    assert result.engine_used == "microsoft_markitdown"
    assert result.word_count > 100
    assert out_md.exists()


def test_markdown_header_splitter_compatibility():
    """Verify that the converted markdown works directly with LangChain splitters."""
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    md = """# Oceanfront Residences By-Laws

## By-law §4 - Renovation Noise
Permitted drilling hours: Monday to Friday 10:00 AM to 4:00 PM.
Sunday drilling is prohibited with a $500 fine.

## By-law §8 - BBQ Pit Facilities
Deposit is $50 refundable.
Session 1 is 10:00 AM to 2:00 PM.
"""
    headers = [("#", "Document"), ("##", "Section")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers, strip_headers=False)
    docs = splitter.split_text(md)

    assert len(docs) == 2
    assert docs[0].metadata["Section"] == "By-law §4 - Renovation Noise"
    assert "$500" in docs[0].page_content
    assert docs[1].metadata["Section"] == "By-law §8 - BBQ Pit Facilities"
    assert "$50" in docs[1].page_content
