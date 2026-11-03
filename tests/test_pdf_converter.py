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


def test_clean_unicode_and_navigation_artifacts():
    """Verify normalization of soft hyphens, non-breaking spaces, and footer anchors."""
    raw = "BY\xadLAWS of Estate\xa0MCST\xad3656\u200b\n\n**CONTENTS**\n\n## Section 1\n**CONTENTS BACK**"
    cleaned = clean_markdown_artifacts(raw)

    assert "\xad" not in cleaned
    assert "BY-LAWS" in cleaned
    assert "MCST-3656" in cleaned
    assert "\u200b" not in cleaned
    assert "**CONTENTS**" not in cleaned
    assert "**CONTENTS BACK**" not in cleaned


def test_extract_headings_with_html_tags():
    """Verify that HTML tags and styling in headings are cleaned."""
    md = "# <u>ANNEX ­ A</u>\n\n## 1. **_DEFINITIONS_**\n\n### <b>Permitted Hours</b>"
    headings = extract_headings(md)

    assert len(headings) == 3
    assert headings[0] == {"level": 1, "title": "ANNEX - A"}
    assert headings[1] == {"level": 2, "title": "1. DEFINITIONS"}
    assert headings[2] == {"level": 3, "title": "Permitted Hours"}


def test_table_preservation_fidelity():
    """Verify markdown table headers, delimiters, and cells align properly without broken lines."""
    table_md = """## Facility Schedule

| Facility | Opening Hours | Deposit Fee | Max Capacity |
|---|---|---|---|
| Swimming Pool | 7:00 AM - 10:00 PM | $0 | 30 pax |
| BBQ Pit 1 & 2 | 10:00 AM - 10:00 PM | $50 | 12 pax |
| Tennis Court | 8:00 AM - 9:00 PM | $10 | 4 pax |

End of schedule.
"""
    assert count_markdown_tables(table_md) == 1

    # Check downstream LangChain compatibility
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "Section")], strip_headers=False)
    docs = splitter.split_text(table_md)

    assert len(docs) == 1
    content = docs[0].page_content
    assert "| Facility | Opening Hours | Deposit Fee | Max Capacity |" in content
    assert "|---|---|---|---|" in content
    assert "| Swimming Pool | 7:00 AM - 10:00 PM | $0 | 30 pax |" in content


def test_auto_engine_prefers_structural_pymupdf():
    """Verify that auto engine retains font-based headings and accurate page counts."""
    pdf_path = Path("output/slides/Condo_AI_Agent_Telegram_Bot.pdf")
    if not pdf_path.exists():
        pytest.skip("Test slide PDF not found")

    res = convert_pdf_to_markdown(pdf_path, engine="auto")
    assert res.engine_used == "local_pymupdf"
    assert res.page_count == 10
    assert len(res.headings) > 0
    assert len(res.pages) == 10


def test_semantic_rag_integration_with_converted_markdown(tmp_path: Path):
    """Verify that converted markdown integrates seamlessly with SemanticRAGEngine."""
    from condo_classifier.semantic_rag import SemanticRAGEngine

    # Write a clean markdown document simulating PDF conversion output
    sample_md = """---
source_document: test_bylaws.pdf
page_count: 2
table_count: 1
word_count: 85
engine: local_pymupdf
---

# Marine Heights Rules

## By-law §5 - Renovation Hours
Renovation works involving hacking or drilling are strictly restricted:
- Monday to Friday: 9:00 AM to 5:00 PM
- Saturday: 9:00 AM to 1:00 PM
- Sunday and Public Holidays: Strictly Prohibited

| Work Category | Permitted Hours | Fine for Non-Compliance |
|---|---|---|
| Heavy Hacking | 10:00 AM - 4:00 PM | $500 |
| General Works | 9:00 AM - 5:00 PM | $200 |
"""
    kb_file = tmp_path / "test_bylaws.md"
    kb_file.write_text(sample_md, encoding="utf-8")

    engine = SemanticRAGEngine(kb_path=kb_file)
    engine.initialize()

    assert len(engine.index.chunks) >= 1
    # Retrieve query
    matches = engine.index.retrieve("What is the fine for heavy hacking noise violation?")
    assert len(matches) > 0
    best_chunk = matches[0].chunk
    assert "$500" in best_chunk.content
    assert best_chunk.has_table is True

