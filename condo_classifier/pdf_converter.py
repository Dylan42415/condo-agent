"""
Lossless PDF to Structured Markdown Conversion Pipeline.
Converts PDFs (by-laws, regulations, handbooks, reports) into high-fidelity
GitHub-Flavored Markdown (.md) preserving 100% of text, tables, numbers, and headers.
Supports:
1. Fast local deterministic parsing via PyMuPDF4LLM.
2. Multimodal Gemini Vision fallback for scanned documents or complex diagrams.
3. Layout artifact clean-up (page numbers, headers/footers) while preserving tables.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class PDFConversionResult:
    source_path: str
    markdown_text: str
    page_count: int
    table_count: int
    word_count: int
    headings: list[dict[str, Any]] = field(default_factory=list)
    pages: list[dict[str, Any]] = field(default_factory=list)
    engine_used: str = "local"


def clean_markdown_artifacts(text: str) -> str:
    """
    Cleans layout noise without dropping any actual content:
    - Normalizes soft hyphens, non-breaking spaces, and zero-width artifacts
    - Removes picture text wrappers (<!-- Start of picture text -->)
    - Removes navigation anchor links (e.g. '**CONTENTS**', '**CONTENTS BACK**')
    - Removes running page numbers (e.g. 'Page 12 of 50', 'Page 12', '--- 12 ---')
    - Cleans duplicate blank lines
    - Preserves all tables (| ... |), headers (#, ##), and inline formatting
    """
    # Normalize unicode noise
    text = text.replace("\xad", "-")
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "").replace("\ufeff", "")

    # Remove PyMuPDF picture text tags
    text = re.sub(r"<!--\s*Start of picture text\s*-->", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<!--\s*End of picture text\s*-->", "", text, flags=re.IGNORECASE)

    # Remove repeated navigation anchors (e.g. handbook bottom links)
    text = re.sub(r"^\s*\*\*(?:CONTENTS|CONTENTS\s+BACK)\*\*\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)

    lines = text.splitlines()
    cleaned_lines: list[str] = []

    # Regex for running page numbers
    page_num_pattern = re.compile(
        r"^\s*(?:page\s+\d+(?:\s*(?:of|\/)\s*\d+)?|\d+\s*(?:of|\/)\s*\d+|[-–—]\s*\d+\s*[-–—]|\d+)\s*$",
        re.IGNORECASE,
    )

    for line in lines:
        stripped = line.strip()
        # Skip pure running page number lines unless they are part of a table row
        if "|" not in line and page_num_pattern.match(stripped):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    # Collapse 3+ consecutive newlines into 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_headings(markdown_text: str) -> list[dict[str, Any]]:
    """Extracts all markdown headings (#, ##, ###) for TOC and section breadcrumbs."""
    headings = []
    for line in markdown_text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            # Clean HTML tags (e.g. <u>, </b>)
            clean_title = re.sub(r"<[^>]+>", "", title)
            # Clean markdown bold/italic/strikethrough and soft hyphens
            clean_title = re.sub(r"[\*_~]", "", clean_title).replace("\xad", "-").strip()
            headings.append({"level": level, "title": clean_title})
    return headings


def count_markdown_tables(markdown_text: str) -> int:
    """Counts the number of GFM tables in markdown text."""
    # A GFM table has a separator row like |---|---|
    table_separators = re.findall(r"^\|(?:\s*:?-+:?\s*\|)+\s*$", markdown_text, re.MULTILINE)
    return len(table_separators)


def convert_microsoft_markitdown(file_path: Path) -> PDFConversionResult:
    """Conversion using Microsoft's MarkItDown library (supports PDF, Office, audio, and images)."""
    from markitdown import MarkItDown

    md = MarkItDown()
    res = md.convert(str(file_path))
    raw_text = res.text_content or ""
    cleaned = clean_markdown_artifacts(raw_text)
    table_count = count_markdown_tables(cleaned)
    headings = extract_headings(cleaned)
    word_count = len(cleaned.split())

    page_count = 1
    try:
        import pymupdf as fitz
        doc = fitz.open(str(file_path))
        page_count = len(doc)
        doc.close()
    except Exception:
        pass

    return PDFConversionResult(
        source_path=str(file_path),
        markdown_text=cleaned,
        page_count=page_count,
        table_count=table_count,
        word_count=word_count,
        headings=headings,
        pages=[{"page_number": 1, "text": cleaned}],
        engine_used="microsoft_markitdown",
    )


def convert_local_pymupdf(pdf_path: Path) -> PDFConversionResult:
    """Fast, local deterministic conversion via pymupdf4llm."""
    from collections import Counter
    import pymupdf as fitz
    import pymupdf4llm

    doc = fitz.open(str(pdf_path))
    page_count = len(doc)
    doc.close()

    # Extract pages with page chunks
    page_data = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    pages: list[dict[str, Any]] = []
    full_text_parts: list[str] = []

    # Detect repeated running headers across pages (appearing on >= 3 pages or >= 30% of pages)
    repeated_headers: set[str] = set()
    if len(page_data) >= 3:
        first_lines: list[str] = []
        for item in page_data:
            lines = [ln.strip() for ln in item.get("text", "").splitlines() if ln.strip()]
            if lines:
                first_lines.append(lines[0])
        counts = Counter(first_lines)
        repeated_headers = {line for line, cnt in counts.items() if cnt >= 3 and len(line) > 5}

    for idx, item in enumerate(page_data):
        p_num = item.get("metadata", {}).get("page", 0) + 1
        raw_text = item.get("text", "")

        # Remove repeated running header on subsequent pages (keep on page 1 as document title)
        if idx > 0 and repeated_headers:
            lines = raw_text.splitlines()
            filtered_lines = [ln for ln in lines if ln.strip() not in repeated_headers]
            raw_text = "\n".join(filtered_lines)

        cleaned_p_text = clean_markdown_artifacts(raw_text)
        pages.append({"page_number": p_num, "text": cleaned_p_text})
        full_text_parts.append(cleaned_p_text)

    full_markdown = "\n\n".join(full_text_parts)
    full_markdown = clean_markdown_artifacts(full_markdown)

    table_count = count_markdown_tables(full_markdown)
    headings = extract_headings(full_markdown)
    word_count = len(full_markdown.split())

    return PDFConversionResult(
        source_path=str(pdf_path),
        markdown_text=full_markdown,
        page_count=page_count,
        table_count=table_count,
        word_count=word_count,
        headings=headings,
        pages=pages,
        engine_used="local_pymupdf",
    )


def convert_gemini_vision(pdf_path: Path) -> PDFConversionResult:
    """Multimodal Gemini Vision conversion for scanned/complex PDFs."""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required for gemini vision conversion engine.")

    page_count = 1
    try:
        import pymupdf as fitz
        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        doc.close()
    except Exception:
        pass

    client = genai.Client(api_key=api_key)
    pdf_bytes = pdf_path.read_bytes()

    prompt = (
        "You are an expert document transcription engine. Convert the attached PDF into "
        "high-fidelity GitHub-Flavored Markdown (GFM).\n\n"
        "STRICT PRESERVATION RULES:\n"
        "1. Complete Fidelity: Transcribe all text, numbers, clauses, and units verbatim. "
        "Do NOT summarize, truncate, or omit any section.\n"
        "2. Tables: Convert all visual tables into GFM Markdown tables (| Col 1 | Col 2 |).\n"
        "3. Document Structure: Use # for main titles, ## for sections, ### for subsections.\n"
        "4. Exact Values: Maintain exact figures, fees, percentages, and currencies "
        "($50, 75 dBA, etc.).\n"
        "5. Omit Only: Running page numbers and repeated header/footer margins.\n"
    )

    models_to_try = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-flash-latest"]
    res_text = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                    prompt,
                ],
            )
            if response.text and len(response.text.strip()) > 50:
                res_text = response.text.strip()
                break
        except Exception as e:
            logger.warning(f"Gemini model {model_name} failed during PDF conversion: {e}")
            continue

    if not res_text:
        raise RuntimeError("All Gemini models failed to convert PDF.")

    # Strip markdown backticks wrapper if returned
    if res_text.startswith("```markdown"):
        res_text = res_text[11:]
    elif res_text.startswith("```"):
        res_text = res_text[3:]
    if res_text.endswith("```"):
        res_text = res_text[:-3]

    cleaned_markdown = clean_markdown_artifacts(res_text.strip())
    table_count = count_markdown_tables(cleaned_markdown)
    headings = extract_headings(cleaned_markdown)
    word_count = len(cleaned_markdown.split())

    return PDFConversionResult(
        source_path=str(pdf_path),
        markdown_text=cleaned_markdown,
        page_count=page_count,
        table_count=table_count,
        word_count=word_count,
        headings=headings,
        pages=[{"page_number": 1, "text": cleaned_markdown}],
        engine_used="gemini_vision",
    )


def convert_pdf_to_markdown(
    pdf_path: str | Path,
    engine: str = "auto",
    output_path: str | Path | None = None,
    add_frontmatter: bool = True,
) -> PDFConversionResult:
    """
    Main entry point for lossless PDF-to-Markdown conversion.

    Args:
        pdf_path: Path to the PDF file.
        engine: 'auto', 'local', 'markitdown', or 'gemini'.
            'auto' uses PyMuPDF4LLM for structural headings and tables,
            falling back to Gemini Vision OCR if scanned (< 10 words/page),
            or MarkItDown if local PyMuPDF parsing fails.
        output_path: Optional path to save the resulting .md file.
        add_frontmatter: If True, adds YAML metadata header to saved markdown file.

    Returns:
        PDFConversionResult with clean markdown and metadata.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    result: PDFConversionResult

    if engine == "gemini":
        result = convert_gemini_vision(pdf_file)
    elif engine == "markitdown":
        result = convert_microsoft_markitdown(pdf_file)
    elif engine == "local":
        result = convert_local_pymupdf(pdf_file)
    else:  # "auto"
        try:
            # Default to PyMuPDF4LLM for high-fidelity structural PDF parsing (headings, tables, page chunks)
            try:
                result = convert_local_pymupdf(pdf_file)
            except Exception as pe:
                logger.info(f"PyMuPDF failed ({pe}), falling back to Microsoft MarkItDown...")
                result = convert_microsoft_markitdown(pdf_file)

            # Low word density (< 10 words/page or 0 words) indicates scanned document / images
            avg_words_per_page = result.word_count / max(1, result.page_count)
            if (avg_words_per_page < 10 or result.word_count == 0) and os.getenv("GEMINI_API_KEY"):
                logger.info(
                    f"Low text density ({avg_words_per_page:.1f} words/page) detected. "
                    "Switching to Gemini Vision OCR..."
                )
                try:
                    gemini_result = convert_gemini_vision(pdf_file)
                    if gemini_result.word_count > result.word_count:
                        result = gemini_result
                except Exception as ex:
                    logger.warning(f"Gemini Vision fallback failed, keeping local result: {ex}")
        except Exception as e:
            logger.warning(f"Local conversion failed ({e}), falling back to Gemini Vision...")
            result = convert_gemini_vision(pdf_file)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        final_output = ""
        if add_frontmatter:
            final_output += (
                f"---\n"
                f"source_document: {pdf_file.name}\n"
                f"page_count: {result.page_count}\n"
                f"table_count: {result.table_count}\n"
                f"word_count: {result.word_count}\n"
                f"engine: {result.engine_used}\n"
                f"---\n\n"
            )
        final_output += result.markdown_text
        out.write_text(final_output, encoding="utf-8")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lossless PDF to Structured Markdown conversion for RAG pipelines."
    )
    parser.add_argument("pdf_path", help="Path to input PDF file")
    parser.add_argument("-o", "--output", help="Path to output markdown file (.md)")
    parser.add_argument(
        "--engine",
        choices=["auto", "markitdown", "local", "gemini"],
        default="auto",
        help="Conversion engine (default: auto)",
    )
    args = parser.parse_args()

    out_file = args.output
    if not out_file:
        out_file = Path(args.pdf_path).with_suffix(".md")

    sys.stdout.reconfigure(encoding="utf-8")
    print(f"Converting '{args.pdf_path}' using engine '{args.engine}'...")
    res = convert_pdf_to_markdown(args.pdf_path, engine=args.engine, output_path=out_file)
    print(
        f"Done! Converted {res.page_count} pages, {res.table_count} tables, "
        f"{res.word_count} words -> Saved to '{out_file}' (Engine: {res.engine_used})"
    )


if __name__ == "__main__":
    main()
