"""
Unit tests for the Semantic Chunking RAG Engine.
Tests cover:
1. Sentence and table splitting logic.
2. Structure-aware markdown header chunking and breadcrumb metadata.
3. Deterministic embedding generation and cosine similarity calculation.
4. Hybrid vector + exact token retrieval and scoring.
5. Grounded synthesis and citation extraction.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from condo_classifier.semantic_rag import (
    EmbeddingProvider,
    HybridSemanticIndex,
    SemanticChunk,
    SemanticChunker,
    SemanticRAGEngine,
    cosine_similarity,
    split_into_sentences,
)


def test_split_into_sentences_preserves_tables():
    markdown = """
Here is rule number one. Here is rule number two!

| Facility | Hours | Fee |
|---|---|---|
| BBQ Pit | 10:00-14:00 | $20 |
| Gym | 06:00-23:00 | Free |

Here is another sentence after table.
"""
    sentences = split_into_sentences(markdown)
    assert len(sentences) >= 3

    # Check that table lines are grouped together
    table_parts = [s for s in sentences if "| BBQ Pit |" in s]
    assert len(table_parts) == 1
    assert "| Gym |" in table_parts[0]
    assert "| Facility |" in table_parts[0]


def test_embedding_provider_deterministic_hash():
    provider = EmbeddingProvider()
    v1 = provider._local_deterministic_embedding("renovation drilling noise")
    v2 = provider._local_deterministic_embedding("renovation drilling noise")
    v3 = provider._local_deterministic_embedding("swimming pool barbeque")

    assert len(v1) == 256
    assert v1 == v2  # Deterministic

    # Cosine similarity between identical texts is ~1.0
    sim_identical = cosine_similarity(v1, v2)
    assert pytest.approx(sim_identical, 0.01) == 1.0

    # Cosine similarity with different topic is significantly lower
    sim_diff = cosine_similarity(v1, v3)
    assert sim_diff < sim_identical


def test_semantic_chunker_structure_and_breadcrumbs():
    text = """# Marina Heights Handbook

## 3. Renovation Rules

### 3.1 Permissible Hours
Renovation is allowed Mon-Fri 09:00 to 18:00. Heavy drilling must stop at 16:00.
Drilling is strictly prohibited on Saturdays, Sundays, and Public Holidays.

### 3.2 Fines and Penalties
The second infraction carries a S$500 penalty deducted from the deposit.
Sunday hacking incurs an immediate S$1,000 fine.
"""
    chunker = SemanticChunker(min_chunk_chars=30)
    chunks = chunker.chunk_markdown_text(text, source_file="test_rules.md")

    assert len(chunks) >= 2
    hours_chunk = next(c for c in chunks if "3.1 Permissible Hours" in c.content or "09:00 to 18:00" in c.content)
    assert "Renovation Rules" in hours_chunk.header_breadcrumb
    assert hours_chunk.source_file == "test_rules.md"

    fines_chunk = next(c for c in chunks if "S$500 penalty" in c.content or "S$1,000 fine" in c.content)
    assert "Fines and Penalties" in fines_chunk.header_breadcrumb


def test_hybrid_semantic_index_retrieval_and_boosting():
    index = HybridSemanticIndex()
    chunks = [
        SemanticChunk(
            chunk_id="chunk_01",
            content="Heavy mechanical drilling is restricted to Monday-Friday 10:00 to 16:00. Strictly prohibited on Saturdays.",
            source_file="handbook.md",
            section="Renovation",
            subsection="Drilling Hours",
        ),
        SemanticChunk(
            chunk_id="chunk_02",
            content="Second noise infraction results in a S$500 fine deducted from deposit. Sunday drilling incurs S$1,000 fine.",
            source_file="handbook.md",
            section="Renovation",
            subsection="Noise Fines",
        ),
        SemanticChunk(
            chunk_id="chunk_03",
            content="Swimming pool is open 07:00 to 22:00 daily. Proper swimwear is required at all times.",
            source_file="handbook.md",
            section="Facilities",
            subsection="Swimming Pool",
        ),
    ]
    index.index_chunks(chunks)

    # Query 1: Drilling on Saturday
    results = index.retrieve("Can my contractor drill walls on Saturday?", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chunk_01"

    # Query 2: Fine amount for noisy renovation
    results_fine = index.retrieve("How much is the fine for noise? Is it $500?", top_k=1)
    assert results_fine[0].chunk.chunk_id == "chunk_02"
    assert "S$500 fine" in results_fine[0].chunk.content


def test_semantic_rag_engine_end_to_end(tmp_path: Path):
    sample_md = tmp_path / "test_condo.md"
    sample_md.write_text(
        """# Sample Estate By-laws

## 1. Noise and Drilling
Heavy drilling is allowed Monday to Friday from 10:00 to 16:00 only under By-law §4.
Unapproved drilling incurs a S$500 penalty.

## 2. Facilities
BBQ Pit 1 booking fee is S$20 with S$50 deposit. Operating hours are 10:00 to 14:00 and 17:00 to 22:00.

## 3. Water Leaks
Under BMSMA Section 101, ceiling leaks are presumed to originate from the upper unit.
""",
        encoding="utf-8",
    )

    engine = SemanticRAGEngine(kb_path=sample_md)
    engine.initialize()
    assert len(engine.index.chunks) >= 3

    # Test query drilling
    res = engine.query("Can we drill on Saturday?")
    assert res.grounded is True
    assert "By-law §4" in " ".join(res.by_laws_cited) or "By-law" in res.answer
    assert "10:00" in res.answer or "16:00" in res.answer or "prohibited" in res.answer.lower()

    # Test query water leak
    res_leak = engine.query("Who pays for ceiling water leak from upper floor?")
    assert res_leak.grounded is True
    assert "BMSMA Section 101" in " ".join(res_leak.by_laws_cited) or "upper unit" in res_leak.answer.lower()


def test_semantic_rag_empty_or_no_match(tmp_path: Path):
    empty_md = tmp_path / "empty.md"
    empty_md.write_text("# Empty Document\n\nNo rules here.", encoding="utf-8")

    engine = SemanticRAGEngine(kb_path=empty_md)
    engine.initialize()
    res = engine.query("What is the rocket launch schedule?")
    assert len(res.retrieved_chunks) >= 1  # Returns top retrieved chunk from doc
    assert res.latency_seconds > 0.0
