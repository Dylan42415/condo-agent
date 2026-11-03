"""
Unit tests for the RAG Architecture Comparator.
"""

from __future__ import annotations

import pytest
from condo_classifier.rag_comparator import RAGComparator, compare_query


def test_rag_comparator_policy_query():
    comparator = RAGComparator()
    comp = comparator.compare("Can my contractor do drilling works on Saturday afternoon?")

    assert comp.recommended_engine == "Semantic Chunking RAG"
    assert "by-laws" in comp.summary_verdict.lower() or "policy" in comp.summary_verdict.lower()
    assert comp.semantic_rag["grounded"] is True
    assert len(comp.semantic_rag["citations"]) > 0 or "By-law" in comp.semantic_rag["answer"]


def test_rag_comparator_topological_query():
    comparator = RAGComparator()
    comp = comparator.compare("Which isolation valve shuts off water for Unit #08-04 in Stack 04?")

    assert comp.recommended_engine == "Hybrid Graph RAG"
    assert "topological" in comp.summary_verdict.lower() or "relational" in comp.summary_verdict.lower()
    assert comp.graph_rag["grounded"] is True
    assert len(comp.graph_rag["expanded_nodes"]) > 0
