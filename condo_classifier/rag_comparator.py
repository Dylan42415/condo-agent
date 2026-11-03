"""
RAG Architecture Comparator: Semantic Chunking RAG vs. Hybrid Graph RAG.

Enables side-by-side evaluation and benchmark analysis to compare:
1. Regular Semantic Chunking RAG (LangChain Markdown Splitter + Embeddings + Hybrid Retrieval)
2. Hybrid Knowledge Graph RAG (Multi-hop BFS across Wikilinks, Nodes, and Riser Stacks)

Demonstrates to stakeholders and lecturers exactly where each approach excels:
- Semantic Chunking RAG: Ideal for flat policy retrieval, tariff lookups, by-law compliance, and resident handbooks.
- Hybrid Graph RAG: Ideal for multi-hop topological queries (e.g. Stack 04 -> Valve V-T2-08B -> Unit #08-04).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from condo_classifier.graph_rag import EstateVaultGraph, get_estate_vault
from condo_classifier.semantic_rag import SemanticRAGEngine, SemanticRAGResult


@dataclass
class DualRAGComparison:
    query: str
    semantic_rag: dict[str, Any]
    graph_rag: dict[str, Any]
    summary_verdict: str
    recommended_engine: str


class RAGComparator:
    """Executes identical user queries against both RAG architectures and generates side-by-side analytics."""

    def __init__(self, kb_path: Path | str | None = None, vault_dir: Path | str | None = None) -> None:
        self.semantic_engine = SemanticRAGEngine(kb_path=kb_path)
        self.vault_graph = EstateVaultGraph(vault_dir=vault_dir)

    def compare(self, query: str) -> DualRAGComparison:
        """Runs both RAG engines on the same query and compares results."""
        # 1. Run Semantic Chunking RAG
        t0 = time.perf_counter()
        sem_res: SemanticRAGResult = self.semantic_engine.query(query)
        sem_time = sem_res.latency_seconds

        # 2. Run Hybrid Graph RAG
        t1 = time.perf_counter()
        graph_res = self.vault_graph.query(query)
        graph_time = time.perf_counter() - t1

        # 3. Determine recommended engine based on query topology
        q_lower = query.lower()
        is_topological_multihop = any(
            token in q_lower for token in ["valve", "stack", "riser", "pipe", "neighbor", "below", "above", "#08-04", "#09-04"]
        )

        if is_topological_multihop:
            recommended = "Hybrid Graph RAG"
            verdict = (
                "Query involves topological/relational infrastructure (units, stacks, isolation valves). "
                "Hybrid Graph RAG traverses BFS relations to identify physical dependencies."
            )
        else:
            recommended = "Semantic Chunking RAG"
            verdict = (
                "Query involves estate by-laws, hours, fines, or facility policies. "
                "Semantic Chunking RAG retrieves exact policy sections with minimal complexity."
            )

        return DualRAGComparison(
            query=query,
            semantic_rag={
                "answer": sem_res.answer,
                "latency_seconds": round(sem_time, 3),
                "citations": sem_res.by_laws_cited,
                "chunks_retrieved": len(sem_res.retrieved_chunks),
                "top_sources": sem_res.source_sections[:2],
                "grounded": sem_res.grounded,
                "model_used": sem_res.model_used,
            },
            graph_rag={
                "answer": graph_res.answer,
                "latency_seconds": round(graph_time, 3),
                "citations": graph_res.by_laws_cited,
                "seed_nodes": graph_res.seed_nodes,
                "expanded_nodes": graph_res.expanded_nodes,
                "edges_traversed": len(graph_res.edges),
                "grounded": graph_res.grounded,
            },
            summary_verdict=verdict,
            recommended_engine=recommended,
        )


def compare_query(query: str) -> DualRAGComparison:
    """Convenience functional interface."""
    comparator = RAGComparator()
    return comparator.compare(query)
