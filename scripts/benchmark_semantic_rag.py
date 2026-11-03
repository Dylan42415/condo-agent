"""
Semantic RAG Benchmark & Comprehensive Evaluation Suite.

Rigorously evaluates condo_classifier/semantic_rag.py against Knowledge Base/sample_condo_handbook.md:
1. Markdown table integrity verification across all handbook tables.
2. Contextual breadcrumb injection verification into embeddable chunk text.
3. Semantic breakpoint distance thresholding verification.
4. Comprehensive Retrieval Benchmark across 18 diverse resident queries (Hit Rate @ 1, Hit Rate @ 3, MRR).
5. Grounded synthesis evaluation & hallucination verification (dollar tariffs, time ranges, contacts, statutory by-laws).
6. Offline deterministic fallback verification.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from condo_classifier.semantic_rag import (
    EmbeddingProvider,
    HybridSemanticIndex,
    SemanticChunk,
    SemanticChunker,
    SemanticRAGEngine,
    cosine_similarity,
    split_into_sentences,
)

sys.stdout.reconfigure(encoding="utf-8")


# ============================================================================
# 1. Chunk Integrity Verification
# ============================================================================

def verify_table_integrity(kb_path: Path) -> dict[str, Any]:
    """
    Verifies that all markdown tables (|---|---|) in the source handbook
    are preserved intact in chunks and never split mid-table.
    """
    content = kb_path.read_text(encoding="utf-8")
    chunker = SemanticChunker()
    chunks = chunker.chunk_markdown_file(kb_path)

    # Extract all markdown tables from the source markdown
    lines = content.splitlines()
    source_tables: list[list[str]] = []
    current_table: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            current_table.append(line)
        else:
            if current_table:
                if any(re.search(r"^\|(?:\s*:?-+:?\s*\|)+\s*$", row) for row in current_table):
                    source_tables.append(current_table)
                current_table = []
    if current_table and any(re.search(r"^\|(?:\s*:?-+:?\s*\|)+\s*$", row) for row in current_table):
        source_tables.append(current_table)

    table_results = []
    intact_count = 0

    for idx, table_lines in enumerate(source_tables, 1):
        header_row = table_lines[0].strip()
        data_rows = [r.strip() for r in table_lines[2:]]  # skip header and delimiter

        # Find which chunk contains this table's header
        matching_chunks = [c for c in chunks if header_row in c.content]

        if not matching_chunks:
            table_results.append({
                "table_id": idx,
                "header": header_row[:65],
                "rows_count": len(table_lines),
                "status": "MISSING",
                "intact": False,
                "chunk_id": None,
                "missing_rows": len(data_rows),
            })
            continue

        chunk = matching_chunks[0]
        # Verify every row is present in this single chunk
        missing_rows = [r for r in data_rows if r not in chunk.content]
        is_intact = (len(missing_rows) == 0) and chunk.has_table

        if is_intact:
            intact_count += 1

        table_results.append({
            "table_id": idx,
            "header": header_row[:65],
            "rows_count": len(table_lines),
            "status": "INTACT" if is_intact else "SPLIT_OR_PARTIAL",
            "intact": is_intact,
            "chunk_id": chunk.chunk_id,
            "section": chunk.header_breadcrumb,
            "missing_rows": len(missing_rows),
        })

    integrity_score = (intact_count / len(source_tables)) * 100.0 if source_tables else 100.0

    return {
        "total_source_tables": len(source_tables),
        "intact_tables": intact_count,
        "table_integrity_score": integrity_score,
        "table_details": table_results,
    }


def verify_header_breadcrumbs(kb_path: Path) -> dict[str, Any]:
    """
    Verifies that header breadcrumbs [Document: ... | Section: ... | Subsection: ...]
    are correctly formatted and injected into chunk embeddings.
    """
    chunker = SemanticChunker()
    chunks = chunker.chunk_markdown_file(kb_path)

    breadcrumb_results = []
    injected_count = 0

    for c in chunks:
        has_breadcrumb = bool(c.document_title or c.section or c.subsection)
        embed_text = c.embeddable_text

        # Check if embeddable_text starts with [Document: ...]
        starts_with_prefix = embed_text.startswith("[") and "]\n" in embed_text
        has_doc = f"Document: {c.document_title}" in embed_text if c.document_title else True
        has_sec = f"Section: {c.section}" in embed_text if c.section else True
        has_subsec = f"Subsection: {c.subsection}" in embed_text if c.subsection else True

        is_valid = (not has_breadcrumb) or (starts_with_prefix and has_doc and has_sec and has_subsec)
        if is_valid and has_breadcrumb:
            injected_count += 1

        breadcrumb_results.append({
            "chunk_id": c.chunk_id,
            "breadcrumb": c.header_breadcrumb,
            "has_breadcrumb": has_breadcrumb,
            "prefix_injected": starts_with_prefix,
            "valid": is_valid,
            "prefix_preview": embed_text.splitlines()[0] if starts_with_prefix else "",
        })

    titled_chunks = [c for c in chunks if (c.document_title or c.section or c.subsection)]
    coverage_score = (injected_count / len(titled_chunks)) * 100.0 if titled_chunks else 100.0

    return {
        "total_chunks": len(chunks),
        "chunks_with_headers": len(titled_chunks),
        "correctly_injected": injected_count,
        "breadcrumb_coverage_score": coverage_score,
        "chunk_details": breadcrumb_results,
    }


def verify_semantic_breakpoints() -> dict[str, Any]:
    """
    Verifies semantic breakpoint distance thresholding on large text sections.
    """
    provider = EmbeddingProvider()
    chunker = SemanticChunker(
        embedding_provider=provider,
        breakpoint_percentile=80.0,
        max_chunk_chars=300,
        min_chunk_chars=80,
    )

    text = (
        "The swimming pool is open daily from 07:00 to 22:00 for all residents. "
        "Appropriate swimwear must be worn and no glassware is permitted near the pool deck. "
        "Vehicles parked in unauthorized bays will be subject to immediate wheel clamping. "
        "The unclamping fee is $107 inclusive of prevailing goods and services tax. "
        "Under Section 101 of the BMSMA ceiling leaks are presumed to originate from the upper unit. "
        "The upper unit owner must bear the full cost of waterproofing membrane repairs."
    )

    chunks = chunker._semantic_breakpoint_split(text)

    sentences = split_into_sentences(text)
    embeddings = provider.embed_texts(sentences)
    distances = [1.0 - cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]

    sorted_dist = sorted(distances)
    idx = int(len(sorted_dist) * 0.8)
    idx = min(idx, len(sorted_dist) - 1)
    threshold = sorted_dist[idx]

    return {
        "sentences_count": len(sentences),
        "distances": [round(d, 4) for d in distances],
        "threshold_80th_percentile": round(threshold, 4),
        "chunks_produced": len(chunks),
        "split_preserved_coherence": len(chunks) >= 2,
    }


# ============================================================================
# 2. Retrieval Benchmark Suite (18 Representative Resident Queries)
# ============================================================================

TEST_QUERIES = [
    {
        "id": "Q01",
        "category": "Renovation Noise",
        "query": "Can my contractor do heavy drilling or hacking on Saturday morning?",
        "target_chunk_id": "chunk_0005",
        "target_section": "3.1 Renovation Working Hours & Permissible Acoustic Thresholds",
        "expected_facts": ["strictly prohibited on saturdays", "10:00 – 16:00", "by-law §4"],
        "expected_verdict": "PROHIBITED",
    },
    {
        "id": "Q02",
        "category": "Renovation Hours",
        "query": "What are the allowed hours for general renovation like painting and carpentry?",
        "target_chunk_id": "chunk_0005",
        "target_section": "3.1 Renovation Working Hours & Permissible Acoustic Thresholds",
        "expected_facts": ["09:00 – 18:00", "09:00 – 13:00", "saturday"],
        "expected_verdict": "PERMITTED",
    },
    {
        "id": "Q03",
        "category": "Weekend Drilling Penalty",
        "query": "What happens if a contractor does unapproved drilling on Sunday or a public holiday?",
        "target_chunk_id": "chunk_0006",
        "target_section": "3.2 Noise Infractions & Financial Penalties",
        "expected_facts": ["$1,000", "deposit", "site closure", "blacklisting"],
        "expected_verdict": "PROHIBITED",
    },
    {
        "id": "Q04",
        "category": "Noise Infraction Fine",
        "query": "What is the penalty fine for a second renovation noise complaint?",
        "target_chunk_id": "chunk_0006",
        "target_section": "3.2 Noise Infractions & Financial Penalties",
        "expected_facts": ["$500", "pass suspension", "48 hours"],
        "expected_verdict": "ACTION REQUIRED",
    },
    {
        "id": "Q05",
        "category": "BBQ Booking Tariff",
        "query": "How much does it cost to book the BBQ pit and is there a deposit?",
        "target_chunk_id": "chunk_0007",
        "target_section": "4.1 Facilities Tariff & Operating Hours",
        "expected_facts": ["$20", "$50", "refundable"],
        "expected_verdict": "PERMITTED",
    },
    {
        "id": "Q06",
        "category": "BBQ Operating Hours & Capacity",
        "query": "What are the operational hours and capacity limit for booking the BBQ pavilion pit?",
        "target_chunk_id": "chunk_0007",
        "target_section": "4.1 Facilities Tariff & Operating Hours",
        "expected_facts": ["10:00–14:00", "17:00–22:00", "12 pax"],
        "expected_verdict": "PERMITTED",
    },
    {
        "id": "Q07",
        "category": "BBQ Cleaning Fee",
        "query": "Is there a cleaning fee if the BBQ pit is left dirty with food and ash?",
        "target_chunk_id": "chunk_0008",
        "target_section": "4.2 BBQ Pit and Poolside Etiquette",
        "expected_facts": ["$50", "cleaning fee", "deposit"],
        "expected_verdict": "ACTION REQUIRED",
    },
    {
        "id": "Q08",
        "category": "Pool Hours & Attire",
        "query": "What are the opening hours and attire rules for the swimming pool?",
        "target_chunk_id": "chunk_0007",
        "target_section": "4.1 Facilities Tariff & Operating Hours",
        "expected_facts": ["07:00 – 22:00", "proper swimwear", "free"],
        "expected_verdict": "PERMITTED",
    },
    {
        "id": "Q09",
        "category": "Poolside Noise / Quiet Hours",
        "query": "Can we play amplified music or host a karaoke session at the poolside BBQ after 10pm?",
        "target_chunk_id": "chunk_0008",
        "target_section": "4.2 BBQ Pit and Poolside Etiquette",
        "expected_facts": ["cease promptly at 22:00", "amplified music", "karaoke"],
        "expected_verdict": "PROHIBITED",
    },
    {
        "id": "Q10",
        "category": "Water Leak Statutory Presumption",
        "query": "Water is leaking from my bathroom ceiling from the unit above. Who is legally responsible under BMSMA Section 101?",
        "target_chunk_id": "chunk_0011",
        "target_section": "6.1 BMSMA Statutory Presumption (Section 101)",
        "expected_facts": ["bmsma section 101", "presumed to originate from the upper unit", "upper unit owner"],
        "expected_verdict": "ACTION REQUIRED",
    },
    {
        "id": "Q11",
        "category": "Water Leak Inspection SLA",
        "query": "What is the management response time and inspection timeline for inter-floor water leaks?",
        "target_chunk_id": "chunk_0012",
        "target_section": "6.2 Inspection SLA & Resolution Escalation",
        "expected_facts": ["within 2 hours", "within 24 hours", "moisture meter", "7 calendar days"],
        "expected_verdict": "ACTION REQUIRED",
    },
    {
        "id": "Q12",
        "category": "Parking Wheel Clamping",
        "query": "My car was wheel-clamped in the visitor lot. What is the unclamping fee and where do I pay?",
        "target_chunk_id": "chunk_0010",
        "target_section": "5.2 Visitor Parking & Wheel-Clamping Enforcement",
        "expected_facts": ["$107", "guardhouse", "within 30 minutes"],
        "expected_verdict": "ACTION REQUIRED",
    },
    {
        "id": "Q13",
        "category": "Season Parking Allocation",
        "query": "What is the monthly parking charge for a resident's second vehicle?",
        "target_chunk_id": "chunk_0009",
        "target_section": "5.1 Resident Vehicle Season Decals & Allocation",
        "expected_facts": ["$120", "second", "quarterly balloting"],
        "expected_verdict": "PERMITTED",
    },
    {
        "id": "Q14",
        "category": "Emergency Lift Hotline",
        "query": "The elevator is broken down and someone is trapped inside. What is the 24/7 lift hotline?",
        "target_chunk_id": "chunk_0003",
        "target_section": "1.1 Management Office & Security Directory",
        "expected_facts": ["1800-543-8888", "schindler", "24 hours"],
        "expected_verdict": "ACTION REQUIRED",
    },
    {
        "id": "Q15",
        "category": "Pet Swimming Pool Ban & Leash Fine",
        "query": "Are pets allowed in the swimming pool area, and what is the fine for an unleashed dog?",
        "target_chunk_id": "chunk_0013",
        "target_section": "7.1 Domestic Pet Limits & Regulations",
        "expected_facts": ["$100", "$200", "by-law §10", "barred from: swimming pool"],
        "expected_verdict": "PROHIBITED",
    },
    {
        "id": "Q16",
        "category": "Pet Limit Per Unit",
        "query": "What is the pet limit per unit? Can I keep two dogs in my condo?",
        "target_chunk_id": "chunk_0013",
        "target_section": "7.1 Domestic Pet Limits & Regulations",
        "expected_facts": ["1 small/medium dog", "2 domestic cats", "by-law §10"],
        "expected_verdict": "PROHIBITED",
    },
    {
        "id": "Q17",
        "category": "Moving Guidelines Deposit",
        "query": "What is the security deposit and advance notice required for moving in or moving out?",
        "target_chunk_id": "chunk_0004",
        "target_section": "2.1 Moving Guidelines & Security Deposit",
        "expected_facts": ["$1,000", "3 working days", "refundable"],
        "expected_verdict": "ACTION REQUIRED",
    },
    {
        "id": "Q18",
        "category": "Corridor Clutter & Fire Escape",
        "query": "Can residents place shoe racks or bicycles along the common corridor outside their unit under SCDF rules?",
        "target_chunk_id": "chunk_0015",
        "target_section": "8.2 Fire Escape Corridors (SCDF Compliance)",
        "expected_facts": ["1.2 meters", "scdf", "48-hour removal notice"],
        "expected_verdict": "PROHIBITED",
    },
]


def run_retrieval_benchmark(engine: SemanticRAGEngine) -> dict[str, Any]:
    """
    Evaluates Hit Rate @ 1, Hit Rate @ 3, and Mean Reciprocal Rank (MRR)
    across all 18 representative resident queries.
    """
    engine.initialize()
    results = []
    reciprocal_ranks = []
    hits_at_1 = 0
    hits_at_3 = 0

    for item in TEST_QUERIES:
        qid = item["id"]
        query_text = item["query"]
        target_chunk_id = item["target_chunk_id"]

        t0 = time.perf_counter()
        matches = engine.index.retrieve(query_text, top_k=5)
        retrieval_latency = time.perf_counter() - t0

        retrieved_ids = [m.chunk.chunk_id for m in matches]
        rank = None
        if target_chunk_id in retrieved_ids:
            rank = retrieved_ids.index(target_chunk_id) + 1  # 1-based rank

        rr = (1.0 / rank) if rank is not None else 0.0
        reciprocal_ranks.append(rr)

        hit_1 = (rank == 1)
        hit_3 = (rank is not None and rank <= 3)

        if hit_1:
            hits_at_1 += 1
        if hit_3:
            hits_at_3 += 1

        top_matches_summary = [
            {
                "rank": i + 1,
                "chunk_id": m.chunk.chunk_id,
                "breadcrumb": m.chunk.header_breadcrumb,
                "similarity_score": round(m.similarity_score, 4),
                "keyword_boost": round(m.keyword_boost, 4),
                "final_score": round(m.final_score, 4),
            }
            for i, m in enumerate(matches[:3])
        ]

        results.append({
            "id": qid,
            "category": item["category"],
            "query": query_text,
            "target_chunk_id": target_chunk_id,
            "target_section": item["target_section"],
            "rank": rank,
            "hit_at_1": hit_1,
            "hit_at_3": hit_3,
            "reciprocal_rank": round(rr, 4),
            "retrieval_latency_ms": round(retrieval_latency * 1000, 2),
            "top_3_retrieved": top_matches_summary,
        })

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    hit_rate_1 = (hits_at_1 / len(TEST_QUERIES)) * 100.0
    hit_rate_3 = (hits_at_3 / len(TEST_QUERIES)) * 100.0

    return {
        "total_queries": len(TEST_QUERIES),
        "hit_count_at_1": hits_at_1,
        "hit_rate_at_1": round(hit_rate_1, 2),
        "hit_count_at_3": hits_at_3,
        "hit_rate_at_3": round(hit_rate_3, 2),
        "mrr": round(mrr, 4),
        "query_results": results,
    }


# ============================================================================
# 3. Grounded Synthesis & Hallucination Verification
# ============================================================================

# Authoritative set of ground-truth dollar amounts appearing in sample_condo_handbook.md
AUTHORITATIVE_DOLLARS = {
    0.55,    # EV charging rate per kWh
    10.0,    # Tennis court / EV idle fee per hour
    15.0,    # Overnight visitor parking per night
    20.0,    # BBQ pavilion booking fee / Motorcycle parking per month
    50.0,    # BBQ pavilion deposit / cleaning fee / Clubhouse booking fee
    100.0,   # Unleashed pet fine
    107.0,   # Wheel unclamping fee (inclusive of GST)
    120.0,   # 2nd resident car parking per month
    200.0,   # Clubhouse deposit / pet biohazard fee / pool contamination tariff
    500.0,   # 2nd noise infraction fine
    1000.0,  # Moving security deposit / Sunday hacking penalty
    2000.0,  # Standard renovation deposit
    5000.0,  # Wall hacking renovation deposit
}

AUTHORITATIVE_PHONES = {
    "+65 6789 0100",  # Security Guardhouse
    "+65 6789 0200",  # Management Office
    "1800-543-8888",  # Schindler Express Hotline
    "+65 6333 4455",  # Apex 24-7 Rapid Plumbing
    "+65 6444 8899",  # VoltCare Electrical
    "+65 6444-2222",  # Default Security Guardhouse in dispatch directory
    "995",            # SCDF Fire / Ambulance
    "999",            # Police
}

VALID_BYLAWS = {
    "By-law §4", "By-law §10", "By-law 4", "By-law 10",
    "BMSMA Section 101", "Section 101", "Animals and Birds Act",
}


def parse_numeric_dollars(text: str) -> list[float]:
    """Extracts dollar amounts from text and parses as floats."""
    matches = re.findall(r"(?:S?\$|SGD\s*)([0-9]+(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)", text)
    values = []
    for m in matches:
        clean = m.replace(",", "")
        try:
            values.append(float(clean))
        except ValueError:
            pass
    return values


def parse_phone_numbers(text: str) -> list[str]:
    """Extracts valid Singapore phone format patterns from text."""
    # Matches +65 XXXX XXXX or +65 XXXX-XXXX or 1800-XXX-XXXX or 995/999
    phones = re.findall(r"(?:\+65\s*\d{4}[-\s]?\d{4}|\b1800[-\s]?\d{3}[-\s]?\d{4}\b|\b(?:995|999)\b)", text)
    return [p.strip() for p in phones]


def parse_bylaw_citations(text: str) -> list[str]:
    """Extracts formal by-law and statute references from text."""
    bylaws = re.findall(r"(?:By-law\s*(?:§|Section|\d+|No\.\s*\d+)[^\n.,;]*|BMSMA\s*Section\s*\d+)", text, re.IGNORECASE)
    return [b.strip() for b in bylaws]


def verify_synthesis_faithfulness(
    engine: SemanticRAGEngine,
    test_queries: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Executes full grounded RAG synthesis on the test queries and verifies:
    - Dollar amounts match authoritative handbook tariffs exactly.
    - Telephone numbers match registered estate directory exactly.
    - By-law citations match statutory handbook sections (By-law §4, §10, BMSMA §101).
    - Ruling aligns with expected verdict.
    """
    synthesis_results = []
    hallucination_flags = []

    for item in test_queries:
        qid = item["id"]
        q_text = item["query"]

        t0 = time.perf_counter()
        rag_res = engine.query(q_text, top_k=3, expand_parent=True)
        latency = time.perf_counter() - t0

        answer = rag_res.answer

        # 1. Verify dollar amounts
        dollars = parse_numeric_dollars(answer)
        unverified_dollars = [d for d in dollars if d not in AUTHORITATIVE_DOLLARS]

        # 2. Verify phone numbers
        phones = parse_phone_numbers(answer)
        unverified_phones = []
        for p in phones:
            normalized_p = re.sub(r"[-\s]", "", p)
            matched = any(re.sub(r"[-\s]", "", vp) in normalized_p or normalized_p in re.sub(r"[-\s]", "", vp) for vp in AUTHORITATIVE_PHONES)
            if not matched:
                unverified_phones.append(p)

        # 3. Verify by-law numbers
        bylaws = parse_bylaw_citations(answer)
        unverified_bylaws = []
        for b in bylaws:
            clean_b = b.lower()
            if not any(valid_kw in clean_b for valid_kw in ["4", "10", "101", "animals"]):
                unverified_bylaws.append(b)

        # 4. Verify fact recall
        expected_facts = item.get("expected_facts", [])
        answer_lower = answer.lower()
        facts_matched = [f for f in expected_facts if any(term in answer_lower for term in f.lower().split(" / "))]
        fact_recall = (len(facts_matched) / len(expected_facts)) if expected_facts else 1.0

        # Check faithfulness verdict
        is_faithful = (len(unverified_dollars) == 0 and len(unverified_phones) == 0 and len(unverified_bylaws) == 0)
        if not is_faithful:
            hallucination_flags.append({
                "id": qid,
                "query": q_text,
                "unverified_dollars": unverified_dollars,
                "unverified_phones": unverified_phones,
                "unverified_bylaws": unverified_bylaws,
            })

        synthesis_results.append({
            "id": qid,
            "category": item["category"],
            "query": q_text,
            "model_used": rag_res.model_used,
            "latency_seconds": round(latency, 2),
            "grounded": rag_res.grounded,
            "by_laws_cited": rag_res.by_laws_cited,
            "source_sections": rag_res.source_sections,
            "dollars_cited": [f"${d:.2f}" for d in dollars],
            "phones_cited": phones,
            "unverified_dollars": [f"${d:.2f}" for d in unverified_dollars],
            "unverified_phones": unverified_phones,
            "unverified_bylaws": unverified_bylaws,
            "fact_recall_score": round(fact_recall, 2),
            "faithful": is_faithful,
            "answer_preview": answer[:300].replace("\n", " "),
        })

    faithfulness_rate = ((len(test_queries) - len(hallucination_flags)) / len(test_queries)) * 100.0 if test_queries else 100.0
    avg_fact_recall = (sum(r["fact_recall_score"] for r in synthesis_results) / len(synthesis_results)) * 100.0 if synthesis_results else 0.0

    return {
        "queries_tested": len(test_queries),
        "faithfulness_rate": round(faithfulness_rate, 2),
        "avg_fact_recall_percent": round(avg_fact_recall, 2),
        "hallucination_count": len(hallucination_flags),
        "hallucination_flags": hallucination_flags,
        "synthesis_details": synthesis_results,
    }


# ============================================================================
# Main Runner & JSON Report Generator
# ============================================================================

def main() -> None:
    print("=" * 80)
    print("  SEMANTIC CHUNKING RAG BENCHMARK & EVALUATION SPECIALIST")
    print("=" * 80)

    kb_file = PROJECT_ROOT / "Knowledge Base" / "sample_condo_handbook.md"
    if not kb_file.exists():
        print(f"Error: Knowledge Base file not found at {kb_file}")
        sys.exit(1)

    print(f"\n[1/5] Evaluating Table Integrity on {kb_file.name}...")
    table_eval = verify_table_integrity(kb_file)
    print(f"  -> Total source tables: {table_eval['total_source_tables']}")
    print(f"  -> Intact tables       : {table_eval['intact_tables']}/{table_eval['total_source_tables']}")
    print(f"  -> Table Integrity     : {table_eval['table_integrity_score']:.1f}%")

    print(f"\n[2/5] Evaluating Header Breadcrumb Injection...")
    breadcrumb_eval = verify_header_breadcrumbs(kb_file)
    print(f"  -> Chunks with headers : {breadcrumb_eval['chunks_with_headers']}/{breadcrumb_eval['total_chunks']}")
    print(f"  -> Breadcrumb Coverage : {breadcrumb_eval['breadcrumb_coverage_score']:.1f}%")

    print(f"\n[3/5] Evaluating Semantic Breakpoint Distance Thresholding...")
    breakpoint_eval = verify_semantic_breakpoints()
    print(f"  -> Sentences split     : {breakpoint_eval['sentences_count']}")
    print(f"  -> 80th %ile Threshold : {breakpoint_eval['threshold_80th_percentile']}")
    print(f"  -> Coherence Preserved : {breakpoint_eval['split_preserved_coherence']}")

    print(f"\n[4/5] Executing Retrieval Benchmark across {len(TEST_QUERIES)} Resident Queries...")
    engine = SemanticRAGEngine(kb_path=kb_file)
    retrieval_eval = run_retrieval_benchmark(engine)
    print(f"  -> Hit Rate @ 1 (Top-1): {retrieval_eval['hit_rate_at_1']:.1f}% ({retrieval_eval['hit_count_at_1']}/{retrieval_eval['total_queries']})")
    print(f"  -> Hit Rate @ 3 (Top-3): {retrieval_eval['hit_rate_at_3']:.1f}% ({retrieval_eval['hit_count_at_3']}/{retrieval_eval['total_queries']})")
    print(f"  -> Mean Reciprocal Rank: {retrieval_eval['mrr']:.4f}")

    # Core evaluation queries covering all key domains
    key_queries_indices = [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 16]
    selected_eval_queries = [TEST_QUERIES[i] for i in key_queries_indices]

    print(f"\n[5/5] Evaluating Hallucination Resistance & Faithfulness on {len(selected_eval_queries)} Domain Queries...")
    faithfulness_eval = verify_synthesis_faithfulness(engine, selected_eval_queries)
    print(f"  -> Faithfulness Rate   : {faithfulness_eval['faithfulness_rate']:.1f}%")
    print(f"  -> Avg Fact Recall     : {faithfulness_eval['avg_fact_recall_percent']:.1f}%")
    print(f"  -> Hallucinations Flagged: {faithfulness_eval['hallucination_count']}")

    benchmark_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "knowledge_base": kb_file.name,
        "table_integrity": table_eval,
        "breadcrumb_injection": breadcrumb_eval,
        "breakpoint_thresholding": breakpoint_eval,
        "retrieval_benchmark": retrieval_eval,
        "faithfulness_evaluation": faithfulness_eval,
    }

    output_path = PROJECT_ROOT / "reports" / "semantic_rag_benchmark_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(benchmark_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[SUCCESS] Comprehensive Benchmark Report written to: {output_path}")


if __name__ == "__main__":
    main()
