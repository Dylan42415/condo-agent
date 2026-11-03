import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from condo_classifier.semantic_rag import (
    SemanticChunker,
    HybridSemanticIndex,
    SemanticRAGEngine,
    DEFAULT_KB_FILE,
)

def run_evaluation():
    print(f"Loading and analyzing Knowledge Base: {DEFAULT_KB_FILE}")
    chunker = SemanticChunker()
    chunks = chunker.chunk_markdown_file(DEFAULT_KB_FILE)

    sizes_chars = [len(c.content) for c in chunks]
    sizes_tokens = [c.token_count for c in chunks]
    
    print("\n" + "="*80)
    print("INGESTION PIPELINE OBSERVATIONS")
    print("="*80)
    print(f"Total chunks generated: {len(chunks)}")
    print(f"Average chunk size: {sum(sizes_chars)/len(sizes_chars):.1f} chars ({sum(sizes_tokens)/len(sizes_tokens):.1f} tokens)")
    
    min_chunk = min(chunks, key=lambda c: len(c.content))
    max_chunk = max(chunks, key=lambda c: len(c.content))
    print(f"Smallest chunk: {min_chunk.chunk_id} ({len(min_chunk.content)} chars, {min_chunk.token_count} tokens)")
    print(f"  Breadcrumb: {min_chunk.header_breadcrumb}")
    print(f"  Content: {repr(min_chunk.content[:120])}")
    
    print(f"Largest chunk: {max_chunk.chunk_id} ({len(max_chunk.content)} chars, {max_chunk.token_count} tokens)")
    print(f"  Breadcrumb: {max_chunk.header_breadcrumb}")
    print(f"  Content snippet: {repr(max_chunk.content[:150])}")

    table_chunks = [c for c in chunks if c.has_table]
    print(f"\nChunks containing markdown tables: {len(table_chunks)}")
    for tc in table_chunks:
        print(f"  - [{tc.chunk_id}] {tc.header_breadcrumb} ({len(tc.content)} chars, {tc.token_count} tokens)")

    # Breadcrumb structure analysis
    print("\n" + "="*80)
    print("HEADING & BREADCRUMB METADATA ANALYSIS")
    print("="*80)
    sample_breadcrumbs = list(dict.fromkeys(c.header_breadcrumb for c in chunks))[:12]
    for bc in sample_breadcrumbs:
        print(f"  - {bc}")

    # Test realistic resident questions
    questions = [
        "Can visitors park overnight?",
        "What happens if someone parks in a reserved lot?",
        "How much does management charge for a particular violation?",
        "Who should residents contact regarding maintenance?",
        "What are the restrictions on using common facilities?"
    ]

    print("\n" + "="*80)
    print("RESIDENT QUESTION EVALUATION & CHUNK INSPECTION")
    print("="*80)

    engine = SemanticRAGEngine(kb_path=DEFAULT_KB_FILE)

    eval_output = []

    for idx, q in enumerate(questions, 1):
        print(f"\n[{idx}/5] TEST QUERY: {q}")
        res = engine.query(q)
        print(f"Latency: {res.latency_seconds:.2f}s | Model: {res.model_used} | Grounded: {res.grounded}")
        print(f"Sections Retrieved: {res.source_sections}")
        print(f"By-laws / Rules Cited: {res.by_laws_cited}")
        
        print(f"\n--- Retrieved Chunks ({len(res.retrieved_chunks)}) ---")
        chunk_details = []
        for c_idx, chunk in enumerate(res.retrieved_chunks, 1):
            print(f"  Chunk #{c_idx}: [{chunk.chunk_id}] {chunk.header_breadcrumb}")
            print(f"    Size: {len(chunk.content)} chars | Table: {chunk.has_table}")
            snippet = chunk.content[:250].replace("\n", " ")
            print(f"    Snippet: {snippet}...")
            chunk_details.append({
                "chunk_id": chunk.chunk_id,
                "breadcrumb": chunk.header_breadcrumb,
                "has_table": chunk.has_table,
                "chars": len(chunk.content),
                "snippet": chunk.content[:400]
            })

        print(f"\n--- Gemini Synthesized Answer ---")
        print(res.answer)
        print("-" * 60)

        eval_output.append({
            "query": q,
            "grounded": res.grounded,
            "model_used": res.model_used,
            "latency_seconds": res.latency_seconds,
            "by_laws_cited": res.by_laws_cited,
            "source_sections": res.source_sections,
            "retrieved_chunks": chunk_details,
            "answer": res.answer
        })

    with open("scripts/rag_eval_results.json", "w", encoding="utf-8") as out_f:
        json.dump(eval_output, out_f, indent=2)
    print("\nEvaluation results saved to scripts/rag_eval_results.json")

if __name__ == "__main__":
    run_evaluation()
