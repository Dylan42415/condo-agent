import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from condo_classifier.semantic_rag import get_semantic_rag

sys.stdout.reconfigure(encoding="utf-8")

leak_queries = [
    "I live on the 8th floor in unit #08-04 and water is actively dripping from my bathroom ceiling. Is the 9th floor owner upstairs responsible, and what should I do right now?",
    "I have water seepage on the 4th floor ceiling coming from the 5th floor unit above. How quickly does management inspect it and what if the upstairs neighbor refuses to fix it?",
    "There is a pipe burst causing heavy flooding on the 8th floor corridor near the lifts. Who can shut off the main water valve immediately?"
]

engine = get_semantic_rag()
engine.initialize()

for i, q in enumerate(leak_queries, 1):
    print("=" * 70)
    print(f"LEAK TEST CASE {i}: {q}")
    print("=" * 70)
    res = engine.query(q, top_k=3)
    print(f"Latency        : {res.latency_seconds:.2f}s")
    print(f"By-Laws Cited  : {res.by_laws_cited}")
    print(f"Top Sources    : {res.source_sections}")
    print("\nAnswer:\n" + res.answer)
    print("\n")
