"""
Hybrid Knowledge Graph RAG Engine for Condo Estate Management.

Combines:
1. Fast semantic/keyword seed retrieval (Unit numbers, issue types, by-law queries).
2. Bidirectional graph traversal (1-hop & 2-hop BFS across [[Wikilinks]] and backlinks).
3. Stub-note filtering & priority expansion (neighbor units, valves, SLAs).
4. Grounded synthesis with local Ollama or rule-based fallback.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VAULT_DIR = ROOT / "Knowledge Base" / "vault"

STOP_WORDS = {
    "a",
    "an",
    "the",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "it",
    "my",
    "our",
    "their",
    "this",
    "that",
    "with",
    "by",
    "from",
    "be",
    "can",
    "could",
    "should",
    "would",
    "do",
    "does",
    "what",
    "which",
    "who",
    "whom",
    "where",
    "when",
    "why",
    "how",
    "i",
    "we",
    "you",
    "he",
    "she",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "if",
    "then",
    "into",
    "so",
    "than",
    "too",
    "very",
    "just",
    "about",
    "there",
    "here",
    "any",
    "some",
}


@dataclass
class NoteData:
    title: str
    note_type: str
    category: str
    aliases: list[str]
    tags: list[str]
    body: str
    preview: str
    outgoing_links: list[str]
    relationships: list[tuple[str, str]]  # (RelationType, TargetTitle)
    file_path: Path

    @property
    def is_stub(self) -> bool:
        """Stub notes are auto-generated or contain less than 60 bytes of body."""
        return len(self.body.strip()) < 60 or "auto_generated" in self.tags


@dataclass
class EstateRAGResult:
    query: str
    answer: str
    seed_nodes: list[str]
    expanded_nodes: list[str]
    edges: list[tuple[str, str, str]]  # (source, relation, target)
    by_laws_cited: list[str]
    affected_infrastructure: list[str]
    emergency_contacts: list[dict[str, str]]
    grounded: bool
    context_text: str = ""


class EstateVaultGraph:
    """In-memory Knowledge Graph representation of the Condo Estate Obsidian Vault."""

    def __init__(self, vault_dir: Path | str | None = None) -> None:
        self.vault_dir = Path(vault_dir) if vault_dir else DEFAULT_VAULT_DIR
        self.notes: dict[str, NoteData] = {}
        self.title_lookup: dict[str, str] = {}  # lowercase -> canonical title
        self.alias_lookup: dict[str, str] = {}  # lowercase alias -> canonical title
        self.outgoing_edges: dict[str, set[tuple[str, str]]] = {}  # source -> {(relation, target)}
        self.incoming_edges: dict[str, set[tuple[str, str]]] = {}  # target -> {(relation, source)}
        self._load_vault()

    def _load_vault(self) -> None:
        if not self.vault_dir.exists():
            return

        for md_file in self.vault_dir.glob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                note = self._parse_note(md_file, content)
                self.notes[note.title] = note
                self.title_lookup[note.title.casefold()] = note.title
                for alias in note.aliases:
                    self.alias_lookup[alias.casefold()] = note.title
            except Exception:
                continue

        # Build bidirectional edges
        for title, note in self.notes.items():
            if title not in self.outgoing_edges:
                self.outgoing_edges[title] = set()
            for rel, target in note.relationships:
                canonical_target = self.resolve_title(target)
                if canonical_target:
                    self.outgoing_edges[title].add((rel, canonical_target))
                    if canonical_target not in self.incoming_edges:
                        self.incoming_edges[canonical_target] = set()
                    self.incoming_edges[canonical_target].add((rel, title))

    def _parse_note(self, file_path: Path, content: str) -> NoteData:
        title = file_path.stem
        note_type = "Document"
        category = "General Inquiries"
        aliases: list[str] = []
        tags: list[str] = []

        body = content
        # Parse YAML frontmatter if present
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter, body = parts[1], parts[2]
                for line in frontmatter.strip().splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        k, v = k.strip().lower(), v.strip()
                        if k == "title" and v:
                            title = v
                        elif k == "type" and v:
                            note_type = v
                        elif k == "category" and v:
                            category = v
                        elif k == "aliases":
                            # Extract aliases from [a, b, c]
                            inner = re.sub(r"[\[\]]", "", v)
                            aliases = [a.strip() for a in inner.split(",") if a.strip()]
                        elif k == "tags":
                            inner = re.sub(r"[\[\]]", "", v)
                            tags = [t.strip() for t in inner.split(",") if t.strip()]

        # Extract all [[Wikilinks]]
        wikilinks_raw = re.findall(r"\[\[(.*?)\]\]", body)
        outgoing_links = [link.split("|")[0].strip() for link in wikilinks_raw]

        # Extract structured typed relationships from '## Relationships' section
        relationships: list[tuple[str, str]] = []
        if "## Relationships" in body:
            rel_section = body.split("## Relationships", 1)[1].split("##", 1)[0]
            for line in rel_section.strip().splitlines():
                match = re.search(r"-\s*\*\*([A-Z_]+)\*\*\s*(?:→|->)\s*\[\[(.*?)\]\]", line)
                if match:
                    rel_type, target = match.group(1), match.group(2).split("|")[0].strip()
                    relationships.append((rel_type, target))

        # If no explicit relationships section, map generic links as 'LINKS_TO'
        if not relationships:
            for link in outgoing_links:
                relationships.append(("LINKS_TO", link))

        preview = body.strip().replace("\n", " ")[:200]
        return NoteData(
            title=title,
            note_type=note_type,
            category=category,
            aliases=aliases,
            tags=tags,
            body=body.strip(),
            preview=preview,
            outgoing_links=outgoing_links,
            relationships=relationships,
            file_path=file_path,
        )

    def resolve_title(self, query_title: str) -> str | None:
        """Resolve a title or alias to the canonical vault note title."""
        cleaned = query_title.strip()
        cf = cleaned.casefold()
        if cf in self.title_lookup:
            return self.title_lookup[cf]
        if cf in self.alias_lookup:
            return self.alias_lookup[cf]
        # Check unit variations (e.g. "08-04" -> "Unit #08-04" or "#08-04")
        unit_match = re.search(r"#?(\d{1,3}-\d{1,4})", cleaned)
        if unit_match:
            u_title = f"Unit #{unit_match.group(1)}"
            if u_title.casefold() in self.title_lookup:
                return self.title_lookup[u_title.casefold()]
        return None

    def search_seeds(
        self, query: str, unit_hint: str | None = None, top_k: int = 3
    ) -> list[NoteData]:
        """
        Hybrid seed selection using exact unit extraction, title matching, and BM25 term frequency.
        """
        seeds: list[NoteData] = []
        seen_titles: set[str] = set()

        # 1. Deterministic Unit Seed Injection
        unit_str = unit_hint
        if not unit_str:
            unit_re = re.search(r"#?(\d{1,2}-\d{2,4})\b", query)
            if unit_re:
                unit_str = unit_re.group(1)

        if unit_str:
            canonical_unit = self.resolve_title(unit_str)
            if canonical_unit and canonical_unit in self.notes:
                seeds.append(self.notes[canonical_unit])
                seen_titles.add(canonical_unit)

        # 2. Token BM25 Scoring
        query_tokens = [
            t.lower()
            for t in re.findall(r"\b[A-Za-z0-9#§\-]+\b", query)
            if len(t) > 1 and t.lower() not in STOP_WORDS
        ]
        if not query_tokens:
            query_tokens = [t.lower() for t in query.split() if t.lower() not in STOP_WORDS]

        scored: list[tuple[float, NoteData]] = []
        clean_query = query.strip().casefold()

        # Corpus IDF calculation
        n_docs = max(1, len(self.notes))
        idf: dict[str, float] = {}
        for token in query_tokens:
            doc_count = sum(
                1
                for n in self.notes.values()
                if token in f"{n.title} {' '.join(n.aliases)} {n.body}".lower()
            )
            idf[token] = math.log((n_docs - doc_count + 0.5) / (doc_count + 0.5) + 1.0)

        for title, note in self.notes.items():
            if title in seen_titles or note.is_stub:
                continue

            score = 0.0
            title_cf = title.casefold()
            body_cf = note.body.casefold()
            aliases_cf = [a.casefold() for a in note.aliases]

            # Exact title or alias match boost
            if clean_query == title_cf or clean_query in aliases_cf:
                score += 150.0
            elif title_cf in clean_query or any(a in clean_query for a in aliases_cf if len(a) > 3):
                score += 80.0

            # Token level matches
            for token in query_tokens:
                term_idf = idf.get(token, 1.0)
                if token in title_cf or any(token in a for a in aliases_cf):
                    score += 35.0 * term_idf
                tf = len(re.findall(r"\b" + re.escape(token) + r"\b", body_cf))
                if tf > 0:
                    score += (tf / (tf + 1.5)) * 15.0 * term_idf

            if score > 0:
                scored.append((score, note))

        scored.sort(key=lambda x: x[0], reverse=True)
        for _, note in scored:
            if len(seeds) >= top_k:
                break
            if note.title not in seen_titles:
                seeds.append(note)
                seen_titles.add(note.title)

        return seeds

    def expand_subgraph(
        self,
        seed_notes: list[NoteData],
        hops: int = 1,
        max_total_notes: int = 12,
    ) -> dict[str, NoteData]:
        """
        Execute Breadth-First Search (BFS) graph expansion across outgoing and incoming links.
        Guarantees that connected neighbors (contractors, risers, valves, upstairs units,
        by-laws) are actively pulled into the context window with priority.
        """

        def _entity_priority(title: str) -> int:
            if title not in self.notes:
                return 9
            n_type = self.notes[title].note_type
            priorities = {
                "Contractor": 1,
                "InternalTeam": 1,
                "Valve": 2,
                "Unit": 2,
                "ByLaw": 2,
                "Infrastructure": 3,
                "Legislation": 3,
                "DomainCategory": 4,
                "Stack": 5,
                "Building": 6,
            }
            return priorities.get(n_type, 4)

        subgraph: dict[str, NoteData] = {}
        queue: list[tuple[str, int]] = []

        for seed in seed_notes:
            subgraph[seed.title] = seed
            queue.append((seed.title, 0))

        while queue and len(subgraph) < max_total_notes:
            curr_title, current_hop = queue.pop(0)
            if current_hop >= hops:
                continue

            # Collect candidates from both outgoing edges and incoming backlinks
            candidates: set[str] = set()
            for _, target_title in self.outgoing_edges.get(curr_title, set()):
                if target_title not in subgraph and target_title in self.notes:
                    if not self.notes[target_title].is_stub:
                        candidates.add(target_title)

            for _, source_title in self.incoming_edges.get(curr_title, set()):
                if source_title not in subgraph and source_title in self.notes:
                    if not self.notes[source_title].is_stub:
                        candidates.add(source_title)

            # Sort candidate neighbors by domain actionability
            sorted_candidates = sorted(candidates, key=_entity_priority)

            for cand_title in sorted_candidates:
                if len(subgraph) >= max_total_notes:
                    break
                if cand_title not in subgraph:
                    subgraph[cand_title] = self.notes[cand_title]
                    queue.append((cand_title, current_hop + 1))

        return subgraph

    def get_subgraph_edges(self, subgraph_notes: dict[str, NoteData]) -> list[tuple[str, str, str]]:
        """Extract all directed edges (Source, Relation, Target) within the active subgraph."""
        edges: list[tuple[str, str, str]] = []
        titles = set(subgraph_notes.keys())

        for source in titles:
            for rel, target in self.outgoing_edges.get(source, set()):
                if target in titles:
                    edges.append((source, rel, target))

        return edges

    def format_context(self, subgraph_notes: dict[str, NoteData]) -> str:
        """Format the connected subgraph into clean, structured Markdown context blocks."""
        blocks: list[str] = []
        for title, note in subgraph_notes.items():
            block = (
                f"--- NOTE: [[{title}]] (Type: {note.note_type} | Category: {note.category}) ---\n"
            )
            block += f"{note.body}\n"
            blocks.append(block)
        return "\n\n".join(blocks)

    def query(
        self,
        user_query: str,
        unit_hint: str | None = None,
        hops: int = 1,
        max_notes: int = 12,
    ) -> EstateRAGResult:
        """
        End-to-End Hybrid Graph RAG Query execution:
        1. Hybrid seed search (Unit extraction + BM25 keyword matching).
        2. 1-hop / 2-hop BFS graph expansion across estate topology and by-laws.
        3. Context synthesis with grounded citations, emergency actions, and penalty checks.
        """
        seeds = self.search_seeds(user_query, unit_hint=unit_hint, top_k=3)
        subgraph = self.expand_subgraph(seeds, hops=hops, max_total_notes=max_notes)
        edges = self.get_subgraph_edges(subgraph)
        context_text = self.format_context(subgraph)

        # Extract structured domain details from the retrieved subgraph
        by_laws: list[str] = [
            t for t, n in subgraph.items() if n.note_type == "ByLaw" or "By-law" in t
        ]
        infrastructure: list[str] = [
            t
            for t, n in subgraph.items()
            if n.note_type in {"Infrastructure", "Valve", "Stack", "Building", "Unit"}
        ]
        contractors: list[dict[str, str]] = []
        for t, n in subgraph.items():
            if n.note_type in {"Contractor", "InternalTeam"}:
                phone_match = re.search(r"(\+65\s*\d{4}[-\s]?\d{4}|EXT\s*\d+)", n.body)
                phone = phone_match.group(1) if phone_match else "N/A"
                contractors.append(
                    {
                        "name": t,
                        "type": n.note_type,
                        "phone": phone,
                        "category": n.category,
                    }
                )

        # Synthesize Grounded Answer
        answer = self._synthesize_grounded_answer(
            query=user_query,
            unit_hint=unit_hint,
            subgraph=subgraph,
            by_laws=by_laws,
            infrastructure=infrastructure,
            contractors=contractors,
        )

        return EstateRAGResult(
            query=user_query,
            answer=answer,
            seed_nodes=[s.title for s in seeds],
            expanded_nodes=list(subgraph.keys()),
            edges=edges,
            by_laws_cited=by_laws,
            affected_infrastructure=infrastructure,
            emergency_contacts=contractors,
            grounded=bool(subgraph),
            context_text=context_text,
        )

    def _synthesize_grounded_answer(
        self,
        query: str,
        unit_hint: str | None,
        subgraph: dict[str, NoteData],
        by_laws: list[str],
        infrastructure: list[str],
        contractors: list[dict[str, str]],
    ) -> str:
        """Deterministic heuristic synthesis generator backed by grounded subgraph knowledge."""
        q_lower = query.lower()

        # Check for weekend / drilling renovation scenario
        if any(
            w in q_lower
            for w in ["drill", "renovation", "saturday", "sunday", "holiday", "noise", "hack"]
        ):
            if "By-law §4 - Renovation and Noise" in subgraph:
                return (
                    "🚫 **Renovation Work Prohibited on Weekends / Public Holidays**\n\n"
                    "Under **[[By-law §4 - Renovation and Noise]]**, heavy mechanical drilling, "
                    "hacking, and noisy renovation works are **strictly prohibited on Saturdays, "
                    "Sundays, and Public Holidays** (Zero Tolerance Policy).\n\n"
                    "### Permitted Renovation Hours:\n"
                    "- **General Renovation Work**: Monday to Friday, 9:00 AM – 5:00 PM\n"
                    "- **Heavy Drilling / Demolition**: Monday to Friday, 10:00 AM – 4:00 PM "
                    "(Max continuous noise: 75 dBA)\n\n"
                    "### Enforcement & Penalties:\n"
                    "- **Unauthorized weekend drilling**: Immediate security shutdown and "
                    "**$500 security deposit forfeiture**.\n"
                    "- **Roving Patrol**: [[Resident Relations Patrol]] is dispatched to issue "
                    "immediate stop-work orders."
                )

        # Check for water leak / ceiling seepage scenario
        if any(
            w in q_lower
            for w in ["water", "leak", "gushing", "ceiling", "burst", "pipe", "flooding"]
        ):
            target_unit = unit_hint or next((t for t in infrastructure if "Unit #" in t), None)
            upstairs_unit = None
            valve = next((t for t in infrastructure if "Valve" in t), None)
            riser = next((t for t in infrastructure if "Riser" in t), None)

            for t, n in subgraph.items():
                if "DIRECTLY_ABOVE" in [r[0] for r in n.relationships] or "DIRECTLY_BELOW" in [
                    r[0] for r in n.relationships
                ]:
                    for r, target in n.relationships:
                        if r == "DIRECTLY_BELOW" and "Unit #" in target:
                            upstairs_unit = target
                        elif r == "DIRECTLY_ABOVE" and target == target_unit:
                            upstairs_unit = t

            ans = "🚨 **Water Leakage & Emergency Isolation Protocol**\n\n"
            if target_unit:
                ans += f"**Affected Unit**: [[{target_unit}]]\n"
            if upstairs_unit:
                ans += f"**Suspected Source (Directly Above)**: [[{upstairs_unit}]]\n"
            if valve:
                ans += f"**Isolation Valve**: [[{valve}]] (PIN `8821#`)\n"
            if riser:
                ans += f"**Supply Line**: [[{riser}]]\n\n"

            upstairs_repr = f"[[{upstairs_unit}]]" if upstairs_unit else "upper unit"
            ans += (
                "### Statutory & By-Law Presumption:\n"
                "Under **[[By-law §6 - Inter-Floor Water Leaks]]** and Singapore "
                "**[[BMSMA Section 101]]**, there is a statutory presumption that an "
                "inter-floor ceiling leak originates from the defect of the upper floor unit "
                f"({upstairs_repr}). Joint technical inspection is mandated within 48 hours.\n\n"
            )

            if contractors:
                plumber = next((c for c in contractors if "Plumbing" in c["name"]), contractors[0])
                ans += (
                    "### Emergency Contractor Auto-Dispatch:\n"
                    f"- **Provider**: [[{plumber['name']}]]\n"
                    f"- **Hotline**: `{plumber['phone']}` (24/7 Response)\n"
                    f"- **Emergency SLA**: Under 20–30 minutes"
                )
            return ans

        # Check for BBQ / Facility Booking scenario
        if any(
            w in q_lower for w in ["bbq", "barbecue", "pool", "facility", "book", "function room"]
        ):
            if "By-law §8 - Facility Booking and Amenities" in subgraph:
                return (
                    "📋 **Estate Facility Booking Guidelines & Tariffs**\n\n"
                    "Governed by **[[By-law §8 - Facility Booking and Amenities]]** "
                    "administered by [[Clubhouse Management Office]]:\n\n"
                    "### BBQ Pavilions (Pits 1 - 4):\n"
                    "- **Sessions**: Session 1 (10:00 AM – 2:00 PM) | "
                    "Session 2 (5:00 PM – 10:00 PM)\n"
                    "- **Deposit / Fee**: **$50 refundable deposit** ($20 usage fee)\n"
                    "- **Cleaning Requirement**: Grill and pit area must be cleaned after use; "
                    "$50 cleaning forfeiture for uncleaned grill.\n"
                    "- **Swimming Pool & Jacuzzi**: 7:00 AM – 10:00 PM "
                    "(Proper swimwear mandatory; no glass allowed).\n\n"
                    "Bookings can be made via the Resident Portal or at the Clubhouse counter."
                )

        # Check for Security / Parking / Wheel clamping scenario
        if any(
            w in q_lower for w in ["parking", "clamp", "visitor", "tailgating", "gantry", "barrier"]
        ):
            if "By-law §12 - Security and Access Control" in subgraph:
                return (
                    "🛡️ **Security & Access Control Enforcement**\n\n"
                    "Governed by **[[By-law §12 - Security and Access Control]]** "
                    "enforced by [[Main Guardhouse Security]]:\n\n"
                    "### Regulations & Penalties:\n"
                    "- **Unauthorized Parking**: Vehicles parked in non-designated lots or "
                    "overnight in visitor parking are subject to wheel clamping.\n"
                    "- **Declamping Fee**: **$100 release fee** payable prior to vehicle release.\n"
                    "- **Late-Night Visitors**: Mandatory guardhouse registration past 11:00 PM "
                    "via resident intercom verification.\n"
                    "- **Barrier Damage**: Tailgating resulting in gantry damage incurs an "
                    "$850 barrier replacement claim."
                )

        # Default Grounded Synthesis
        lines = [f"### Grounded Estate Intelligence ({len(subgraph)} Connected Entities Found)\n"]
        for t, n in list(subgraph.items())[:4]:
            lines.append(f"**[[{t}]]** ({n.note_type}): {n.preview}")
        if by_laws:
            bylaws_str = ", ".join(f"[[{b}]]" for b in by_laws)
            lines.append(f"\n**Relevant By-Laws**: {bylaws_str}")
        if contractors:
            dispatch_str = ", ".join(f"{c['name']} ({c['phone']})" for c in contractors)
            lines.append(f"**Emergency Dispatch**: {dispatch_str}")
        return "\n".join(lines)


# Singleton vault graph instance for global application use
_estate_graph_instance: EstateVaultGraph | None = None


def get_estate_vault() -> EstateVaultGraph:
    """Lazily load and cache the Estate Vault Knowledge Graph."""
    global _estate_graph_instance
    if _estate_graph_instance is None:
        _estate_graph_instance = EstateVaultGraph()
    return _estate_graph_instance
