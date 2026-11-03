"""
Semantic Chunking RAG Engine for Condominium Estate Management.

Optimized Two-Stage Semantic Chunking Pipeline:
1. Ingests high-fidelity Markdown files (converted from PDFs or estate vaults).
2. Performs structure-aware splitting via LangChain MarkdownHeaderTextSplitter.
3. Injects hierarchical breadcrumbs into embeddable text (Anthropic-style Contextual Retrieval).
4. Maintains parent document section references for Small-to-Big context expansion.
5. Performs query normalization/expansion to bridge colloquial speech to legal by-law terminology.
6. Indexes chunks into an in-memory hybrid vector store (Vector Similarity + Priority Token Boosting).
7. Decision-first grounded synthesis using Gemini (gemini-3.1-flash-lite) with deterministic fallback.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KB_FILE = ROOT / "Knowledge Base" / "sample_condo_handbook.md"
DEFAULT_VAULT_DIR = ROOT / "Knowledge Base" / "vault"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SemanticChunk:
    chunk_id: str
    content: str
    source_file: str
    document_title: str = ""
    section: str = ""
    subsection: str = ""
    headings_path: list[str] = field(default_factory=list)
    token_count: int = 0
    embedding: list[float] | None = None
    has_table: bool = False
    parent_section_id: str = ""
    parent_content: str = ""

    @property
    def header_breadcrumb(self) -> str:
        parts = [p for p in [self.document_title, self.section, self.subsection] if p]
        return " > ".join(parts) if parts else "General"

    @property
    def embeddable_text(self) -> str:
        """Context-injected text for dense embedding and sparse matching."""
        parts = []
        if self.document_title:
            parts.append(f"Document: {self.document_title}")
        if self.section:
            parts.append(f"Section: {self.section}")
        if self.subsection:
            parts.append(f"Subsection: {self.subsection}")
        header_prefix = " | ".join(parts)
        if header_prefix:
            return f"[{header_prefix}]\n{self.content}"
        return self.content


@dataclass
class RetrievedChunkMatch:
    chunk: SemanticChunk
    similarity_score: float
    keyword_boost: float
    final_score: float


@dataclass
class SemanticRAGResult:
    query: str
    answer: str
    retrieved_chunks: list[SemanticChunk]
    by_laws_cited: list[str]
    source_sections: list[str]
    grounded: bool
    context_text: str = ""
    model_used: str = "deterministic"
    latency_seconds: float = 0.0
    expanded_query: str = ""


# ============================================================================
# Embeddings: Gemini API with Local Deterministic Fallback
# ============================================================================

class EmbeddingProvider:
    """Provides vector embeddings using Gemini API with deterministic local fallback."""

    def __init__(self, model: str = "gemini-embedding-001") -> None:
        self.model = model
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self._client: Any = None
        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini Client: {e}. Falling back to local embeddings.")

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embeds a list of texts into vector representations."""
        if not texts:
            return []

        if self._client:
            try:
                embeddings: list[list[float]] = []
                batch_size = 32
                for i in range(0, len(texts), batch_size):
                    batch = [t[:8000] for t in texts[i:i + batch_size]]
                    res = self._client.models.embed_content(
                        model=self.model,
                        contents=batch
                    )
                    for emb in res.embeddings:
                        embeddings.append(emb.values)
                return embeddings
            except Exception as e:
                logger.warning(f"Gemini embedding API call failed: {e}. Falling back to deterministic embedding.")

        return [self._local_deterministic_embedding(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        """Embeds a single query string."""
        return self.embed_texts([query])[0]

    @staticmethod
    def _local_deterministic_embedding(text: str, dim: int = 256) -> list[float]:
        """
        Fast deterministic hash-based embedding with character n-grams.
        Guarantees reproducible cosine similarities offline for unit testing.
        """
        import hashlib
        vector = [0.0] * dim
        words = re.findall(r"\w+", text.lower())
        if not words:
            return vector

        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vector[idx] += sign

        # Also hash 2-word bigrams for local semantic context
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            h = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vector[idx] += sign * 1.5

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 1e-9:
            vector = [v / norm for v in vector]
        return vector


def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Computes cosine similarity between two vectors."""
    if len(v1) != len(v2) or not v1:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    return max(0.0, min(1.0, dot / (norm1 * norm2)))


# ============================================================================
# Two-Stage Semantic Chunking Pipeline
# ============================================================================

def split_into_sentences(text: str) -> list[str]:
    """Splits text into sentences while keeping bullet points and tables intact."""
    lines = text.splitlines()
    sentences: list[str] = []
    table_buffer: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            table_buffer.append(line)
            continue
        elif table_buffer:
            sentences.append("\n".join(table_buffer))
            table_buffer = []

        if not stripped:
            continue

        # Split normal text by sentence terminal punctuation
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'§])", stripped)
        for p in parts:
            if p.strip():
                sentences.append(p.strip())

    if table_buffer:
        sentences.append("\n".join(table_buffer))

    return sentences


class SemanticChunker:
    """
    Two-Stage Semantic Chunker:
    1. Structure-Aware Markdown Header Splitting (#, ##, ###).
    2. Small-to-Big Parent Context Association.
    3. Semantic Breakpoint Detection (sentence-level cosine distance thresholding).
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        breakpoint_percentile: float = 90.0,
        max_chunk_chars: int = 1200,
        min_chunk_chars: int = 150,
    ) -> None:
        self.embedding_provider = embedding_provider or EmbeddingProvider()
        self.breakpoint_percentile = breakpoint_percentile
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars

    def chunk_markdown_file(self, file_path: Path | str) -> list[SemanticChunk]:
        """Loads and semantically chunks a markdown file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {path}")

        content = path.read_text(encoding="utf-8")
        return self.chunk_markdown_text(content, source_file=path.name)

    def chunk_markdown_text(self, markdown_text: str, source_file: str = "document.md") -> list[SemanticChunk]:
        """
        Executes two-stage semantic chunking on markdown content.
        Preserves parent section relationships for Small-to-Big retrieval.
        """
        # Stage 1: Structural Markdown Header Splitting
        from langchain_text_splitters import MarkdownHeaderTextSplitter

        headers_to_split_on = [
            ("#", "Document"),
            ("##", "Section"),
            ("###", "Subsection"),
        ]
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        sections = splitter.split_text(markdown_text)

        all_chunks: list[SemanticChunk] = []
        chunk_idx = 1

        for sec in sections:
            meta = sec.metadata
            sec_text = sec.page_content.strip()
            if not sec_text:
                continue

            doc_title = meta.get("Document", "")
            section = meta.get("Section", "")
            subsection = meta.get("Subsection", "")
            parent_id = f"{source_file}:{doc_title}:{section}:{subsection}"

            # If section contains a GFM table or is already compact, keep as atomic semantic chunk
            has_table = bool(re.search(r"^\|(?:\s*:?-+:?\s*\|)+\s*$", sec_text, re.MULTILINE))
            if has_table or len(sec_text) <= self.max_chunk_chars:
                chunk = SemanticChunk(
                    chunk_id=f"chunk_{chunk_idx:04d}",
                    content=sec_text,
                    source_file=source_file,
                    document_title=doc_title,
                    section=section,
                    subsection=subsection,
                    headings_path=[p for p in [doc_title, section, subsection] if p],
                    token_count=len(sec_text.split()),
                    has_table=has_table,
                    parent_section_id=parent_id,
                    parent_content=sec_text,
                )
                all_chunks.append(chunk)
                chunk_idx += 1
                continue

            # Stage 2: Breakpoint Semantic Splitting on large textual sections
            sub_chunks = self._semantic_breakpoint_split(sec_text)
            for sub_text in sub_chunks:
                chunk = SemanticChunk(
                    chunk_id=f"chunk_{chunk_idx:04d}",
                    content=sub_text,
                    source_file=source_file,
                    document_title=doc_title,
                    section=section,
                    subsection=subsection,
                    headings_path=[p for p in [doc_title, section, subsection] if p],
                    token_count=len(sub_text.split()),
                    has_table="|" in sub_text,
                    parent_section_id=parent_id,
                    parent_content=sec_text,
                )
                all_chunks.append(chunk)
                chunk_idx += 1

        return all_chunks

    def _semantic_breakpoint_split(self, text: str) -> list[str]:
        """Splits large text at semantic distance breakpoints between adjacent sentences."""
        sentences = split_into_sentences(text)
        if len(sentences) <= 2:
            return [text]

        # Embed sentences
        embeddings = self.embedding_provider.embed_texts(sentences)

        # Compute distances between adjacent sentences (1.0 - cosine_similarity)
        distances: list[float] = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(embeddings[i], embeddings[i + 1])
            distances.append(1.0 - sim)

        if not distances:
            return [text]

        # Determine threshold at configured percentile
        sorted_dist = sorted(distances)
        percentile_idx = int(len(sorted_dist) * (self.breakpoint_percentile / 100.0))
        percentile_idx = min(percentile_idx, len(sorted_dist) - 1)
        threshold = sorted_dist[percentile_idx]

        # Group sentences into semantic chunks
        chunks: list[str] = []
        current_sentences: list[str] = [sentences[0]]

        for i, dist in enumerate(distances):
            sentence = sentences[i + 1]
            current_len = sum(len(s) for s in current_sentences)

            # Split if distance exceeds threshold and chunk meets minimum size, or if exceeding max size
            if (dist >= threshold and current_len >= self.min_chunk_chars) or (current_len + len(sentence) > self.max_chunk_chars):
                chunks.append("\n\n".join(current_sentences))
                current_sentences = [sentence]
            else:
                current_sentences.append(sentence)

        if current_sentences:
            chunks.append("\n\n".join(current_sentences))

        return chunks


# ============================================================================
# Hybrid Vector Store (Vector Similarity + BM25/Exact Token Boosting)
# ============================================================================

class HybridSemanticIndex:
    """In-memory vector and keyword index with Contextual Breadcrumb embeddings."""

    def __init__(self, embedding_provider: EmbeddingProvider | None = None) -> None:
        self.embedding_provider = embedding_provider or EmbeddingProvider()
        self.chunks: list[SemanticChunk] = []

    def index_chunks(self, chunks: list[SemanticChunk]) -> None:
        """Indexes chunks, generating contextual breadcrumb embeddings in batch."""
        self.chunks = chunks
        texts_to_embed = [c.embeddable_text for c in self.chunks]
        embeddings = self.embedding_provider.embed_texts(texts_to_embed)
        for chunk, emb in zip(self.chunks, embeddings):
            chunk.embedding = emb

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        vector_weight: float = 0.70,
        keyword_weight: float = 0.30,
    ) -> list[RetrievedChunkMatch]:
        """Hybrid retrieval combining dense contextual vector similarity with sparse exact token reinforcement."""
        if not self.chunks:
            return []

        query_vec = self.embedding_provider.embed_query(query)
        query_tokens = set(re.findall(r"\b\w+\b", query.lower()))

        # Specific high-value token patterns: numbers, fines, times, by-laws
        priority_tokens = set(re.findall(r"\b(?:\$\d+|\d+:\d+|\d+|by-law|bmsma|deposit|fine|clamp|penalty|unclamp)\b", query.lower()))

        matches: list[RetrievedChunkMatch] = []

        for chunk in self.chunks:
            # 1. Dense Cosine Similarity (over contextual embeddable text)
            sim = 0.0
            if chunk.embedding:
                sim = cosine_similarity(query_vec, chunk.embedding)

            # 2. Sparse / Keyword Boost (matched against embeddable text)
            contextual_lower = chunk.embeddable_text.lower()

            token_matches = sum(1 for t in query_tokens if t in contextual_lower)
            base_kw_score = (token_matches / max(1, len(query_tokens)))

            # Priority exact token matching (e.g. "$500", "10:00", "Saturday", "BMSMA")
            priority_matches = sum(1 for pt in priority_tokens if pt in contextual_lower)
            priority_bonus = 0.25 * priority_matches

            kw_score = min(1.0, base_kw_score + priority_bonus)

            final_score = (vector_weight * sim) + (keyword_weight * kw_score)

            matches.append(
                RetrievedChunkMatch(
                    chunk=chunk,
                    similarity_score=sim,
                    keyword_boost=kw_score,
                    final_score=final_score,
                )
            )

        matches.sort(key=lambda m: m.final_score, reverse=True)
        return matches[:top_k]


# ============================================================================
# Grounded Synthesis Engine (Gemini with Fallback)
# ============================================================================

class SemanticRAGEngine:
    """
    Main Semantic Chunking RAG Interface for Condo Inquiries.
    Features:
    - Query Normalization & Colloquial Expansion
    - Small-to-Big Parent Section Context Expansion
    - Contextual Breadcrumb Injection
    - Decision-First Structured Synthesis
    """

    def __init__(
        self,
        kb_path: Path | str | None = None,
        model_name: str = "gemini-3.1-flash-lite",
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.kb_path = Path(kb_path) if kb_path else DEFAULT_KB_FILE
        self.model_name = model_name
        self.embedding_provider = embedding_provider or EmbeddingProvider()
        self.chunker = SemanticChunker(embedding_provider=self.embedding_provider)
        self.index = HybridSemanticIndex(embedding_provider=self.embedding_provider)
        self._initialized = False

    def initialize(self) -> None:
        """Loads and indexes the knowledge base."""
        if self._initialized:
            return

        if not self.kb_path.exists():
            # If default sample doesn't exist, check vault directory
            if DEFAULT_VAULT_DIR.exists():
                all_chunks: list[SemanticChunk] = []
                for md in DEFAULT_VAULT_DIR.glob("*.md"):
                    all_chunks.extend(self.chunker.chunk_markdown_file(md))
                self.index.index_chunks(all_chunks)
                self._initialized = True
                return
            raise FileNotFoundError(f"Knowledge Base path does not exist: {self.kb_path}")

        chunks = self.chunker.chunk_markdown_file(self.kb_path)
        self.index.index_chunks(chunks)
        self._initialized = True
        logger.info(f"Initialized Semantic RAG with {len(chunks)} chunks from {self.kb_path.name}")

    def expand_query(self, query: str) -> str:
        """
        Expands colloquial resident inquiries with relevant by-law keywords and statutes
        to bridge colloquial speech to formal estate by-law terminology.
        """
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return query

        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = (
                "You are an expert Condominium Legal Assistant. Given a resident inquiry, output 3 to 6 "
                "formal estate terminology search keywords (e.g. By-law §, BMSMA Section, permissible hours, "
                "quiet hours, deposit, penalty fine, wheel clamp, pet restriction). "
                "Output ONLY the keywords on a single line separated by spaces.\n\n"
                f"Resident Inquiry: {query}\n"
                "Keywords:"
            )
            res = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={"temperature": 0.0}
            )
            expanded_terms = res.text.strip().replace("\n", " ")
            if expanded_terms:
                return f"{query} {expanded_terms}"
        except Exception as e:
            logger.debug(f"Query expansion skipped: {e}")

        return query

    def query(self, user_question: str, top_k: int = 3, expand_parent: bool = True) -> SemanticRAGResult:
        """Answers a user inquiry using optimized Semantic Chunking RAG."""
        start_time = time.perf_counter()
        self.initialize()

        # Step 1: Query Normalization / Expansion
        search_query = self.expand_query(user_question)

        # Step 2: Retrieve top chunks using enriched search query
        matches = self.index.retrieve(search_query, top_k=top_k)
        retrieved_chunks = [m.chunk for m in matches]

        if not retrieved_chunks:
            return SemanticRAGResult(
                query=user_question,
                answer="No relevant estate by-laws or house rules were found in the knowledge base.",
                retrieved_chunks=[],
                by_laws_cited=[],
                source_sections=[],
                grounded=False,
                latency_seconds=time.perf_counter() - start_time,
                expanded_query=search_query,
            )

        # Step 3: Small-to-Big Parent Context Assembly (deduplicating parent sections)
        seen_parents = set()
        context_parts = []
        source_sections = []

        for i, m in enumerate(matches, 1):
            c = m.chunk
            header = c.header_breadcrumb
            source_sections.append(header)

            pid = c.parent_section_id
            if expand_parent and pid:
                if pid in seen_parents:
                    continue
                seen_parents.add(pid)
                # Deliver full parent section context to the LLM to preserve complete tables and rules
                text_to_include = c.parent_content or c.content
            else:
                text_to_include = c.content

            context_parts.append(
                f"[Source {i}: {header}]\n{text_to_include}"
            )

        context_text = "\n\n---\n\n".join(context_parts)

        # Step 4: Decision-First Grounded Answer Synthesis
        answer, model_used, by_laws_cited = self._synthesize(user_question, context_text)

        elapsed = time.perf_counter() - start_time
        return SemanticRAGResult(
            query=user_question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            by_laws_cited=by_laws_cited,
            source_sections=list(dict.fromkeys(source_sections)),
            grounded=bool(by_laws_cited or "By-law" in answer or "Section" in answer or "$" in answer),
            context_text=context_text,
            model_used=model_used,
            latency_seconds=elapsed,
            expanded_query=search_query,
        )

    def _synthesize(self, query: str, context: str) -> tuple[str, str, list[str]]:
        """Synthesizes structured, decision-first grounded answer using Gemini or deterministic rules."""
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if api_key:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)

                system_prompt = (
                    "You are the official Condominium Management AI Assistant. Answer resident questions strictly "
                    "using the provided estate handbook and by-law context.\n\n"
                    "FORMAT YOUR RESPONSE WITH THIS CLEAR DECISION-FIRST STRUCTURE:\n"
                    "1. **RULING**: Start with a direct verdict: **PERMITTED**, **PROHIBITED**, or **ACTION REQUIRED**.\n"
                    "2. **BY-LAW & STATUTORY AUTHORITY**: Cite specific By-laws and legal acts (e.g., By-law §4, By-law §10, BMSMA Section 101).\n"
                    "3. **EXACT CONSTRAINTS & TARIFFS**: Quote exact schedules, permissible hours, dollar fines ($500, $1,000, $107), deposits ($2,000, $50), and notice periods from the context.\n"
                    "4. **ACTIONABLE NEXT STEPS & CONTACTS**: Give practical instructions, specific phone numbers, and guardhouse/management locations.\n\n"
                    "Ground your answer strictly on the provided context without speculating."
                )

                from condo_classifier.dispatch import load_vendor_directory
                vendors = load_vendor_directory()
                lines = ["OFFICIAL ESTATE DIRECTORY & CONTACTS:"]
                for cat, vdata in vendors.items():
                    if isinstance(vdata, dict):
                        for sub, v in vdata.items():
                            if isinstance(v, dict) and "vendor_name" in v and "phone" in v:
                                cov = f" ({v.get('coverage', '')})" if v.get("coverage") else ""
                                lines.append(f"- {v['vendor_name']}: {v['phone']}{cov}")
                directory_context = "\n".join(lines)

                prompt = (
                    f"{directory_context}\n\n"
                    f"ESTATE RULES CONTEXT:\n{context}\n\n"
                    f"RESIDENT QUESTION:\n{query}\n\n"
                    "OFFICIAL MANAGEMENT RESPONSE:"
                )

                res = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={"system_instruction": system_prompt, "temperature": 0.1}
                )
                answer_text = res.text.strip()
                cited = self._extract_citations(answer_text + "\n" + context)
                return answer_text, f"gemini:{self.model_name}", cited
            except Exception as e:
                logger.warning(f"Gemini generation call failed ({e}); using deterministic synthesis.")

        # Deterministic Offline Rule-based Synthesis
        answer_text, cited = self._deterministic_synthesis(query, context)
        return answer_text, "deterministic_rule_engine", cited

    @staticmethod
    def _extract_citations(text: str) -> list[str]:
        """Extracts cited by-laws, acts, or sections."""
        citations = []
        bylaws = re.findall(r"By-law\s*(?:§|Section|\d+|No\.\s*\d+)[^\n.,]*", text, re.IGNORECASE)
        bmsma = re.findall(r"BMSMA\s*Section\s*\d+", text, re.IGNORECASE)
        citations.extend([b.strip() for b in bylaws])
        citations.extend([s.strip() for s in bmsma])
        return list(dict.fromkeys(citations))

    def _deterministic_synthesis(self, query: str, context: str) -> tuple[str, list[str]]:
        """
        Dynamic rule extraction and structured synthesis when offline.
        Strictly parses retrieved context lines and tables rather than using hardcoded scenario branches.
        """
        cited = self._extract_citations(context)
        if not context.strip():
            return "No relevant estate by-laws found in handbook.", cited

        # Extract candidate lines and table rows from context
        lines = [line.strip() for line in context.splitlines() if line.strip() and not line.strip().startswith("#")]

        # Tokenize query for dynamic relevance scoring
        q_tokens = set(re.findall(r"\w+", query.lower()))
        stopwords = {"can", "we", "i", "is", "the", "a", "an", "for", "on", "in", "to", "at", "my", "our", "what", "how", "who", "when", "does"}
        informative_tokens = q_tokens - stopwords

        # Score each context line/row by token overlap
        scored_lines: list[tuple[float, str]] = []
        for line in lines:
            line_tokens = set(re.findall(r"\w+", line.lower()))
            overlap = len(informative_tokens & line_tokens)
            if overlap > 0:
                has_numbers = bool(re.search(r"\$\d+|\d{1,2}:\d{2}|\b\d+\s*(?:days|hours|sessions)\b", line))
                score = overlap + (1.5 if has_numbers else 0.0)
                scored_lines.append((score, line))

        scored_lines.sort(key=lambda x: x[0], reverse=True)
        top_relevant = [l for _, l in scored_lines[:5]]

        # Determine dynamic ruling from relevant context lines
        ruling = "**ACTION REQUIRED**"
        context_window = " ".join(top_relevant).lower() if top_relevant else context.lower()
        if any(w in context_window for w in ["strictly prohibited", "not permitted", "prohibited", "penalty", "clamp"]):
            if "with prior" in context_window or "permit required" in context_window or "upon application" in context_window:
                ruling = "**PERMITTED (WITH PERMIT / CONDITIONS)**"
            else:
                ruling = "**PROHIBITED**"
        elif any(w in context_window for w in ["permitted", "allowed", "may be booked", "open daily", "complimentary"]):
            ruling = "**PERMITTED**"

        # By-law authority: cite detected sections or top citation
        if cited:
            authority = ", ".join(cited)
        else:
            headers = re.findall(r"#+\s*(.+)", context)
            authority = headers[0] if headers else "Estate By-laws & Regulations"

        # Constraints: format top relevant lines/table rows
        constraint_bullets: list[str] = []
        for line in top_relevant:
            if line.startswith("|") and line.endswith("|"):
                cells = [c.strip() for c in line.split("|")[1:-1] if c.strip()]
                if cells and not all(c.startswith("-") for c in cells):
                    constraint_bullets.append("- " + " — ".join(cells))
            else:
                constraint_bullets.append(f"- {line}")

        if not constraint_bullets:
            constraint_bullets = [f"- {lines[0]}"] if lines else ["- Refer to management handbook."]

        constraints_str = "\n".join(constraint_bullets[:4])

        # Actionable steps: dynamically find contacts from vendor directory
        from condo_classifier.dispatch import load_vendor_directory
        vendors = load_vendor_directory()
        contact_items: list[str] = []
        for cat, data in vendors.items():
            if isinstance(data, dict):
                for sub, item in data.items():
                    if isinstance(item, dict) and "phone" in item and "vendor_name" in item:
                        vname = item["vendor_name"]
                        phone = item["phone"]
                        if any(kw in query.lower() for kw in item.get("keywords", [])):
                            contact_items.append(f"Contact {vname} at {phone} for immediate assistance.")

        if not contact_items:
            sec = vendors.get("security", {}).get("default", {})
            sec_name = sec.get("vendor_name", "Guardhouse")
            sec_phone = sec.get("phone", "+65 6444-2222")
            contact_items.append(f"Report or verify with {sec_name} ({sec_phone}).")

        action_str = "\n".join(dict.fromkeys(contact_items[:2]))

        structured_response = (
            f"**RULING**: {ruling}\n\n"
            f"**BY-LAW AUTHORITY**: {authority}\n\n"
            f"**EXACT CONSTRAINTS & DETAILS**:\n"
            f"{constraints_str}\n\n"
            f"**ACTIONABLE NEXT STEPS**:\n"
            f"{action_str}"
        )
        return structured_response, cited


_DEFAULT_SEMANTIC_ENGINE: SemanticRAGEngine | None = None


def get_semantic_rag(kb_path: Path | str | None = None) -> SemanticRAGEngine:
    """Singleton getter for the estate Semantic RAG Engine."""
    global _DEFAULT_SEMANTIC_ENGINE
    if _DEFAULT_SEMANTIC_ENGINE is None:
        _DEFAULT_SEMANTIC_ENGINE = SemanticRAGEngine(kb_path=kb_path)
        _DEFAULT_SEMANTIC_ENGINE.initialize()
    return _DEFAULT_SEMANTIC_ENGINE
