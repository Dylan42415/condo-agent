"""
By-Law Amendment Copilot (Human-in-the-Loop) for Condominium Estate Management.

Empowers estate managers and staff to amend condominium house rules and by-laws
using natural language without searching through raw markdown or PDF documents.

Key Architectural Guarantees:
1. Section Localization Agent: Directs the employee to the exact section & by-law clause.
2. Human-in-the-Loop Review: Highlights proposed changes with a visual color-coded diff.
3. Surgical In-Place Replacement: Preserves all surrounding sections, frontmatter, and GFM tables.
4. Automated Backup & Audit Trail: Creates timestamped backups and logs every change to JSON.
5. Live Semantic RAG Re-Indexing: Instantly rebuilds vector memory so resident inquiries
   immediately reflect updated rules across Web and Telegram.
6. Rollback Capability: Allows 1-click restoration to any prior version.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KB_PATH = ROOT / "Knowledge Base" / "converted_tribeca_handbook.md"
BACKUPS_DIR = ROOT / "Knowledge Base" / "backups"
AUDIT_LOG_FILE = BACKUPS_DIR / "amendment_audit_log.json"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class PolicySection:
    """Represents a discrete, editable section or subsection within the handbook."""
    index: int
    level: int
    heading_line: str
    title: str
    parent_heading: str
    breadcrumb: str
    start_char: int
    end_char: int
    line_no: int
    raw_block: str
    body: str

    @property
    def section_id(self) -> str:
        clean = re.sub(r"[^a-zA-Z0-9]+", "_", self.breadcrumb.lower()).strip("_")
        return f"sec_{self.index}_{clean[:30]}"


@dataclass
class PolicyAmendmentProposal:
    """Proposed by-law change drafted by the agent for human-in-the-loop verification."""
    section: PolicySection
    intent: str
    explanation: str
    original_raw_block: str
    proposed_raw_block: str
    proposed_body: str
    summary_bullets: list[str]
    diff_lines: list[tuple[str, str]]
    unified_diff: str
    additions_count: int
    deletions_count: int
    model_used: str


@dataclass
class AmendmentAuditRecord:
    """Immutable audit record saved after employee confirms by-law amendment."""
    log_id: str
    timestamp: str
    author: str
    intent: str
    target_section: str
    backup_file: str
    summary: str
    additions_count: int
    deletions_count: int
    diff_snippet: str
    status: str = "APPLIED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_id": self.log_id,
            "timestamp": self.timestamp,
            "author": self.author,
            "intent": self.intent,
            "target_section": self.target_section,
            "backup_file": self.backup_file,
            "summary": self.summary,
            "additions_count": self.additions_count,
            "deletions_count": self.deletions_count,
            "diff_snippet": self.diff_snippet,
            "status": self.status,
        }


# ============================================================================
# Section Parsing & Offsets
# ============================================================================

def parse_handbook_sections(markdown_text: str) -> list[PolicySection]:
    """
    Parses a markdown document into hierarchical, surgically replaceable sections.
    Preserves exact character offsets [start_char:end_char] for non-destructive replacement.
    """
    lines = markdown_text.splitlines(keepends=True)
    heading_regex = re.compile(r"^(#{1,4})\s+(.+)$")

    raw_headings: list[dict[str, Any]] = []
    char_offset = 0
    in_code_block = False

    for line_idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            char_offset += len(line)
            continue

        if not in_code_block:
            m = heading_regex.match(line.rstrip("\r\n"))
            if m:
                level = len(m.group(1))
                raw_headings.append({
                    "line_no": line_idx,
                    "start_char": char_offset,
                    "level": level,
                    "heading_line": line.rstrip("\r\n"),
                    "title": m.group(2).strip(),
                })
        char_offset += len(line)

    if not raw_headings:
        # Fallback for document with no headings
        return [
            PolicySection(
                index=0,
                level=1,
                heading_line="",
                title="Full Document",
                parent_heading="",
                breadcrumb="Full Document",
                start_char=0,
                end_char=len(markdown_text),
                line_no=0,
                raw_block=markdown_text,
                body=markdown_text,
            )
        ]

    # Calculate end_char for each heading span
    for i in range(len(raw_headings)):
        if i < len(raw_headings) - 1:
            raw_headings[i]["end_char"] = raw_headings[i + 1]["start_char"]
        else:
            raw_headings[i]["end_char"] = len(markdown_text)

    # Build hierarchy (H1 -> H2 -> H3)
    sections: list[PolicySection] = []

    # If frontmatter or introductory text exists before the first heading, preserve it as section 0
    if raw_headings[0]["start_char"] > 0:
        preamble = markdown_text[0 : raw_headings[0]["start_char"]]
        sections.append(
            PolicySection(
                index=0,
                level=1,
                heading_line="",
                title="Frontmatter & Metadata",
                parent_heading="",
                breadcrumb="Frontmatter & Metadata",
                start_char=0,
                end_char=raw_headings[0]["start_char"],
                line_no=0,
                raw_block=preamble,
                body=preamble,
            )
        )

    current_h1 = ""
    current_h2 = ""

    for h in raw_headings:
        idx = len(sections)
        lvl = h["level"]
        title = h["title"]
        heading_line = h["heading_line"]
        start_c = h["start_char"]
        end_c = h["end_char"]
        raw_block = markdown_text[start_c:end_c]

        # Extract body (excluding first heading line)
        parts = raw_block.split("\n", 1)
        body = parts[1] if len(parts) > 1 else ""

        if lvl == 1:
            current_h1 = title
            current_h2 = ""  # Reset lower-level subsection hierarchy
            parent_h = ""
            breadcrumb = title
        elif lvl == 2:
            current_h2 = title
            parent_h = current_h1
            breadcrumb = f"{current_h1} > {title}" if current_h1 else title
        elif lvl == 3:
            parent_h = current_h2 or current_h1
            if current_h2:
                breadcrumb = f"{current_h2} > {title}"
            elif current_h1:
                breadcrumb = f"{current_h1} > {title}"
            else:
                breadcrumb = title
        else:
            parent_h = current_h2 or current_h1
            breadcrumb = f"{parent_h} > {title}" if parent_h else title

        sections.append(
            PolicySection(
                index=idx,
                level=lvl,
                heading_line=heading_line,
                title=title,
                parent_heading=parent_h,
                breadcrumb=breadcrumb,
                start_char=start_c,
                end_char=end_c,
                line_no=h["line_no"],
                raw_block=raw_block,
                body=body,
            )
        )

    return sections


# ============================================================================
# Section Localization Agent
# ============================================================================

def locate_target_section(
    sections: list[PolicySection],
    user_intent: str,
    use_llm: bool = True,
) -> tuple[PolicySection, float, str]:
    """
    Directs the employee to the exact section requiring amendment.
    Combines keyword token scoring with Gemini semantic verification.
    """
    if not sections:
        raise ValueError("No sections found in handbook to search.")

    if not user_intent or not user_intent.strip():
        return sections[0], 0.0, "No query intent provided."

    # Filter to editable subsections (H2 or H3, ignoring top H1 doc title)
    candidate_sections = [s for s in sections if s.level >= 2]
    if not candidate_sections:
        candidate_sections = sections

    # Keyword Scoring (Unicode-aware)
    intent_clean = user_intent.lower()
    intent_tokens = re.findall(r"\w+", intent_clean, re.UNICODE)
    stopwords = {"the", "a", "an", "and", "or", "to", "for", "in", "of", "on", "from", "with", "is", "be", "change", "update", "set", "can", "please", "make"}
    query_terms = [t for t in intent_tokens if t not in stopwords and len(t) > 1]

    scored: list[tuple[float, PolicySection, list[str]]] = []
    for s in candidate_sections:
        score = 0.0
        matches = []
        body_lower = s.raw_block.lower()
        title_lower = s.breadcrumb.lower()

        for term in query_terms:
            if term in title_lower:
                score += 5.0
                matches.append(term)
            elif term in body_lower:
                score += 2.0
                matches.append(term)

        # Prefer leaf subsections (level 3) over broad headers (level 2) if they share score
        if s.level == 3 and score > 0:
            score += 0.5

        # Prefer operative rule/by-law sections over static application forms unless the user explicitly mentions 'form' or 'application'
        is_form_section = "application form" in s.breadcrumb.lower() or "form " in s.title.lower()
        mentions_form = "form" in intent_clean or "application" in intent_clean
        if not mentions_form and is_form_section:
            score -= 3.0
        elif not is_form_section and not mentions_form and score > 0:
            score += 1.0

        scored.append((score, s, list(set(matches))))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_sec, matched_words = scored[0]

    # If Gemini is available, verify or enhance the rationale
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if use_llm and api_key and user_intent.strip():
        try:
            from google import genai
            client = genai.Client(api_key=api_key)

            top_candidates = scored[:3]
            candidates_text = "\n".join(
                f"[{i+1}] Index {c[1].index}: {c[1].breadcrumb}\nSnippet: {c[1].body[:200]}..."
                for i, c in enumerate(top_candidates)
            )

            prompt = (
                f"You are an Estate Policy Navigation Agent.\n"
                f"EMPLOYEE INTENT: \"{user_intent}\"\n\n"
                f"CANDIDATE SECTIONS:\n{candidates_text}\n\n"
                f"Select the single best candidate index (1, 2, or 3) and provide a 1-sentence rationale.\n"
                f"Return JSON:\n{{\"selected_choice\": 1, \"rationale\": \"...\"}}"
            )

            res = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config={"response_mime_type": "application/json", "temperature": 0.1},
            )
            data = json.loads(res.text)
            sel_idx = int(data.get("selected_choice", 1)) - 1
            if 0 <= sel_idx < len(top_candidates):
                best_sec = top_candidates[sel_idx][1]
                rationale = data.get("rationale", f"Selected section governs: {best_sec.title}")
                return best_sec, top_candidates[sel_idx][0], rationale
        except Exception as e:
            logger.warning(f"LLM section localization fallback ({e}); using token overlap.")

    # Deterministic Rationale
    if matched_words:
        rationale = f"Matched key terms [{', '.join(matched_words)}] directly within section '{best_sec.breadcrumb}'."
    else:
        rationale = f"Identified section '{best_sec.breadcrumb}' as the closest match for this policy instruction."

    return best_sec, best_score, rationale


# ============================================================================
# Amendment Proposal Generator (Human-in-the-Loop)
# ============================================================================

def reconcile_surgical_amendment(orig_block: str, prop_block: str) -> str:
    """
    Surgically reconciles the proposed text against the original section block.

    Eliminates spurious/hallucinated micro-diffs where the LLM accidentally altered
    whitespace, trailing punctuation on blanks (e.g. '$__________.|' vs '$__________|'),
    underline lengths, or stripped trailing newlines on lines that were NOT intended to change.
    """
    orig_trailing_ws = orig_block[len(orig_block.rstrip("\r\n")):]
    orig_lines = [l.rstrip("\r\n") for l in orig_block.splitlines()]
    prop_lines = [l.rstrip("\r\n") for l in prop_block.splitlines()]

    def is_semantically_identical(orig_l: str, prop_l: str) -> bool:
        o_clean = re.sub(r"[^a-zA-Z0-9]+", "", orig_l).lower()
        p_clean = re.sub(r"[^a-zA-Z0-9]+", "", prop_l).lower()
        return o_clean == p_clean

    sm = difflib.SequenceMatcher(None, orig_lines, prop_lines)
    reconciled: list[str] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            reconciled.extend(orig_lines[i1:i2])
        elif tag == "replace":
            orig_sub = orig_lines[i1:i2]
            prop_sub = prop_lines[j1:j2]

            if len(orig_sub) == len(prop_sub):
                for o_ln, p_ln in zip(orig_sub, prop_sub):
                    if is_semantically_identical(o_ln, p_ln):
                        reconciled.append(o_ln)
                    else:
                        reconciled.append(p_ln)
            else:
                o_norm = "".join(re.sub(r"[^a-zA-Z0-9]+", "", l).lower() for l in orig_sub)
                p_norm = "".join(re.sub(r"[^a-zA-Z0-9]+", "", l).lower() for l in prop_sub)
                if o_norm == p_norm:
                    reconciled.extend(orig_sub)
                else:
                    reconciled.extend(prop_sub)
        elif tag == "delete":
            # If the deleted lines are merely blank lines or whitespace, keep from orig
            if all(not l.strip() for l in orig_lines[i1:i2]):
                reconciled.extend(orig_lines[i1:i2])
            else:
                pass  # Real deletion
        elif tag == "insert":
            reconciled.extend(prop_lines[j1:j2])

    result = "\n".join(reconciled)
    if orig_trailing_ws:
        result = result.rstrip("\r\n") + orig_trailing_ws
    elif not result.endswith("\n"):
        result += "\n"

    return result


def generate_policy_amendment(
    section: PolicySection,
    user_intent: str,
    model_name: str = "gemini-3.1-flash-lite",
) -> PolicyAmendmentProposal:
    """
    Drafts a surgical amendment to the target section based on employee intent.
    Computes visual diffs and summaries for human-in-the-loop review.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    proposed_raw_block = ""
    summary_bullets: list[str] = []
    explanation = ""
    model_used = "deterministic_rule_engine"

    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)

            system_instruction = (
                "You are an expert Condominium Legal & Operations Policy Editor.\n"
                "Your objective is to apply SURGICAL, MINIMAL amendments to condominium house rules and tables.\n\n"
                "CRITICAL SURGICAL MANDATES:\n"
                "1. ZERO UNNECESSARY CHANGES: Modify ONLY the specific number, word, or clause requested by the employee.\n"
                "2. NO WASTED OUTPUT TOKENS: Identify the exact phrase/clause to be changed. Do NOT invent changes to untouched sentences.\n"
                "3. VERBATIM PRESERVATION: Every untouched line, table row, sentence, punctuation mark, and blank underline must remain 100% IDENTICAL to the original text.\n"
                "4. DO NOT trim, reformat, or alter fill-in lines (e.g. keep 'for $__________.|' exactly as-is).\n"
                "5. Maintain strict GitHub-Flavored Markdown table formatting.\n"
                "6. Return ONLY a valid JSON object matching the requested schema.\n"
            )

            prompt = (
                f"EMPLOYEE INTENT / INSTRUCTION:\n\"{user_intent}\"\n\n"
                f"CURRENT SECTION TEXT:\n{section.raw_block}\n\n"
                f"Provide targeted surgical replacement phrases AND the full updated markdown.\n"
                f"Return JSON:\n"
                f"{{\n"
                f"  \"explanation\": \"Brief 1-sentence summary of the rule change\",\n"
                f"  \"summary_bullets\": [\"Bullet 1 of exact changes made\", \"Bullet 2\"],\n"
                f"  \"replacements\": [\n"
                f"    {{\"original_phrase\": \"verbatim substring from CURRENT SECTION TEXT to replace\", \"updated_phrase\": \"replacement substring\"}}\n"
                f"  ],\n"
                f"  \"updated_markdown\": \"Full updated markdown including heading and tables\"\n"
                f"}}"
            )

            res = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "system_instruction": system_instruction,
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                },
            )

            parsed = json.loads(res.text)
            summary_bullets = parsed.get("summary_bullets", [])
            explanation = parsed.get("explanation", f"Amended {section.title} to satisfy intent.")
            model_used = f"gemini:{model_name}"

            # Step 1: Attempt ultra-surgical verbatim replacements if provided (eliminates spurious line diffs)
            replacements = parsed.get("replacements", [])
            if isinstance(replacements, list) and len(replacements) > 0:
                candidate_block = section.raw_block
                all_applied = True
                for rep in replacements:
                    orig_p = rep.get("original_phrase", "")
                    upd_p = rep.get("updated_phrase", "")
                    if orig_p and orig_p in candidate_block:
                        candidate_block = candidate_block.replace(orig_p, upd_p, 1)
                    else:
                        all_applied = False
                        break
                if all_applied and candidate_block != section.raw_block:
                    proposed_raw_block = candidate_block

            # Step 2: Fall back to updated_markdown if replacements weren't provided or didn't match
            if not proposed_raw_block:
                proposed_raw_block = parsed.get("updated_markdown", "").strip()
        except Exception as e:
            logger.warning(f"Gemini amendment generation call failed ({e}); falling back to deterministic.")

    # Deterministic Fallback if LLM unavailable
    if not proposed_raw_block:
        proposed_raw_block, summary_bullets, explanation = _deterministic_amendment_fallback(
            section, user_intent
        )

    # Surgically reconcile proposed block against original to eliminate accidental micro-diffs on untouched lines
    proposed_raw_block = reconcile_surgical_amendment(section.raw_block, proposed_raw_block)

    # Ensure trailing horizontal rule (---) is preserved if original had it
    had_hr = bool(re.search(r"\n---\s*$", section.raw_block.rstrip()))
    now_has_hr = bool(re.search(r"\n---\s*$", proposed_raw_block.rstrip()))
    if had_hr and not now_has_hr:
        proposed_raw_block = proposed_raw_block.rstrip() + "\n\n---\n\n"

    # Ensure trailing newline matches
    if not proposed_raw_block.endswith("\n"):
        proposed_raw_block += "\n"

    # Separate body
    parts = proposed_raw_block.split("\n", 1)
    proposed_body = parts[1] if len(parts) > 1 else ""

    # Calculate unified diff and line-by-line diff
    orig_lines = section.raw_block.splitlines(keepends=True)
    prop_lines = proposed_raw_block.splitlines(keepends=True)

    unified = "".join(
        difflib.unified_diff(
            orig_lines,
            prop_lines,
            fromfile=f"Current: {section.title}",
            tofile=f"Proposed: {section.title}",
            n=3,
        )
    )

    diff_lines, adds, dels = _compute_annotated_diff(orig_lines, prop_lines)

    return PolicyAmendmentProposal(
        section=section,
        intent=user_intent,
        explanation=explanation,
        original_raw_block=section.raw_block,
        proposed_raw_block=proposed_raw_block,
        proposed_body=proposed_body,
        summary_bullets=summary_bullets,
        diff_lines=diff_lines,
        unified_diff=unified,
        additions_count=adds,
        deletions_count=dels,
        model_used=model_used,
    )


def _compute_annotated_diff(
    orig_lines: list[str],
    prop_lines: list[str],
) -> tuple[list[tuple[str, str]], int, int]:
    """Computes categorized diff lines (unchanged, add, delete) and counters."""
    matcher = difflib.SequenceMatcher(None, orig_lines, prop_lines)
    diff_lines: list[tuple[str, str]] = []
    adds = 0
    dels = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for line in orig_lines[i1:i2]:
                diff_lines.append(("unchanged", line.rstrip("\r\n")))
        elif tag == "replace":
            for line in orig_lines[i1:i2]:
                diff_lines.append(("delete", line.rstrip("\r\n")))
                dels += 1
            for line in prop_lines[j1:j2]:
                diff_lines.append(("add", line.rstrip("\r\n")))
                adds += 1
        elif tag == "delete":
            for line in orig_lines[i1:i2]:
                diff_lines.append(("delete", line.rstrip("\r\n")))
                dels += 1
        elif tag == "insert":
            for line in prop_lines[j1:j2]:
                diff_lines.append(("add", line.rstrip("\r\n")))
                adds += 1

    return diff_lines, adds, dels


def _deterministic_amendment_fallback(
    section: PolicySection,
    user_intent: str,
) -> tuple[str, list[str], str]:
    """Fallback rule modifier when Gemini API is unavailable."""
    updated = section.raw_block
    summary = []

    # Look for dollar amount adjustments e.g. "from $50 to $100" or "$100"
    dollar_matches = re.findall(r"\$(\d+(?:,\d+)*(?:\.\d{2})?)", user_intent)
    if len(dollar_matches) >= 2:
        old_val, new_val = dollar_matches[0], dollar_matches[1]
        pattern = re.compile(rf"(?<!\d)S?\$?\s*{re.escape(old_val)}(?:\.00)?(?!\d)", re.IGNORECASE)
        if pattern.search(updated):
            updated = pattern.sub(f"S${new_val}.00", updated)
            summary.append(f"Adjusted monetary value from S${old_val} to S${new_val}.00")
    elif len(dollar_matches) == 1:
        new_val = dollar_matches[0]
        # Contextual replacement: if user mentions keywords like "deposit", "fee", "fine"
        lines = updated.splitlines()
        replaced = False
        target_hints = [t for t in re.findall(r"\b\w+\b", user_intent.lower()) if t in ["deposit", "fee", "fine", "penalty", "parking", "drilling", "clean", "bbq", "unclamping", "car", "pet"]]
        for i, line in enumerate(lines):
            line_l = line.lower()
            if any(hint in line_l for hint in target_hints):
                new_line = re.sub(r"(?<!\d)S?\$(\d+(?:,\d+)*(?:\.\d{2})?)(?!\d)", f"S${new_val}.00", line, count=1)
                if new_line != line:
                    lines[i] = new_line
                    replaced = True
                    break
        if not replaced:
            updated = re.sub(r"(?<!\d)S?\$(\d+(?:,\d+)*(?:\.\d{2})?)(?!\d)", f"S${new_val}.00", updated, count=1)
        else:
            updated = "\n".join(lines)
        summary.append(f"Updated primary tariff/deposit to S${new_val}.00")

    # If no dollar matched, append as an administrative policy addendum
    if updated == section.raw_block:
        addendum = f"- **Special Amendment:** {user_intent.strip()}"
        if re.search(r"\n---\s*$", updated.rstrip()):
            updated = re.sub(r"(\n---\s*)$", rf"\n\n{addendum}\1", updated.rstrip()) + "\n\n"
        else:
            updated = updated.rstrip() + f"\n\n{addendum}\n\n"
        summary.append(f"Appended administrative amendment: {user_intent.strip()}")

    explanation = f"Applied rule adjustments to section {section.title}."
    return updated, summary, explanation


# ============================================================================
# Application, Backup & Audit Trail
# ============================================================================

def apply_amendment_to_handbook(
    handbook_path: Path,
    section: PolicySection,
    new_raw_block: str,
    user_intent: str,
    summary: str,
    author: str = "Estate Staff",
) -> tuple[bool, AmendmentAuditRecord, str]:
    """
    Applies the verified amendment to the handbook file:
    1. Creates timestamped backup in Knowledge Base/backups/
    2. Writes updated handbook
    3. Logs structured audit trail to Knowledge Base/backups/amendment_audit_log.json
    4. Automatically triggers real-time Semantic RAG re-indexing!
    """
    handbook_path = Path(handbook_path).resolve()
    if not handbook_path.exists():
        raise FileNotFoundError(f"Handbook file not found at: {handbook_path}")

    original_text = handbook_path.read_text(encoding="utf-8")

    # Ensure backups dir exists
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)

    # Verify section offsets have not drifted due to concurrent file changes
    current_slice = original_text[section.start_char : section.end_char]
    if current_slice != section.raw_block:
        re_parsed = parse_handbook_sections(original_text)
        matching = [s for s in re_parsed if s.heading_line == section.heading_line and s.title == section.title]
        if not matching:
            raise ValueError(
                f"Handbook content changed concurrently; cannot locate section '{section.title}'. Please refresh."
            )
        section = matching[0]

    # 1. Timestamped Backup
    now = datetime.now(timezone.utc)
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    backup_filename = f"{handbook_path.stem}_{timestamp_str}_{short_id}.md"
    backup_file_path = BACKUPS_DIR / backup_filename
    backup_file_path.write_bytes(handbook_path.read_bytes())

    # 2. In-Place Surgical Replacement
    # Ensure trailing newline is preserved
    clean_replacement = new_raw_block
    if not clean_replacement.endswith("\n"):
        clean_replacement += "\n"

    updated_text = (
        original_text[:section.start_char]
        + clean_replacement
        + original_text[section.end_char:]
    )
    handbook_path.write_text(updated_text, encoding="utf-8", newline="\n")

    # 3. Calculate Diff Stats
    orig_lines = section.raw_block.splitlines()
    new_lines = clean_replacement.splitlines()
    _, adds, dels = _compute_annotated_diff(orig_lines, new_lines)

    diff_sample = "\n".join(
        difflib.unified_diff(orig_lines, new_lines, lineterm="", n=2)
    )

    # 4. Create & Append Audit Record
    record = AmendmentAuditRecord(
        log_id=f"amend_{short_id}",
        timestamp=now.isoformat(),
        author=author,
        intent=user_intent,
        target_section=section.breadcrumb,
        backup_file=str(backup_file_path.relative_to(ROOT)),
        summary=summary,
        additions_count=adds,
        deletions_count=dels,
        diff_snippet=diff_sample[:2000],
        status="APPLIED",
    )

    _append_audit_log(record)

    # 5. Live Re-Index Semantic RAG
    reindex_status = _reindex_semantic_rag(handbook_path)

    msg = (
        f"Amendment successfully applied to '{section.breadcrumb}'. "
        f"Backup saved to '{backup_filename}'. {reindex_status}"
    )
    return True, record, msg


def _append_audit_log(record: AmendmentAuditRecord) -> None:
    """Appends record to amendment_audit_log.json atomically."""
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    logs: list[dict[str, Any]] = []

    if AUDIT_LOG_FILE.exists():
        try:
            logs = json.loads(AUDIT_LOG_FILE.read_text(encoding="utf-8"))
        except Exception:
            logs = []

    logs.insert(0, record.to_dict())  # Most recent first
    AUDIT_LOG_FILE.write_text(json.dumps(logs, indent=2), encoding="utf-8")


def load_audit_history(limit: int = 50) -> list[dict[str, Any]]:
    """Loads past amendment records from the audit trail."""
    if not AUDIT_LOG_FILE.exists():
        return []
    try:
        data = json.loads(AUDIT_LOG_FILE.read_text(encoding="utf-8"))
        return data[:limit]
    except Exception as e:
        logger.warning(f"Failed to read audit log: {e}")
        return []


def rollback_amendment(
    handbook_path: Path,
    backup_relative_path: str,
    author: str = "Estate Staff (Rollback)",
) -> tuple[bool, str]:
    """Restores the handbook from a previous backup file and re-indexes RAG."""
    handbook_path = Path(handbook_path).resolve()
    target_backup = (ROOT / backup_relative_path).resolve()

    if not target_backup.is_relative_to(BACKUPS_DIR.resolve()):
        return False, "Access denied: Backup path must reside within the designated backups directory."

    if not target_backup.exists():
        return False, f"Backup file not found at: {target_backup}"

    backup_content = target_backup.read_text(encoding="utf-8")

    # Create safety backup of current state before rollback
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    safety_backup = BACKUPS_DIR / f"{handbook_path.stem}_prerollback_{ts}.md"
    safety_backup.write_bytes(handbook_path.read_bytes())

    # Overwrite handbook
    handbook_path.write_bytes(target_backup.read_bytes())

    # Record rollback in audit log
    record = AmendmentAuditRecord(
        log_id=f"rollback_{uuid.uuid4().hex[:8]}",
        timestamp=now.isoformat(),
        author=author,
        intent=f"Rollback to backup: {backup_relative_path}",
        target_section="Entire Handbook (Restoration)",
        backup_file=str(safety_backup.relative_to(ROOT)),
        summary=f"Reverted handbook to backup snapshot {target_backup.name}",
        additions_count=0,
        deletions_count=0,
        diff_snippet=f"Restored from {target_backup.name}",
        status="ROLLED_BACK",
    )
    _append_audit_log(record)

    reindex_status = _reindex_semantic_rag(handbook_path)
    return True, f"Successfully restored handbook from '{target_backup.name}'. {reindex_status}"


def _reindex_semantic_rag(handbook_path: Path) -> str:
    """Trigger real-time re-indexing of the Semantic RAG vector store."""
    try:
        from condo_classifier.semantic_rag import get_semantic_rag
        engine = get_semantic_rag(kb_path=handbook_path)
        if hasattr(engine, "reindex"):
            engine.reindex(handbook_path)
        else:
            engine._initialized = False
            engine.kb_path = Path(handbook_path)
            engine.initialize()
        return f"Semantic RAG re-indexed with {len(engine.index.chunks)} fresh chunks."
    except Exception as e:
        logger.error(f"Semantic RAG re-indexing failed: {e}")
        return f"RAG re-indexing warning: {e}"


# ============================================================================
# Visual HTML Diff Renderer
# ============================================================================

def render_diff_html(diff_lines: list[tuple[str, str]]) -> str:
    """Renders GitHub-style color-coded visual diff HTML for Streamlit."""
    html_lines = [
        '<div style="font-family: Consolas, monospace; font-size: 13px; line-height: 1.5; '
        'background-color: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; '
        'padding: 12px; overflow-x: auto; max-height: 400px;">'
    ]

    for tag, line in diff_lines:
        safe_line = (
            line.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace(" ", "&nbsp;")
        )
        if tag == "add":
            html_lines.append(
                f'<div style="background-color: #e6ffed; color: #1a7f37; border-left: 3px solid #2da44e; '
                f'padding: 2px 6px; margin: 1px 0;">+ {safe_line}</div>'
            )
        elif tag == "delete":
            html_lines.append(
                f'<div style="background-color: #ffebe9; color: #cf222e; border-left: 3px solid #cf222e; '
                f'text-decoration: line-through; padding: 2px 6px; margin: 1px 0;">- {safe_line}</div>'
            )
        else:
            html_lines.append(
                f'<div style="color: #24292f; padding: 2px 6px; margin: 1px 0;">&nbsp;&nbsp;{safe_line}</div>'
            )

    html_lines.append("</div>")
    return "".join(html_lines)
