"""
Pytest integration test for By-Law Amendment Copilot & Live Semantic RAG Synchronization.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import pytest

from condo_classifier.policy_editor import (
    DEFAULT_KB_PATH,
    ROOT,
    AUDIT_LOG_FILE,
    parse_handbook_sections,
    locate_target_section,
    generate_policy_amendment,
    apply_amendment_to_handbook,
    rollback_amendment,
    load_audit_history,
)
from condo_classifier.semantic_rag import get_semantic_rag


def test_bylaw_amendment_lifecycle_and_rag_sync():
    """Verifies end-to-end amendment, backup, audit trail, live RAG sync, and rollback."""
    assert DEFAULT_KB_PATH.exists()
    original_text = DEFAULT_KB_PATH.read_text(encoding="utf-8")
    original_hash = hashlib.sha256(DEFAULT_KB_PATH.read_bytes()).hexdigest()

    try:
        # 1. Parse and locate BBQ section
        sections = parse_handbook_sections(original_text)
        intent = "Increase BBQ deposit from $50 to $120"
        target_sec, score, _ = locate_target_section(sections, intent)
        assert "4.1" in target_sec.title or "Facilities Tariff" in target_sec.title or "Barbecue" in target_sec.title

        # 2. Draft proposal
        proposal = generate_policy_amendment(target_sec, intent)
        assert "120" in proposal.proposed_raw_block

        # 3. Apply amendment
        success, record, _ = apply_amendment_to_handbook(
            handbook_path=DEFAULT_KB_PATH,
            section=target_sec,
            new_raw_block=proposal.proposed_raw_block,
            user_intent=intent,
            summary="Increased BBQ deposit to $120.00",
            author="Pytest Verifier",
        )
        assert success is True
        assert record.status == "APPLIED"

        # Assert backup created and matches original hash
        backup_file = ROOT / record.backup_file
        assert backup_file.exists()
        assert hashlib.sha256(backup_file.read_bytes()).hexdigest() == original_hash

        # Assert audit trail
        history = load_audit_history()
        assert len(history) >= 1
        assert history[0]["log_id"] == record.log_id
        assert history[0]["status"] == "APPLIED"

        # 4. Assert live Semantic RAG immediately reflects $120
        res1 = get_semantic_rag().query("What is the security deposit for BBQ Pit 1?")
        assert "$120" in res1.answer or "120.00" in res1.answer

        # 5. Rollback
        rollback_ok, _ = rollback_amendment(DEFAULT_KB_PATH, record.backup_file)
        assert rollback_ok is True

        # Assert handbook restored
        restored_hash = hashlib.sha256(DEFAULT_KB_PATH.read_bytes()).hexdigest()
        assert restored_hash == original_hash

        # Assert audit trail shows ROLLED_BACK
        history2 = load_audit_history()
        assert history2[0]["status"] == "ROLLED_BACK"

        # Assert live Semantic RAG immediately reflects original deposit again
        res2 = get_semantic_rag().query("What is the security deposit for BBQ Pit 1?")
        assert "$150" in res2.answer or "150.00" in res2.answer or "$50" in res2.answer or "50.00" in res2.answer
        assert "$120" not in res2.answer

    finally:
        # Guarantee restoration
        if DEFAULT_KB_PATH.read_text(encoding="utf-8") != original_text:
            DEFAULT_KB_PATH.write_text(original_text, encoding="utf-8", newline="\n")
            get_semantic_rag().reindex(DEFAULT_KB_PATH)
