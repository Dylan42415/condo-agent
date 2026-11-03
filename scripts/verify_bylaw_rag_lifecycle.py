#!/usr/bin/env python3
"""
Integration Verification Script for By-Law Amendment Copilot & Real-Time Semantic RAG Sync.
Tests the full amendment and rollback lifecycle on Knowledge Base/sample_condo_handbook.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Set up project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from condo_classifier.policy_editor import (
    DEFAULT_KB_PATH,
    BACKUPS_DIR,
    AUDIT_LOG_FILE,
    parse_handbook_sections,
    locate_target_section,
    generate_policy_amendment,
    apply_amendment_to_handbook,
    rollback_amendment,
    load_audit_history,
)
from condo_classifier.semantic_rag import get_semantic_rag


def file_sha256(path: Path) -> str:
    """Computes SHA256 checksum of file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_verification() -> dict:
    scorecard = {
        "status": "FAILED",
        "stages_passed": 0,
        "total_stages": 5,
        "metrics": {},
        "errors": [],
    }

    print("======================================================================")
    print("BY-LAW AMENDMENT COPILOT & SEMANTIC RAG REAL-TIME SYNC VERIFICATION")
    print("======================================================================")
    
    # 0. Pre-Flight Checks & Baseline Recording
    assert DEFAULT_KB_PATH.exists(), f"Handbook missing: {DEFAULT_KB_PATH}"
    original_text = DEFAULT_KB_PATH.read_text(encoding="utf-8")
    original_hash = file_sha256(DEFAULT_KB_PATH)
    print(f"[PRE-FLIGHT] Target handbook: {DEFAULT_KB_PATH.name} ({len(original_text)} bytes, hash: {original_hash[:10]}...)")
    assert "S$50.00" in original_text, "Expected original BBQ deposit S$50.00 in handbook"

    # Query initial RAG state
    t0 = time.perf_counter()
    engine = get_semantic_rag(DEFAULT_KB_PATH)
    init_query_res = engine.query("What is the security deposit for BBQ Pit 1?")
    rag_init_latency = time.perf_counter() - t0
    print(f"[PRE-FLIGHT] Initial RAG Query Latency: {rag_init_latency:.3f}s")
    print(f"[PRE-FLIGHT] Initial Answer Snippet: {init_query_res.answer[:120].strip()}...")
    assert "$50" in init_query_res.answer or "50.00" in init_query_res.answer, (
        f"Initial RAG query did not return $50 deposit: {init_query_res.answer}"
    )
    scorecard["metrics"]["initial_rag_query_latency_s"] = round(rag_init_latency, 4)

    backup_path = None
    applied_record = None

    try:
        # ====================================================================
        # Stage 1: Section Localization & Proposal Generation
        # ====================================================================
        print("\n--- STAGE 1: Section Localization & Proposal Generation ---")
        sections = parse_handbook_sections(original_text)
        assert len(sections) >= 5, f"Expected >= 5 sections, got {len(sections)}"

        intent = "Increase BBQ deposit from $50 to $120"
        t_loc_start = time.perf_counter()
        target_sec, loc_score, loc_rationale = locate_target_section(sections, intent)
        loc_latency = time.perf_counter() - t_loc_start
        print(f"[STAGE 1] Located target section: '{target_sec.breadcrumb}' (Score: {loc_score}, Latency: {loc_latency:.3f}s)")
        assert "4.1" in target_sec.title or "Facilities Tariff" in target_sec.title, (
            f"Unexpected section matched: {target_sec.title}"
        )

        t_prop_start = time.perf_counter()
        proposal = generate_policy_amendment(target_sec, intent)
        prop_latency = time.perf_counter() - t_prop_start
        print(f"[STAGE 1] Proposal drafted via {proposal.model_used} (Latency: {prop_latency:.3f}s)")
        print(f"[STAGE 1] Diff stats: +{proposal.additions_count} / -{proposal.deletions_count} lines")
        assert proposal.additions_count > 0 or proposal.deletions_count > 0
        assert "120" in proposal.proposed_raw_block, f"Proposal missing $120: {proposal.proposed_raw_block}"
        scorecard["stages_passed"] += 1
        scorecard["metrics"]["proposal_generation_latency_s"] = round(prop_latency, 4)

        # ====================================================================
        # Stage 2: In-Place Application & Timestamped Backup Verification
        # ====================================================================
        print("\n--- STAGE 2: In-Place Amendment Application & Backup Verification ---")
        t_apply_start = time.perf_counter()
        success, record, msg = apply_amendment_to_handbook(
            handbook_path=DEFAULT_KB_PATH,
            section=target_sec,
            new_raw_block=proposal.proposed_raw_block,
            user_intent=intent,
            summary="Increased BBQ Pavilion deposit from S$50.00 to S$120.00",
            author="Auditor / Estate Staff (Integration Test)",
        )
        apply_latency = time.perf_counter() - t_apply_start
        applied_record = record
        print(f"[STAGE 2] Amendment applied: {success} (Latency: {apply_latency:.3f}s)")
        print(f"[STAGE 2] Result message: {msg}")
        assert success is True

        # Check handbook content updated
        amended_text = DEFAULT_KB_PATH.read_text(encoding="utf-8")
        assert "S$120.00" in amended_text or "$120" in amended_text, "Amended handbook missing S$120.00"
        print(f"[STAGE 2] Handbook successfully updated in-place with new S$120.00 clause.")

        # Check backup file
        backup_path = ROOT / record.backup_file
        print(f"[STAGE 2] Verifying backup file at: {backup_path}")
        assert backup_path.exists(), f"Backup file not found at: {backup_path}"
        backup_text = backup_path.read_text(encoding="utf-8")
        backup_hash = file_sha256(backup_path)
        assert backup_hash == original_hash, (
            f"Backup hash mismatch! Expected {original_hash}, got {backup_hash}"
        )
        print(f"[STAGE 2] Backup integrity verified: 100% bit-exact match with pre-amendment state (SHA256: {backup_hash[:10]}...)")
        scorecard["stages_passed"] += 1
        scorecard["metrics"]["amendment_apply_latency_s"] = round(apply_latency, 4)
        scorecard["metrics"]["backup_file"] = record.backup_file
        scorecard["metrics"]["backup_hash_verified"] = True

        # ====================================================================
        # Stage 3: Audit Trail Log Verification
        # ====================================================================
        print("\n--- STAGE 3: Audit Trail Log Verification ---")
        assert AUDIT_LOG_FILE.exists(), f"Audit log missing: {AUDIT_LOG_FILE}"
        audit_history = load_audit_history(limit=10)
        assert len(audit_history) >= 1, "Audit history is empty"
        latest_entry = audit_history[0]
        print(f"[STAGE 3] Latest audit entry: ID={latest_entry['log_id']}, Status={latest_entry['status']}")
        assert latest_entry["log_id"] == record.log_id
        assert latest_entry["status"] == "APPLIED"
        assert latest_entry["intent"] == intent
        assert latest_entry["additions_count"] == record.additions_count
        assert latest_entry["deletions_count"] == record.deletions_count
        assert latest_entry["backup_file"] == record.backup_file
        print(f"[STAGE 3] Audit trail verified: Log ID {latest_entry['log_id']}, status 'APPLIED', diff +{latest_entry['additions_count']}/-{latest_entry['deletions_count']}")
        scorecard["stages_passed"] += 1

        # ====================================================================
        # Stage 4: Live Semantic RAG Re-indexing & Query Synchronization
        # ====================================================================
        print("\n--- STAGE 4: Live Semantic RAG Re-indexing & Immediate Reflection ---")
        t_rag_start = time.perf_counter()
        amended_rag_res = get_semantic_rag().query("What is the security deposit for BBQ Pit 1?")
        rag_amended_latency = time.perf_counter() - t_rag_start
        print(f"[STAGE 4] RAG Query Latency after amendment: {rag_amended_latency:.3f}s")
        print(f"[STAGE 4] Live RAG Answer:\n{amended_rag_res.answer}")

        # Assert answer immediately reflects $120 deposit
        assert "$120" in amended_rag_res.answer or "120.00" in amended_rag_res.answer, (
            f"FAIL: Semantic RAG did NOT reflect $120 deposit! Answer was: {amended_rag_res.answer}"
        )
        print(f"[STAGE 4] PASS: Live Semantic RAG answer immediately reflects the amended $120 deposit.")
        scorecard["stages_passed"] += 1
        scorecard["metrics"]["amended_rag_query_latency_s"] = round(rag_amended_latency, 4)

        # ====================================================================
        # Stage 5: Rollback Execution & Live RAG Restoration
        # ====================================================================
        print("\n--- STAGE 5: Rollback Execution & Live RAG Restoration ---")
        t_rollback_start = time.perf_counter()
        rollback_ok, rollback_msg = rollback_amendment(
            handbook_path=DEFAULT_KB_PATH,
            backup_relative_path=record.backup_file,
            author="Auditor / Estate Staff (Rollback Verification)",
        )
        rollback_latency = time.perf_counter() - t_rollback_start
        print(f"[STAGE 5] Rollback status: {rollback_ok} (Latency: {rollback_latency:.3f}s)")
        print(f"[STAGE 5] Rollback message: {rollback_msg}")
        assert rollback_ok is True

        # Assert handbook restored to original
        restored_text = DEFAULT_KB_PATH.read_text(encoding="utf-8")
        restored_hash = file_sha256(DEFAULT_KB_PATH)
        assert restored_hash == original_hash, (
            f"Restored hash mismatch! Expected {original_hash}, got {restored_hash}"
        )
        print(f"[STAGE 5] Handbook restored: 100% bit-exact match with original SHA256.")

        # Assert audit log records 'ROLLED_BACK'
        audit_history_after = load_audit_history(limit=10)
        latest_rollback_entry = audit_history_after[0]
        print(f"[STAGE 5] Audit after rollback: ID={latest_rollback_entry['log_id']}, Status={latest_rollback_entry['status']}")
        assert latest_rollback_entry["status"] == "ROLLED_BACK"
        assert "Rollback" in latest_rollback_entry["intent"]

        # Assert Semantic RAG immediately reflects original $50 deposit again
        t_rag2_start = time.perf_counter()
        restored_rag_res = get_semantic_rag().query("What is the security deposit for BBQ Pit 1?")
        rag_restored_latency = time.perf_counter() - t_rag2_start
        print(f"[STAGE 5] Restored RAG Query Latency: {rag_restored_latency:.3f}s")
        print(f"[STAGE 5] Restored RAG Answer:\n{restored_rag_res.answer}")

        assert "$50" in restored_rag_res.answer or "50.00" in restored_rag_res.answer, (
            f"FAIL: Semantic RAG did NOT reflect original $50 deposit after rollback! Answer: {restored_rag_res.answer}"
        )
        assert "$120" not in restored_rag_res.answer, (
            f"FAIL: Stale $120 deposit still present in RAG after rollback! Answer: {restored_rag_res.answer}"
        )
        print(f"[STAGE 5] PASS: Live Semantic RAG answer immediately reflects the restored $50 deposit.")
        scorecard["stages_passed"] += 1
        scorecard["metrics"]["rollback_latency_s"] = round(rollback_latency, 4)
        scorecard["metrics"]["restored_rag_query_latency_s"] = round(rag_restored_latency, 4)

        scorecard["status"] = "PASSED"
        print("\n======================================================================")
        print(f"VERIFICATION COMPLETE: ALL {scorecard['stages_passed']}/{scorecard['total_stages']} STAGES PASSED!")
        print("======================================================================")

    except Exception as e:
        print(f"\n[FATAL ERROR IN LIFECYCLE]: {e}")
        scorecard["errors"].append(str(e))
        import traceback
        traceback.print_exc()

        # Emergency cleanup if test failed midway
        if DEFAULT_KB_PATH.read_text(encoding="utf-8") != original_text:
            print("[CLEANUP] Reverting handbook to original state after failure...")
            DEFAULT_KB_PATH.write_text(original_text, encoding="utf-8", newline="\n")
            get_semantic_rag().reindex(DEFAULT_KB_PATH)

    return scorecard


if __name__ == "__main__":
    scorecard = run_verification()
    sys.exit(0 if scorecard["status"] == "PASSED" else 1)
