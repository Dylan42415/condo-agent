"""End-to-End Full Pipeline Lifecycle Integration Test for Condo Management Agent.

Verifies the complete 6-stage lifecycle:
1. Resident Input (Telegram / Web)
2. Classification & Priority Triage
3. SQLite Central Persistence with WAL mode
4. Automated Contractor Dispatch & Guardhouse Pre-Clearance & SLA Tracking
5. Semantic RAG Policy Retrieval & Grounded Synthesis
6. Status Transitions, Resident Notification, and Feedback Sign-Off
"""

import sqlite3
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from condo_classifier.database import (
    create_ticket,
    get_ticket,
    get_timeline,
    get_dispatches,
    init_db,
    get_connection,
    update_ticket_status,
    record_resident_feedback,
    generate_audit_report,
)
from condo_classifier.dispatch import (
    resolve_vendor_assignment,
    generate_guardhouse_pass,
    trigger_auto_dispatch,
    calculate_sla_metrics,
)
from condo_classifier.semantic_rag import get_semantic_rag
from condo_classifier.service import Classifier
from condo_classifier.backends import BaselineBackend
from condo_classifier.schema import ResidentRequest
from condo_classifier.notifications import notify_resident_status_change
import telegram_bot


def test_sqlite_wal_mode_and_schema_integrity(tmp_path: Path):
    """Stage 3 verification: SQLite DB Persistence with WAL mode & Foreign Keys."""
    db_file = tmp_path / "e2e_tickets.db"
    init_db(db_file)

    with get_connection(db_file) as conn:
        cursor = conn.cursor()
        
        # Verify WAL mode
        cursor.execute("PRAGMA journal_mode;")
        journal_mode = cursor.fetchone()[0]
        assert journal_mode.lower() == "wal"

        # Verify Foreign keys enabled
        cursor.execute("PRAGMA foreign_keys;")
        foreign_keys = cursor.fetchone()[0]
        assert foreign_keys == 1

        # Verify schema tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        assert {"tickets", "dispatches", "timeline_events"}.issubset(tables)

        # Verify columns in tickets table
        cursor.execute("PRAGMA table_info(tickets);")
        columns = {row[1] for row in cursor.fetchall()}
        assert "rating" in columns
        assert "feedback" in columns
        assert "photo_path" in columns


def test_end_to_end_emergency_lifecycle(tmp_path: Path, monkeypatch):
    """Full lifecycle execution for critical emergency request."""
    db_file = tmp_path / "e2e_emergency.db"
    init_db(db_file)
    from condo_classifier import database, dispatch
    monkeypatch.setattr(database, "DEFAULT_DB_PATH", db_file)
    monkeypatch.setattr(dispatch, "DEFAULT_DB_PATH", db_file)

    # Stage 1: Resident Input
    resident_msg = "EMERGENCY: Kitchen water pipe burst in unit #14-02! Active flooding into hallway!"
    unit_no = "#14-02"

    # Stage 2: Classification & Priority Triage (Offline baseline)
    classifier = Classifier(BaselineBackend())
    req = ResidentRequest(text=resident_msg, location=unit_no, channel="Telegram")
    classification = classifier.classify(req)
    
    assert classification.category.value == "maintenance"
    assert classification.priority.value in ["emergency", "high"]

    # Stage 3 & 4: Vendor Resolution & Auto-Dispatch & SLA
    vendor_info = resolve_vendor_assignment("maintenance", resident_msg)
    assert "Apex" in vendor_info["vendor_name"]
    
    ticket_id = "WO-20261103-E2E01"
    dispatch_res = trigger_auto_dispatch(
        ticket_id=ticket_id,
        category=classification.category.value,
        priority="emergency",
        issue_summary="Burst kitchen pipe with flooding",
        raw_message=resident_msg,
        unit_number=unit_no,
        assigned_vendor=vendor_info,
        db_path=db_file,
    )
    
    assert dispatch_res["vendor_name"] == vendor_info["vendor_name"]
    assert "PASS-1402-" in dispatch_res["passcode"]
    assert dispatch_res["sla_minutes"] == 25

    # Stage 4b: Real-time SLA tracking
    ticket = get_ticket(ticket_id, db_path=db_file)
    assert ticket is not None
    assert ticket["status"] == "DISPATCHED"
    
    sla_metrics = calculate_sla_metrics(ticket["created_at"], dispatch_res["sla_minutes"])
    assert sla_metrics["is_breached"] is False
    assert sla_metrics["remaining_minutes"] <= 25

    # Stage 5: Semantic RAG By-law Retrieval & Decision Synthesis
    sem_rag = get_semantic_rag()
    rag_res = sem_rag.query("emergency water leak isolation protocol and unit owner responsibility")
    assert rag_res.grounded is True
    assert len(rag_res.retrieved_chunks) > 0
    assert any("By-law" in b for b in rag_res.by_laws_cited) or "By-law" in rag_res.answer

    # Stage 6: Status Transitions & Proactive Resident Notification
    with patch("httpx.Client") as mock_client:
        mock_post = MagicMock()
        mock_post.status_code = 200
        mock_client.return_value.__enter__.return_value.post.return_value = mock_post
        
        # Staff transitions ticket to IN_PROGRESS
        ok1 = update_ticket_status(
            ticket_id,
            "IN_PROGRESS",
            actor="STAFF_CHENG",
            notes="Plumber on-site at #14-02",
            notify=True,
            db_path=db_file,
        )
        assert ok1 is True

        # Contractor finishes work -> RESOLVED
        ok2 = update_ticket_status(
            ticket_id,
            "RESOLVED",
            actor="CONTRACTOR_APEX",
            notes="Replaced burst copper joint, floor dried.",
            notify=True,
            db_path=db_file,
        )
        assert ok2 is True

    # Stage 7: Resident Feedback & Audit Trail
    record_resident_feedback(ticket_id, rating=5, feedback="Fast 20-min arrival, great work!", db_path=db_file)
    
    final_ticket = get_ticket(ticket_id, db_path=db_file)
    assert final_ticket["status"] == "RESOLVED"
    assert final_ticket["rating"] == 5
    assert final_ticket["feedback"] == "Fast 20-min arrival, great work!"

    timeline = get_timeline(ticket_id, db_path=db_file)
    assert len(timeline) >= 4  # Created, Dispatched, In-Progress, Resolved, Feedback
    
    dispatches = get_dispatches(ticket_id, db_path=db_file)
    assert len(dispatches) == 1
    assert dispatches[0]["delivery_status"] == "DELIVERED"

    # Audit Report Generation
    audit_report = generate_audit_report(ticket_id, db_path=db_file)
    assert ticket_id in audit_report
    assert "Apex 24/7 Rapid Plumbing" in audit_report
    assert "⭐⭐⭐⭐⭐" in audit_report
    assert "Replaced burst copper joint" in audit_report


def test_end_to_end_routine_inquiry_lifecycle(tmp_path: Path, monkeypatch):
    """Full lifecycle for routine facilities inquiry."""
    db_file = tmp_path / "e2e_routine.db"
    init_db(db_file)
    from condo_classifier import database, dispatch
    monkeypatch.setattr(database, "DEFAULT_DB_PATH", db_file)
    monkeypatch.setattr(dispatch, "DEFAULT_DB_PATH", db_file)

    # Stage 1: Inquiry Input
    query_text = "What is the deposit and hours to book BBQ Pit 1?"

    # Stage 2: Semantic RAG direct answer
    sem_rag = get_semantic_rag()
    rag_res = sem_rag.query(query_text)
    assert rag_res.grounded is True
    assert "$" in rag_res.answer or "deposit" in rag_res.answer.lower()
    assert len(rag_res.retrieved_chunks) >= 1

    # Stage 3: Request logged as work order
    ticket_id = "WO-20261103-INQ01"
    create_ticket(
        ticket_id=ticket_id,
        source="TELEGRAM",
        category="facilities",
        priority="normal",
        issue_summary="BBQ Pit 1 booking inquiry",
        raw_message=query_text,
        assigned_entity="Clubhouse Management Office",
        entity_contact="+65 6222-0000",
        eta_minutes=480,
        status="SUBMITTED",
        db_path=db_file,
    )

    t = get_ticket(ticket_id, db_path=db_file)
    assert t["status"] == "SUBMITTED"
    assert t["priority"] == "normal"
