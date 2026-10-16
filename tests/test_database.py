"""Unit tests for the centralized condo ticket and dispatch database."""

from pathlib import Path

from condo_classifier.database import (
    create_dispatch,
    create_ticket,
    generate_audit_report,
    get_dispatches,
    get_stats,
    get_ticket,
    get_timeline,
    init_db,
    list_tickets,
    record_resident_feedback,
    update_ticket_status,
)


def test_database_crud_and_timeline(tmp_path: Path):
    db_file = tmp_path / "test_tickets.db"
    init_db(db_file)

    # 1. Create ticket
    t_id = "WO-TEST-001"
    created = create_ticket(
        ticket_id=t_id,
        source="TELEGRAM",
        resident_name="Test Resident",
        unit_number="#08-14",
        category="maintenance",
        priority="emergency",
        issue_summary="Pipe burst in bathroom",
        raw_message="Pipe is burst and flooding unit #08-14!",
        assigned_entity="Apex Plumbing",
        entity_contact="+65 6789-0123",
        eta_minutes=25,
        status="DISPATCHED",
        db_path=db_file,
    )
    assert created == t_id

    # 2. Get ticket
    ticket = get_ticket(t_id, db_path=db_file)
    assert ticket is not None
    assert ticket["unit_number"] == "#08-14"
    assert ticket["priority"] == "emergency"
    assert ticket["status"] == "DISPATCHED"

    # 3. Create dispatch
    disp_id = create_dispatch(
        dispatch_id="DISP-TEST-001",
        ticket_id=t_id,
        contractor_name="Apex Plumbing",
        recipient_phone="+65 6789-0123",
        message_text="Urgent dispatch to #08-14",
        db_path=db_file,
    )
    assert disp_id == "DISP-TEST-001"

    dispatches = get_dispatches(t_id, db_path=db_file)
    assert len(dispatches) == 1
    assert dispatches[0]["contractor_name"] == "Apex Plumbing"

    # 4. Check timeline
    timeline = get_timeline(t_id, db_path=db_file)
    assert len(timeline) >= 2  # Created + Dispatched
    actors = [e["actor"] for e in timeline]
    assert "AI_AGENT" in actors
    assert "DISPATCH_ENGINE" in actors

    # 5. Update status
    updated = update_ticket_status(
        t_id, "RESOLVED", actor="STAFF", notes="Plumber repaired pipe", db_path=db_file
    )
    assert updated is True
    ticket_resolved = get_ticket(t_id, db_path=db_file)
    assert ticket_resolved["status"] == "RESOLVED"

    # 6. Check stats
    stats = get_stats(db_path=db_file)
    assert stats["total"] == 1
    assert stats["emergency"] == 1
    assert stats["resolved"] == 1


def test_list_tickets_filtering(tmp_path: Path):
    db_file = tmp_path / "test_filter.db"
    init_db(db_file)

    create_ticket(
        ticket_id="T-1",
        source="WEB",
        category="maintenance",
        priority="emergency",
        issue_summary="Elevator down",
        raw_message="Elevator broke",
        status="DISPATCHED",
        db_path=db_file,
    )
    create_ticket(
        ticket_id="T-2",
        source="TELEGRAM",
        category="noise",
        priority="normal",
        issue_summary="Loud music",
        raw_message="Loud party",
        status="SUBMITTED",
        db_path=db_file,
    )

    all_tickets = list_tickets(db_path=db_file)
    assert len(all_tickets) == 2

    emergency_only = list_tickets(priority="emergency", db_path=db_file)
    assert len(emergency_only) == 1
    assert emergency_only[0]["ticket_id"] == "T-1"

    noise_only = list_tickets(category="noise", db_path=db_file)
    assert len(noise_only) == 1
    assert noise_only[0]["ticket_id"] == "T-2"


def test_resident_feedback_and_audit_report(tmp_path: Path):
    db_file = tmp_path / "test_feedback.db"
    init_db(db_file)

    t_id = "WO-TEST-999"
    create_ticket(
        ticket_id=t_id,
        source="TELEGRAM",
        resident_name="Alice Tan",
        unit_number="#15-02",
        category="maintenance",
        priority="emergency",
        issue_summary="Ceiling pipe leak",
        raw_message="Water leaking from ceiling",
        assigned_entity="Apex Plumbing",
        entity_contact="+65 6789-0123",
        eta_minutes=20,
        status="RESOLVED",
        db_path=db_file,
    )

    create_dispatch(
        dispatch_id="DISP-999",
        ticket_id=t_id,
        contractor_name="Apex Plumbing",
        recipient_phone="+65 6789-0123",
        message_text="Urgent dispatch to #15-02",
        db_path=db_file,
    )

    ok = record_resident_feedback(
        t_id, rating=5, feedback="Plumber arrived in 15 mins, excellent!", db_path=db_file
    )
    assert ok is True

    ticket = get_ticket(t_id, db_path=db_file)
    assert ticket["rating"] == 5
    assert "excellent" in ticket["feedback"]

    report = generate_audit_report(t_id, db_path=db_file)
    assert "# Incident & Work Order Audit Report: WO-TEST-999" in report
    assert "Apex Plumbing" in report
    assert "⭐⭐⭐⭐⭐" in report
    assert "excellent" in report
