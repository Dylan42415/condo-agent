from datetime import UTC, datetime, timedelta
from pathlib import Path

from condo_classifier.database import get_dispatches, init_db
from condo_classifier.dispatch import (
    calculate_sla_metrics,
    generate_guardhouse_pass,
    resolve_vendor_assignment,
    trigger_auto_dispatch,
)


def test_resolve_vendor_assignment():
    plumb = resolve_vendor_assignment("maintenance", "The kitchen water pipe burst")
    assert plumb["vendor_name"] == "Apex 24/7 Rapid Plumbing"
    assert plumb["emergency_sla_minutes"] == 25

    lift = resolve_vendor_assignment("maintenance", "Resident trapped in lift B")
    assert lift["vendor_name"] == "Schindler Express Response"

    electric = resolve_vendor_assignment("maintenance", "Electric sparks from power socket")
    assert electric["vendor_name"] == "VoltCare Electrical Hotline"

    sec = resolve_vendor_assignment("security", "Intruder spotted near boundary wall")
    assert sec["vendor_name"] == "Main Guardhouse & Patrol"


def test_generate_guardhouse_pass():
    passcode = generate_guardhouse_pass("#08-14")
    assert passcode.startswith("PASS-0814-")
    assert len(passcode) == 16


def test_trigger_auto_dispatch(tmp_path: Path):
    db_file = tmp_path / "dispatch_test.db"
    init_db(db_file)

    res = trigger_auto_dispatch(
        ticket_id="WO-20261015-8888",
        category="maintenance",
        priority="emergency",
        issue_summary="Burst pipe in kitchen",
        raw_message="Flooding from burst pipe",
        unit_number="#10-02",
        db_path=db_file,
    )

    assert res["vendor_name"] == "Apex 24/7 Rapid Plumbing"
    assert "PASS-1002-" in res["sms_text"]
    assert "SLA: 25 mins" in res["sms_text"]

    dispatches = get_dispatches("WO-20261015-8888", db_path=db_file)
    assert len(dispatches) == 1
    assert dispatches[0]["contractor_name"] == "Apex 24/7 Rapid Plumbing"
    assert dispatches[0]["delivery_status"] == "DELIVERED"


def test_calculate_sla_metrics():
    now = datetime.now(UTC)
    created_20m_ago = (now - timedelta(minutes=20)).isoformat()

    metrics = calculate_sla_metrics(created_20m_ago, sla_minutes=30)
    assert 19 <= metrics["elapsed_minutes"] <= 21
    assert 9 <= metrics["remaining_minutes"] <= 11
    assert metrics["is_breached"] is False

    created_40m_ago = (now - timedelta(minutes=40)).isoformat()
    breached_metrics = calculate_sla_metrics(created_40m_ago, sla_minutes=30)
    assert breached_metrics["is_breached"] is True
    assert breached_metrics["remaining_minutes"] < 0
