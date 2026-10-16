"""
Automated Contractor Dispatch Engine, Guardhouse Clearance, and SLA Tracking.
"""

import json
import logging
import random
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from condo_classifier.backends import ROOT
from condo_classifier.database import (
    DEFAULT_DB_PATH,
    create_dispatch,
    create_ticket,
    get_ticket,
)

logger = logging.getLogger(__name__)
DEFAULT_VENDORS_FILE = ROOT / "data" / "vendors.json"


def load_vendor_directory(vendors_path: Path | str = DEFAULT_VENDORS_FILE) -> dict[str, Any]:
    """Loads the vendor routing dictionary from vendors.json."""
    path = Path(vendors_path)
    if not path.is_file():
        logger.warning("Vendors file not found at %s; returning empty directory.", path)
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_vendor_assignment(
    category: str,
    message_text: str = "",
    vendors_path: Path | str = DEFAULT_VENDORS_FILE,
) -> dict[str, Any]:
    """Resolves the best contractor assignment based on category and textual signals."""
    directory = load_vendor_directory(vendors_path)
    cat_lower = category.lower().strip()
    cat_vendors = directory.get(cat_lower, {})

    if not cat_vendors:
        # Fallback to general maintenance
        cat_vendors = directory.get("maintenance", {})

    # Detect subtopic keywords within maintenance
    text_lower = message_text.lower()
    if cat_lower == "maintenance":
        plumbing_words = ["pipe", "leak", "flood", "water", "plumb", "sink", "toilet", "tap"]
        if any(w in text_lower for w in plumbing_words):
            return cat_vendors.get("plumbing", cat_vendors.get("default", {}))
        if any(w in text_lower for w in ["lift", "elevator", "stuck", "entrap"]):
            return cat_vendors.get("lift", cat_vendors.get("default", {}))
        electrical_words = ["electric", "power", "spark", "blackout", "circuit", "fuse"]
        if any(w in text_lower for w in electrical_words):
            return cat_vendors.get("electrical", cat_vendors.get("default", {}))

    return cat_vendors.get("default", {})


def generate_guardhouse_pass(unit_number: str | None = None) -> str:
    """Generates an authorized security access code for contractor entry at the main guardhouse."""
    pin = random.randint(100000, 999999)
    prefix = re.sub(r"[^A-Za-z0-9]", "", unit_number or "ESTATE")
    return f"PASS-{prefix[:6]}-{pin}"


def build_dispatch_sms(
    ticket_id: str,
    unit_number: str | None,
    issue_summary: str,
    priority: str,
    sla_minutes: int,
    passcode: str,
    condo_name: str = "Oceanfront Residences",
) -> str:
    """Constructs the standard contractor dispatch SMS payload."""
    unit_str = unit_number or "Common Area"
    return (
        f"🚨 [URGENT DISPATCH - {condo_name}]\n"
        f"Ticket: {ticket_id} | Unit: {unit_str}\n"
        f"Fault: {issue_summary}\n"
        f"Priority: {priority.upper()} (SLA: {sla_minutes} mins)\n"
        f"Access: Pre-cleared at Guardhouse (Pass: {passcode})\n"
        "Reply 'CONFIRM' with your ETA."
    )


def trigger_auto_dispatch(
    ticket_id: str,
    category: str,
    priority: str,
    issue_summary: str,
    raw_message: str = "",
    unit_number: str | None = None,
    channel: str = "SMS",
    condo_name: str = "Oceanfront Residences",
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Automates vendor routing, guardhouse pass generation, and dispatch persistence."""
    vendor = resolve_vendor_assignment(category, raw_message)
    vendor_name = vendor.get("vendor_name", "Duty Emergency Contractor")
    phone = vendor.get(
        "emergency_phone" if priority.lower() == "emergency" else "phone", "+65 6000-0000"
    )
    sla_minutes = vendor.get(
        "emergency_sla_minutes" if priority.lower() == "emergency" else "routine_sla_minutes",
        30,
    )

    passcode = generate_guardhouse_pass(unit_number)
    sms_text = build_dispatch_sms(
        ticket_id=ticket_id,
        unit_number=unit_number,
        issue_summary=issue_summary,
        priority=priority,
        sla_minutes=sla_minutes,
        passcode=passcode,
        condo_name=condo_name,
    )

    # Ensure ticket exists in database to satisfy foreign key constraint
    if not get_ticket(ticket_id, db_path=db_path):
        create_ticket(
            ticket_id=ticket_id,
            source="DISPATCH_SYSTEM",
            category=category,
            priority=priority,
            issue_summary=issue_summary,
            raw_message=raw_message or issue_summary,
            unit_number=unit_number,
            status="DISPATCHED",
            assigned_entity=vendor_name,
            entity_contact=phone,
            eta_minutes=sla_minutes,
            db_path=db_path,
        )

    dispatch_id = f"DISP-{ticket_id.split('-')[-1]}"
    create_dispatch(
        dispatch_id=dispatch_id,
        ticket_id=ticket_id,
        contractor_name=vendor_name,
        recipient_phone=phone,
        message_text=sms_text,
        contact_channel=channel,
        delivery_status="DELIVERED",
        db_path=db_path,
    )

    return {
        "dispatch_id": dispatch_id,
        "vendor_name": vendor_name,
        "phone": phone,
        "sla_minutes": sla_minutes,
        "passcode": passcode,
        "sms_text": sms_text,
    }


def calculate_sla_metrics(created_at_iso: str, sla_minutes: int) -> dict[str, Any]:
    """Calculates live SLA remaining time, elapsed minutes, and breach status."""
    try:
        created_time = datetime.fromisoformat(created_at_iso)
        if created_time.tzinfo is None:
            created_time = created_time.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        elapsed_minutes = max(0, int((now - created_time).total_seconds() / 60))
        remaining_minutes = sla_minutes - elapsed_minutes
        is_breached = remaining_minutes < 0
        progress = min(1.0, max(0.0, elapsed_minutes / max(1, sla_minutes)))

        return {
            "elapsed_minutes": elapsed_minutes,
            "remaining_minutes": remaining_minutes,
            "is_breached": is_breached,
            "progress_percent": round(progress * 100, 1),
        }
    except Exception as exc:
        logger.error("Error calculating SLA metrics: %s", exc)
        return {
            "elapsed_minutes": 0,
            "remaining_minutes": sla_minutes,
            "is_breached": False,
            "progress_percent": 0.0,
        }
