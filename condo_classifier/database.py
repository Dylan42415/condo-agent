"""
Centralized SQLite Database Manager for Condo Help Tickets, Dispatches, and Audit Timelines.
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from condo_classifier.backends import ROOT
from condo_classifier.notifications import notify_resident_status_change

DEFAULT_DB_PATH = ROOT / "data" / "tickets.db"
EVIDENCE_DIR = ROOT / "data" / "evidence"


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> Path:
    path = Path(db_path)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                telegram_chat_id INTEGER,
                resident_name TEXT,
                unit_number TEXT,
                contact_number TEXT,
                category TEXT NOT NULL,
                priority TEXT NOT NULL,
                issue_summary TEXT NOT NULL,
                raw_message TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'SUBMITTED',
                assigned_entity TEXT,
                entity_contact TEXT,
                eta_minutes INTEGER,
                safety_instruction TEXT,
                photo_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dispatches (
                dispatch_id TEXT PRIMARY KEY,
                ticket_id TEXT NOT NULL,
                contractor_name TEXT NOT NULL,
                contact_channel TEXT NOT NULL,
                recipient_phone TEXT NOT NULL,
                message_text TEXT NOT NULL,
                delivery_status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id)
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS timeline_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL,
                actor TEXT NOT NULL,
                event_type TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id)
            );
            """
        )
        cursor.execute("PRAGMA table_info(tickets);")
        columns = [col[1] for col in cursor.fetchall()]
        if "rating" not in columns:
            cursor.execute("ALTER TABLE tickets ADD COLUMN rating INTEGER;")
        if "feedback" not in columns:
            cursor.execute("ALTER TABLE tickets ADD COLUMN feedback TEXT;")
        conn.commit()
    return path


def create_ticket(
    ticket_id: str,
    source: str,
    category: str,
    priority: str,
    issue_summary: str,
    raw_message: str,
    unit_number: str | None = None,
    resident_name: str | None = None,
    contact_number: str | None = None,
    telegram_chat_id: int | None = None,
    assigned_entity: str | None = None,
    entity_contact: str | None = None,
    eta_minutes: int | None = None,
    safety_instruction: str | None = None,
    photo_path: str | None = None,
    status: str = "SUBMITTED",
    db_path: Path | str = DEFAULT_DB_PATH,
) -> str:
    now_iso = datetime.now(UTC).isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO tickets (
                ticket_id, source, telegram_chat_id, resident_name, unit_number,
                contact_number, category, priority, issue_summary, raw_message,
                status, assigned_entity, entity_contact, eta_minutes,
                safety_instruction, photo_path, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_id,
                source,
                telegram_chat_id,
                resident_name,
                unit_number,
                contact_number,
                category,
                priority,
                issue_summary,
                raw_message,
                status,
                assigned_entity,
                entity_contact,
                eta_minutes,
                safety_instruction,
                photo_path,
                now_iso,
                now_iso,
            ),
        )
        created_desc = (
            f"Ticket created via {source}. Priority: {priority.upper()}, "
            f"Category: {category.title()}."
        )
        cursor.execute(
            """
            INSERT INTO timeline_events (ticket_id, actor, event_type, description, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                ticket_id,
                "AI_AGENT" if source == "TELEGRAM" else "RESIDENT",
                "TICKET_CREATED",
                created_desc,
                now_iso,
            ),
        )
        conn.commit()
    return ticket_id


def create_dispatch(
    dispatch_id: str,
    ticket_id: str,
    contractor_name: str,
    recipient_phone: str,
    message_text: str,
    contact_channel: str = "SMS",
    delivery_status: str = "SENT",
    db_path: Path | str = DEFAULT_DB_PATH,
) -> str:
    now_iso = datetime.now(UTC).isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO dispatches (
                dispatch_id, ticket_id, contractor_name, contact_channel,
                recipient_phone, message_text, delivery_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dispatch_id,
                ticket_id,
                contractor_name,
                contact_channel,
                recipient_phone,
                message_text,
                delivery_status,
                now_iso,
            ),
        )
        cursor.execute(
            """
            INSERT INTO timeline_events (ticket_id, actor, event_type, description, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                ticket_id,
                "DISPATCH_ENGINE",
                "CONTRACTOR_DISPATCHED",
                f"Auto-dispatched {contractor_name} via {contact_channel} ({recipient_phone}).",
                now_iso,
            ),
        )
        conn.commit()
    return dispatch_id


def get_ticket(ticket_id: str, db_path: Path | str = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def list_tickets(
    status: str | None = None,
    priority: str | None = None,
    category: str | None = None,
    limit: int = 100,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    query = "SELECT * FROM tickets WHERE 1=1"
    params: list[Any] = []
    if status and status != "ALL":
        query += " AND status = ?"
        params.append(status)
    if priority and priority != "ALL":
        query += " AND priority = ?"
        params.append(priority.lower())
    if category and category != "ALL":
        query += " AND category = ?"
        params.append(category.lower())

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def update_ticket_status(
    ticket_id: str,
    new_status: str,
    actor: str = "STAFF",
    notes: str = "",
    notify: bool = True,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> bool:
    now_iso = datetime.now(UTC).isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
        ticket_row = cursor.fetchone()
        if not ticket_row:
            return False

        cursor.execute(
            """
            UPDATE tickets
            SET status = ?, updated_at = ?
            WHERE ticket_id = ?
            """,
            (new_status, now_iso, ticket_id),
        )

        desc = f"Status updated to {new_status}."
        if notes:
            desc += f" Notes: {notes}"

        cursor.execute(
            """
            INSERT INTO timeline_events (ticket_id, actor, event_type, description, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ticket_id, actor, "STATUS_UPDATE", desc, now_iso),
        )

        if notify:
            ticket_dict = dict(ticket_row)
            try:
                sent = notify_resident_status_change(ticket_dict, new_status, notes)
                if sent:
                    cursor.execute(
                        """
                        INSERT INTO timeline_events (
                            ticket_id, actor, event_type, description, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            ticket_id,
                            "SYSTEM",
                            "RESIDENT_NOTIFIED",
                            "Proactive Telegram status update sent to resident.",
                            now_iso,
                        ),
                    )
            except Exception:
                pass

        conn.commit()
        return True


def get_timeline(ticket_id: str, db_path: Path | str = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM timeline_events
            WHERE ticket_id = ?
            ORDER BY created_at ASC
            """,
            (ticket_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_dispatches(ticket_id: str, db_path: Path | str = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM dispatches
            WHERE ticket_id = ?
            ORDER BY created_at DESC
            """,
            (ticket_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_stats(db_path: Path | str = DEFAULT_DB_PATH) -> dict[str, int]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tickets")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM tickets WHERE priority = 'emergency'")
        emergencies = cursor.fetchone()[0]

        active_query = (
            "SELECT COUNT(*) FROM tickets WHERE status IN "
            "('SUBMITTED', 'DISPATCHED', 'IN_PROGRESS')"
        )
        cursor.execute(active_query)
        active = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM tickets WHERE status = 'RESOLVED'")
        resolved = cursor.fetchone()[0]

        return {
            "total": total,
            "emergency": emergencies,
            "active": active,
            "resolved": resolved,
        }


def seed_demo_tickets(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Pre-populates sample tickets if the database is currently empty."""
    stats = get_stats(db_path)
    if stats["total"] > 0:
        return

    demos = [
        {
            "ticket_id": "WO-20261014-1001",
            "source": "TELEGRAM",
            "resident_name": "Resident #08-14",
            "unit_number": "#08-14",
            "category": "maintenance",
            "priority": "emergency",
            "issue_summary": "Kitchen water pipe burst causing active floor flooding",
            "raw_message": (
                "Emergency! The water pipe under our kitchen sink in #08-14 just burst "
                "and water is flooding the floor!"
            ),
            "status": "IN_PROGRESS",
            "assigned_entity": "Apex 24/7 Rapid Plumbing",
            "entity_contact": "+65 6789-0123",
            "eta_minutes": 25,
            "safety_instruction": "Please locate main water shutoff valve and turn clockwise.",
        },
        {
            "ticket_id": "WO-20261014-1002",
            "source": "TELEGRAM",
            "resident_name": "Resident #12-03",
            "unit_number": "#12-03",
            "category": "security",
            "priority": "high",
            "issue_summary": "Lost electronic lobby access card, requires security replacement",
            "raw_message": (
                "Lost my access card in the carpark. Need replacement urgently for access."
            ),
            "status": "SUBMITTED",
            "assigned_entity": "Main Guardhouse & Patrol",
            "entity_contact": "+65 6444-2222",
            "eta_minutes": 15,
            "safety_instruction": "Please visit the main guardhouse with photo identification.",
        },
        {
            "ticket_id": "WO-20261014-1003",
            "source": "WEB",
            "resident_name": "Resident #04-18",
            "unit_number": "#04-18",
            "category": "facilities",
            "priority": "normal",
            "issue_summary": "Barbecue pit reservation inquiry for upcoming Saturday",
            "raw_message": "Can I book the BBQ pit 2 for next Saturday evening from 6pm to 10pm?",
            "status": "RESOLVED",
            "assigned_entity": "Clubhouse Management Office",
            "entity_contact": "+65 6222-0000",
            "eta_minutes": 240,
            "safety_instruction": "Booking confirmed in resident portal.",
        },
    ]

    for d in demos:
        create_ticket(**d, db_path=db_path)
        if d["priority"] == "emergency":
            create_dispatch(
                dispatch_id="DISP-" + d["ticket_id"].split("-")[-1],
                ticket_id=d["ticket_id"],
                contractor_name=d["assigned_entity"],
                recipient_phone=d["entity_contact"],
                message_text=f"🚨 Emergency dispatch for {d['unit_number']}: {d['issue_summary']}",
                db_path=db_path,
            )


def record_resident_feedback(
    ticket_id: str,
    rating: int,
    feedback: str = "",
    db_path: Path | str = DEFAULT_DB_PATH,
) -> bool:
    """Stores resident 1-5 star rating and optional comments for a ticket."""
    now_iso = datetime.now(UTC).isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE tickets
            SET rating = ?, feedback = ?, updated_at = ?
            WHERE ticket_id = ?
            """,
            (rating, feedback, now_iso, ticket_id),
        )
        if cursor.rowcount == 0:
            return False

        stars = "⭐" * rating
        comment_note = f" Feedback: '{feedback}'" if feedback else ""
        cursor.execute(
            """
            INSERT INTO timeline_events (ticket_id, actor, event_type, description, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                ticket_id,
                "RESIDENT",
                "FEEDBACK_SUBMITTED",
                f"Resident rated service: {stars} ({rating}/5).{comment_note}",
                now_iso,
            ),
        )
        conn.commit()
        return True


def generate_audit_report(ticket_id: str, db_path: Path | str = DEFAULT_DB_PATH) -> str:
    """Generates an official post-incident audit summary in Markdown format."""
    ticket = get_ticket(ticket_id, db_path=db_path)
    if not ticket:
        return "Work Order not found."

    timeline = get_timeline(ticket_id, db_path=db_path)
    dispatches = get_dispatches(ticket_id, db_path=db_path)

    report_lines = [
        f"# Incident & Work Order Audit Report: {ticket['ticket_id']}",
        f"**Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## 1. Work Order Overview",
        f"- **Work Order ID:** `{ticket['ticket_id']}`",
        f"- **Source Channel:** {ticket['source']}",
        f"- **Unit Number:** {ticket['unit_number'] or 'Common Property'}",
        f"- **Resident:** {ticket['resident_name'] or 'Resident'}",
        f"- **Category:** {ticket['category'].title()}",
        f"- **Priority:** {ticket['priority'].upper()}",
        f"- **Current Status:** {ticket['status']}",
        f"- **Created At:** {ticket['created_at']}",
        f"- **Last Updated:** {ticket['updated_at']}",
        "",
        "## 2. Incident Summary & Safety Instruction",
        f"- **Issue Summary:** {ticket['issue_summary']}",
        f'- **Raw Resident Message:** "{ticket["raw_message"]}"',
        f"- **Safety Instruction Issued:** {ticket['safety_instruction'] or 'None'}",
        "",
        "## 3. Contractor Dispatch & Routing",
        f"- **Assigned Entity:** {ticket['assigned_entity'] or 'None'}",
        f"- **Contact:** `{ticket['entity_contact'] or 'None'}`",
        f"- **Target SLA:** {ticket['eta_minutes']} minutes",
    ]

    if dispatches:
        report_lines.append("\n### Outbound Dispatch Log:")
        for d in dispatches:
            report_lines.append(
                f"- `[{d['created_at'][:19]}]` {d['contractor_name']} via "
                f"{d['contact_channel']} ({d['recipient_phone']}): {d['delivery_status']}"
            )
            report_lines.append(f'  > "{d["message_text"]}"')

    report_lines.append("\n## 4. Complete Audit Timeline")
    for ev in timeline:
        report_lines.append(
            f"- `[{ev['created_at'][:19]}]` **[{ev['actor']}]** "
            f"{ev['event_type']}: {ev['description']}"
        )

    if ticket.get("rating"):
        report_lines.append("\n## 5. Resident Satisfaction & Resolution Sign-Off")
        report_lines.append(
            f"- **Resident Rating:** {'⭐' * ticket['rating']} ({ticket['rating']}/5)"
        )
        if ticket.get("feedback"):
            report_lines.append(f'- **Resident Feedback:** "{ticket["feedback"]}"')

    return "\n".join(report_lines)
