"""
Notification service for resident status alerts via Telegram.
"""

import html
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def notify_resident_status_change(
    ticket: dict[str, Any],
    new_status: str,
    notes: str = "",
) -> bool:
    """Sends a proactive push notification to the resident via Telegram
    if telegram_chat_id is present."""
    chat_id = ticket.get("telegram_chat_id")
    if not chat_id:
        return False

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        logger.warning("TELEGRAM_BOT_TOKEN not configured; cannot send resident push notification.")
        return False

    status_emojis = {
        "SUBMITTED": "📋",
        "IN_PROGRESS": "🔧",
        "RESOLVED": "✅",
        "CANCELLED": "❌",
    }
    emoji = status_emojis.get(new_status, "ℹ️")
    ticket_id = html.escape(str(ticket.get("ticket_id", "")))
    unit = html.escape(str(ticket.get("unit_number") or "N/A"))
    summary = html.escape(str(ticket.get("issue_summary") or ""))

    msg_lines = [
        f"{emoji} <b>Work Order Update: {ticket_id}</b>",
        "",
        f"<b>Unit:</b> {unit}",
        f"<b>Status:</b> {new_status.replace('_', ' ').title()}",
        f"<b>Issue:</b> {summary}",
    ]

    if ticket.get("assigned_entity"):
        contractor = html.escape(str(ticket["assigned_entity"]))
        msg_lines.append(f"<b>Assigned:</b> {contractor}")

    if notes:
        clean_notes = html.escape(notes)
        msg_lines.append(f"<b>Staff Note:</b> <i>{clean_notes}</i>")

    if new_status == "RESOLVED":
        msg_lines.append("\n🌟 <b>How was our service?</b> Please rate below:")

    msg_lines.append(f"\nYou can check anytime using <code>/status {ticket_id}</code>.")
    text = "\n".join(msg_lines)

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    }

    if new_status == "RESOLVED":
        payload["reply_markup"] = {
            "inline_keyboard": [
                [
                    {"text": "⭐ 1", "callback_data": f"rate_{ticket_id}_1"},
                    {"text": "⭐⭐ 2", "callback_data": f"rate_{ticket_id}_2"},
                    {"text": "⭐⭐⭐ 3", "callback_data": f"rate_{ticket_id}_3"},
                    {"text": "⭐⭐⭐⭐ 4", "callback_data": f"rate_{ticket_id}_4"},
                    {"text": "⭐⭐⭐⭐⭐ 5", "callback_data": f"rate_{ticket_id}_5"},
                ]
            ]
        }

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                logger.info("Successfully sent status update to Telegram chat %s", chat_id)
                return True
            logger.error("Failed to send Telegram notification: %s %s", resp.status_code, resp.text)
            return False
    except Exception as exc:
        logger.error("Exception sending Telegram notification: %s", exc)
        return False
