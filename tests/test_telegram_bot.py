"""Integration and unit tests for telegram_bot.py handlers and workflows."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import telegram_bot
from condo_classifier.database import (
    init_db,
    create_ticket,
    get_ticket,
    list_tickets,
)


@pytest.fixture
def mock_db(tmp_path: Path, monkeypatch):
    test_db = tmp_path / "test_telegram.db"
    init_db(test_db)
    from condo_classifier import database, dispatch
    monkeypatch.setattr(database, "DEFAULT_DB_PATH", test_db)
    monkeypatch.setattr(dispatch, "DEFAULT_DB_PATH", test_db)
    return test_db


@pytest.mark.anyio
async def test_start_and_help_commands():
    # Test start_command
    update = MagicMock()
    update.message = AsyncMock()
    context = MagicMock()

    await telegram_bot.start_command(update, context)
    update.message.reply_text.assert_awaited_once()
    reply_start = update.message.reply_text.call_args[0][0]
    assert "Welcome to" in reply_start and ("Tribeca" in reply_start or "Oceanfront" in reply_start)

    # Test help_command
    update_help = MagicMock()
    update_help.message = AsyncMock()
    await telegram_bot.help_command(update_help, context)
    update_help.message.reply_text.assert_awaited_once()
    reply_help = update_help.message.reply_text.call_args[0][0]
    assert "Welcome to" in reply_help and ("Tribeca" in reply_help or "Oceanfront" in reply_help)


@pytest.mark.anyio
async def test_ask_command():
    # 1. Without args
    update_no_args = MagicMock()
    update_no_args.message = AsyncMock()
    context_no_args = MagicMock()
    context_no_args.args = []

    await telegram_bot.ask_command(update_no_args, context_no_args)
    update_no_args.message.reply_text.assert_awaited_once()
    assert "Estate Knowledge Graph Assistant" in update_no_args.message.reply_text.call_args[0][0]

    # 2. With query
    update_q = MagicMock()
    update_q.message = AsyncMock()
    update_q.message.chat = AsyncMock()
    context_q = MagicMock()
    context_q.args = ["Can", "contractor", "drill", "Saturday?"]

    await telegram_bot.ask_command(update_q, context_q)
    update_q.message.reply_text.assert_awaited_once()
    reply = update_q.message.reply_text.call_args[0][0]
    assert "Estate Knowledge Base" in reply


@pytest.mark.anyio
async def test_report_command(mock_db):
    # 1. /report without args shows guidance
    update_no_args = MagicMock()
    update_no_args.message = AsyncMock()
    update_no_args.message.photo = None
    context_no_args = MagicMock()
    context_no_args.args = []

    await telegram_bot.report_command(update_no_args, context_no_args)
    update_no_args.message.reply_text.assert_awaited_once()
    assert "Report an Estate Defect" in update_no_args.message.reply_text.call_args[0][0]

    # 2. /report with issue text triggers triage and creates ticket
    update_report = MagicMock()
    update_report.message = AsyncMock()
    update_report.message.photo = []
    update_report.message.chat = AsyncMock()
    update_report.message.chat.id = 998877
    update_report.message.chat_id = 998877
    update_report.message.from_user = MagicMock()
    update_report.message.from_user.full_name = "Alice Test"
    context_report = MagicMock()
    context_report.args = ["Water", "pipe", "burst", "in", "#08-14!"]

    await telegram_bot.report_command(update_report, context_report)
    update_report.message.reply_text.assert_awaited_once()
    reply = update_report.message.reply_text.call_args[0][0]
    assert "Ticket:" in reply
    assert "Plumbing" in reply or "Maintenance" in reply


@pytest.mark.anyio
async def test_status_command(mock_db):
    update = MagicMock()
    update.message = AsyncMock()
    context = MagicMock()

    # 1. No ticket ID provided
    context.args = []
    await telegram_bot.status_command(update, context)
    assert "Please provide a ticket ID" in update.message.reply_text.call_args[0][0]

    # 2. Ticket not found
    context.args = ["WO-NONEXISTENT"]
    await telegram_bot.status_command(update, context)
    assert "not found" in update.message.reply_text.call_args[0][0]

    # 3. Valid ticket
    t_id = "WO-20261103-9001"
    create_ticket(
        ticket_id=t_id,
        source="TELEGRAM",
        category="maintenance",
        priority="emergency",
        issue_summary="Pipe burst in unit",
        raw_message="Urgent pipe burst!",
        status="DISPATCHED",
        db_path=mock_db,
    )
    context.args = [t_id]
    await telegram_bot.status_command(update, context)
    reply = update.message.reply_text.call_args[0][0]
    assert t_id in reply
    assert "DISPATCHED" in reply


@pytest.mark.anyio
async def test_mytickets_command(mock_db):
    update = MagicMock()
    update.message = AsyncMock()
    update.message.chat = MagicMock()
    update.message.chat.id = 123456789
    context = MagicMock()

    # 1. No tickets
    await telegram_bot.mytickets_command(update, context)
    assert "no active work orders" in update.message.reply_text.call_args[0][0]

    # 2. With matching ticket
    t_id = "WO-20261103-9002"
    create_ticket(
        ticket_id=t_id,
        source="TELEGRAM",
        telegram_chat_id=123456789,
        category="noise",
        priority="normal",
        issue_summary="Loud music at midnight",
        raw_message="Neighbors playing loud party",
        status="SUBMITTED",
        db_path=mock_db,
    )
    await telegram_bot.mytickets_command(update, context)
    reply = update.message.reply_text.call_args[0][0]
    assert t_id in reply
    assert "Recent Condominium Tickets" in reply


@pytest.mark.anyio
async def test_handle_resident_message_emergency(mock_db):
    update = MagicMock()
    update.message = AsyncMock()
    update.message.text = "Emergency! Water pipe burst in kitchen #08-14, flooding floor!"
    update.message.caption = None
    update.message.photo = []
    update.message.chat = AsyncMock()
    update.message.chat.id = 987654321
    update.message.chat_id = 987654321
    update.message.from_user = MagicMock()
    update.message.from_user.full_name = "Jane Resident"
    context = MagicMock()

    await telegram_bot.handle_resident_message(update, context)

    update.message.reply_text.assert_awaited_once()
    reply = update.message.reply_text.call_args[0][0]
    assert "AUTOMATIC DISPATCH CONFIRMED" in reply or "TICKET RECEIVED" in reply
    assert "#08-14" in reply
    assert "Guardhouse Clearance" in reply

    # Verify ticket was persisted in SQLite DB
    tickets = list_tickets(db_path=mock_db)
    assert len(tickets) >= 1
    t = tickets[0]
    assert t["priority"] == "emergency"
    assert t["unit_number"] == "#08-14"
    assert t["assigned_entity"] is not None


@pytest.mark.anyio
async def test_handle_resident_message_general_inquiry(mock_db):
    update = MagicMock()
    update.message = AsyncMock()
    update.message.text = "Can I book the BBQ pit for next Saturday evening?"
    update.message.caption = None
    update.message.photo = []
    update.message.chat = AsyncMock()
    update.message.chat.id = 987654321
    update.message.chat_id = 987654321
    update.message.from_user = MagicMock()
    update.message.from_user.full_name = "John Doe"
    context = MagicMock()

    await telegram_bot.handle_resident_message(update, context)

    update.message.reply_text.assert_awaited_once()
    reply = update.message.reply_text.call_args[0][0]
    assert "TICKET RECEIVED & QUEUED" in reply
    assert "Ticket:" in reply


@pytest.mark.anyio
async def test_handle_callback_query_actions(mock_db):
    # 1. Guide water
    update1 = MagicMock()
    update1.callback_query = AsyncMock()
    update1.callback_query.data = "guide_water"
    update1.callback_query.message = AsyncMock()
    context1 = MagicMock()

    await telegram_bot.handle_callback_query(update1, context1)
    update1.callback_query.answer.assert_awaited_once()
    assert "Water Isolation" in update1.callback_query.message.reply_text.call_args[0][0]

    # 2. Call tech
    update2 = MagicMock()
    update2.callback_query = AsyncMock()
    update2.callback_query.data = "call_+65 6789-0123"
    update2.callback_query.message = AsyncMock()
    await telegram_bot.handle_callback_query(update2, context1)
    assert "+65 6789-0123" in update2.callback_query.message.reply_text.call_args[0][0]

    # 3. View Work Order
    t_id = "WO-20261103-9003"
    create_ticket(
        ticket_id=t_id,
        source="TELEGRAM",
        category="maintenance",
        priority="emergency",
        issue_summary="Elevator trapped #02-01",
        raw_message="Elevator stuck",
        assigned_entity="Schindler Lift",
        entity_contact="+65 6888-4321",
        status="DISPATCHED",
        db_path=mock_db,
    )
    update3 = MagicMock()
    update3.callback_query = AsyncMock()
    update3.callback_query.data = f"view_{t_id}"
    update3.callback_query.message = AsyncMock()
    await telegram_bot.handle_callback_query(update3, context1)
    assert t_id in update3.callback_query.message.reply_text.call_args[0][0]

    # 4. Resident Feedback Rating
    update4 = MagicMock()
    update4.callback_query = AsyncMock()
    update4.callback_query.data = f"rate_{t_id}_5"
    update4.callback_query.message = AsyncMock()
    await telegram_bot.handle_callback_query(update4, context1)
    assert "⭐⭐⭐⭐⭐" in update4.callback_query.message.reply_text.call_args[0][0]

    # Verify feedback was written to database
    ticket = get_ticket(t_id, db_path=mock_db)
    assert ticket["rating"] == 5


@pytest.mark.anyio
async def test_handle_resident_message_with_photo(mock_db, tmp_path: Path, monkeypatch):
    test_evidence = tmp_path / "evidence"
    monkeypatch.setattr(telegram_bot, "EVIDENCE_DIR", test_evidence)

    update = MagicMock()
    update.message = AsyncMock()
    update.message.text = None
    update.message.caption = "Broken lock on lobby door #01-01"
    
    mock_photo_file = AsyncMock()
    async def mock_download(stream):
        stream.write(b"fake-jpeg-binary-data")
    mock_photo_file.download_to_memory = mock_download
    
    photo_item = MagicMock()
    photo_item.get_file = AsyncMock(return_value=mock_photo_file)
    update.message.photo = [photo_item]

    update.message.chat = AsyncMock()
    update.message.chat.id = 987654321
    update.message.chat_id = 987654321
    update.message.from_user = MagicMock()
    update.message.from_user.full_name = "Jane Resident"
    context = MagicMock()

    await telegram_bot.handle_resident_message(update, context)

    update.message.reply_text.assert_awaited_once()
    reply = update.message.reply_text.call_args[0][0]
    assert "Ticket:" in reply

    # Verify photo was saved
    saved_photos = list(test_evidence.glob("*.jpg"))
    assert len(saved_photos) == 1
    assert saved_photos[0].read_bytes() == b"fake-jpeg-binary-data"


@pytest.mark.anyio
async def test_process_triage_offline_fallback(monkeypatch):
    monkeypatch.setattr(telegram_bot, "genai_client", None)
    decision = await telegram_bot.process_triage("Emergency! Water pipe burst in kitchen #08-14")
    assert decision.is_emergency is True
    assert decision.category == "maintenance"
    assert decision.priority == "emergency"
    assert decision.unit_number == "#08-14"
    assert ("Apex" in decision.assigned_entity or "Plumbing" in decision.assigned_entity or "Hai Yong" in decision.assigned_entity)
    assert decision.eta_minutes == 25


@pytest.mark.anyio
async def test_process_triage_gemini_exception_fallback(monkeypatch):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = TimeoutError("Simulated network timeout")
    monkeypatch.setattr(telegram_bot, "genai_client", mock_client)

    decision = await telegram_bot.process_triage("Lift stuck between floors 5 and 6 #05-02")
    assert decision.is_emergency is True
    assert decision.category == "maintenance"
    assert ("Schindler" in decision.assigned_entity or "Lift" in decision.assigned_entity or "KONE" in decision.assigned_entity)
    assert decision.eta_minutes == 20

