from unittest.mock import MagicMock, patch

from condo_classifier.notifications import notify_resident_status_change


def test_notify_resident_no_chat_id():
    ticket = {"ticket_id": "WO-TEST-001"}
    assert notify_resident_status_change(ticket, "IN_PROGRESS") is False


def test_notify_resident_no_token(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    ticket = {"ticket_id": "WO-TEST-001", "telegram_chat_id": 12345}
    assert notify_resident_status_change(ticket, "IN_PROGRESS") is False


def test_notify_resident_success(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "mock_token")
    ticket = {
        "ticket_id": "WO-TEST-001",
        "telegram_chat_id": 12345,
        "unit_number": "#08-14",
        "issue_summary": "Pipe leak",
        "assigned_entity": "Apex Plumbing",
    }
    with patch("httpx.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_instance.post.return_value = mock_resp
        mock_client_cls.return_value.__enter__.return_value = mock_instance

        res = notify_resident_status_change(ticket, "RESOLVED", notes="Fixed completely")
        assert res is True
        mock_instance.post.assert_called_once()
