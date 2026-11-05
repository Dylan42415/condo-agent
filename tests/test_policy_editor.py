"""
Unit and Integration Tests for By-Law Amendment Copilot (condo_classifier/policy_editor.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from condo_classifier.policy_editor import (
    AUDIT_LOG_FILE,
    BACKUPS_DIR,
    apply_amendment_to_handbook,
    generate_policy_amendment,
    load_audit_history,
    locate_target_section,
    parse_handbook_sections,
    reconcile_surgical_amendment,
    render_diff_html,
    rollback_amendment,
    _compute_annotated_diff,
)

SAMPLE_HANDBOOK_TEXT = """# Marina Heights Condominium - By-laws

Welcome to the handbook.

## 1. Moving In & Out

### 1.1 Moving Guidelines
Moving is allowed Mon-Fri 09:00-17:00.
Deposit is S$1,000.

## 2. Recreational Facilities

### 2.1 Facilities Tariff & Operating Hours
| Facility | Hours | Booking Fee | Security Deposit |
|---|---|---|---|
| BBQ Pit 1 | 10:00–22:00 | S$20.00 | S$50.00 |
| Function Room | 10:00–22:00 | S$50.00 | S$200.00 |

### 2.2 Pool Rules
No glassware allowed within 3m of the pool.

## 3. Renovation & Noise

### 3.1 Permissible Hours
Monday to Friday: 09:00 – 18:00.

### 3.2 Infractions & Fines
Unapproved Sunday drilling carries an immediate S$1,000 deduction from deposit.
"""


def test_parse_handbook_sections():
    sections = parse_handbook_sections(SAMPLE_HANDBOOK_TEXT)
    assert len(sections) >= 6

    # Verify offsets are contiguous and cover entire text
    assert sections[0].start_char == 0
    assert sections[-1].end_char == len(SAMPLE_HANDBOOK_TEXT)

    # Check breadcrumb structure
    tariff_sec = next(s for s in sections if "Tariff" in s.title)
    assert "Recreational Facilities" in tariff_sec.breadcrumb
    assert tariff_sec.level == 3
    assert "BBQ Pit 1" in tariff_sec.raw_block


def test_locate_target_section_bbq():
    sections = parse_handbook_sections(SAMPLE_HANDBOOK_TEXT)
    intent = "Increase BBQ pit deposit from $50 to $100"
    best_sec, score, rationale = locate_target_section(sections, intent, use_llm=False)

    assert "Facilities Tariff" in best_sec.title or "Recreational" in best_sec.breadcrumb
    assert score > 0
    assert "deposit" in rationale.lower() or "bbq" in rationale.lower() or "tariff" in rationale.lower()


def test_locate_target_section_drilling():
    sections = parse_handbook_sections(SAMPLE_HANDBOOK_TEXT)
    intent = "Change Sunday heavy drilling fine to $2,000"
    best_sec, score, rationale = locate_target_section(sections, intent, use_llm=False)

    assert "Renovation" in best_sec.breadcrumb or "Infractions" in best_sec.title
    assert score > 0


def test_generate_policy_amendment_deterministic():
    sections = parse_handbook_sections(SAMPLE_HANDBOOK_TEXT)
    target_sec = next(s for s in sections if "Tariff" in s.title)

    intent = "Increase BBQ deposit from $50 to $100"
    proposal = generate_policy_amendment(target_sec, intent)

    assert proposal.section == target_sec
    assert proposal.intent == intent
    assert proposal.additions_count > 0 or proposal.deletions_count > 0
    assert len(proposal.unified_diff) > 0
    assert len(proposal.diff_lines) > 0

    # Verify diff HTML rendering
    html = render_diff_html(proposal.diff_lines)
    assert "font-family: Consolas" in html


def test_apply_amendment_and_audit_trail(tmp_path: Path, monkeypatch):
    test_kb = tmp_path / "test_handbook.md"
    test_kb.write_text(SAMPLE_HANDBOOK_TEXT, encoding="utf-8")

    test_backups = tmp_path / "backups"
    test_audit = test_backups / "amendment_audit_log.json"

    # Patch module constants
    monkeypatch.setattr("condo_classifier.policy_editor.ROOT", tmp_path)
    monkeypatch.setattr("condo_classifier.policy_editor.BACKUPS_DIR", test_backups)
    monkeypatch.setattr("condo_classifier.policy_editor.AUDIT_LOG_FILE", test_audit)

    sections = parse_handbook_sections(test_kb.read_text(encoding="utf-8"))
    target_sec = next(s for s in sections if "Tariff" in s.title)

    new_block = target_sec.raw_block.replace("S$50.00", "S$100.00")
    assert "S$100.00" in new_block

    success, record, msg = apply_amendment_to_handbook(
        handbook_path=test_kb,
        section=target_sec,
        new_raw_block=new_block,
        user_intent="Increase BBQ deposit to $100",
        summary="Adjusted BBQ deposit to S$100.00",
        author="John Doe (Estate Manager)",
    )

    assert success is True
    assert record.log_id.startswith("amend_")
    assert record.author == "John Doe (Estate Manager)"

    # Check updated handbook file
    updated_text = test_kb.read_text(encoding="utf-8")
    assert "S$100.00" in updated_text
    assert "S$50.00" not in updated_text
    # Verify other sections were NOT modified
    assert "No glassware allowed within 3m" in updated_text
    assert "Deposit is S$1,000." in updated_text

    # Check backup file created
    backup_file = tmp_path / record.backup_file
    assert backup_file.exists()
    assert "S$50.00" in backup_file.read_text(encoding="utf-8")

    # Check audit log JSON
    history = load_audit_history()
    assert len(history) == 1
    assert history[0]["log_id"] == record.log_id
    assert history[0]["intent"] == "Increase BBQ deposit to $100"

    # Test Rollback
    rollback_ok, rollback_msg = rollback_amendment(test_kb, record.backup_file)
    assert rollback_ok is True
    assert "Successfully restored" in rollback_msg

    # Confirm handbook restored to original
    restored_text = test_kb.read_text(encoding="utf-8")
    assert "S$50.00" in restored_text
    assert "S$100.00" not in restored_text

    # Confirm audit log records the rollback
    new_history = load_audit_history()
    assert len(new_history) == 2
    assert new_history[0]["status"] == "ROLLED_BACK"


def test_reconcile_surgical_amendment_eliminates_spurious_diffs():
    """Verify that accidental micro-diffs on untouched lines and trailing blank deletions are eliminated."""
    orig_block = (
        "|3.00 pm / 4.00 pm to 10.00 pm). I/We hereby place the deposit of **$150.00** by way of <br>Cheque No.______________ . In addition to the refundable deposit, I/We also enclose a **$15.00**|\n"
        "|I/We acknowledge the return of my Cheque No._____________ for $_________.|\n"
        "|I/We acknowledge receipt of the refund of $_______in *cash/Cheque No._________.|\n\n"
    )
    # LLM altered $150 to $300 (intentional), dropped a trailing period on line 2 (accidental), and trimmed trailing newline
    prop_block_from_llm = (
        "|3.00 pm / 4.00 pm to 10.00 pm). I/We hereby place the deposit of **$300.00** by way of <br>Cheque No.______________ . In addition to the refundable deposit, I/We also enclose a **$15.00**|\n"
        "|I/We acknowledge the return of my Cheque No._____________ for $_________|\n"
        "|I/We acknowledge receipt of the refund of $_______in *cash/Cheque No._________.|"
    )

    reconciled = reconcile_surgical_amendment(orig_block, prop_block_from_llm)

    # Line 2 should have been snapped back to orig_block character-for-character
    assert "|I/We acknowledge the return of my Cheque No._____________ for $_________.|" in reconciled
    # Trailing newline/blank should be preserved
    assert reconciled.endswith("\n\n")

    # Compute diff lines
    orig_lines = orig_block.splitlines(keepends=True)
    rec_lines = reconciled.splitlines(keepends=True)
    diff_lines, adds, dels = _compute_annotated_diff(orig_lines, rec_lines)

    # Exactly 1 add and 1 del for the deposit change ($150 -> $300); zero spurious diffs
    assert adds == 1
    assert dels == 1
    non_unchanged = [d for d in diff_lines if d[0] != "unchanged"]
    assert len(non_unchanged) == 2
    assert "$150.00" in non_unchanged[0][1]
    assert "$300.00" in non_unchanged[1][1]


def test_locate_target_section_prioritizes_rules_over_forms():
    """Verify that operative rules are prioritized over application forms unless 'form' is requested."""
    text = Path("Knowledge Base/converted_tribeca_handbook.md").read_text(encoding="utf-8")
    sections = parse_handbook_sections(text)

    intent = "When booking a barbecue pit, Resident shall deposit from $150.00 per pit to 300$"
    best_sec, score, rationale = locate_target_section(sections, intent, use_llm=False)

    # Must select the operative Barbecue Area rule section, NOT Form E
    assert "Barbecue Area" in best_sec.title
    assert "Form" not in best_sec.title
    assert "Application" not in best_sec.breadcrumb
    assert score > 10.0

