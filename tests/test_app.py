from streamlit.testing.v1 import AppTest

from condo_classifier.backends import ROOT


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_streamlit_classify_and_example_change():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()
    assert not app.exception
    next(item for item in app.text_input if item.label == "Location (optional)").set_value(
        "Block A"
    )
    button(app, "Classify request").click().run()
    assert not app.exception
    assert app.session_state.single_result["category"] == "maintenance"
    example = next(item for item in app.selectbox if item.label == "Try an example")
    example.select("Emergency report").run()
    assert app.session_state.request_location == ""
    button(app, "Classify request").click().run()
    assert not app.exception
    assert app.session_state.single_result["priority"] == "emergency"
    assert any("does not dispatch" in item.value for item in app.error)


def test_streamlit_evaluation_and_invalid_input():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()
    button(app, "Evaluate offline baseline").click().run()
    assert not app.exception
    assert app.session_state.evaluation["rows"] == 55
    app.text_area[0].set_value("   ")
    button(app, "Classify request").click().run()
    assert not app.exception
    assert app.error


def test_changing_engine_clears_result_from_previous_backend():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()
    button(app, "Classify request").click().run()
    assert app.session_state.single_result["backend"] == "baseline"
    next(item for item in app.selectbox if item.label == "Classification engine").select(
        "Ollama"
    ).run()
    assert not app.exception
    assert "single_result" not in app.session_state


def test_streamlit_dispatch_management_tab():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()
    assert not app.exception

    # Filter priority selectbox
    prio_sb = next(s for s in app.selectbox if s.label == "Filter Priority")
    prio_sb.select("Emergency").run()
    assert not app.exception

    # Filter status selectbox
    status_sb = next(s for s in app.selectbox if s.label == "Filter Status")
    status_sb.select("ALL").run()
    assert not app.exception

    # Select ticket selectbox
    ticket_sb = next(s for s in app.selectbox if s.label == "Select a Work Order to manage")
    assert ticket_sb.value is not None

    # Work Order actions
    prog_btn = next((b for b in app.button if "Mark In Progress" in b.label), None)
    res_btn = next((b for b in app.button if "Mark Resolved" in b.label), None)
    assert prog_btn is not None or res_btn is not None


def test_streamlit_estate_policy_rag_tab():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()
    assert not app.exception

    # Select a preset query
    preset_sb = next(s for s in app.selectbox if s.label == "Quick Query Presets")
    preset_sb.select(preset_sb.options[1]).run()
    assert not app.exception

    # Click Run Semantic RAG Query
    rag_btn = next(b for b in app.button if "Semantic" in b.label)
    rag_btn.click().run()
    assert not app.exception
    assert len(app.success) >= 1

    # Browse chunk selector
    browse_sb = next(s for s in app.selectbox if s.label == "Browse Indexed Chunk")
    assert browse_sb is not None


def test_streamlit_batch_processing_tab():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()
    assert not app.exception

    # Upload test CSV
    csv_bytes = b"text\nWater pipe burst in kitchen #08-14\nLoud music next door\n"
    csv_uploader = next(u for u in app.file_uploader if "Request CSV" in u.label)
    csv_uploader.upload("test_requests.csv", csv_bytes).run()
    assert not app.exception

    # Click Classify batch requests
    batch_btn = next(b for b in app.button if "Classify" in b.label and "requests" in b.label)
    batch_btn.click().run()
    assert not app.exception
    assert app.session_state.get("batch_result") is not None


def test_streamlit_bylaw_amendment_copilot():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()
    assert not app.exception

    # Find preset template
    copilot_sb = next(s for s in app.selectbox if s.label == "Quick Amendment Templates")
    assert len(copilot_sb.options) > 1
    copilot_sb.select(copilot_sb.options[1]).run()
    assert not app.exception

    # Directly set or ensure text area has intent
    intent_area = next(t for t in app.text_area if "Describe the by-law" in t.label)
    if not intent_area.value:
        intent_area.set_value(copilot_sb.options[1]).run()

    # Click Guide Me & Draft Changes
    draft_btn = next(b for b in app.button if "Guide Me" in b.label)
    draft_btn.click().run()
    assert not app.exception

    proposal = app.session_state.get("copilot_proposal")
    assert proposal is not None
    assert proposal.section is not None
    assert len(proposal.diff_lines) > 0

    # Verify Confirm button is available
    confirm_btn = next((b for b in app.button if "Confirm & Apply" in b.label), None)
    assert confirm_btn is not None


