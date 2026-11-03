import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from condo_classifier.backends import ROOT, BaselineBackend, OllamaBackend
from condo_classifier.batch import classify_batch, export_csv, parse_batch, spreadsheet_cell
from condo_classifier.database import (
    generate_audit_report,
    get_dispatches,
    get_stats,
    get_timeline,
    init_db,
    list_tickets,
    seed_demo_tickets,
    update_ticket_status,
)
from condo_classifier.dispatch import calculate_sla_metrics
from condo_classifier.evaluation import evaluate
from condo_classifier.schema import ResidentRequest
from condo_classifier.policy_editor import (
    DEFAULT_KB_PATH,
    PolicyAmendmentProposal,
    apply_amendment_to_handbook,
    generate_policy_amendment,
    load_audit_history,
    locate_target_section,
    parse_handbook_sections,
    render_diff_html,
    rollback_amendment,
)
from condo_classifier.semantic_rag import get_semantic_rag
from condo_classifier.service import Classifier
from condo_classifier.taxonomy import LABELS

load_dotenv(ROOT / ".env")
init_db()
seed_demo_tickets()
st.set_page_config(
    page_title="Condo Agent | Autonomous Estate Coordinator", page_icon="🏢", layout="wide"
)
st.html("""
<style>
    .block-container { max-width: 1240px; padding-top: 2.5rem; }
    [data-testid="stSidebar"] { border-right: 1px solid #d4ded6; }
    h1 { letter-spacing: -0.04em; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
    .eyebrow { font-size: .76rem; letter-spacing: .16em; font-weight: 700; color: #17665e; }
    .intro { font-size: 1.08rem; color: #52665d; max-width: 760px; }
    .brand { font-size: 1.1rem; font-weight: 750; line-height: 1.4; }
    @media (max-width: 1100px) {
        [data-testid="stMain"] [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        [data-testid="stMain"] [data-testid="stColumn"] {
            flex: 1 1 260px;
            min-width: min(100%, 260px);
        }
    }
</style>
""")


@st.cache_resource
def baseline() -> BaselineBackend:
    return BaselineBackend()


with st.sidebar:
    st.html('<div class="brand">CONDO<br>AGENT</div>')
    st.caption("Autonomous Estate Operations · Multi-Agent Platform")
    st.divider()
    backend_name = st.selectbox("Classification engine", ["Offline baseline", "Ollama"])
    threshold = st.slider("Review threshold", 0.0, 1.0, 0.60, 0.05)
    st.caption(
        "Results below this score are flagged for review. Scores are uncalibrated estimates."
    )
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "gemma3:4b")
    if backend_name == "Ollama":
        host = st.text_input("Ollama server", value=host)
        model = st.text_input("Model name", value=model)
        st.caption("Cloud credentials are read from OLLAMA_API_KEY in your local .env file.")
        st.info(
            "Ollama sends entered text to the configured server. All LLM results need staff review."
        )
    else:
        st.caption(
            "Trained on 140 synthetic examples. Runs locally without an API key or download."
        )
    st.divider()
    st.markdown("**Current scope**")
    st.caption("Classify · Triage · Auto-Dispatch")
    st.caption(
        "Live operations hub with SQLite state, auto-contractor dispatch, and Telegram alerts."
    )


@st.cache_resource
def get_classifier(backend: str, srv_host: str, mdl_name: str, thresh: float) -> Classifier:
    if backend == "Offline baseline":
        return Classifier(baseline(), thresh)
    return Classifier(
        OllamaBackend(
            host=srv_host,
            model=mdl_name,
            api_key=os.getenv("OLLAMA_API_KEY", ""),
            timeout=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "45")),
        ),
        thresh,
    )


def service() -> Classifier:
    return get_classifier(backend_name, host, model, threshold)


active_settings = (backend_name, host, model, threshold)
if st.session_state.get("classification_settings") != active_settings:
    st.session_state.pop("single_result", None)
    st.session_state["classification_settings"] = active_settings


st.html('<div class="eyebrow">CONDO AGENT / AUTONOMOUS ESTATE OPERATIONS</div>')
st.title("Condo Agent: Resident Requests & Autonomous Dispatch")
st.html(
    '<p class="intro">Help property-management staff make sense of residents\' requests. '
    "Each message becomes a category, priority and suggested team, "
    "with clear reasons for review.</p>"
)
st.caption("10 categories  ·  English starter dataset  ·  Human review built in")
live_tab, single_tab, rag_tab, batch_tab, evaluation_tab, guide_tab, showcase_tab = st.tabs(
    [
        "Live Operations & Dispatch",
        "Classify a request",
        "Estate Knowledge Base & Semantic RAG",
        "Batch review",
        "Evaluation",
        "Category guide",
        "Showcase & setup",
    ]
)

with live_tab:
    st.subheader("Live Operations & Auto-Dispatch Hub")
    st.caption(
        "Central monitoring for resident requests, Telegram reports, "
        "and emergency contractor dispatches."
    )

    stats = get_stats()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Tickets", stats["total"])
    m2.metric("Active Work Orders", stats["active"])
    m3.metric("Emergency Dispatches", stats["emergency"])
    m4.metric("Resolved", stats["resolved"])

    st.divider()

    f_col1, f_col2, f_col3, f_col4 = st.columns([1, 1, 1, 1])
    with f_col1:
        status_filter = st.selectbox(
            "Filter Status",
            ["ALL", "SUBMITTED", "IN_PROGRESS", "RESOLVED"],
            index=0,
            key="live_filter_status",
        )
    with f_col2:
        priority_filter = st.selectbox(
            "Filter Priority",
            ["ALL", "Emergency", "High", "Normal", "Low"],
            index=0,
            key="live_filter_priority",
        )
    with f_col3:
        category_options = ["ALL"] + [label.name for label in LABELS.values()]
        selected_cat_name = st.selectbox(
            "Filter Category",
            category_options,
            index=0,
            key="live_filter_category",
        )
        cat_filter = "ALL"
        if selected_cat_name != "ALL":
            for k, v in LABELS.items():
                if v.name == selected_cat_name:
                    cat_filter = k.value
                    break
    with f_col4:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Data", key="refresh_tickets_btn"):
            st.rerun()

    tickets = list_tickets(
        status=status_filter if status_filter != "ALL" else None,
        priority=priority_filter.lower() if priority_filter != "ALL" else None,
        category=cat_filter if cat_filter != "ALL" else None,
    )

    if not tickets:
        st.info("No tickets found matching the selected filters.")
    else:
        left_queue, right_detail = st.columns([1.1, 1], gap="medium")

        with left_queue:
            st.markdown(f"**Work Orders Queue** ({len(tickets)} tickets)")
            df_display = pd.DataFrame(
                [
                    {
                        "ID": t["ticket_id"],
                        "Source": "🤖 Telegram" if t["source"] == "TELEGRAM" else "🌐 Web",
                        "Unit": t["unit_number"] or "—",
                        "Category": t["category"].title(),
                        "Priority": t["priority"].upper(),
                        "Status": t["status"],
                        "Assigned": t["assigned_entity"] or "Pending",
                    }
                    for t in tickets
                ]
            )
            st.dataframe(
                df_display,
                hide_index=True,
                width="stretch",
            )

            ticket_ids = [t["ticket_id"] for t in tickets]
            selected_ticket_id = st.selectbox(
                "Select a Work Order to manage",
                ticket_ids,
                key="selected_ticket_id",
            )

        with right_detail:
            selected_ticket = next(
                (t for t in tickets if t["ticket_id"] == selected_ticket_id), tickets[0]
            )
            with st.container(border=True):
                st.markdown(f"### {selected_ticket['ticket_id']}")
                b1, b2, b3 = st.columns(3)
                p_emoji = (
                    "🚨"
                    if selected_ticket["priority"] == "emergency"
                    else ("⚠️" if selected_ticket["priority"] == "high" else "ℹ️")
                )
                b1.markdown(f"**Priority:** {p_emoji} `{selected_ticket['priority'].upper()}`")
                b2.markdown(f"**Status:** `{selected_ticket['status']}`")
                b3.markdown(f"**Unit:** `{selected_ticket['unit_number'] or 'N/A'}`")

                st.markdown(f"**Issue Summary:** {selected_ticket['issue_summary']}")
                st.caption(f'Resident text: "{selected_ticket["raw_message"]}"')

                if selected_ticket["assigned_entity"]:
                    st.info(
                        f"**Dispatched Contractor:** {selected_ticket['assigned_entity']}  \n"
                        f"📞 Contact: `{selected_ticket['entity_contact'] or 'N/A'}` | "
                        f"⏱️ ETA: `{selected_ticket['eta_minutes']} mins`"
                    )
                    if selected_ticket["status"] != "RESOLVED":
                        target_sla = selected_ticket.get("eta_minutes") or 30
                        sla_info = calculate_sla_metrics(selected_ticket["created_at"], target_sla)
                        rem = sla_info["remaining_minutes"]
                        if sla_info["is_breached"]:
                            st.error(f"⏱️ **SLA Breached:** Exceeded target by {abs(rem)} minutes!")
                        else:
                            st.success(
                                f"⏱️ **SLA Active:** {rem} minutes remaining (Target: {target_sla}m)"
                            )
                        st.progress(min(1.0, sla_info["progress_percent"] / 100))

                if selected_ticket["safety_instruction"]:
                    st.warning(f"**Safety Instructions:** {selected_ticket['safety_instruction']}")

                photo_rel = selected_ticket.get("photo_path")
                if photo_rel:
                    photo_full = (
                        ROOT / photo_rel if not os.path.isabs(photo_rel) else Path(photo_rel)
                    )
                    if photo_full.exists():
                        st.image(
                            str(photo_full),
                            caption=f"Evidence Photo: {selected_ticket['ticket_id']}",
                            use_container_width=True,
                        )

                st.markdown("**Staff Work Order Actions**")
                action_c1, action_c2 = st.columns(2)
                notes_text = st.text_input(
                    "Optional update note to resident",
                    placeholder="e.g. Technician arrived at unit",
                    key=f"notes_{selected_ticket['ticket_id']}",
                )
                with action_c1:
                    is_in_prog = selected_ticket["status"] == "IN_PROGRESS"
                    if st.button(
                        "🔧 Mark In Progress",
                        key=f"prog_{selected_ticket['ticket_id']}",
                        disabled=is_in_prog,
                    ):
                        update_ticket_status(
                            selected_ticket["ticket_id"],
                            "IN_PROGRESS",
                            actor="STAFF",
                            notes=notes_text,
                        )
                        st.success("Ticket marked IN_PROGRESS & Resident notified!")
                        st.rerun()
                with action_c2:
                    is_resolved = selected_ticket["status"] == "RESOLVED"
                    if st.button(
                        "✅ Mark Resolved",
                        key=f"res_{selected_ticket['ticket_id']}",
                        disabled=is_resolved,
                    ):
                        update_ticket_status(
                            selected_ticket["ticket_id"],
                            "RESOLVED",
                            actor="STAFF",
                            notes=notes_text,
                        )
                        st.success("Ticket marked RESOLVED & Resident notified!")
                        st.rerun()

                with st.expander("Audit Timeline & History"):
                    timeline = get_timeline(selected_ticket["ticket_id"])
                    for ev in timeline:
                        st.markdown(
                            f"• `{ev['created_at'][:19]}` **[{ev['actor']}]** "
                            f"{ev['event_type']}: {ev['description']}"
                        )

                dispatches = get_dispatches(selected_ticket["ticket_id"])
                if dispatches:
                    with st.expander(f"Contractor Dispatches ({len(dispatches)})"):
                        for disp in dispatches:
                            st.write(
                                f"• **{disp['contractor_name']}** via {disp['contact_channel']} "
                                f"({disp['recipient_phone']})"
                            )
                            st.caption(
                                f"Status: {disp['delivery_status']} · "
                                f'Text: "{disp["message_text"]}"'
                            )

                if selected_ticket.get("rating"):
                    st.divider()
                    st.markdown(
                        f"**Resident Feedback:** {'⭐' * selected_ticket['rating']} "
                        f"({selected_ticket['rating']}/5)"
                    )
                    if selected_ticket.get("feedback"):
                        st.caption(f'"{selected_ticket["feedback"]}"')

                audit_report_md = generate_audit_report(selected_ticket["ticket_id"])
                st.download_button(
                    "📄 Download Incident Audit Report (.md)",
                    audit_report_md,
                    file_name=f"audit_report_{selected_ticket['ticket_id']}.md",
                    mime="text/markdown",
                    key=f"dl_audit_{selected_ticket['ticket_id']}",
                )

examples = {label.name: label.example for label in LABELS.values()}
examples["Emergency report"] = "My child is trapped in the lift at Block B."
examples["Multiple requests"] = "The tap in #05-12 is leaking. Also please send my payment receipt."


def load_example() -> None:
    if st.session_state.example in examples:
        st.session_state.request_text = examples[st.session_state.example]
        st.session_state.request_location = ""
        st.session_state.pop("single_result", None)


with single_tab:
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.subheader("Resident message")
        st.selectbox(
            "Try an example",
            list(examples),
            index=None,
            placeholder="Choose a sample request",
            key="example",
            on_change=load_example,
        )
        if "request_text" not in st.session_state:
            st.session_state.request_text = examples["Maintenance"]
        with st.form("request_form"):
            text = st.text_area("Request", key="request_text", height=180, max_chars=4000)
            channel_col, location_col = st.columns(2)
            with channel_col:
                channel = st.selectbox("Channel", ["Web", "WhatsApp", "Email", "Phone transcript"])
            with location_col:
                location = st.text_input(
                    "Location (optional)",
                    key="request_location",
                    max_chars=120,
                    placeholder="Block / unit / area",
                )
            submitted = st.form_submit_button("Classify request", type="primary", width="stretch")
        st.caption(
            "Classification inference sandbox. Live tickets and contractor dispatches "
            "are managed in the Live Operations Hub."
        )
        if submitted:
            st.session_state.pop("single_result", None)
            try:
                request = ResidentRequest(text=text, channel=channel, location=location)
                with st.spinner("Classifying the request…"):
                    result = service().classify(request)
                st.session_state.single_result = result.model_dump(mode="json")
                st.session_state.classified_text = request.text
            except ValidationError:
                st.error(
                    "Enter a request of 3–4,000 characters and a location of up to 120 characters."
                )
            except ValueError as exc:
                st.error(str(exc))
    with right:
        st.subheader("Classification")
        result = st.session_state.get("single_result")
        if not result:
            with st.container(border=True):
                st.markdown("**Start with a resident request.**")
                st.write(
                    "Enter a resident request or choose an example "
                    "to see its category, priority and suggested team."
                )
                st.caption("Ambiguous messages and urgent reports are flagged for staff review.")
        else:
            with st.container(border=True):
                if result["priority"] == "emergency":
                    st.error(
                        "Possible emergency. Contact on-site security or emergency services now. "
                        "This app does not dispatch help."
                    )
                elif result["review_required"]:
                    st.warning("Staff review required")
                else:
                    st.success("Classification ready for review")
                st.markdown(f"### {LABELS[result['category']].name}")
                a, b = st.columns(2)
                a.metric("Priority", result["priority"].title())
                score = result["confidence"]
                b.metric("Model score", f"{score:.0%}" if score is not None else "N/A")
                st.write(f"**Suggested team:** {result['suggested_team']}")
                st.write(f"**Location:** {result['location'] or 'Not provided'}")
                for reason in result["review_reasons"]:
                    st.write(f"• {reason}")
                if result["missing_details"]:
                    st.markdown("**Details to request**")
                    for detail in result["missing_details"]:
                        st.write(f"• {detail}")
                st.caption(
                    f"{result['backend']} · {result['model']} · {result['elapsed_ms']:.0f} ms"
                )
            st.caption(
                "Scores are uncalibrated estimates, not the probability "
                "that this decision is correct."
            )
            with st.expander("Inspect the classified message and JSON"):
                st.text(st.session_state.classified_text)
                st.json(result)
            if result["category_scores"]:
                with st.expander("Compare category scores"):
                    scores = pd.DataFrame(
                        [
                            {"Category": LABELS[key].name, "Score": value}
                            for key, value in result["category_scores"].items()
                        ]
                    ).sort_values("Score", ascending=False)
                    st.bar_chart(scores, x="Category", y="Score", horizontal=True, color="#17665E")
            st.download_button(
                "Download classification JSON",
                json.dumps(result, indent=2),
                "classification.json",
                "application/json",
            )

with rag_tab:
    st.subheader("Estate Knowledge Base & Semantic Chunking RAG Explorer")
    st.caption(
        "Production Semantic Chunking RAG pipeline over condominium by-laws, facility booking rules, and incident protocols. "
        "Splits documents via LangChain MarkdownHeaderTextSplitter with semantic breakpoint detection."
    )

    sem_engine = get_semantic_rag()
    chunks = sem_engine.index.chunks

    g_m1, g_m2, g_m3, g_m4 = st.columns(4)
    g_m1.metric("Indexed Chunks", len(chunks))
    section_count = len({c.section for c in chunks if c.section})
    g_m2.metric("Topic Sections", section_count)
    table_count = sum(1 for c in chunks if c.has_table)
    g_m3.metric("Atomic GFM Tables", table_count)
    g_m4.metric("Chunking Method", "LangChain + Semantic")

    with st.expander("📤 Ingest New Estate Handbook / By-Law (PDF)", expanded=False):
        st.markdown("**Upload a new Condominium By-Law, AGM Circular, or Resident Handbook PDF:**")
        st.caption(
            "Uses PyMuPDF4LLM & Gemini Vision OCR to convert PDFs into clean GitHub-Flavored Markdown, "
            "preserving tables, schedules, and headings, then automatically indexes it into vector memory."
        )
        uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"], key="rag_pdf_uploader")
        if uploaded_pdf:
            st.caption(f"Selected: `{uploaded_pdf.name}` ({len(uploaded_pdf.getvalue()) / 1024:.1f} KB)")
            if st.button(f"⚡ Convert & Index '{uploaded_pdf.name}' into RAG", type="primary"):
                with st.spinner("Converting PDF to lossless markdown and indexing semantic chunks..."):
                    from condo_classifier.pdf_converter import convert_pdf_to_markdown
                    upload_dir = ROOT / "Knowledge Base" / "uploads"
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    temp_pdf = upload_dir / uploaded_pdf.name
                    temp_pdf.write_bytes(uploaded_pdf.getvalue())

                    target_md = ROOT / "Knowledge Base" / f"converted_{Path(uploaded_pdf.name).stem}.md"
                    conv_res = convert_pdf_to_markdown(temp_pdf, output_path=target_md, engine="auto")

                    sem_engine = get_semantic_rag(kb_path=target_md)
                    sem_engine.initialize()

                st.success(f"Successfully converted & indexed **{uploaded_pdf.name}**!")
                c_c1, c_c2, c_c3, c_c4 = st.columns(4)
                c_c1.metric("Pages Processed", conv_res.page_count)
                c_c2.metric("Sections Indexed", len(conv_res.headings))
                c_c3.metric("Tables Extracted", conv_res.table_count)
                c_c4.metric("Engine Used", conv_res.engine_used)
                st.rerun()

    with st.expander("⚖️ AI By-Law Amendment Copilot (Human-in-the-Loop)", expanded=False):
        st.markdown("**Amend Rules, Tariffs & By-Laws in Natural Language:**")
        st.caption(
            "Instead of manually searching and editing raw markdown handbook files, tell the agent what rule to change. "
            "The agent guides you to the exact by-law section, drafts surgical amendments, and highlights the changes (diff) "
            "for human review before applying. Automated timestamped backup and audit logging are maintained for complete compliance."
        )

        active_kb_path = Path(getattr(sem_engine, "kb_path", DEFAULT_KB_PATH))
        if not active_kb_path.exists():
            active_kb_path = DEFAULT_KB_PATH

        st.caption(f"Active Handbook: `{active_kb_path.name}`")

        # Quick preset suggestions
        copilot_presets = [
            "Select an amendment instruction preset...",
            "Increase BBQ Pavilion deposit from $50 to $100 and allow booking 45 days in advance",
            "Increase unapproved Sunday drilling penalty from $1,000 to $2,000 with 72-hour pass suspension",
            "Increase overnight visitor parking fee from $15 to $25 per night",
            "Increase common corridor unleashed pet fine from $100 to $200",
            "Allow moving in on Saturdays up to 15:00 instead of 13:00",
        ]
        def _load_copilot_preset() -> None:
            preset = st.session_state.get("copilot_preset_select")
            if preset and preset != "Select an amendment instruction preset...":
                st.session_state.copilot_intent_input = preset

        selected_copilot_preset = st.selectbox(
            "Quick Amendment Templates",
            copilot_presets,
            index=0,
            key="copilot_preset_select",
            on_change=_load_copilot_preset,
        )

        if "copilot_intent_input" not in st.session_state:
            st.session_state.copilot_intent_input = ""

        user_intent_input = st.text_area(
            "Describe the by-law or house rule amendment:",
            key="copilot_intent_input",
            placeholder="e.g. Increase BBQ pit deposit to $100 and require booking 3 days in advance",
            height=80,
        )

        col_draft, col_clear = st.columns([1.5, 1])
        with col_draft:
            draft_clicked = st.button("🧭 1. Guide Me to Section & Draft Changes", type="primary", key="btn_draft_amendment")
        with col_clear:
            if st.button("🔄 Reset Draft", key="btn_reset_draft"):
                st.session_state.pop("copilot_proposal", None)
                st.session_state.pop("copilot_applied_msg", None)
                st.rerun()

        if draft_clicked and user_intent_input.strip():
            with st.spinner("Agent analyzing handbook, locating target section, and drafting proposal..."):
                hb_text = active_kb_path.read_text(encoding="utf-8")
                sections = parse_handbook_sections(hb_text)
                target_sec, match_score, rationale = locate_target_section(sections, user_intent_input.strip())
                proposal = generate_policy_amendment(target_sec, user_intent_input.strip())
                st.session_state.copilot_proposal = proposal
                st.session_state.copilot_rationale = rationale

        # Display Proposal for Human-in-the-Loop Verification
        proposal: PolicyAmendmentProposal | None = st.session_state.get("copilot_proposal")
        if proposal:
            st.divider()
            st.markdown("#### 🎯 Step 2: Agent Guidance & Human Review")
            st.info(f"📍 **Target Section Located:** `{proposal.section.breadcrumb}`")
            if "copilot_rationale" in st.session_state:
                st.caption(f"💡 **Agent Rationale:** {st.session_state.copilot_rationale}")

            st.markdown(f"**Proposed Amendment Summary:** {proposal.explanation}")
            if proposal.summary_bullets:
                for b in proposal.summary_bullets:
                    st.markdown(f"- {b}")

            # Diff Highlighting
            st.markdown("##### 🔍 Proposed Changes Diff (GitHub-Style)")
            d_col1, d_col2, d_col3 = st.columns([1, 1, 2])
            d_col1.metric("Lines Added", f"+{proposal.additions_count}")
            d_col2.metric("Lines Removed", f"-{proposal.deletions_count}")
            d_col3.caption(f"Engine: `{proposal.model_used}` · Human-in-the-Loop review active")

            st.markdown(render_diff_html(proposal.diff_lines), unsafe_allow_html=True)

            with st.expander("✏️ Human-in-the-Loop: Inspect & Fine-Tune Markdown Before Applying", expanded=False):
                edited_markdown = st.text_area(
                    "You can tweak the proposed markdown below before saving:",
                    value=proposal.proposed_raw_block,
                    height=220,
                    key="copilot_edited_markdown",
                )

            # Confirm and Apply Button
            col_apply, col_discard = st.columns([2, 1])
            with col_apply:
                apply_clicked = st.button(
                    "✅ 2. Confirm & Apply Changes to Handbook",
                    type="primary",
                    key="btn_confirm_amendment",
                )
            with col_discard:
                if st.button("❌ Discard", key="btn_discard_draft"):
                    st.session_state.pop("copilot_proposal", None)
                    st.rerun()

            if apply_clicked:
                with st.spinner("Creating timestamped backup, updating handbook, and re-indexing Semantic RAG..."):
                    text_to_save = st.session_state.get("copilot_edited_markdown", proposal.proposed_raw_block)
                    ok, record, msg = apply_amendment_to_handbook(
                        handbook_path=active_kb_path,
                        section=proposal.section,
                        new_raw_block=text_to_save,
                        user_intent=proposal.intent,
                        summary=proposal.explanation,
                        author="Estate Staff (UI)",
                    )
                    if ok:
                        st.session_state.copilot_applied_msg = {
                            "text": msg,
                            "backup": record.backup_file,
                            "log_id": record.log_id,
                        }
                        st.session_state.pop("copilot_proposal", None)
                        st.rerun()

        if "copilot_applied_msg" in st.session_state:
            applied_info = st.session_state.copilot_applied_msg
            st.success(f"🎉 **{applied_info['text']}**")
            st.caption(f"Audit Log ID: `{applied_info['log_id']}` · Backup File: `{applied_info['backup']}`")

        # Audit History & Rollback Sub-Expander
        with st.expander("📜 Amendment Audit Trail & Version Rollback", expanded=False):
            history = load_audit_history(limit=10)
            if not history:
                st.caption("No amendments recorded in this session yet.")
            else:
                for item in history:
                    with st.container(border=True):
                        h_col1, h_col2 = st.columns([3, 1])
                        with h_col1:
                            st.markdown(f"**{item.get('status', 'APPLIED')}** — `{item.get('log_id')}` ({item.get('timestamp')[:19]})")
                            st.markdown(f"**Intent:** {item.get('intent')}")
                            st.caption(f"Section: `{item.get('target_section')}` · Author: `{item.get('author')}`")
                            st.caption(f"Summary: {item.get('summary')}")
                        with h_col2:
                            st.caption(f"+{item.get('additions_count', 0)} / -{item.get('deletions_count', 0)} lines")
                            if item.get("status") != "ROLLED_BACK" and item.get("backup_file"):
                                if st.button(f"↩️ Rollback", key=f"rb_{item['log_id']}"):
                                    rb_ok, rb_msg = rollback_amendment(active_kb_path, item["backup_file"])
                                    if rb_ok:
                                        st.success(rb_msg)
                                        st.rerun()
                                    else:
                                        st.error(rb_msg)

    st.divider()

    st.markdown("### 🔍 Test Semantic Chunking RAG")
    preset_scenarios = [
        "Select a test query...",
        "🎾 What are the tennis court operating hours, session limits, and booking quotas per week?",
        "🔑 How many access cards is a 3-bedroom apartment entitled to, and what is the replacement fee?",
        "🔨 What are the permissible hours for contractor renovation works, and how much is the security deposit?",
        "🚗 What is the maximum vehicle height limit for Basement 1 and Basement 2 carparks?",
        "🚿 Can I wash my car in my regular parking lot, or are there designated wash bays?",
        "🐕 What are the rules on keeping dogs, and are pets allowed in the clubhouse or swimming pool?",
        "📞 Who should I contact for a lift breakdown, plumbing emergency, or the security guardhouse?",
        "🍖 How much does it cost to book a BBQ pit, and what is the gas cylinder fee and deposit?",
    ]
    selected_preset = st.selectbox("Quick Query Presets", preset_scenarios, index=0)

    default_q = ""
    if selected_preset != "Select a test query...":
        default_q = selected_preset.split(" ", 1)[1] if " " in selected_preset else selected_preset

    user_rag_query = st.text_input("Resident Question or By-Law Inquiry", value=default_q)

    topk_col, btn_col = st.columns([1, 2])
    with topk_col:
        top_k = st.slider("Top Chunks to Retrieve (k)", 1, 5, 3)
    with btn_col:
        st.write("")
        st.write("")
        run_rag = st.button("🚀 Run Semantic Chunking RAG Query", type="primary")

    if user_rag_query and run_rag:
        with st.spinner("Retrieving semantic chunks and synthesizing answer..."):
            rag_result = sem_engine.query(user_rag_query, top_k=top_k)

        st.success(
            f"Grounded Context Retrieved ({len(rag_result.retrieved_chunks)} Semantic Chunks in {rag_result.latency_seconds:.2f}s)"
        )

        ans_col, chunk_col = st.columns([1.1, 0.9])
        with ans_col:
            st.markdown("#### Grounded Answer Synthesis")
            st.markdown(rag_result.answer)

            if rag_result.by_laws_cited:
                b_str = ", ".join(f"`{b}`" for b in rag_result.by_laws_cited)
                st.markdown(f"**Cited By-Laws:** {b_str}")
            if rag_result.source_sections:
                s_str = ", ".join(f"`{s}`" for s in rag_result.source_sections)
                st.markdown(f"**Source Sections:** {s_str}")

        with chunk_col:
            st.markdown("#### Retrieved Semantic Chunks")
            for i, c in enumerate(rag_result.retrieved_chunks, 1):
                with st.expander(f"Chunk {i}: {c.header_breadcrumb}", expanded=(i == 1)):
                    st.caption(f"Source: `{c.source_file}` | Tokens: ~{c.token_count} | Has Table: {c.has_table}")
                    st.markdown(c.content)

        with st.expander("Inspect Raw Grounded Context Prompt"):
            st.code(rag_result.context_text, language="markdown")

    st.divider()
    st.markdown("### 📚 Estate Knowledge Base Chunks Browser")
    if chunks:
        chunk_options = [f"[{c.chunk_id}] {c.header_breadcrumb}" for c in chunks]
        selected_chunk_idx = st.selectbox(
            "Browse Indexed Chunk",
            range(len(chunks)),
            format_func=lambda idx: chunk_options[idx],
            index=0,
        )
        if selected_chunk_idx is not None:
            sel_chunk = chunks[selected_chunk_idx]
            st.markdown(f"**Section Breadcrumb:** `{sel_chunk.header_breadcrumb}`")
            st.caption(f"File: `{sel_chunk.source_file}` | Words: {sel_chunk.token_count} | Has Table: {sel_chunk.has_table}")
            st.markdown(sel_chunk.content)

with batch_tab:
    st.subheader("Classify requests together")
    st.write("Upload a UTF-8 CSV with a `text` column. Optional columns: `channel`, `location`.")
    st.caption("Up to 100 requests and 2 MB. Each row is checked independently.")
    st.download_button(
        "Download sample CSV",
        (ROOT / "data/batch_example.csv").read_bytes(),
        "batch_example.csv",
        "text/csv",
    )
    upload = st.file_uploader("Request CSV", type="csv")
    if upload:
        content = upload.getvalue()
        signature = hashlib.sha256(
            content + f"{backend_name}|{host}|{model}|{threshold}".encode()
        ).hexdigest()
        try:
            rows = parse_batch(content)
            st.caption(f"{len(rows)} requests · {backend_name}")
            if st.button(f"Classify {len(rows)} requests", type="primary"):
                with st.spinner("Classifying the inbox…"):
                    batch_results = classify_batch(service(), rows)
                st.session_state.batch_result = (signature, batch_results)
            stored_batch = st.session_state.get("batch_result")
            if stored_batch and stored_batch[0] == signature:
                st.dataframe(
                    pd.DataFrame(stored_batch[1])
                    .reindex(columns=["row", "text", "category", "priority", "review_required"])
                    .map(spreadsheet_cell),
                    column_config={
                        "row": st.column_config.NumberColumn("Row", width="small"),
                        "text": st.column_config.TextColumn("Resident request", width="medium"),
                        "category": st.column_config.TextColumn("Category", width="small"),
                        "priority": st.column_config.TextColumn("Priority", width="small"),
                        "review_required": st.column_config.CheckboxColumn(
                            "Review required", width="small"
                        ),
                    },
                    hide_index=True,
                    width="stretch",
                )
                st.download_button(
                    "Download classifications CSV",
                    export_csv(stored_batch[1]),
                    "classified_requests.csv",
                    "text/csv",
                )
        except ValueError as exc:
            st.error(str(exc))

with evaluation_tab:
    st.subheader("Evaluate the classifier")
    st.write(
        "Evaluate the offline baseline on 55 held-out synthetic requests, "
        "including five emergencies."
    )
    st.info(
        "These examples test the starter workflow. "
        "They do not measure performance on real resident requests."
    )
    if st.button("Evaluate offline baseline"):
        with st.spinner("Evaluating the held-out examples…"):
            st.session_state.evaluation = evaluate(Classifier(baseline(), threshold))
    report = st.session_state.get("evaluation")
    if report:
        a, b, c, d = st.columns(4)
        a.metric("Category accuracy", f"{report['category_accuracy']:.1%}")
        b.metric("Macro F1", f"{report['macro_f1']:.3f}")
        c.metric("Review rate", f"{report['review_rate']:.1%}")
        d.metric("Emergency recall", f"{report['emergency_recall']:.0%}")
        st.caption(
            f"{report['rows']} synthetic examples · threshold {report['review_threshold']:.2f} · "
            f"{report['provider_errors']} provider errors · includes deterministic safety screening"
        )
        st.markdown(
            "**Confusion matrix** · Rows are expected categories; columns are predicted categories."
        )
        st.dataframe(
            pd.DataFrame(
                report["confusion_matrix"], index=report["labels"], columns=report["labels"]
            ),
            width="stretch",
        )
        misses = [
            row
            for row in report["predictions"]
            if row["category"] != row["expected_category"]
            or row["priority"] != row["expected_priority"]
        ]
        with st.expander(f"Inspect {len(misses)} category or priority mismatches"):
            st.dataframe(pd.DataFrame(misses), hide_index=True)
        st.download_button(
            "Download evaluation report",
            json.dumps(report, indent=2),
            "evaluation.json",
            "application/json",
        )

with guide_tab:
    st.subheader("Understand the request categories")
    st.write(
        "Use these definitions to classify requests consistently and prepare labelled examples."
    )
    for key, label in LABELS.items():
        with st.expander(f"{label.name} · {label.team}"):
            st.write(label.description)
            st.caption(f"Example: {label.example}")
            st.code(key.value, language=None)
    st.info(
        "One primary category is returned. Messages with several requests need review. "
        "A booking is facilities; a broken facility is maintenance; its refund is billing."
    )

with showcase_tab:
    st.subheader("Project showcase")
    st.write(
        "Explore the project through presentation slides and a narrated walkthrough. "
        "To try it yourself, open Classify a request and choose an engine in the sidebar."
    )
    slide_folder = ROOT / "output/slides"
    slide_images = sorted(slide_folder.glob("slide-[0-9][0-9].png"))
    if slide_images:
        slide_number = st.select_slider(
            "Showcase slide", options=list(range(1, len(slide_images) + 1)), value=1
        )
        st.image(str(slide_images[slide_number - 1]), width="stretch")
        for extension, title, mime in (
            (
                "pptx",
                "Download PowerPoint",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ),
            ("pdf", "Download slides PDF", "application/pdf"),
        ):
            deck = slide_folder / f"Management_Agent_Classifier_Showcase.{extension}"
            if deck.is_file():
                st.download_button(title, deck.read_bytes(), deck.name, mime)
    else:
        st.caption("The slide deck has not been generated in this copy of the project yet.")

    demo_video = ROOT / "output/demo/Management_Agent_Classifier_Demo.mp4"
    if demo_video.is_file():
        with st.expander("Watch the 79-second video"):
            st.video(str(demo_video))
            st.caption(
                "The video shows the original offline demo. The slides include the later "
                "local Ollama results. Historical measurements use synthetic examples."
            )
    st.subheader("Start the project on Windows")
    st.write(
        "After installing dependencies and Ollama, double-click start_project.cmd "
        "in the project folder to start both services. For the offline classifier, "
        "follow the setup instructions below."
    )
    st.code(".\\start_project.cmd", language="powershell")
    st.write(
        "Open http://localhost:8501. The offline classifier is ready immediately. "
        "Select Ollama to use gemma3:4b; its first request may take longer while the model loads. "
        "Keep the launcher window open. Press Ctrl+C there to stop the website."
    )
    with st.expander("Offline startup and first-time setup"):
        st.code(
            "# First-time dependency setup, if .venv is missing\n"
            "uv sync --locked --python 3.12\n\n"
            "# Start the website without starting Ollama\n"
            ".\\start_project.cmd -SkipOllama",
            language="powershell",
        )
        st.caption(
            "The app runs locally. It classifies requests and exports results; "
            "ticket creation and resident messaging are future work."
        )
