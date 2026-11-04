# Condo Agent: Autonomous Condominium Emergency Triage & Management Platform

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-303%20passed-brightgreen.svg)](tests/)
[![RAG Benchmark](https://img.shields.io/badge/RAG%20Hit%20Rate@3-100%25-success.svg)](reports/TRIBECA_RAG_BENCHMARK_REPORT.md)
[![Table Preservation](https://img.shields.io/badge/Table%20Preservation-100%25-success.svg)](reports/TRIBECA_RAG_BENCHMARK_REPORT.md)
[![Codebase Knowledge Graph](https://img.shields.io/badge/Knowledge%20Graph-1%2C634%20notes-purple.svg)](graphify-out/obsidian/)

A production-grade, multi-agent AI operations platform for condominium resident services, emergency maintenance triage, and contractor auto-dispatch. Powered by **Multi-Stage Semantic RAG (Gemini Embeddings + Dense/Sparse Hybrid Retrieval)**, **Document AI Human-in-the-Loop (HITL) Policy Management**, and real-time **Multimodal Telegram Bot Coordination** (Voice & Vision).

Grounded and rigorously benchmarked against authentic Singapore condominium handbooks (**Tribeca Condominium**, 60 Kim Seng Road), achieving **100% table atomicity**, **100% breadcrumb coverage**, and **100% Hit Rate @ 3**.

---

## Architecture Overview

```text
                                 RESIDENT CHANNELS
                     ┌───────────────────┴───────────────────┐
                     │                                       │
            Telegram Bot (/report, /ask)               Web Portal / Dashboard
          [Photos · Voice Memos · Singlish]           [Operations Console · RAG]
                     │                                       │
                     ▼                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   CORE AGENT PIPELINE                                   │
├─────────────────────────────────────┬───────────────────────────────────────────────────┤
│ 1. Multimodal Perception & Intake   │ • Audio Transcription (Whisper / Gemini Voice)    │
│                                     │ • Visual Damage Diagnosis (Gemini Vision API)     │
│                                     │ • Multilingual Normalization (Singlish/ZH/MS/TA)  │
├─────────────────────────────────────┼───────────────────────────────────────────────────┤
│ 2. Autonomous Emergency Triage      │ • Urgency Classification (P1 Critical to P4 Low) │
│                                     │ • Category Routing & Safety Escalation Gates      │
│                                     │ • SQLite WAL Persistent State & Audit Trails      │
├─────────────────────────────────────┼───────────────────────────────────────────────────┤
│ 3. Automated Contractor Dispatch    │ • SLA Countdown Timers (1h P1 / 4h P2 / 24h P3)   │
│                                     │ • Contractor Directory Matching (Hai Yong, etc.)  │
│                                     │ • Work-Order Notifications & Trello Integration   │
├─────────────────────────────────────┼───────────────────────────────────────────────────┤
│ 4. Multi-Stage Semantic RAG         │ • Two-Stage Structure & Breakpoint Chunking       │
│                                     │ • 100% Atomic Markdown Table Preservation         │
│                                     │ • Contextual Breadcrumbs ([Doc | Sec | Sub])      │
│                                     │ • Hybrid Retrieval (Dense Vector + BM25 Boosting) │
│                                     │ • Decision-First Grounded Synthesis (Gemini 2.5)  │
├─────────────────────────────────────┼───────────────────────────────────────────────────┤
│ 5. Document AI Living Policy Editor │ • AST Character-Offset Section Localization       │
│    (Human-in-the-Loop)              │ • Constrained Generative Rewriting & Color Diffs  │
│                                     │ • Immutable JSON Audit Logs & 1-Click Rollback    │
│                                     │ • Zero-Downtime Hot Vector Re-Indexing            │
└─────────────────────────────────────┴───────────────────────────────────────────────────┘
```

---

## Key Pillars & Technical Capabilities

### 1. Multi-Stage Semantic RAG & Estate Legal Brain
Traditional RAG breaks on condominium handbooks because tables get severed and clauses lose hierarchical context. Our custom Semantic RAG engine (`condo_classifier/semantic_rag.py`) introduces:
* **Two-Stage Chunking**: Structure-aware Markdown AST header splitting combined with sentence-level semantic breakpoint distance clustering (90th-percentile cosine distance).
* **100% Table Atomicity**: GFM tables (contractor directories, parking clearance tables, fee schedules) are detected and preserved as indivisible semantic units.
* **Contextual Breadcrumb Injection**: Prepends `[Document: Tribeca | Section: General > Facilities]` into dense embeddings, ensuring chunks retain full structural awareness.
* **Hybrid Retrieval**: Fuses dense semantic vector search (`gemini-embedding-001`) with sparse token and domain regex boosting (`$150`, `22:00`, `BMSMA`, `deposit`).
* **Small-to-Big Parent Context**: Matches on concise semantic chunks but automatically injects deduplicated parent sections (`parent_content`) into the synthesis prompt.
* **Decision-First Synthesis**: Enforces structured rulings: **PERMITTED**, **PROHIBITED**, or **ACTION REQUIRED**, with mandatory by-law citations and zero mock contact hallucination.

### 2. Multimodal Resident Experience (Telegram Bot)
The resident bot (`telegram_bot.py` & `condo_classifier/telegram_bot.py`) provides an interactive interface for residents:
* **/report**: Submit maintenance issues with photos or voice memos. Gemini Vision assesses pipe burst / structural damage severity.
* **/ask**: Natural language estate handbook and by-law inquiries with instant grounded answers.
* **/status & /mytickets**: Live tracking of reported issues with assigned contractor details and SLA countdowns.
* **Multilingual & Singlish Support**: Robust handling of Singaporean colloquialisms (*"aircon leaking water very jialat downstairs"*).

### 3. Autonomous Triage, SLA & Contractor Auto-Dispatch
Operational management engine (`condo_classifier/dispatch.py` & `condo_classifier/database.py`):
* **4-Tier Priority Matrix**:
  * `P1 Critical` (SLA: 1 hour) — Pipe bursts, gas leaks, power outages, elevator entrapment.
  * `P2 High` (SLA: 4 hours) — Common area lighting failure, security barrier malfunction.
  * `P3 Medium` (SLA: 24 hours) — Facility booking disputes, gym equipment fault.
  * `P4 Low / Routine` (SLA: 48 hours) — General feedback, administrative queries.
* **Automated Trade Dispatching**: Dispatches verified contractors directly from the estate registry:
  * Plumbing & Gas: *Hai Yong Engineering Pte Ltd* (`6553 0503`)
  * ACMV & Electrical: *King Wan Construction Pte Ltd* (`6368 4300`)
  * Lift Systems: *KONE Pte Ltd* (`6424 6246`)
* **Enterprise Persistence**: SQLite WAL database with foreign-key constraints, audit logs, and Trello card integration.

### 4. Living Policy Editor & Document AI (Human-in-the-Loop)
Condo managers can update rules without touching raw Markdown files (`condo_classifier/policy_editor.py`):
1. **Natural Language Amendment**: Manager types: *"Increase BBQ deposit to $200 and add a $15 cleaning fee."*
2. **Section Localization Agent**: AST parser locates the exact section boundaries and character offsets.
3. **Constrained Rewriting**: Generates surgical clause revisions while preserving surrounding tables and formatting.
4. **Visual Diff Review**: Highlights unified additions/deletions for staff verification.
5. **Atomic Write & Hot Re-Index**: Saves a timestamped backup, updates the Markdown handbook, appends an immutable JSON audit log, and immediately hot-reindexes RAG memory with **zero server downtime**.
6. **1-Click Rollback**: Instantly restores previous rule versions from the backup archive.

### 5. Interactive Operations Console (Streamlit)
The web console (`app.py`) serves as property management command central:
* **Live Ticket Queue & Kanban**: Real-time SLA breach countdowns, priority filtering, and manual contractor overrides.
* **Estate Knowledge Base Explorer**: Interactive RAG query tester loaded with authentic showcase presets (parking clearances, tennis booking quotas, renovation deposits).
* **Living Policy Editor Studio**: Visual editor for amending rules and inspecting audit trails.
* **Operations Analytics**: Historical failure trends, contractor response SLAs, and resident sentiment breakdown.

---

## Real-Document Verification & Benchmarks

The RAG and dispatch pipelines were independently audited across authentic Tribeca Condominium documents:

| Benchmark Metric | Result | Target | Status |
| :--- | :---: | :---: | :---: |
| **Total Chunks Generated** | **145** | 120–160 | **OPTIMAL** |
| **Table Preservation Rate** | **100% (13/13 tables intact)** | 100% | **PASS** |
| **Breadcrumb Coverage** | **100% (145/145 chunks titled)** | 100% | **PASS** |
| **RAG Hit Rate @ 3** | **100%** | > 95% | **PASS** |
| **Mean Reciprocal Rank (MRR)** | **0.9333** | > 0.85 | **PASS** |
| **Faithfulness / Grounding** | **100%** | 100% | **PASS** |
| **Hallucinated Mock Numbers** | **0** | 0 | **PASS** |
| **Adversarial UX Scenarios** | **16/16 Passed** | 100% | **PASS** |
| **Automated Test Suite** | **303 Passed / 0 Failed** | 100% | **PASS** |

---

## Quick Start Guide

### 1. Prerequisites
- Python 3.12+
- [uv package manager](https://docs.astral.sh/uv/) (recommended) or standard `pip`
- Git

### 2. Clone & Install Dependencies

```bash
git clone https://github.com/Dylan42415/condo-agent.git
cd condo-agent/condo-request-classifier

# Install dependencies using uv
uv sync
```

### 3. Environment Configuration
Create or configure your `.env` file in `condo-request-classifier/`:

```env
# Gemini API Key (Required for Semantic RAG embeddings and vision analysis)
GEMINI_API_KEY=your_gemini_api_key_here

# Telegram Bot Token (Optional, for live resident bot)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

# Default Estate Knowledge Base
DEFAULT_KB_PATH=Knowledge Base/converted_tribeca_handbook.md
```

### 4. Launch the Management Dashboard

```bash
uv run streamlit run app.py
```
Open **`http://localhost:8501`** in your browser. Navigate to:
- **Operations & Tickets**: View live tickets, urgent alerts, and dispatch actions.
- **Estate Knowledge Base & Semantic RAG**: Test natural language questions against the authentic Tribeca handbook.
- **Policy Editor (HITL)**: Amend rules and review live diffs.

### 5. Launch the Resident Telegram Bot

```bash
uv run python telegram_bot.py
```
Residents can now message the bot:
- `/ask What are the car park height limits?`
- `/ask Can visitors park overnight?`
- `/report` (with photo or voice memo of maintenance issue)
- `/status` (track ticket updates)

---

## Verification & Testing

Run the full automated test suite (303 tests):

```bash
uv run pytest tests/ -v
```

Run the RAG Evaluation Benchmark against real estate documents:

```bash
uv run python scripts/evaluate_tribeca_rag.py
```

Run the Adversarial UX & Persona Simulation:

```bash
uv run python scripts/run_adversarial_ux_audit.py
```

Update and Export the Codebase Knowledge Graph:

```bash
uv run graphify update .
uv run graphify export obsidian
uv run graphify export wiki
```

---

## Project Structure

```text
condo-request-classifier/
├── Knowledge Base/
│   ├── converted_tribeca_handbook.md   # Authentic Tribeca Resident Handbook
│   └── backups/                         # Timestamped backups & amendment audit logs
├── condo_classifier/
│   ├── semantic_rag.py                 # Multi-Stage Semantic RAG & Hybrid Index
│   ├── policy_editor.py                # Human-in-the-Loop Living Policy Editor
│   ├── telegram_bot.py                 # Resident Telegram Bot & Command Handlers
│   ├── classifier.py                   # 4-tier emergency triage classifier
│   ├── dispatch.py                     # SLA calculation & contractor dispatch engine
│   ├── database.py                     # SQLite WAL persistence layer
│   ├── vision.py                       # Gemini Vision damage analysis
│   ├── voice.py                        # Whisper/Gemini voice memo transcription
│   └── pdf_converter.py                # Automated PDF-to-Markdown ingestion
├── data/
│   └── vendors.json                    # Authentic contractor & emergency directory
├── graphify-out/
│   ├── obsidian/                       # Obsidian interactive visual knowledge graph
│   └── wiki/                           # Structured markdown architecture wiki
├── reports/
│   ├── TRIBECA_RAG_BENCHMARK_REPORT.md # Grounded RAG benchmark audit report
│   └── adversarial_ux_audit_report.md  # 16-persona UX adversarial audit report
├── scripts/
│   ├── evaluate_tribeca_rag.py         # Automated RAG evaluation benchmark
│   └── run_adversarial_ux_audit.py     # Adversarial resident UX simulation
├── tests/                              # 303 unit, integration & end-to-end tests
├── app.py                              # Streamlit management operations dashboard
├── telegram_bot.py                     # Telegram bot launcher
└── pyproject.toml                      # Project dependencies and configuration
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
