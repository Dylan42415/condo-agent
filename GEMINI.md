# Project Instructions: Condo Management Agent

## Strict Quality & Review Rules

### 1. Mandatory Code Review Execution
Whenever a new feature, bugfix, or enhancement is implemented:
1. **Lint & Style Review**: Run `uv run ruff check .` and ensure 0 errors.
2. **Test Suite Verification**: Run `uv run pytest` and ensure all tests pass (0 failures).
3. **Architectural & Security Audit**: Audit the code changes with the `code-reviewer` agent to ensure input sanitization, safe state management, and edge-case handling.
4. **Zero Regressions**: Any newly added feature must not break existing test contracts or baseline classification pipelines.

### 2. Environment & Dependency Management
- Always use `uv` commands (`uv run`, `uv add`, `uv sync`).
- Never run global `pip install`.
- Preserve environment secrets in `.env` (never hardcode tokens or API keys).
