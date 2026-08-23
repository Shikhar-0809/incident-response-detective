# Benchmark Results Snapshot

Provenance for the committed `benchmark_results.json` file.

| Field | Value |
|-------|-------|
| **Generated** | 2026-08-23 |
| **Command** | `python benchmark.py` |
| **API** | Verified live Groq API (`GROQ_API_KEY` set) |
| **Historical LLM models** | `llama-3.3-70b-versatile`, `llama-3.1-8b-instant` (deprecated by Groq 2026-08-16) |
| **Current LLM models** | `openai/gpt-oss-120b`, `openai/gpt-oss-20b` |
| **Runs per cell** | 5 |
| **Not fallback** | LLM rows in the Aug 2026 snapshot are from real API calls, not `deterministic_fallback()` |

Re-run with current models:

```bash
GROQ_API_KEY=gsk_... python benchmark.py
```

Result rows will be labeled `openai/gpt-oss-120b` and `openai/gpt-oss-20b`.

Without `GROQ_API_KEY`, `benchmark.py` labels LLM rows `deterministic_fallback (not openai/gpt-oss-120b)` / `deterministic_fallback (not openai/gpt-oss-20b)` and prints a warning.
