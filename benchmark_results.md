# Benchmark Results Snapshot

Provenance for the committed `benchmark_results.json` file.

| Field | Value |
|-------|-------|
| **Generated** | 2026-08-23 |
| **Command** | `python benchmark.py` |
| **API** | Verified live Groq API (`GROQ_API_KEY` set) |
| **LLM model** | `llama-3.3-70b-versatile` (labeled `llama-3.3-70b` in results) |
| **Runs per cell** | 5 |
| **Not fallback** | Rows labeled `llama-3.3-70b` are from real API calls, not `deterministic_fallback()` |

Re-run:

```bash
GROQ_API_KEY=gsk_... python benchmark.py
```

Without `GROQ_API_KEY`, `benchmark.py` labels the LLM row `deterministic_fallback (not llama-3.3-70b)` and prints a warning.
