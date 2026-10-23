# AI and external-code use

ENPM703 permits LLM-generated code/text and code from public repositories in the final
project, provided it is clearly attributed and we can explain it in our own words.
Log every such use here. Commits made with an AI assistant also carry a
`Co-Authored-By:` trailer.

| Date | Who | Tool / source | What it produced | Files |
|---|---|---|---|---|
| 2026-10-15 | Marcus | Claude Code (Claude Opus 5.5) | Repository skeleton: package layout, shared types, `build_context` dispatcher, stubs, contract tests, docs scaffolding | initial commit |
| 2026-10-18 | Marcus | Claude Code (Claude Opus 5.5) | Prior-work summaries (RepoFuse, Repoformer, ProCC) from a project review session; import-spacing lint fixes | `docs/literature_review.md`, `src/rac/` |
| 2026-10-19 | Kenneth | ChatGPT | Assisted with code_tokens implementation, explanation, and unit tests; documented ASCII and numeric token behavior | `src/rac/context/tokenize.py`, `tests/test_tokenize.py` |
| 2026-10-22 | Maria | ChatGPT | Assisted with query_text and rank implementation improvements. Unit test generation. | `src/rac/context/bm25.py`, `tests/test_bm25.py` |