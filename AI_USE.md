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
| 2026-10-19 | Kenneth | ChatGPT | Assisted with fill_to_cutoff implementation, unit tests, and a manual DeepSeek tokenizer validation script | `src/rac/context/fill.py`, `tests/test_fill.py` |
| 2026-10-20 | Marcus | Claude Code (Claude Opus 5.5) | Reviewed the datasets-server rows-API loader draft; rewrote `load_examples` on `datasets.load_dataset` with a pinned revision after the rows API rate-limited (429); wrote its unit test | `src/rac/data/repobench.py`, `tests/test_repobench.py` |
| 2026-10-20 | Marcus | Claude Code (Claude Opus 5.5) | Unit tests for `sample_ids`, `save_sample`, `load_sample` (seeding, distinct IDs, oversize `n`, JSON round trip); flagged that `sample_ids` reseeded the global RNG and suggested the one-line `random.Random(seed)` fix. The helpers themselves were written by hand | `tests/test_repobench.py` |
| 2026-10-29 | Kenneth | ChatGPT | Assisted with Unicode truncation fix, regression tests, and 27 real-tokenizer Unicode checks | `src/rac/context/fill.py`, `tests/test_fill.py` |
