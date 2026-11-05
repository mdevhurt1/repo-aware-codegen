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
| 2026-10-22 | Maria | ChatGPT | Assisted with query_text and rank implementation improvements (functions were originally written by hand). Generated unit tests. | `src/rac/context/bm25.py`, `tests/test_bm25.py` |
| 2026-10-29 | Kenneth | ChatGPT | Assisted with Unicode truncation fix, regression tests, and 27 real-tokenizer Unicode checks | `src/rac/context/fill.py`, `tests/test_fill.py` |
| 2026-10-30 | Marcus | Claude Code (Claude Opus 5.5) | Reviewed the handwritten `paired_bootstrap_es`/`mcnemar_em` and helped debug; wrote the unit tests | `src/rac/eval/stats.py`, `tests/test_stats.py` |
| 2026-10-30 | Marcus | Claude Code (Claude Opus 5.5) | Wrote the `build_prompt` unit tests; reviewed the hand-written draft, suggesting the `max_new_tokens` reserve and the empty-imports guard | `tests/test_prompt.py`, `src/rac/prompt.py` |
| 2026-11-03 | Marcus | Claude Code (Claude Fable 5.1) | Reviewed the `build-prompt` branch; updated the `build_prompt` tests based on the special character count decission and added a regression test | `tests/test_prompt.py` |
| 2026-10-30 | Marcus | Claude Code (Claude Opus 5.5) | `exact_match` and `edit_similarity` checked against RepoBench's `evaluation/metrics.py` (whitespace-split EM, `fuzz.ratio` ES), with unit tests | `src/rac/eval/metrics.py`, `tests/test_metrics.py` |
| 2026-11-3 | Maria | ChatGPT | Reviewed bm25_short and oracle and recommended improvements. Assisted with new version of oracle with fallback retrieval rules. Generated unit and integration tests. | `src/rac/context/methods.py`, `tests/test_bm25_short.py`, `tests/test_oracle.py` |
| 2026-11-03 | Maria | ChatGPT | Reviewed rank validation improvements. Assisted with modifying existing unit tests with new function signature. Generated unit tests for new validation error cases. | `src/rac/context/bm25.py`, `tests/test_bm25.py` |
| 2026-11-04 | Marcus | Claude Code (Claude Fable 5.1) | Changed `rank` to score 19 examples (empty snippets, token-less queries) 0.0 instead of raising and upated tests | `src/rac/context/bm25.py`, `tests/test_bm25.py` |
| 2026-11-04 | Marcus | Claude Code (Claude Fable 5.1); CVPR 2017 author kit | Report scaffold: `report.tex` skeleton, `references.bib` (six entries, metadata from the arXiv abstract pages), build notes. `cvpr.sty` and `ieee.bst` are copied unmodified from the CVPR 2017 author kit | `docs/report/` |
| 2026-10-30 | Marcus | Claude Code (Claude Opus 5.5) | Assisted with `load_model` (NF4 4-bit via bitsandbytes) and timed greedy `generate` that stops at the first newline; CPU unit tests with stand-in model and tokenizer | `src/rac/models/load.py`, `tests/test_load.py` |
| 2026-11-03 | Marcus | Claude Code (Claude Opus 5.5) | Reviewed `load.py` changes; Assisted with block comment handling logic; Added unit tests for load.py newline and comment exclusions | `src/rac/models/load.py`, `tests/test_load.py` |
