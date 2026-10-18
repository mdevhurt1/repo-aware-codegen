# repo-aware-codegen

ENPM703 Group 2 term project: repository-aware code completion under a context budget.

Code completion models mostly see the open file, but real code calls functions and
classes defined elsewhere in the repository. Cross-file context helps, but the context
window has a limited budget. We ask two questions:

1. **Pass 1:** what should fill a budget of N tokens for a small, frozen code model?
2. **Pass 2:** can a cheap classifier predict the best choice per completion?

**Data**: RepoBench v1.1 Python (`cross_file_first`); CrossCodeEval Python as the final test.

**Model**: DeepSeek-Coder-1.3B in 4-bit.

## Layout

| Path | What |
|---|---|
| `src/rac/types.py` | `Example`, `Snippet`, `ResultRow`: the shared data contract |
| `src/rac/context/__init__.py` | `build_context(example, method, N, tokenizer)`: the context builder |
| `src/rac/data/` | RepoBench loader, seeded sample |
| `src/rac/models/`, `src/rac/prompt.py` | 4-bit model load, timed generation, prompt builder |
| `src/rac/eval/` | metrics, runner, paired stats |
| `src/rac/context/` (tokenize, bm25, fill, methods) | retrieval and budget fill |
| `src/rac/context/signatures.py`, `src/rac/selector/` | signature map, pass 2 selector |
| `configs/` | one YAML per run |
| `data/samples/` | committed sample ID lists, so everyone uses the same examples |
| `outputs/` | run results (gitignored) |
| `docs/` | architecture and design decisions |

## Setup

Laptop GPU (Linux/WSL, CUDA):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[gpu,dev]'
pytest -q
```

Colab: open [notebooks/colab_quickstart.ipynb](notebooks/colab_quickstart.ipynb). Put a Hugging
Face token in Colab secrets as `HF_TOKEN`; never commit tokens.

Run an experiment:

```bash
python -m rac run configs/m1_n512.yaml   # -> outputs/m1_n512/
```

## Working together

- One branch per feature, PR into `main`, CI must pass.
- Changing `types.py` or the `build_context` signature affects everyone, so flag it in the PR title.
- Add tests next to your code's contract in `tests/` (use `tests/fixtures/repobench_tiny.jsonl`;
  do not commit real dataset rows).
- Log any LLM- or GitHub-sourced code in [AI_USE.md](AI_USE.md); course policy requires it.
