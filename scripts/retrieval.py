"""Why the methods differ: how often BM25 kept the gold snippet, and what that was worth.

Usage: python scripts/retrieval.py

Rebuilds contexts; no GPU. Writes outputs/report/retrieval.json. Gold counts as kept if
the start of its first line is in the context.
"""

from __future__ import annotations

import collections
import json
import statistics as st
from pathlib import Path

from transformers import AutoTokenizer

from rac.context import build_context
from rac.data.repobench import load_examples
from rac.eval.stats import paired_bootstrap_es

REVISION = "8a7cf0c8942cc1aa066bf261839650ac55a2ff79"
MODEL = "deepseek-ai/deepseek-coder-1.3b-base"
BUDGETS = [256, 512, 1024, 2048]
LEVELS = ["2k", "4k", "8k", "12k", "16k", "24k", "32k", "64k", "128k"]


def rows(pattern):
    table = {}
    for path in Path("outputs").glob(f"{pattern}/predictions.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            table[row["method"], row["budget"], row["example_id"]] = row
    return table


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    first = {e.example_id: e for e in load_examples("cross_file_first", REVISION)}
    ids = json.loads(Path("data/samples/cff_500_seed0.json").read_text())
    ids += json.loads(Path("data/samples/cff_extra1500_seed1.json").read_text())
    table = {**rows("s500_n*"), **rows("extra1500_n*")}
    none = {i: table["none", 512, i] for i in ids}
    report = {"n": len(ids), "recall": {}, "gain_by_gold_kept": {}, "recall_by_level": {}}
    for method in ("bm25_long", "bm25_short"):
        for budget in BUDGETS:
            kept = {}
            for i in ids:
                example = first[i]
                gold = (
                    example.context[example.gold_snippet_index].snippet.strip().split("\n")[0][:60]
                )
                kept[i] = gold in build_context(example, method, budget, tokenizer)
            key = f"{method}@{budget}"
            report["recall"][key] = 100 * st.mean(kept.values())
            split = {}
            for label, chosen in (
                ("kept", [i for i in ids if kept[i]]),
                ("missed", [i for i in ids if not kept[i]]),
            ):
                if len(chosen) >= 10:
                    split[label] = {
                        "n": len(chosen),
                        "es_gain": paired_bootstrap_es(
                            [table[method, budget, i]["es"] for i in chosen],
                            [none[i]["es"] for i in chosen],
                        ),
                    }
            report["gain_by_gold_kept"][key] = split
            levels = collections.defaultdict(list)
            for i in ids:
                levels[first[i].level].append(kept[i])
            report["recall_by_level"][key] = {
                k: 100 * st.mean(levels[k]) for k in LEVELS if len(levels[k]) >= 10
            }
            print(key, round(report["recall"][key], 1), flush=True)
    Path("outputs/report/retrieval.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
