"""Turn the run directories into report data: results.json, CSV tables, and LaTeX tables.

Usage: python scripts/analyze.py

Run this any time: it reports only what has finished, and lists what has not.
"""

from __future__ import annotations

import collections
import csv
import json
import statistics as st
from pathlib import Path

import numpy as np

from rac.eval.stats import mcnemar_em, paired_bootstrap_es

OUT = Path("outputs/report")
RUNS = Path("outputs")
M1 = RUNS / "m1_n512" / "predictions.jsonl"
BUDGETS = [256, 512, 1024, 2048, 4096]
LABEL = {
    "none": "No context",
    "bm25_long": "BM25 long",
    "bm25_short": "BM25 short",
    "oracle_long": "Oracle (BM25 long)",
    "oracle_short": "Oracle (BM25 short)",
    "oracle": "Gold only (variant)",
    "signatures": "Signature map",
}


def read_rows(pattern):
    rows = []
    for path in sorted(RUNS.glob(f"{pattern}/predictions.jsonl")):
        complete = (path.parent / "summary.json").exists()
        if complete:
            rows += [json.loads(line) for line in path.read_text().splitlines()]
    return rows


def index(rows):
    """{(method, budget): {example_id: row}}; `none` is copied to every budget missing it."""
    table = collections.defaultdict(dict)
    for row in rows:
        table[row["method"], row["budget"]][row["example_id"]] = row
    return table


def bootstrap_mean(values, n_boot=10_000, seed=0):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def holm(pvalues):
    """Holm-adjusted p-values, in the input order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def method_row(cell, baseline, ids):
    rows = [cell[i] for i in ids]
    em = [r["em"] for r in rows]
    es = [r["es"] for r in rows]
    seconds = [r["gen_seconds"] for r in rows]
    tokens = [r["context_tokens"] for r in rows]
    result = {
        "n": len(ids),
        "em_pct": bootstrap_mean([100.0 * x for x in em]),
        "es": bootstrap_mean(es),
        "median_seconds": st.median(seconds),
        "median_seconds_ci": [
            float(x)
            for x in np.percentile(
                np.median(
                    np.asarray(seconds)[
                        np.random.default_rng(0).integers(0, len(seconds), (2000, len(seconds)))
                    ],
                    axis=1,
                ),
                [2.5, 97.5],
            )
        ],
        "mean_context_tokens": st.mean(tokens),
        "median_context_tokens": st.median(tokens),
        "empty_predictions": sum(not r["prediction"].strip() for r in rows),
    }
    if baseline is not None:
        base = [baseline[i] for i in ids]
        wins = sum(r["em"] and not b["em"] for r, b in zip(rows, base, strict=True))
        losses = sum(b["em"] and not r["em"] for r, b in zip(rows, base, strict=True))
        result["es_vs_none"] = paired_bootstrap_es(es, [b["es"] for b in base])
        result["em_vs_none_p"] = mcnemar_em(em, [b["em"] for b in base])
        result["em_wins"], result["em_losses"] = wins, losses
        gain = result["es_vs_none"][0]
        mean_tokens = result["mean_context_tokens"]
        result["es_gain_per_1k_tokens"] = 1000 * gain / mean_tokens if mean_tokens else None
        changed = sum(r["prediction"] != b["prediction"] for r, b in zip(rows, base, strict=True))
        result["prediction_changed"] = changed
    return result


def sweep(table, ids, budgets=BUDGETS):
    """Every method x budget against the same-example no-context run, Holm-corrected per budget."""
    results = {}
    none = next((table["none", b] for b in budgets if ("none", b) in table), None)
    methods = sorted({m for m, _ in table if m != "none"}, key=list(LABEL).index)
    for budget in budgets:
        cells = {}
        for method in methods:
            cell = table.get((method, budget))
            if cell and all(i in cell for i in ids):
                cells[method] = method_row(cell, none, ids)
        if cells:
            adjusted = holm([c["em_vs_none_p"] for c in cells.values()])
            for cell, p in zip(cells.values(), adjusted, strict=True):
                cell["em_vs_none_p_holm"] = p
            results[budget] = cells
    if none:
        results["none"] = method_row(none, None, ids)
    return results


def pairs(table, ids, comparisons):
    results = []
    for (a, budget_a), (b, budget_b) in comparisons:
        cell_a, cell_b = table.get((a, budget_a)), table.get((b, budget_b))
        if not cell_a or not cell_b or not all(i in cell_a and i in cell_b for i in ids):
            continue
        x, y = [cell_a[i] for i in ids], [cell_b[i] for i in ids]
        es = paired_bootstrap_es([r["es"] for r in x], [r["es"] for r in y])
        p = mcnemar_em([r["em"] for r in x], [r["em"] for r in y])
        results.append(
            {
                "a": a,
                "budget_a": budget_a,
                "b": b,
                "budget_b": budget_b,
                "n": len(ids),
                "es_diff": es,
                "em_p": p,
                "only_a": sum(r["em"] and not s["em"] for r, s in zip(x, y, strict=True)),
                "only_b": sum(s["em"] and not r["em"] for r, s in zip(x, y, strict=True)),
            }
        )
    return results


def selector_labels(table, ids, budgets):
    """Pass 2 labels per the plan: highest ES, ties to the option with fewest context tokens."""
    methods = ["none", "bm25_long", "bm25_short", "signatures"]
    labels = {}
    for i in ids:
        options = []
        none = next(table["none", b][i] for b in budgets if ("none", b) in table)
        options.append(("none", 0, none))
        for method in methods[1:]:
            for budget in budgets:
                cell = table.get((method, budget))
                if cell and i in cell:
                    options.append((method, budget, cell[i]))
        best = max(r["es"] for *_, r in options)
        winners = [o for o in options if o[2]["es"] == best]
        method, budget, row = min(winners, key=lambda o: (o[2]["context_tokens"], o[1]))
        labels[i] = {
            "method": method,
            "budget": budget,
            "es": row["es"],
            "em": row["em"],
            "tokens": row["context_tokens"],
            "tied_with_none": none["es"] == best,
            "none_strictly_best": sum(o[2]["es"] == best for o in options) == 1
            and none["es"] == best,
        }
    fixed = {}
    for method in methods[1:]:
        for budget in budgets:
            cell = table.get((method, budget))
            if cell and all(i in cell for i in ids):
                fixed[f"{method}@{budget}"] = st.mean(cell[i]["es"] for i in ids)
    best_fixed = max(fixed, key=fixed.get) if fixed else None
    return {
        "labels": labels,
        "counts": collections.Counter(
            f"{v['method']}@{v['budget']}" if v["method"] != "none" else "none"
            for v in labels.values()
        ).most_common(),
        "hindsight_es": st.mean(v["es"] for v in labels.values()),
        "hindsight_em_pct": 100 * st.mean(v["em"] for v in labels.values()),
        "hindsight_mean_tokens": st.mean(v["tokens"] for v in labels.values()),
        "best_fixed": best_fixed,
        "best_fixed_es": fixed.get(best_fixed),
        "none_is_a_best_choice": sum(v["tied_with_none"] for v in labels.values()),
        "none_strictly_best": sum(v["none_strictly_best"] for v in labels.values()),
    }


def by_level(table, method, budget, ids, examples):
    """ES and EM gain over no context by RepoBench length bucket."""
    order = ["2k", "4k", "8k", "12k", "16k", "24k", "32k", "64k", "128k"]
    none = next(table["none", b] for b in BUDGETS if ("none", b) in table)
    cell = table.get((method, budget))
    if not cell:
        return {}
    out = {}
    for level in order:
        chosen = [i for i in ids if examples[i].level == level]
        if len(chosen) >= 10:
            es, low, high = paired_bootstrap_es(
                [cell[i]["es"] for i in chosen], [none[i]["es"] for i in chosen]
            )
            out[level] = {
                "n": len(chosen),
                "none_es": st.mean(none[i]["es"] for i in chosen),
                "es_gain": es,
                "low": low,
                "high": high,
            }
    return out


def qualitative(table, ids, examples, method="bm25_long", budget=512):
    """Short, readable worked completions, chosen by rule so nothing is cherry-picked by hand."""
    none = next(table["none", b] for b in BUDGETS if ("none", b) in table)
    cell = table.get((method, budget))
    picks = {"fixed": [], "misled": [], "both_wrong_context_closer": []}
    if not cell:
        return picks
    for i in ids:
        ex, a, b = examples[i], cell[i], none[i]
        entry = {
            "example_id": i,
            "repo": ex.repo_name,
            "file": ex.file_path,
            "last_lines": "\n".join(ex.cropped_code.rstrip("\n").splitlines()[-4:]),
            "target": ex.next_line,
            "none": b["prediction"],
            "with_context": a["prediction"],
            "es_none": b["es"],
            "es_context": a["es"],
            "gold_identifier": ex.context[ex.gold_snippet_index].identifier,
            "gold_in_context": None,
        }
        if a["em"] and not b["em"]:
            picks["fixed"].append(entry)
        elif b["em"] and not a["em"]:
            picks["misled"].append(entry)
        elif not a["em"] and not b["em"] and a["es"] - b["es"] >= 25:
            picks["both_wrong_context_closer"].append(entry)
    return picks


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def latex_main_table(results, budget, path, caption, label):
    lines = [
        r"\begin{table}[t]",
        r"\begin{center}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{|l|c|c|c|c|}",
        r"\hline",
        r"Method & EM (\%) & ES & $\Delta$ES vs none & Time (s) \\",
        r"\hline\hline",
    ]

    def fmt(name, cell, delta):
        em, es = cell["em_pct"], cell["es"]
        return (
            f"{name} & {em[0]:.1f} [{em[1]:.1f}, {em[2]:.1f}]"
            f" & {es[0]:.1f} [{es[1]:.1f}, {es[2]:.1f}]"
            f" & {delta} & {cell['median_seconds']:.2f} \\\\"
        )

    lines.append(fmt("No context", results["none"], "--"))
    for method, cell in results[budget].items():
        d = cell["es_vs_none"]
        lines.append(fmt(LABEL[method], cell, f"{d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]"))
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{center}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n")


def gpu_by_run():
    """Peak GPU memory per run, from the sampler's clock times and the queue's start/done lines."""
    log, samples = Path("outputs/progress.log"), Path("outputs/gpu.csv")
    if not log.exists() or not samples.exists():
        return {}

    def clock(text, state):
        h, m, s = (int(x) for x in text.split(":"))
        now = h * 3600 + m * 60 + s
        state["day"] += now < state["last"]
        state["last"] = now
        return now + 86400 * state["day"]

    spans, opened, state = {}, {}, {"day": 0, "last": 0}
    for line in log.read_text().splitlines():
        parts = line.split()
        when = clock(parts[0], state)
        if parts[1] == "start":
            opened[parts[2]] = when
        elif parts[1] == "done" and parts[2] in opened:
            spans[parts[2]] = (opened[parts[2]], when)
    peaks = {}
    state = {"day": 0, "last": 0}
    for line in samples.read_text().splitlines():
        fields = line.split(",", 3)
        if len(fields) < 4:
            continue
        when = clock(fields[0], state)
        ours = sum(
            int(app.rsplit(",", 1)[1])
            for app in fields[3].split(";")
            if "enpm703" not in app and app.count(",") == 2
        )
        for name, (start, stop) in spans.items():
            if start <= when <= stop:
                peak = peaks.setdefault(name, {"total_mib": 0, "process_mib": 0})
                peak["total_mib"] = max(peak["total_mib"], int(fields[1]))
                peak["process_mib"] = max(peak["process_mib"], ours)
    return peaks


def latex_sweep_table(results, path, caption, label):
    """Delta ES against no context for every method and budget, in one column."""
    budgets = [b for b in BUDGETS if b in results]
    methods = [m for m in LABEL if any(m in results[b] for b in budgets)]
    lines = [
        r"\begin{table}[t]",
        r"\begin{center}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{|l|" + "c" * len(budgets) + "|}",
        r"\hline",
        "Method & " + " & ".join(str(b) for b in budgets) + r" \\",
        r"\hline\hline",
    ]
    for method in methods:
        cells = [
            f"{results[b][method]['es_vs_none'][0]:+.2f}" if method in results[b] else "--"
            for b in budgets
        ]
        lines.append(f"{LABEL[method]} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{center}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    from rac.data.repobench import load_examples

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    revision = "8a7cf0c8942cc1aa066bf261839650ac55a2ff79"
    all_first = {e.example_id: e for e in load_examples("cross_file_first", revision)}
    ids_500 = json.loads(Path("data/samples/cff_500_seed0.json").read_text())
    report = {"status": {}}

    core_rows = read_rows("s500_n*") + read_rows("sig500_n*")
    core = index(core_rows)

    # Determinism: the same config and examples as the Milestone run.
    repro = {}
    if M1.exists():
        m1 = index(json.loads(line) for line in M1.read_text().splitlines())
        for key in m1:
            if key in core:
                same = sum(m1[key][i]["prediction"] == core[key][i]["prediction"] for i in ids_500)
                repro[f"{key[0]}@{key[1]}"] = {"identical": same, "of": len(ids_500)}
    report["determinism_vs_m1"] = repro

    report["sweep_500"] = sweep(core, ids_500)

    budgets_done = [b for b in BUDGETS if b in report["sweep_500"]]
    compare = []
    for budget in budgets_done:
        compare += [
            (("bm25_long", budget), ("bm25_short", budget)),
            (("oracle", budget), ("bm25_long", budget)),
            (("oracle_long", budget), ("bm25_long", budget)),
            (("oracle_short", budget), ("bm25_short", budget)),
            (("signatures", budget), ("bm25_long", budget)),
            (("signatures", budget), ("bm25_short", budget)),
        ]
    for method in ("bm25_long", "bm25_short", "oracle_long"):
        for low, high in zip(budgets_done, budgets_done[1:], strict=False):
            compare.append(((method, high), (method, low)))
    report["pairs_500"] = pairs(core, ids_500, compare)
    if budgets_done:
        report["levels_bm25_long"] = {
            str(b): by_level(core, "bm25_long", b, ids_500, all_first) for b in budgets_done
        }
        report["selector_500"] = selector_labels(core, ids_500, budgets_done)
        report["qualitative"] = qualitative(core, ids_500, all_first)

    # Other samples and models, if they have finished.
    extra_ids = Path("data/samples/cff_extra1500_seed1.json")
    if extra_ids.exists():
        extra_rows = read_rows("extra1500_n*") + read_rows("sigextra1500_n*")
        if extra_rows:
            ids_extra = json.loads(extra_ids.read_text())
            ids_2000 = ids_500 + ids_extra
            union = index(core_rows + extra_rows)
            report["sweep_2000"] = sweep(union, ids_2000)
            report["sweep_extra_only"] = sweep(index(extra_rows), ids_extra)
            report["levels_2000"] = {
                str(b): by_level(union, "bm25_long", b, ids_2000, all_first) for b in (512, 2048)
            }
            report["levels_extra_only"] = {
                str(b): by_level(union, "bm25_long", b, ids_extra, all_first) for b in (512, 2048)
            }
            report["pairs_2000"] = pairs(
                union,
                ids_2000,
                [
                    c
                    for b in BUDGETS
                    for c in (
                        (("bm25_long", b), ("bm25_short", b)),
                        (("oracle_long", b), ("bm25_long", b)),
                        (("signatures", b), ("bm25_long", b)),
                        (("signatures", b), ("bm25_short", b)),
                    )
                ],
            )
    rand_rows = read_rows("rand500_n512")
    if rand_rows:
        ids = json.loads(Path("data/samples/crf_500_seed0.json").read_text())
        report["sweep_random_500"] = sweep(index(rand_rows), ids, budgets=[512])
    large_rows = read_rows("large500_n*")
    if large_rows:
        large = index(large_rows)
        report["sweep_large_500"] = sweep(large, ids_500)
        cross = []
        for (method, budget), cell in large.items():
            small = core.get((method, budget))
            if small:
                es = paired_bootstrap_es(
                    [cell[i]["es"] for i in ids_500], [small[i]["es"] for i in ids_500]
                )
                cross.append(
                    {
                        "method": method,
                        "budget": budget,
                        "es_large_minus_small": es,
                        "em_p": mcnemar_em(
                            [cell[i]["em"] for i in ids_500], [small[i]["em"] for i in ids_500]
                        ),
                    }
                )
        report["large_vs_small"] = cross
        report["pairs_large_500"] = pairs(
            large,
            ids_500,
            [
                c
                for b in BUDGETS
                for c in (
                    (("bm25_long", b), ("bm25_short", b)),
                    (("oracle_long", b), ("bm25_long", b)),
                )
            ],
        )

    gpu = Path("outputs/gpu.csv")
    if gpu.exists():
        used = [
            int(line.split(",")[1]) for line in gpu.read_text().splitlines() if line.count(",") >= 2
        ]
        report["gpu_total_used_mib_peak"] = max(used) if used else None
    report["gpu_by_run"] = gpu_by_run()
    report["status"]["budgets_done_500"] = budgets_done
    report["status"]["runs_complete"] = sorted(p.parent.name for p in RUNS.glob("*/summary.json"))
    (OUT / "results.json").write_text(json.dumps(report, indent=1, default=str))

    # Flat tables for the report authors.
    rows = []
    for sweep_name in (
        "sweep_500",
        "sweep_extra_only",
        "sweep_2000",
        "sweep_large_500",
        "sweep_random_500",
    ):
        for budget, cells in report.get(sweep_name, {}).items():
            if budget == "none":
                c = cells
                rows.append(
                    [
                        sweep_name,
                        "none",
                        "",
                        c["n"],
                        *[round(x, 3) for x in c["em_pct"]],
                        *[round(x, 3) for x in c["es"]],
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        round(c["median_seconds"], 3),
                        0,
                        "",
                        c["empty_predictions"],
                    ]
                )
                continue
            for method, c in cells.items():
                d = c["es_vs_none"]
                rows.append(
                    [
                        sweep_name,
                        method,
                        budget,
                        c["n"],
                        *[round(x, 3) for x in c["em_pct"]],
                        *[round(x, 3) for x in c["es"]],
                        *[round(x, 3) for x in d],
                        round(c["em_vs_none_p"], 5),
                        round(c["em_vs_none_p_holm"], 5),
                        c["em_wins"],
                        c["em_losses"],
                        round(c["median_seconds"], 3),
                        round(c["mean_context_tokens"], 1),
                        round(c["es_gain_per_1k_tokens"], 3),
                        c["empty_predictions"],
                    ]
                )
    write_csv(
        OUT / "tables" / "all_results.csv",
        [
            "experiment",
            "method",
            "budget",
            "n",
            "em_pct",
            "em_low",
            "em_high",
            "es",
            "es_low",
            "es_high",
            "es_vs_none",
            "es_vs_none_low",
            "es_vs_none_high",
            "em_p",
            "em_p_holm",
            "em_wins",
            "em_losses",
            "median_seconds",
            "mean_context_tokens",
            "es_gain_per_1k_tokens",
            "empty_predictions",
        ],
        rows,
    )
    if 512 in report["sweep_500"] and "none" in report["sweep_500"]:
        latex_main_table(
            report["sweep_500"],
            512,
            OUT / "tables" / "table_n512.tex",
            "Results at $N=512$ on the frozen sample of 500 \\texttt{cross\\_file\\_first} "
            "examples with "
            "DeepSeek-Coder-1.3B. Brackets are 95\\% bootstrap intervals; $\\Delta$ES is paired "
            "against no context on the same examples. Oracle places the gold snippet first, "
            "then fills "
            "the budget with the named BM25 rule; the gold-only variant stops after the gold "
            "snippet. Time is median generation seconds.",
            "tab:n512",
        )
    if budgets_done:
        latex_sweep_table(
            report["sweep_500"],
            OUT / "tables" / "table_sweep.tex",
            r"Paired $\Delta$ES against no context by budget $N$ on the frozen sample of 500 "
            r"\texttt{cross\_file\_first} examples with DeepSeek-Coder-1.3B.",
            "tab:sweep",
        )
    if report.get("sweep_2000"):
        latex_sweep_table(
            report["sweep_2000"],
            OUT / "tables" / "table_sweep_n2000.tex",
            r"Paired $\Delta$ES against no context by budget $N$ on 2,000 "
            r"\texttt{cross\_file\_first} examples (the frozen 500 plus 1,500 more) with "
            r"DeepSeek-Coder-1.3B.",
            "tab:sweep2000",
        )
    if report.get("sweep_large_500"):
        latex_sweep_table(
            report["sweep_large_500"],
            OUT / "tables" / "table_sweep_5p7b.tex",
            r"Paired $\Delta$ES against no context by budget $N$ with DeepSeek-Coder-5.7B "
            r"(500 examples).",
            "tab:sweep5p7",
        )
    print("budgets done:", budgets_done, "runs:", report["status"]["runs_complete"])


if __name__ == "__main__":
    main()
