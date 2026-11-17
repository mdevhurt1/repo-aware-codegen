"""Draw the report figures from outputs/report/results.json.

Usage: python scripts/figures.py

Each figure is written as PDF (for the LaTeX report) and PNG (for review), sized for one CVPR
column (3.25 in) or both (6.875 in), so type sizes are what they will be in print.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

OUT = Path("outputs/report/figures")
R = json.loads(Path("outputs/report/results.json").read_text())
BUDGETS = [256, 512, 1024, 2048, 4096]

# Hue names the ranking rule, a dashed line means the gold snippet was prepended, so the oracle
# variants need no extra hues. Blue, orange and aqua validate all-pairs; the rest is neutral ink.
COLOR = {
    "bm25_long": "#2a78d6",
    "bm25_short": "#eb6834",
    "signatures": "#1baf7a",
    "oracle_long": "#2a78d6",
    "oracle": "#52514e",
    "oracle_short": "#eb6834",
}
DASH = {"oracle_long": (0, (3, 1.5)), "oracle_short": (0, (3, 1.5)), "oracle": (0, (1, 1.2))}
MARKER = {
    "bm25_long": "o",
    "bm25_short": "s",
    "signatures": "D",
    "oracle_long": "^",
    "oracle": "v",
    "oracle_short": "P",
}
NAME = {
    "bm25_long": "BM25 long",
    "bm25_short": "BM25 short",
    "signatures": "Signature map",
    "oracle_long": "Oracle (BM25 long)",
    "oracle": "Gold only (variant)",
    "oracle_short": "Oracle (BM25 short)",
}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ONE, TWO = 3.25, 6.875
N500, N2000 = "Frozen sample, n = 500", "Pooled sample, n = 2,000"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.linewidth": 0.6,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name)


def budget_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
    ax.minorticks_off()
    ax.set_xlabel("Context budget $N$ (tokens)")


def curves(ax, sweep, methods, key, none_key=None, ci=False):
    """One line per method over budget, with a none reference line."""
    drawn = []
    for method in methods:
        points = [
            (int(b), sweep[str(b)][method][key])
            for b in BUDGETS
            if str(b) in sweep and method in sweep[str(b)]
        ]
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1][0] for p in points]
        if ci:
            ax.fill_between(
                xs,
                [p[1][1] for p in points],
                [p[1][2] for p in points],
                color=COLOR[method],
                alpha=0.10,
                linewidth=0,
            )
        ax.plot(
            xs,
            ys,
            color=COLOR[method],
            marker=MARKER[method],
            label=NAME[method],
            linestyle=DASH.get(method, "-"),
            markeredgecolor="white",
            markeredgewidth=0.4,
        )
        drawn.append(method)
    if none_key and "none" in sweep:
        base = sweep["none"][none_key]
        ax.axhline(base[0], color=INK2, linewidth=0.9)
        ax.annotate(
            "No context",
            xy=(1, base[0]),
            xycoords=("axes fraction", "data"),
            xytext=(-2, 2.5),
            textcoords="offset points",
            ha="right",
            fontsize=7,
            color=INK2,
        )
    return drawn


def fig_es_curve(sweep, name, methods, title=None, width=ONE, height=2.35, legend_cols=2):
    fig, ax = plt.subplots(figsize=(width, height))
    curves(ax, sweep, methods, "es", "es")
    budget_axis(ax)
    ax.set_ylabel("Edit similarity (0-100)")
    if title:
        ax.set_title(title, loc="left", color=INK)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=legend_cols,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    save(fig, name)


def fig_em_curve(sweep, name, methods, title=None, width=ONE, height=2.35):
    fig, ax = plt.subplots(figsize=(width, height))
    if title:
        ax.set_title(title, loc="left", color=INK)
    curves(ax, sweep, methods, "em_pct", "em_pct")
    budget_axis(ax)
    ax.set_ylabel("Exact match (%)")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0, handletextpad=0.4
    )
    save(fig, name)


def fig_delta(sweep, name, methods, title=None, width=ONE, height=2.35):
    """Paired ES gain over no context. Separate from the level plot: this is what was tested."""
    fig, ax = plt.subplots(figsize=(width, height))
    if title:
        ax.set_title(title, loc="left", color=INK)
    offsets = {m: (i - (len(methods) - 1) / 2) * 0.09 for i, m in enumerate(methods)}
    for method in methods:
        xs, ys, low, high = [], [], [], []
        for b in BUDGETS:
            if str(b) in sweep and method in sweep[str(b)]:
                d = sweep[str(b)][method]["es_vs_none"]
                xs.append(b * 2 ** offsets[method])
                ys.append(d[0])
                low.append(d[0] - d[1])
                high.append(d[2] - d[0])
        if xs:
            bars = ax.errorbar(
                xs,
                ys,
                yerr=[low, high],
                color=COLOR[method],
                marker=MARKER[method],
                label=NAME[method],
                capsize=1.2,
                elinewidth=0.6,
                linewidth=0.9,
                linestyle=DASH.get(method, "-"),
                markeredgecolor="white",
                markeredgewidth=0.4,
            )
            bars[2][0].set_linestyle("-")
    ax.axhline(0, color=INK2, linewidth=0.9)
    budget_axis(ax)
    ax.set_ylabel(r"$\Delta$ES vs no context (95% CI)")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0, handletextpad=0.4
    )
    save(fig, name)


def fig_time(sweep, name, methods, title=None):
    fig, ax = plt.subplots(figsize=(ONE, 2.2))
    if title:
        ax.set_title(title, loc="left", color=INK)
    for method in methods:
        points = [
            (b, sweep[str(b)][method])
            for b in BUDGETS
            if str(b) in sweep and method in sweep[str(b)]
        ]
        if points:
            xs = [b for b, _ in points]
            ax.plot(
                xs,
                [c["median_seconds"] for _, c in points],
                color=COLOR[method],
                marker=MARKER[method],
                label=NAME[method],
                linestyle=DASH.get(method, "-"),
                markeredgecolor="white",
                markeredgewidth=0.4,
            )
    if "none" in sweep:
        ax.axhline(sweep["none"]["median_seconds"], color=INK2, linewidth=0.9)
        ax.annotate(
            "No context",
            xy=(1, sweep["none"]["median_seconds"]),
            xycoords=("axes fraction", "data"),
            xytext=(-2, -8),
            ha="right",
            textcoords="offset points",
            fontsize=7,
            color=INK2,
        )
    budget_axis(ax)
    ax.set_ylabel("Median generation time (s)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", ncol=1, handletextpad=0.4)
    save(fig, name)


def fig_tradeoff(sweep, name, methods, title=None):
    """Accuracy against tokens actually used: two budgets can spend the same tokens."""
    fig, ax = plt.subplots(figsize=(ONE, 2.35))
    if title:
        ax.set_title(title, loc="left", color=INK)
    for method in methods:
        points = [
            sweep[str(b)][method] for b in BUDGETS if str(b) in sweep and method in sweep[str(b)]
        ]
        if points:
            ax.plot(
                [c["mean_context_tokens"] for c in points],
                [c["es_vs_none"][0] for c in points],
                color=COLOR[method],
                marker=MARKER[method],
                label=NAME[method],
                linestyle=DASH.get(method, "-"),
                markeredgecolor="white",
                markeredgewidth=0.4,
            )
    ax.axhline(0, color=INK2, linewidth=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Mean context tokens actually used")
    ax.set_ylabel(r"$\Delta$ES vs no context")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0, handletextpad=0.4
    )
    save(fig, name)


def fig_levels(levels, name):
    budgets = [b for b in ("512", "2048") if b in levels and levels[b]]
    if not budgets:
        return
    fig, ax = plt.subplots(figsize=(ONE, 2.2))
    order = list(levels[budgets[0]])
    tone = {"512": "#86b6ef", "2048": "#1c5cab"}
    for j, budget in enumerate(budgets):
        data = levels[budget]
        xs = [order.index(k) + (j - (len(budgets) - 1) / 2) * 0.22 for k in data]
        ys = [data[k]["es_gain"] for k in data]
        ax.errorbar(
            xs,
            ys,
            yerr=[
                [y - data[k]["low"] for y, k in zip(ys, data, strict=True)],
                [data[k]["high"] - y for y, k in zip(ys, data, strict=True)],
            ],
            fmt="o",
            color=tone[budget],
            capsize=1.5,
            elinewidth=0.8,
            markersize=3.5,
            label=f"$N={budget}$",
        )
    ax.axhline(0, color=INK2, linewidth=0.9)
    ax.set_xticks(range(len(order)), order)
    ax.set_xlabel("RepoBench length level")
    ax.set_ylabel(r"$\Delta$ES, BM25 long vs no context")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, name)


def fig_labels(selector, name):
    counts = selector["counts"][:10]
    if not counts:
        return
    total = sum(c for _, c in selector["counts"])
    labels = [
        k.replace("bm25_long", "BM25 long")
        .replace("bm25_short", "BM25 short")
        .replace("signatures", "Signature map")
        .replace("none", "No context")
        .replace("@", ", N=")
        for k, _ in counts
    ][::-1]
    values = [100 * c / total for _, c in counts][::-1]
    fig, ax = plt.subplots(figsize=(ONE, 2.4))
    ax.barh(labels, values, color="#2a78d6", height=0.62)
    for y, value in enumerate(values):
        ax.annotate(
            f"{value:.0f}%",
            (value, y),
            xytext=(2, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
            color=INK2,
        )
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", left=False)
    ax.set_title(
        "Pass 2 label: best option per example (ties to fewest tokens)",
        loc="left",
        color=INK,
        fontsize=7.5,
    )
    save(fig, name)


def fig_models(small, large, name):
    methods = [
        m
        for m in ("bm25_long", "bm25_short", "oracle_long")
        if any(str(b) in large and m in large[str(b)] for b in BUDGETS)
    ]
    if not methods:
        return
    fig, axes = plt.subplots(1, 2, figsize=(TWO, 2.3), sharex=True, sharey=True)
    for ax, sweep, title in (
        (axes[0], small, "DeepSeek-Coder 1.3B"),
        (axes[1], large, "DeepSeek-Coder 5.7B"),
    ):
        curves(ax, sweep, methods, "es", "es")
        budget_axis(ax)
        ax.set_title(title, loc="left", color=INK)
    axes[0].set_ylabel("Edit similarity (0-100)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=len(methods))
    save(fig, name)


def fig_splits(first, rand, name):
    """Paired gain at N=512 on both splits. Different examples, so compare gains, not levels."""
    methods = ["bm25_long", "bm25_short", "oracle_long", "oracle_short", "oracle"]
    methods = [m for m in methods if m in first and m in rand]
    fig, ax = plt.subplots(figsize=(ONE, 2.2))
    for j, (cells, label, tone) in enumerate(
        ((first, "cross_file_first", "#1c5cab"), (rand, "cross_file_random", "#86b6ef"))
    ):
        ys = [len(methods) - 1 - i + (0.14 if j == 0 else -0.14) for i in range(len(methods))]
        d = [cells[m]["es_vs_none"] for m in methods]
        ax.errorbar(
            [x[0] for x in d],
            ys,
            xerr=[[x[0] - x[1] for x in d], [x[2] - x[0] for x in d]],
            fmt="o",
            color=tone,
            capsize=1.2,
            elinewidth=0.7,
            markersize=3.5,
            label=label,
        )
    ax.axvline(0, color=INK2, linewidth=0.9)
    ax.set_yticks(range(len(methods)), [NAME[m] for m in methods][::-1])
    ax.grid(False)
    ax.set_xlabel(r"$\Delta$ES vs no context at $N=512$ (95% CI)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.22), ncol=2)
    save(fig, name)


def main():
    sweep = R.get("sweep_500", {})
    base = ["bm25_long", "bm25_short"]
    with_sig = base + (
        ["signatures"] if any("signatures" in sweep.get(str(b), {}) for b in BUDGETS) else []
    )
    ceiling = ["oracle_long"]
    if sweep:
        fig_es_curve(sweep, "fig1_es_vs_budget", with_sig + ceiling, title=N500)
        fig_em_curve(sweep, "fig2_em_vs_budget", with_sig + ceiling, title=N500)
        fig_delta(sweep, "fig3_delta_es_vs_budget", with_sig + ceiling, title=N500)
        fig_es_curve(
            sweep,
            "fig4_oracle_definitions",
            ["bm25_long", "oracle_long", "oracle_short", "oracle"],
            title=N500,
        )
        fig_time(sweep, "fig5_time_vs_budget", with_sig + ceiling, title=N500)
        fig_tradeoff(sweep, "fig6_gain_vs_tokens", with_sig + ceiling, title=N500)
        fig_levels(R.get("levels_2000", {}), "fig7_gain_by_length")
        if R.get("selector_500"):
            fig_labels(R["selector_500"], "fig8_selector_labels")
    if R.get("sweep_large_500"):
        fig_models(sweep, R["sweep_large_500"], "fig9_model_sizes")
    if R.get("sweep_random_500") and "512" in sweep:
        fig_splits(sweep["512"], R["sweep_random_500"]["512"], "fig11_first_vs_random")
    if R.get("sweep_2000"):
        pooled = R["sweep_2000"]
        three = (
            ["bm25_long", "bm25_short"]
            + (
                ["signatures"]
                if any("signatures" in pooled.get(str(b), {}) for b in BUDGETS)
                else []
            )
            + ["oracle_long"]
        )
        fig_es_curve(pooled, "fig10_es_vs_budget_n2000", three, title=N2000)
        fig_delta(pooled, "fig12_delta_es_n2000", three, title=N2000)
        fig_em_curve(pooled, "fig13_em_n2000", three, title=N2000)
        fig_time(pooled, "fig14_time_n2000", three, title=N2000)
        fig_tradeoff(pooled, "fig15_gain_vs_tokens_n2000", three, title=N2000)


if __name__ == "__main__":
    main()
