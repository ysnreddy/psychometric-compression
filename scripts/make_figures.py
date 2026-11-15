"""Paper figures (PDF for LaTeX + PNG previews) from the experiment JSONs.

Fig 1 — compression curves (EA.2): aggregate MAE and Kendall tau vs subset size, five
selectors. Fig 2 — stratum-collapse exhibit (EA.4): per-stratum error dumbbells,
unconstrained vs stratified, zero-coverage strata flagged.

Design rules applied (dataviz skill): categorical hues in the reference palette's fixed
slot order; secondary encoding (markers/linestyles) so identity is never color-alone;
one axis per panel; thin marks; recessive grid; direct labels for key series; legends
present; no rainbow, no dual axes.
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent.parent
import os
RES = HERE / os.environ.get("RESULTS_DIR", "results")          # e.g. RESULTS_DIR=results/v2
FIG = HERE / os.environ.get("FIG_DIR", "paper/tex/figures")     # e.g. FIG_DIR=paper/tmlr/figures
FIG.mkdir(parents=True, exist_ok=True)

# reference palette, fixed slot order (validated set; see dataviz references/palette.md)
C = {"blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100",
     "green": "#008300", "violet": "#4a3aa7", "orange": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"

plt.rcParams.update({
    "font.size": 8, "font.family": "sans-serif", "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "pdf.fonttype": 42,
})

# series identity: fixed assignment (never re-ordered by rank)
METHODS = [
    ("random",               "Random",              C["blue"],   "o", "-"),
    ("stratified_random",    "Stratified random",   C["aqua"],   "s", "--"),
    ("kmedoids",             "k-medoids anchors",   C["yellow"], "^", ":"),
    ("fisher_unconstrained", "Info (unconstrained)", C["violet"], "D", "-."),
    ("fisher_stratified",    "Info + stratum quotas", C["green"], "v", "-"),
]


def fig1():
    d = json.load(open(RES / "ea2_gp_cal.json"))
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.3))
    for key, label, col, mk, ls in METHODS:
        rows = d["curves"][key]
        n = [r["n_clips"] for r in rows]
        mae = [r["mae_mean"] * 100 for r in rows]
        lo = [r["mae_ci95"][0] * 100 for r in rows]
        hi = [r["mae_ci95"][1] * 100 for r in rows]
        tau = [r["tau_mean"] for r in rows]
        lw = 1.6 if key in ("fisher_stratified", "random") else 1.1
        axes[0].plot(n, mae, ls, color=col, marker=mk, ms=3.5, lw=lw, label=label,
                     markerfacecolor="white", markeredgewidth=1.0)
        if key in ("fisher_stratified", "random"):
            axes[0].fill_between(n, lo, hi, color=col, alpha=0.12, lw=0)
        axes[1].plot(n, tau, ls, color=col, marker=mk, ms=3.5, lw=lw,
                     markerfacecolor="white", markeredgewidth=1.0)
    # direct labels for the two key series (relief rule; identity not color-alone)
    axes[0].set_xlabel("subset size (clips)")
    axes[0].set_ylabel("score MAE (pp)")
    axes[0].set_ylim(0, 2.6)
    axes[1].set_xlabel("subset size (clips)")
    axes[1].set_ylabel("Kendall $\\tau$ (aggregate ranking)")
    axes[1].set_ylim(0.35, 1.0)
    for ax in axes:
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks([50, 100, 150, 200, 300])
    axes[0].legend(loc="lower right", fontsize=6.2, frameon=False, ncol=2,
                   handlelength=2.2, columnspacing=0.9)
    axes[0].set_title("(a) score reconstruction error", fontsize=8, loc="left")
    axes[1].set_title("(b) rank preservation", fontsize=8, loc="left")
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_compression.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("fig1 written")


def fig2():
    # n=200 run (the shipped SteelBench-mini size); ea4_strata.json holds the n=150 sweep
    d = json.load(open(RES / "ea4_strata_n200.json"))
    u = d["methods"]["fisher_unconstrained"]
    s = d["methods"]["fisher_stratified"]
    strata = sorted(u["per_stratum_mae"], key=lambda k: -u["per_stratum_mae"][k])
    uv = np.array([u["per_stratum_mae"][k] * 100 for k in strata])
    sv = np.array([s["per_stratum_mae"][k] * 100 for k in strata])
    zero_u = set(u["zero_cov_strata"])
    ypos = np.arange(len(strata))[::-1]

    fig, ax = plt.subplots(figsize=(3.35, 4.3))
    # dumbbell connectors
    for y, a, b in zip(ypos, uv, sv):
        ax.plot([min(a, b), max(a, b)], [y, y], color=GRID, lw=1.4, zorder=1)
    # unconstrained: open marker when the stratum got ZERO clips (score literally unmeasured)
    for y, v, st in zip(ypos, uv, strata):
        filled = st not in zero_u
        ax.plot(v, y, "o", ms=4.2, color=C["orange"], zorder=3,
                markerfacecolor=C["orange"] if filled else "white", markeredgewidth=1.1)
    ax.plot(sv, ypos, "o", ms=4.2, color=C["blue"], zorder=3)
    def fmt(k):
        vis, layer, band = k.split("|")
        return f"{vis} · L{layer} · {band}"
    labels = [fmt(k) for k in strata]
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=5.6)
    ax.set_xlabel("per-stratum score error (pp)")
    ax.grid(True, axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    # legend with the zero-coverage state encoded (icon + label, not color alone)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="", color=C["orange"], ms=4.2, label="info-max (unconstrained)"),
        Line2D([], [], marker="o", ls="", color=C["orange"], markerfacecolor="white",
               markeredgewidth=1.1, ms=4.2, label="  └ zero clips selected"),
        Line2D([], [], marker="o", ls="", color=C["blue"], ms=4.2, label="info + stratum quotas"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6.2, frameon=False)
    ax.set_title(f"per-stratum error at n = {d['n']} "
                 f"(zero-covered strata: {u['strata_zero_coverage']} vs "
                 f"{s['strata_zero_coverage']})", fontsize=7.5, loc="left")
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_strata.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("fig2 written")


if __name__ == "__main__":
    fig1()
    fig2()
