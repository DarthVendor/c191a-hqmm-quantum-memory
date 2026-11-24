"""Figure 1 of the interim report, drawn from analysis/results/prelim_results.json.
Run:  .venv/bin/python -I analysis/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "analysis" / "results" / "prelim_results.json"
OUT = ROOT / "report" / "figures"

BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (validated, light)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e6e5e1"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "pdf.fonttype": 42,
})


def panel_rank(ax, rec):
    for key, color, marker, label in [("quantum", BLUE, "o", "qubit HQMM"),
                                      ("dephased", ORANGE, "s", "dephased")]:
        s = rec[key]["singular_values"]
        ax.semilogy(range(1, len(s) + 1), s, color=color, marker=marker, ms=4.5,
                    lw=1.5, label=f"{label} (rank {rec[key]['rank']})")
    ax.axhspan(1e-19, 1e-13, color=GRID, lw=0)
    ax.text(1.0, 1e-18, "numerical zero", ha="left", va="center", color=INK2, fontsize=7)
    # headroom above the data holds the legend clear of every curve
    ax.set_ylim(1e-19, 1e8)
    ax.set_yticks([1e0, 1e-5, 1e-10, 1e-15])
    ax.set_xticks(range(1, 7))
    ax.set_xlabel("singular value index")
    ax.set_ylabel("Hankel singular value")
    ax.set_title("(a) Tetrahedral: Hankel spectrum", color=INK, loc="left")
    ax.legend(loc="upper right", fontsize=6.5, borderaxespad=0.2)


def panel_fits(ax, rec, bound, title):
    ns = sorted(int(n) for n in rec["hmm_fits"])
    loss = [rec["hmm_fits"][str(n)]["test_bits"] for n in ns]
    ax.plot(ns, loss, color=BLUE, marker="o", ms=4.5, lw=1.5)
    true = rec["true_model_test_bits"]
    ax.axhline(true, color=INK2, lw=1, ls="--")
    ax.annotate("1-qubit HQMM", xy=(ns[0], true), xytext=(0, 2), textcoords="offset points",
                ha="left", va="bottom", color=INK2, fontsize=6.5)
    ax.axvline(bound, color=MUTED, lw=1, ls=":")
    ax.text(bound + 0.08, max(loss), f"rank bound\nn ≥ {bound}", ha="left", va="top",
            color=INK2, fontsize=7)
    ax.set_xticks(ns)
    ax.set_xlabel("classical hidden states n")
    ax.set_ylabel("held-out log loss (bits/symbol)")
    ax.set_title(title, color=INK, loc="left")


if __name__ == "__main__":
    r = json.loads(RESULTS.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.35), constrained_layout=True)
    panel_rank(axes[0], r["E2_hankel"]["tetrahedral_sic"])
    panel_fits(axes[1], r["E3_E4_fits"]["tetrahedral_sic"], 4, "(b) Tetrahedral: HMM fits")
    panel_fits(axes[2], r["E3_E4_fits"]["trine"], 3, "(c) Trine: HMM fits")
    fig.savefig(OUT / "fig_rank_and_fits.pdf")
    fig.savefig(OUT / "fig_rank_and_fits.png", dpi=200)
    print("wrote", OUT / "fig_rank_and_fits.pdf")
