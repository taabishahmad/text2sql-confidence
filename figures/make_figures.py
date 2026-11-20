#!/usr/bin/env python3
"""Regenerate all manuscript figures in a consistent, publication-grade house
style, reading every number from all_results.json so values match the paper."""
import os, json, numpy as np
_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import FuncFormatter

D = json.load(open(os.path.join(_ROOT,"outputs","all_results.json")))
OUT = os.path.join(_ROOT,"figures")+os.sep
os.makedirs(OUT, exist_ok=True)

# ---------- house style ----------
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.9,
    "axes.grid": True, "grid.color": "#D9D9D9", "grid.linewidth": 0.7,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#333333", "ytick.color": "#333333",
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9.5, "legend.frameon": True, "legend.framealpha": 0.95,
    "legend.edgecolor": "#CCCCCC", "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})
C = {"Heuristic": "#6B7280", "CRC": "#E08A2B", "RCPS": "#2E6FB0", "LtT": "#2E8B57"}
MK = {"Heuristic": "o", "CRC": "s", "RCPS": "D", "LtT": "^"}
LBL = {"Heuristic": "Heuristic", "CRC": "CRC", "RCPS": "RCPS", "LtT": "LtT"}
ALPHAS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

def style_line(ax):
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#444444")
    ax.tick_params(length=3, width=0.8)

def panel_tag(ax, s):
    ax.set_title(s, loc="left", fontsize=12, fontweight="bold", pad=8)

sp = D["datasets"]["spider__gpt-4o-mini"]["backends"]

# ============ FIG 2: validity + price of a guarantee ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.1))
for bk in ["Heuristic", "CRC", "RCPS", "LtT"]:
    y = [(np.nan if sp[f"{a:g}"][bk]["viol_rate"] is None else sp[f"{a:g}"][bk]["viol_rate"] * 100) for a in ALPHAS]
    ax1.plot(ALPHAS, y, marker=MK[bk], color=C[bk], lw=2, ms=6,
             mec="white", mew=0.8, label=LBL[bk])
ax1.axhline(10, ls="--", lw=1.3, color="#333333")
ax1.text(0.405, 11.5, r"allowance $\delta=10\%$", fontsize=8.5, ha="right", color="#333333")
ax1.set_xlabel(r"risk tolerance $\alpha$"); ax1.set_ylabel("per-deployment violation rate (%)")
ax1.set_ylim(-3, 60); panel_tag(ax1, "a  Validity"); style_line(ax1)
ax1.legend(loc="upper right", ncol=2)
for bk in ["Heuristic", "CRC", "RCPS", "LtT"]:
    y = [sp[f"{a:g}"][bk]["coverage"] for a in ALPHAS]
    y = [np.nan if v is None else v for v in y]
    ax2.plot(ALPHAS, y, marker=MK[bk], color=C[bk], lw=2, ms=6,
             mec="white", mew=0.8, label=LBL[bk])
ax2.set_xlabel(r"risk tolerance $\alpha$"); ax2.set_ylabel("coverage (fraction answered)")
ax2.set_ylim(0.15, 1.02); panel_tag(ax2, "b  Price of a guarantee"); style_line(ax2)
ax2.legend(loc="lower right", ncol=2)
fig.tight_layout(); fig.savefig(OUT + "fig2_backends.png"); plt.close(fig)

# ============ FIG 3: risk-coverage trade-off ============
fig, ax = plt.subplots(figsize=(6.4, 4.8))
for bk in ["Heuristic", "CRC", "RCPS", "LtT"]:
    pts = [(sp[f"{a:g}"][bk]["coverage"], sp[f"{a:g}"][bk]["risk"]) for a in ALPHAS]
    pts = [(c, r) for c, r in pts if c is not None and r is not None]
    pts.sort()
    xs, ys = zip(*pts)
    ax.plot(xs, ys, marker=MK[bk], color=C[bk], lw=2, ms=6, mec="white", mew=0.8, label=LBL[bk])
ax.axhline(0.25, ls=":", lw=1.2, color="#999999")
ax.text(0.205, 0.256, r"$\alpha=0.25$ target", fontsize=8.5, color="#777777")
ax.set_xlabel("coverage (fraction answered)"); ax.set_ylabel("realised selective risk")
panel_tag(ax, ""); style_line(ax); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(OUT + "fig3_riskcoverage.png"); plt.close(fig)

# ============ FIG 4: distribution shift (CLEAN bar chart) ============
sh = D["distribution_shift"]["result"]
backends = ["RCPS", "LtT"]
indist = [sh[b]["in_dist"]["viol_rate"] * 100 for b in backends]
shifted = [sh[b]["shifted"]["viol_rate"] * 100 for b in backends]
x = np.arange(len(backends)); w = 0.34
fig, ax = plt.subplots(figsize=(6.6, 4.5))
b1 = ax.bar(x - w/2, indist, w, color="#2E6FB0", label="In-distribution (Spider → Spider)",
            edgecolor="white", linewidth=0.8)
b2 = ax.bar(x + w/2, shifted, w, color="#C0392B", label="Shifted (Spider → BIRD)",
            edgecolor="white", linewidth=0.8)
ax.axhline(10, ls="--", lw=1.3, color="#333333")
ax.text(1.45, 12.5, r"allowance $\delta=10\%$", fontsize=9, ha="right", color="#333333")
for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.0f}%", (rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold", color="#222222")
ax.set_xticks(x); ax.set_xticklabels(backends, fontsize=11)
ax.set_ylabel("per-deployment violation rate (%)"); ax.set_ylim(0, 118)
panel_tag(ax, ""); style_line(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
fig.tight_layout(); fig.savefig(OUT + "fig4_shift.png"); plt.close(fig)

# ============ FIG 5: certifiability by score (CLEAN horizontal bars) ============
sc = D["datasets"]["spider__gpt-4o-mini"]["score_certifiability"]
order = ["Learned fusion", "Token logprob", "Self-consistency", "Semantic entropy"]
disp = {"Learned fusion": "Learned fusion\n(24 features)", "Token logprob": "Token log-probability",
        "Self-consistency": "Self-consistency ($K{=}5$)", "Semantic entropy": "Semantic entropy ($K{=}5$)"}
vals = [sc[k]["cert_rate"] * 100 for k in order]
cols = ["#2E8B57", "#2E6FB0", "#B0413E", "#B0413E"]
fig, ax = plt.subplots(figsize=(7.0, 4.0))
y = np.arange(len(order))[::-1]
bars = ax.barh(y, vals, color=cols, edgecolor="white", linewidth=0.8, height=0.6)
for rect, v in zip(bars, vals):
    if v > 1:
        ax.annotate(f"{v:.1f}%", (v, rect.get_y() + rect.get_height()/2),
                    xytext=(5, 0), textcoords="offset points", va="center", fontsize=10, fontweight="bold")
    else:
        ax.annotate("never certified", (0.5, rect.get_y() + rect.get_height()/2),
                    xytext=(5, 0), textcoords="offset points", va="center", fontsize=9.5,
                    style="italic", color="#B0413E")
ax.set_yticks(y); ax.set_yticklabels([disp[k] for k in order])
ax.set_xlabel("certification rate (% of 300 splits)"); ax.set_xlim(0, 80)
ax.grid(axis="y", visible=False); panel_tag(ax, ""); style_line(ax)
fig.tight_layout(); fig.savefig(OUT + "fig5_scores.png"); plt.close(fig)

# ============ FIG 6: calibration data required (dual axis) ============
cz = D["datasets"]["spider__gpt-4o-mini"]["calibration_size"]
ns = sorted(int(k) for k in cz)
cert = [cz[str(n)]["cert_rate"] * 100 for n in ns]
cov = [cz[str(n)]["coverage"] for n in ns]
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.plot(ns, cert, marker="o", color="#2E6FB0", lw=2.2, ms=7, mec="white", mew=0.9, label="certification rate")
ax.set_xlabel("calibration set size $n$"); ax.set_ylabel("certification rate (%)", color="#2E6FB0")
ax.tick_params(axis="y", labelcolor="#2E6FB0"); ax.set_ylim(0, 70)
ax2 = ax.twinx(); ax2.grid(False)
ax2.plot(ns, cov, marker="D", color="#E08A2B", lw=2.2, ms=6, mec="white", mew=0.9, ls="--", label="certified coverage")
ax2.set_ylabel("certified coverage", color="#E08A2B"); ax2.tick_params(axis="y", labelcolor="#E08A2B")
ax2.set_ylim(0.3, 0.75); ax2.spines["top"].set_visible(False)
l1, lab1 = ax.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, lab1 + lab2, loc="upper left")
style_line(ax); fig.tight_layout(); fig.savefig(OUT + "fig6_calibration.png"); plt.close(fig)

# ============ FIG 7: sensitivity (delta + grid) ============
sens = D["datasets"]["spider__gpt-4o-mini"]["sensitivity"]
fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.2, 4.1))
dk = sorted(float(k) for k in sens["delta"])
dcert = [sens["delta"][f"{k:g}"]["cert_rate"] * 100 for k in dk]
dcov = [sens["delta"][f"{k:g}"]["coverage"] for k in dk]
axa.plot(dk, dcert, marker="o", color="#2E6FB0", lw=2.2, ms=7, mec="white", mew=0.9, label="certification rate")
axa.set_xscale("log"); axa.set_xticks(dk); axa.set_xticklabels([f"{k:g}" for k in dk])
axa.set_xlabel(r"confidence level $\delta$"); axa.set_ylabel("certification rate (%)", color="#2E6FB0")
axa.tick_params(axis="y", labelcolor="#2E6FB0"); axa.set_ylim(0, 80)
axa2 = axa.twinx(); axa2.grid(False)
axa2.plot(dk, dcov, marker="D", ls="--", color="#E08A2B", lw=2, ms=6, mec="white", mew=0.9)
axa2.set_ylabel("certified coverage", color="#E08A2B"); axa2.tick_params(axis="y", labelcolor="#E08A2B")
axa2.set_ylim(0.3, 0.75); axa2.spines["top"].set_visible(False)
panel_tag(axa, "a  Confidence level"); style_line(axa)
gk = sorted(int(k) for k in sens["grid"])
gcert = [sens["grid"][str(k)]["cert_rate"] * 100 for k in gk]
gcov = [sens["grid"][str(k)]["coverage"] for k in gk]
axb.plot(gk, gcert, marker="o", color="#2E6FB0", lw=2.2, ms=7, mec="white", mew=0.9)
axb.set_xscale("log"); axb.set_xticks(gk); axb.set_xticklabels([str(k) for k in gk])
axb.set_xlabel("threshold grid size $m$"); axb.set_ylabel("certification rate (%)", color="#2E6FB0")
axb.tick_params(axis="y", labelcolor="#2E6FB0"); axb.set_ylim(0, 100)
axb2 = axb.twinx(); axb2.grid(False)
axb2.plot(gk, gcov, marker="D", ls="--", color="#E08A2B", lw=2, ms=6, mec="white", mew=0.9)
axb2.set_ylabel("certified coverage", color="#E08A2B"); axb2.tick_params(axis="y", labelcolor="#E08A2B")
axb2.set_ylim(0.3, 0.75); axb2.spines["top"].set_visible(False)
panel_tag(axb, "b  Grid resolution"); style_line(axb)
fig.tight_layout(); fig.savefig(OUT + "fig7_sensitivity.png"); plt.close(fig)

# ============ FIG 8: deployment economics (CLEAN grouped bars, log y) ============
ec = D["datasets"]["spider__gpt-4o-mini"]["economics"]
ratios = ["100/5", "500/5", "1000/10"]
rlab = [r"\$100 / \$5", r"\$500 / \$5", r"\$1000 / \$10"]
no_ab = [ec[r]["no_abstention"] for r in ratios]
cert_ = [ec[r]["certified"] for r in ratios]
save = [ec[r]["saving_pct"] for r in ratios]
x = np.arange(len(ratios)); w = 0.34
fig, ax = plt.subplots(figsize=(7.2, 4.5))
b1 = ax.bar(x - w/2, no_ab, w, color="#B0413E", label="No abstention", edgecolor="white", linewidth=0.8)
b2 = ax.bar(x + w/2, cert_, w, color="#2E8B57", label="Certified ($\\alpha=0.25$)", edgecolor="white", linewidth=0.8)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"${v/1000:.0f}k"))
for rect in list(b1) + list(b2):
    h = rect.get_height()
    ax.annotate(f"${h/1000:.0f}k", (rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                fontsize=8.5, color="#222222")
for i, s in enumerate(save):
    top = max(no_ab[i], cert_[i])
    ax.annotate(f"−{s:.0f}%", (i, top * 1.5), ha="center", fontsize=10,
                fontweight="bold", color="#2E8B57")
ax.set_xticks(x); ax.set_xticklabels(rlab)
ax.set_xlabel(r"error cost / review cost ($c_{\mathrm{err}}$ / $c_{\mathrm{rev}}$)")
ax.set_ylabel("expected daily cost (log scale)")
ax.set_ylim(8e3, 7e5); ax.grid(axis="x", visible=False)
panel_tag(ax, ""); style_line(ax); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(OUT + "fig8_economics.png"); plt.close(fig)

# ============ FIG 9 (NEW, Supplementary): error analysis ============
feat_order = ["agreement_ratio", "semantic_entropy", "struct_agreement_ratio", "n_joins", "frac_columns_valid"]
feat_lbl = ["Execution\nagreement", "Semantic\nentropy", "Structural\nagreement", "Num.\njoins", "Valid-column\nfraction"]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=False)
for ax, dk_, title in [(axes[0], "spider__gpt-4o-mini", "a  Spider"), (axes[1], "bird__gpt-4o-mini", "b  BIRD")]:
    ea = D["datasets"][dk_]["error_analysis"]
    cor = [ea[f]["correct"] for f in feat_order]
    inc = [ea[f]["incorrect"] for f in feat_order]
    x = np.arange(len(feat_order)); w = 0.36
    ax.bar(x - w/2, cor, w, color="#2E8B57", label="Correct", edgecolor="white", linewidth=0.8)
    ax.bar(x + w/2, inc, w, color="#B0413E", label="Incorrect", edgecolor="white", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(feat_lbl, fontsize=8.5)
    ax.set_ylabel("mean feature value"); ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    panel_tag(ax, title); style_line(ax); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=2, fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "fig9_erroranalysis.png"); plt.close(fig)

print("all figures regenerated")
