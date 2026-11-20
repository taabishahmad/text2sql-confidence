#!/usr/bin/env python3
"""Architecture diagram, redesigned as a clean orthogonal flowchart
(draw.io / Visio style): flat rounded boxes, right-angle connectors routed
through the empty band between the two phases so no line crosses a box."""
import os, matplotlib
_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK = "#0F172A"; SUB = "#475569"
GREY_F = "#F6F8FB"; GREY_B = "#B6C2D1"
BLUE_B = "#2563EB"
GREEN_F = "#ECFDF5"; GREEN_B = "#10B981"
RED_F = "#FEF2F2"; RED_B = "#EF4444"
AMBER_F = "#FFFBEB"; AMBER_B = "#F59E0B"
ARROW = "#64748B"
GREEN = "#059669"

fig, ax = plt.subplots(figsize=(12.0, 5.8))
ax.set_xlim(0, 133); ax.set_ylim(0, 62); ax.axis("off")

def box(x, y, w, h, title, sub="", fill=GREY_F, border=GREY_B, tcol=INK,
        tsize=10.3, ssize=8.0, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.8",
                                linewidth=lw, edgecolor=border, facecolor=fill, zorder=3))
    if sub:
        ax.text(x + w/2, y + h*0.62, title, ha="center", va="center",
                fontsize=tsize, fontweight="bold", color=tcol, zorder=4)
        ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center",
                fontsize=ssize, color=SUB, zorder=4, linespacing=1.22)
    else:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center",
                fontsize=tsize, fontweight="bold", color=tcol, zorder=4)

def harrow(x1, x2, y, color=ARROW, lw=1.7):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=14, lw=lw, color=color), zorder=2)

def seg(x1, y1, x2, y2, color=ARROW, lw=1.7, head=False, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>" if head else "-", mutation_scale=14,
                                lw=lw, color=color, linestyle=ls,
                                shrinkA=0, shrinkB=0), zorder=2)

# ---- phase headers with divider rules ----
ax.text(2.5, 59.3, "CALIBRATION", fontsize=11, fontweight="bold", color=BLUE_B,
        family="DejaVu Sans")
ax.text(29, 59.3, "offline · once per deployment", fontsize=9.2, color=SUB, style="italic")
ax.plot([2.5, 127.5], [57.4, 57.4], color="#E2E8F0", lw=1.2, zorder=1)

ax.text(2.5, 27.0, "DEPLOYMENT", fontsize=11, fontweight="bold", color=GREEN,
        family="DejaVu Sans")
ax.text(28, 27.0, "online · per query", fontsize=9.2, color=SUB, style="italic")
ax.plot([2.5, 127.5], [25.1, 25.1], color="#E2E8F0", lw=1.2, zorder=1)

# ===== Lane A : calibration =====
yA = 43.5; hA = 11.0; wA = 19.5
ax1 = [3, 25.5, 48, 70.5]
box(ax1[0], yA, wA, hA, "Calibration set", "n questions\n+ gold SQL")
box(ax1[1], yA, wA, hA, "LLM generation", "K candidate\nqueries each")
box(ax1[2], yA, wA, hA, "Confidence score", "execution · schema ·\ntoken uncertainty")
box(ax1[3], yA, wA, hA, "Risk-control backend", "Heuristic / CRC /\nRCPS / LtT")
for i in range(3):
    harrow(ax1[i] + wA, ax1[i+1], yA + hA/2)

# outputs: certificate (upper) + refuse (lower)
xo = 96.0; wo = 30.0
box(xo, yA + 5.4, wo, 5.6, r"Certificate  $(\alpha,\ \delta,\ \hat\lambda,\ n)$", "",
    fill=GREEN_F, border=GREEN_B, tsize=9.8)
box(xo, yA - 1.1, wo, 5.6, "Refuse", "no certifiable threshold",
    fill=RED_F, border=RED_B, tsize=9.8, ssize=7.6)
seg(ax1[3] + wA, yA + hA/2 + 1.8, xo, yA + 8.2, color=GREEN_B, head=True)
seg(ax1[3] + wA, yA + hA/2 - 1.8, xo, yA + 1.7, color=RED_B, head=True, ls=(0, (4, 2)))

# ===== Lane B : deployment =====
yB = 10.0; hB = 11.0; wB = 19.5
box(ax1[0], yB, wB, hB, "New question", r"$q$")
box(ax1[1], yB, wB, hB, "LLM → SQL", r"query $\hat{y}$," + "\n" + r"score $s(\hat{y})$")
box(ax1[2], yB, wB, hB, "Gate", r"$s(\hat{y}) \geq \hat\lambda$ ?",
    fill=AMBER_F, border=AMBER_B)
harrow(ax1[0] + wB, ax1[1], yB + hB/2)
harrow(ax1[1] + wB, ax1[2], yB + hB/2)

box(ax1[3], yB + 5.4, wB, 5.6, "Answer", r"risk $\leq \alpha$ w.p. $1-\delta$",
    fill=GREEN_F, border=GREEN_B, tsize=10.0, ssize=7.6)
box(ax1[3], yB - 1.1, wB, 5.6, "Abstain", "defer to analyst",
    fill=RED_F, border=RED_B, tsize=10.0, ssize=7.6)
seg(ax1[2] + wB, yB + hB/2 + 1.6, ax1[3], yB + 8.2, color=GREEN_B, head=True)
seg(ax1[2] + wB, yB + hB/2 - 1.6, ax1[3], yB + 1.7, color=RED_B, head=True, ls=(0, (4, 2)))
ax.text(ax1[3] - 1.0, yB + 9.4, "yes", fontsize=8.2, color=GREEN, fontweight="bold", ha="right")
ax.text(ax1[3] - 1.0, yB + 0.8, "no", fontsize=8.2, color=RED_B, fontweight="bold", ha="right")

# ===== orthogonal hand-off: deploy lambda-hat (routed around the right margin,
#       so the connector never crosses the Refuse box or any other) =====
cert_r = xo + wo          # certificate right edge (~126)
cert_cy = yA + 8.2        # certificate vertical centre (~51.7)
mx = 129.0                # right routing margin, clear of all boxes
gx = ax1[2] + wB/2        # gate centre x (~57.75)
ymid = 31.5               # empty band between the two lanes
seg(cert_r, cert_cy, mx, cert_cy, color=GREEN, lw=2.0, ls=(0, (5, 2)))   # right to margin
seg(mx, cert_cy, mx, ymid, color=GREEN, lw=2.0, ls=(0, (5, 2)))          # down the margin
seg(mx, ymid, gx, ymid, color=GREEN, lw=2.0, ls=(0, (5, 2)))            # across the band
seg(gx, ymid, gx, yB + hB, color=GREEN, lw=2.0, head=True, ls=(0, (5, 2)))  # down into gate
ax.text((mx + gx)/2, ymid + 1.8, r"deploy certified threshold $\hat\lambda$",
        fontsize=9.2, color=GREEN, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREEN, lw=1.0), zorder=5)

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(os.path.join(_ROOT,"figures","fig1_architecture.png"), dpi=300, bbox_inches="tight", pad_inches=0.06)
print("fig1 v2 rendered")
