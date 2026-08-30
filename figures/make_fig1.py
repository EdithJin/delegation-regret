#!/usr/bin/env python3
"""Fig 1 — the measured delegation boundary.

Reads ONLY canonical artifacts (nothing hand-typed):
  - figures/beta-star-ladder.json               (Claude leg, measured arms + beta*)
  - results/probe-wide4-s{3,8,15,25}/probe.json (Claude predicted values -> faded)
  - figures/beta-star-ladder-gpt.json           (GPT reasoning-high)

Side effects:
  - writes report/fig1.pdf (vector, for the report) and fig1-preview.png

Panels (one axis each, never dual): (a1) executed dollars vs node size,
(a2) executed minutes vs node size — serial vs max-fanout, refuted
extrapolations faded behind, calibration noise floors as whiskers, size-40
hollow (indicative: serial attribution-invalid); (b) beta*(size) with the
scored price band — Claude crosses between 8 and 15; GPT (forced plans) has
no beta* at any measured size.

Palette validated (dataviz six checks, light surface): serial #0072B2,
fan-out #D55E00; band/GPT in neutral grays; text in ink.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)

BLUE, VERM = "#0072B2", "#D55E00"     # serial, max-fanout (validated)
INK, MUT, BAND = "#1C2126", "#8A93A0", "#E9E7E1"

# ---- data -------------------------------------------------------------------
lad = json.load(open(os.path.join(HERE, "beta-star-ladder.json")))
rungs = {r["node_size"]: r for r in lad["rungs"]}
SIZES = [3, 8, 15, 25]
S40 = 40
floors = lad["floors"]  # dollars / minutes materiality floors

pred = {}
for s in SIZES:
    p = json.load(open(os.path.join(BENCH, "results", f"probe-wide4-s{s}", "probe.json")))
    pred[s] = p["predicted"]

gpt_spine = json.load(open(os.path.join(HERE, "beta-star-ladder-gpt.json")))
gpt_sizes = [row["node_size"] for row in gpt_spine["rungs"]]
assert gpt_spine["model"] == "gpt-5.6-sol"
assert gpt_spine["reasoning_effort"] == "high"
assert gpt_sizes == SIZES
assert all(row["beta_star_per_min"] is None for row in gpt_spine["rungs"])
gpt_packed = gpt_spine["packed_confirmatory"]["runs"]

# ---- figure -----------------------------------------------------------------
plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUT, "axes.linewidth": 0.7,
    "xtick.color": MUT, "ytick.color": MUT,
    "text.color": INK, "axes.labelcolor": INK,
})
fig, (a1, a2, b) = plt.subplots(1, 3, figsize=(5.5, 2.05))

def arm_series(field):
    ser = [rungs[s]["serial"][field] for s in SIZES]
    fan = [rungs[s]["fanout"][field] for s in SIZES]
    return ser, fan

for ax, field, floor, ylabel in (
        (a1, "dollars", floors["dollars"], "executed dollars"),
        (a2, "minutes", floors["minutes"], "executed minutes")):
    ser, fan = arm_series(field)
    pser = [pred[s]["serial"][field] for s in SIZES]
    pfan = [pred[s]["fanout"][field] for s in SIZES]
    ax.plot(SIZES, pser, "--", color=BLUE, alpha=0.30, lw=1.1, zorder=1)
    ax.plot(SIZES, pfan, "--", color=VERM, alpha=0.30, lw=1.1, zorder=1)
    ax.errorbar(SIZES, ser, yerr=floor, color=BLUE, marker="o", ms=3.6,
                lw=1.5, capsize=1.6, elinewidth=0.7, zorder=3, label="serial")
    ax.errorbar(SIZES, fan, yerr=floor, color=VERM, marker="s", ms=3.4,
                lw=1.5, capsize=1.6, elinewidth=0.7, zorder=3, label="max-fanout")
    # size-40: measured but indicative (serial attribution-invalid) -> hollow
    ax.plot([S40], [rungs[S40]["serial"][field]], marker="o", ms=3.6,
            mfc="white", mec=BLUE, lw=0, zorder=3)
    ax.plot([S40], [rungs[S40]["fanout"][field]], marker="s", ms=3.4,
            mfc="white", mec=VERM, lw=0, zorder=3)
    ax.set_xticks(SIZES + [S40])
    ax.set_xlabel("node size")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=MUT, alpha=0.22, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)

a1.legend(frameon=False, fontsize=6.5, handlelength=1.6,
          loc="upper left", borderaxespad=0.2)
a1.text(0.97, 0.05, "faded dashed:\nrefuted extrapolation", transform=a1.transAxes,
        ha="right", va="bottom", fontsize=5.8, color=MUT)
a1.set_title("(a) executed dollars", loc="left")
a2.set_title("(b) executed minutes", loc="left")

# ---- beta* panel -------------------------------------------------------------
bs_sizes = [s for s in SIZES if rungs[s]["beta_star_per_min"] is not None]
bs_vals = [rungs[s]["beta_star_per_min"] for s in bs_sizes]
b.axhspan(0, 1, color=BAND, zorder=0)
b.plot(bs_sizes, bs_vals, "-D", color=INK, ms=3.6, lw=1.5, zorder=3)
b.annotate("Claude", (bs_sizes[0], bs_vals[0]), textcoords="offset points",
           xytext=(6, 1), fontsize=6.8, color=INK)
# dominated points: GPT row at every measured size; Claude only at size 3
b.plot(gpt_sizes, [0.04] * len(gpt_sizes), lw=0, marker="x", ms=4.2, color=MUT,
       zorder=2)
# Floor-clearing GPT packed-mode repeats. Size 3 remains dominated and the two
# nominal size-8 crossings are omitted because their savings fall inside the
# calibration minute floor.
packed_points = [
    (int(size), row["beta_star"])
    for size, rows in gpt_packed.items()
    for row in rows
    if row["floor_cleared"] and row["beta_star"] is not None
]
packed_x = [size + (-0.45 if i % 2 == 0 else 0.45) for i, (size, _) in enumerate(packed_points)]
packed_y = [value for _, value in packed_points]
b.scatter(packed_x, packed_y, marker="o", s=18, facecolors="white",
          edgecolors=MUT, linewidths=0.9, zorder=4)
b.annotate("GPT packed", (packed_x[-1], packed_y[-1]), textcoords="offset points",
           xytext=(6, -2), fontsize=5.8, color=MUT)
b.plot([3], [0.16], marker="x", ms=4.2, color=INK, lw=0, zorder=3)
b.annotate("Claude, size 3", (3, 0.16), textcoords="offset points",
           xytext=(-2, 7), fontsize=5.8, color=INK)
b.text(40, 0.12, "$\\times$ = no $\\beta^*$ (dominated)", fontsize=5.8,
       color=MUT, ha="right", va="bottom")
# size-40 indicative
b.plot([S40], [rungs[S40]["beta_star_per_min"]], marker="D", ms=3.6,
       mfc="white", mec=INK, lw=0, zorder=3)
b.set_xticks(SIZES + [S40])
b.set_xlabel("node size")
b.set_ylabel("$\\beta^*$  (\\$ per saved minute)")
b.set_ylim(-0.08, 2.1)
b.grid(axis="y", color=MUT, alpha=0.22, lw=0.5)
b.spines[["top", "right"]].set_visible(False)
b.set_title("(c) the boundary", loc="left")

fig.tight_layout(pad=0.4, w_pad=1.0)
pdf = os.path.join(BENCH, "report", "fig1.pdf")
fig.savefig(pdf)
fig.savefig(os.path.join(HERE, "fig1-preview.png"), dpi=220)
print("wrote", pdf)
