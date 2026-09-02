#!/usr/bin/env python3
"""Fig 1 — the Claude Opus delegation boundary (single-leg, one encoding).

Reads ONLY canonical artifacts (nothing hand-typed):
  - figures/beta-star-ladder.json               (measured arms + beta*)
  - results/probe-wide4-s{3,8,15,25}/probe.json (predicted values -> faded dashed)
  - results/audit-w15/audit.json                (12 shape representatives, panel d)
  - results/matrix/w15-stated-b1/*              (observed programmatic serial, F15;
                                                 computed via the harness Trace API,
                                                 exactly as opus_findings.py does)

Design rule: one leg per figure. The cross-model comparison lives in the
cross-model census table, where three unlike evidence classes (a curve,
paired repeats, a single floor-cleared point) can each be scoped honestly;
drawing them on one axis implies a comparability the evidence hierarchy
forbids. One color system throughout: BLUE = serial, VERMILLION = max-fanout,
INK = quantities derived from the pair (beta*, the beta=1 objective),
gray = context. Every panel labels its own semantics; headline values are
printed beside their markers, formatted from the loaded data.

Side effects: writes fig1.pdf (to FIG_PDF_DIR, default figures/) and
fig1-preview.png.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)

from harness.calibrate import PriceSheet, TimingModel  # noqa: E402
from harness.calibration import CalibrationResult  # noqa: E402
from harness.trace import Trace  # noqa: E402

BLUE, VERM = "#0072B2", "#D55E00"     # serial, max-fanout (Okabe-Ito, validated)
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

# Bind the drawn boundary to the report macros (F12); a drifted artifact must
# break the figure build rather than silently draw stale values.
assert rungs[3]["beta_star_per_min"] is None                    # \betastarThree
assert round(rungs[8]["beta_star_per_min"], 2) == 1.85          # \betastarEight
assert round(rungs[15]["beta_star_per_min"], 2) == 0.55         # \betastarFifteen
assert round(rungs[25]["beta_star_per_min"], 2) == 0.47         # \betastarTwentyfive

# Shape-complete wide-15 audit (panel d) + observed programmatic serial (F15).
audit = json.load(open(os.path.join(BENCH, "results", "audit-w15", "audit.json")))
audit_rows = [r for r in audit["rows"] if not r["excluded"]]
assert len(audit_rows) == 12  # \auditPlanCount: one per allocation shape
_cal = CalibrationResult.load(os.path.join(BENCH, "results", "cal-opus-v2", "calibration.json"))
_price = PriceSheet(**_cal.price_sheet)
_timing = TimingModel(**_cal.timing_model)
_cell_dir = os.path.join(BENCH, "results", "matrix", "w15-stated-b1")
_cell = json.load(open(os.path.join(_cell_dir, "cell.json")))
assert _cell["model"] == "claude-opus-5"
_serial_objs = []
for _i, _run in enumerate(_cell["runs"]):
    _t = Trace.load(os.path.join(_cell_dir, f"run-{_i}-trace.json"))
    if _t.k == 0 and _t.succeeded:
        _serial_objs.append(_t.objective(_price, _timing, 1.0))
prog_serial = min(_serial_objs)
assert round(prog_serial, 2) == 1.32  # \bestProgrammaticFifteen

# ---- figure -----------------------------------------------------------------
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUT, "axes.linewidth": 0.7,
    "xtick.color": MUT, "ytick.color": MUT,
    "text.color": INK, "axes.labelcolor": INK,
})
fig, ((a1, a2), (b, d)) = plt.subplots(2, 2, figsize=(5.6, 4.35))
fig.suptitle("Claude Opus: executed serial vs. max-fanout arms (single draws)",
             fontsize=9.5, y=0.995)

def arm_series(field):
    ser = [rungs[s]["serial"][field] for s in SIZES]
    fan = [rungs[s]["fanout"][field] for s in SIZES]
    return ser, fan

for ax, field, floor, ylabel in (
        (a1, "dollars", floors["dollars"], "executed dollars"),
        (a2, "minutes", floors["minutes"], "analytic minutes")):
    ser, fan = arm_series(field)
    pser = [pred[s]["serial"][field] for s in SIZES]
    pfan = [pred[s]["fanout"][field] for s in SIZES]
    ax.plot(SIZES, pser, "--", color=BLUE, alpha=0.30, lw=1.1, zorder=1)
    ax.plot(SIZES, pfan, "--", color=VERM, alpha=0.30, lw=1.1, zorder=1)
    ax.errorbar(SIZES, ser, yerr=floor, color=BLUE, marker="o", ms=3.8,
                lw=1.5, capsize=1.6, elinewidth=0.7, zorder=3, label="serial")
    ax.errorbar(SIZES, fan, yerr=floor, color=VERM, marker="s", ms=3.6,
                lw=1.5, capsize=1.6, elinewidth=0.7, zorder=3, label="max-fanout")
    # size-40: measured but indicative (serial attribution-invalid) -> hollow
    ax.plot([S40], [rungs[S40]["serial"][field]], marker="o", ms=3.8,
            mfc="white", mec=BLUE, lw=0, zorder=3)
    ax.plot([S40], [rungs[S40]["fanout"][field]], marker="s", ms=3.6,
            mfc="white", mec=VERM, lw=0, zorder=3)
    ax.set_xticks(SIZES + [S40])
    ax.set_xlabel("node size")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=MUT, alpha=0.22, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)

a1.legend(frameon=False, fontsize=7.5, handlelength=1.6,
          loc="upper left", borderaxespad=0.2)
a1.text(0.97, 0.04, "faded dashed: refuted extrapolation",
        transform=a1.transAxes, ha="right", va="bottom", fontsize=7, color=MUT)
a2.text(0.97, 0.10, "hollow: size 40, attribution-invalid",
        transform=a2.transAxes, ha="right", va="bottom", fontsize=7, color=MUT)
a1.set_title("(a) what each arm cost", loc="left")
a2.set_title("(b) how long each arm took", loc="left")

# ---- (c) the derived boundary ------------------------------------------------
bs_sizes = [s for s in SIZES if rungs[s]["beta_star_per_min"] is not None]
bs_vals = [rungs[s]["beta_star_per_min"] for s in bs_sizes]
b.axhspan(0, 1, color=BAND, zorder=0)
b.text(0.97, 0.90, "scored range $\\beta \\in [0,1]$", transform=b.transAxes,
       ha="right", va="top", fontsize=7, color=MUT)
b.plot(bs_sizes, bs_vals, "-D", color=INK, ms=4.0, lw=1.5, zorder=3)
for s, v in zip(bs_sizes, bs_vals):
    b.annotate(f"\\${v:.2f}", (s, v), textcoords="offset points",
               xytext=(7, 2), fontsize=7.5, color=INK)
# size 3: fan-out dominated, no beta* exists
b.plot([3], [0.06], marker="x", ms=5, color=INK, lw=0, zorder=3)
b.text(2.3, 0.78, "size 3:\ndominated\n(no $\\beta^*$)", fontsize=7,
       color=INK, ha="left", va="center")
# size 40: indicative only
b.plot([S40], [rungs[S40]["beta_star_per_min"]], marker="D", ms=4.0,
       mfc="white", mec=INK, lw=0, zorder=3)
b.annotate("40: excluded", (S40, rungs[S40]["beta_star_per_min"]),
           textcoords="offset points", xytext=(-4, 9), ha="right",
           fontsize=7, color=MUT)
b.set_xticks(SIZES + [S40])
b.set_xlabel("node size")
b.set_ylabel("$\\beta^*$  (\\$ per saved minute)")
b.set_ylim(-0.06, 2.1)
b.grid(axis="y", color=MUT, alpha=0.22, lw=0.5)
b.spines[["top", "right"]].set_visible(False)
b.set_title("(c) break-even price $\\beta^*$", loc="left")

# ---- (d) the shape-complete wide-15 table ------------------------------------
ks = [r["k"] for r in audit_rows]
objs = [r["measured_objective"] for r in audit_rows]
champ = min(audit_rows, key=lambda r: r["measured_objective"])
serial_row = next(r for r in audit_rows if r["k"] == 0)
d.scatter(ks, objs, s=24, facecolors="white", edgecolors=MUT,
          linewidths=1.0, zorder=3, clip_on=False)
d.scatter([champ["k"]], [champ["measured_objective"]], s=32, color=VERM,
          zorder=4, clip_on=False)
d.annotate(f"max-fanout wins ({champ['measured_objective']:.2f})",
           (champ["k"], champ["measured_objective"]),
           textcoords="offset points", xytext=(-7, -13), ha="right",
           fontsize=7.5, color=VERM)
d.scatter([0], [serial_row["measured_objective"]], s=32, color=BLUE, zorder=4)
d.annotate(f"serial ({serial_row['measured_objective']:.2f}; nearest tie)",
           (0, serial_row["measured_objective"]),
           textcoords="offset points", xytext=(7, 4), fontsize=7.5, color=BLUE)
d.axhline(prog_serial, ls="--", lw=1.2, color=INK, alpha=0.75, zorder=2)
d.annotate(f"observed programmatic serial ({prog_serial:.2f})",
           (3.95, prog_serial), textcoords="offset points", xytext=(0, 4),
           ha="right", fontsize=7.5, color=INK)
d.annotate("all partial point estimates are higher", (2.0, 3.62), fontsize=7, color=MUT,
           ha="center")
d.text(0.03, 0.97, "lower is better", transform=d.transAxes, fontsize=7,
       color=MUT, ha="left", va="top")
d.set_xticks([0, 1, 2, 3, 4])
d.set_xlabel("subagents $k$")
d.set_ylabel("$\\beta{=}1$ objective")
d.set_ylim(1.1, 4.25)
d.grid(axis="y", color=MUT, alpha=0.22, lw=0.5)
d.spines[["top", "right"]].set_visible(False)
d.set_title("(d) 12 allocation shapes", loc="left")

fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.3, rect=(0, 0, 1, 0.985))
pdf = os.path.join(os.environ.get("FIG_PDF_DIR", HERE), "fig1.pdf")
fig.savefig(pdf)
fig.savefig(os.path.join(HERE, "fig1-preview.png"), dpi=220)
print("wrote", pdf)
