#!/usr/bin/env python3
"""Fig 2 — the behavioral dissociation: one dot per run, three legs, one protocol.

The cross-model FINDING is behavioral, and behavior is directly comparable
across legs because the observable is the same discrete action under the same
frozen scripts: did the agent spawn (k>0)?  did it batch its spawns into one
turn?  The beta* VALUES are not drawn side by side anywhere: their evidence
differs in kind per leg (a curve / paired repeats / one floor-cleared point)
and lives in Fig. 1c and the prose.

Encoding: an icon array — every run is one dot; filled = spawned (top block)
or packed (bottom block); open = did not.  Exact tallies are printed beside
each strip.  Dot arrays show the small denominators instead of hiding them:
"contrasts, not rates" as a picture.

Data: the nine decision tallies are RECOMPUTED here from the canonical
results/matrix*/*/cell.json artifacts (same field logic as the
per-model audits: cond/shape/size/undisc + the cell's own model field; the
GPT leg reads only matrix-gpt-hi, the canonical reasoning-high mirror).  The
execution-mode tallies are the audited macros' values, validated upstream by
opus_findings.py / gpt_findings.py / kimi_findings.py; each is asserted here
against its macro comment so drift breaks the figure build.

Side effects: writes fig2.pdf (to FIG_PDF_DIR, default figures/) and
fig2-preview.png.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)

INK, MUT = "#1C2126", "#8A93A0"
BLUE = "#0072B2"

# ---- recompute the decision tallies from the cell artifacts ------------------
def leg_runs(model, dirs):
    for pattern in dirs:
        for path in sorted(glob.glob(os.path.join(BENCH, "results", pattern, "*", "cell.json"))):
            cell = json.load(open(path))
            if cell.get("model") != model:
                continue
            meta = cell["cell"]
            for run in cell["runs"]:
                yield meta, run


def tally(model, dirs, cond, min_size=0, exclude_undisc=True):
    spawned = total = 0
    for meta, run in leg_runs(model, dirs):
        if meta["shape"] != "wide" or meta["cond"] != cond:
            continue
        if meta["size"] < min_size:
            continue
        if exclude_undisc and "undisc" in meta["id"]:
            continue
        total += 1
        spawned += run["k"] > 0
    return spawned, total


OPUS = ("claude-opus-5", ["matrix", "matrix-reserve*"])
GPT = ("gpt-5.6-sol", ["matrix-gpt-hi"])      # canonical reasoning-high mirror only
K3 = ("kimi-k3", ["matrix-kimi"])

# rows: (label, per-leg (spawned, total)); Opus blind pools sizes 15-25 (its
# size-3 evidence is the pass-1 control corpus, not a matrix cell).
decision_rows = [
    ("blind", [tally(*OPUS, "blind", min_size=15), tally(*GPT, "blind"), tally(*K3, "blind")]),
    ("stated $\\beta{=}1$", [tally(*OPUS, "stated-b1"), tally(*GPT, "stated-b1"), tally(*K3, "stated-b1")]),
    ("stated $\\beta{=}0$", [tally(*OPUS, "stated-b0"), tally(*GPT, "stated-b0"), tally(*K3, "stated-b0")]),
]
# Bind to the audited macros (F14 / F14-addendum / F16-addendum-2 / F17c).
assert decision_rows[0][1] == [(6, 10), (8, 8), (0, 8)]      # \blindRate \gptHiBlindWide \kimiBlindWide
assert decision_rows[1][1] == [(3, 13), (9, 11), (0, 11)]    # \statedBoneAllWide \gptHiStatedBoneWide \kimiStatedBoneWide
assert decision_rows[2][1] == [(2, 8), (0, 5), (0, 5)]       # \statedBzeroRate \gptHiStatedBzero \kimiStatedBzero

# Execution mode (validated upstream; see F14-addendum, F16 tier 2, F17b):
# packed count / multi-spawn runs.  K3's forced row is the unpoolable ladder
# only (its split-form matrix reference serialized); K3 has no free-choice
# spawns, drawn as an empty strip.
mode_rows = [
    ("forced plans", [(6, 6), (0, 6), (4, 4)]),   # \opusForcedPacked \gptHiForcedAllSerial(serialized) \kimiForcedLadderPacked
    ("free choice", [(14, 14), (21, 21), (0, 0)]),  # \opusFreePacked \gptFreePackedHigh ; K3: \kimiFreeSpawns -> none exist
]

# ---- figure -----------------------------------------------------------------
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9,
    "text.color": INK,
})
fig, axes = plt.subplots(1, 3, figsize=(5.6, 2.15))
LEGS = ["Opus", "GPT (high)", "K3"]

ALL_ROWS = [("\\textbf{spawns?} (wide cells)", None)] if False else []
# y layout: block header, 3 decision rows, gap, block header, 2 mode rows
labels = [r[0] for r in decision_rows] + [r[0] for r in mode_rows]

for col, ax in enumerate(axes):
    ax.set_xlim(0, 24.5)
    ax.set_ylim(-0.6, 7.4)
    ax.axis("off")
    ax.set_title(LEGS[col], loc="left", fontsize=9)
    ax.text(0, 6.55, "spawns? ($k{>}0$)", fontsize=7.5, color=MUT, va="center")
    ax.text(0, 2.35, "packs spawns into one turn?", fontsize=7.5, color=MUT,
            va="center")

    def strip(y, filled, total, mark_color=INK):
        for i in range(total):
            face = mark_color if i < filled else "white"
            ax.plot(i * 0.88 + 0.45, y, marker="o", ms=4.4, mfc=face, mec=mark_color,
                    mew=0.9, lw=0, clip_on=False)
        txt = f"{filled}/{total}" if total else "none exist"
        ax.text(24.3, y, txt, fontsize=7.5, color=INK, ha="right", va="center")

    for j, (label, cells) in enumerate(decision_rows):
        y = 5.7 - j
        filled, total = cells[col]
        strip(y, filled, total, BLUE)
        if col == 0:
            ax.text(-0.4, y + 0.42, label, fontsize=7, color=MUT, va="center")
    for j, (label, cells) in enumerate(mode_rows):
        y = 1.5 - j
        filled, total = cells[col]
        strip(y, filled, total, INK)
        if col == 0:
            ax.text(-0.4, y + 0.42, label, fontsize=7, color=MUT, va="center")

# one legend line under the arrays
fig.text(0.01, 0.015,
         "one dot per run — filled: spawned (blue) / all spawns in one batch (black); open: did not",
         fontsize=7, color=MUT)

fig.tight_layout(pad=0.4, w_pad=0.8, rect=(0, 0.05, 1, 1))
pdf = os.path.join(os.environ.get("FIG_PDF_DIR", HERE), "fig2.pdf")
fig.savefig(pdf)
fig.savefig(os.path.join(HERE, "fig2-preview.png"), dpi=220)
print("wrote", pdf)
