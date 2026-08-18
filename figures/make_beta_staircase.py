"""Emit figures/beta-staircase.svg -- optimal spawn count as a function of beta.

    python -m figures.make_beta_staircase

NOT COMMITTED YET, AND NOT IN THE README. Every breakpoint on the x-axis is a
function of the placeholder cost constants, so the figure would have to be
restated the moment calibration lands -- and a committed chart is exactly the
kind of artifact that goes stale quietly. The script lives here so the figure can
be produced on demand now and published once the constants are real. What will
survive calibration is the SHAPE of each curve (chain flat at zero; wide climbing
then saturating at the cap), not the prices at which the steps happen.

`make_shapes.py` is the figure the README does use: dependency structure only,
nothing priced, nothing to go stale.

The figure is generated from `optimal_k_intervals`, never drawn by hand, so it
cannot drift away from what the oracle actually computes.

SVG by hand rather than matplotlib: the repo has no dependencies and this is one
step function per shape, which is a few dozen line segments.
"""

from __future__ import annotations

import pathlib

from generator.dag import sample_dag
from generator.oracle import CostModel, enumerate_plans, evaluate, optimal_k_intervals

N = 6
SIZES = (3,)
SHAPES = ("wide", "chain", "diamond", "mixed")

# Every breakpoint under the default CostModel falls below $0.25/min, so a linear
# axis to $0.30 shows all of them. The attended persona ($1.00/min) sits far to
# the right of the last one; that is called out on the plot rather than drawn,
# because compressing the axis to reach it would hide the breakpoints entirely.
XMAX = 0.30
KMAX = 5

W, H = 780, 440
L, R, T, B = 92, 200, 92, 70  # margins; the right margin holds the legend

INK = "#1f2328"
MUTED = "#6e7781"
GRID = "#d8dee4"
CARD = "#fbfbfa"
COLORS = {"wide": "#1f6feb", "chain": "#bf3989", "diamond": "#bc4c00", "mixed": "#0f7b6c"}
# Coincident steps would hide each other -- every shape sits at k=0 near beta=0.
# A sub-pixel-scale vertical offset keeps all four readable at the same true k.
OFFSET = {"wide": -3.0, "chain": -1.0, "diamond": 1.0, "mixed": 3.0}


def x(beta: float) -> float:
    return L + (W - L - R) * min(beta, XMAX) / XMAX


def y(k: float) -> float:
    return H - B - (H - B - T) * k / KMAX


def intervals() -> dict[str, list[tuple[float, float, int]]]:
    cm = CostModel()
    out = {}
    for shape in SHAPES:
        dag = sample_dag(shape, N, sizes=SIZES, seed=0)
        results = [evaluate(dag, p, cm) for p in enumerate_plans(dag)]
        out[shape] = optimal_k_intervals(results)
    return out


def build(spans: dict[str, list[tuple[float, float, int]]]) -> str:
    p: list[str] = []
    add = p.append

    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">')
    add(f'<rect width="{W}" height="{H}" rx="6" fill="{CARD}"/>')

    add(f'<text x="{L - 44}" y="30" font-size="15" font-weight="600" fill="{INK}">'
        "How many subagents should the lead spawn?</text>")
    add(f'<text x="{L - 44}" y="52" font-size="12" fill="{MUTED}">'
        "Derived from the oracle, not measured. Six equal subtasks; only the dependency structure differs.</text>")
    add(f'<text x="{L - 44}" y="70" font-size="12" fill="{MUTED}">'
        "Cost constants are placeholders. At the background persona (beta = $0, nobody waiting) every shape "
        "says spawn 0.</text>")

    # gridlines and the k axis
    for k in range(KMAX + 1):
        add(f'<line x1="{L}" y1="{y(k):.1f}" x2="{W - R}" y2="{y(k):.1f}" stroke="{GRID}" stroke-width="1"/>')
        add(f'<text x="{L - 12}" y="{y(k) + 4:.1f}" font-size="12" fill="{MUTED}" text-anchor="end">{k}</text>')
    mid_y = (y(0) + y(KMAX)) / 2
    add(f'<text x="{L - 44}" y="{mid_y:.1f}" font-size="11.5" fill="{INK}" text-anchor="middle" '
        f'transform="rotate(-90 {L - 44} {mid_y:.1f})">subagents spawned</text>')

    # the concurrency cap: no reason to brief a worker that cannot run
    cap = CostModel().concurrency_cap
    add(f'<line x1="{L}" y1="{y(cap):.1f}" x2="{W - R}" y2="{y(cap):.1f}" stroke="{MUTED}" '
        f'stroke-width="1" stroke-dasharray="2 3"/>')
    add(f'<text x="{W - R - 6}" y="{y(cap) - 7:.1f}" font-size="11" fill="{MUTED}" '
        f'text-anchor="end">concurrency cap = {cap}</text>')

    # beta axis
    add(f'<line x1="{L}" y1="{y(0):.1f}" x2="{W - R}" y2="{y(0):.1f}" stroke="{INK}" stroke-width="1.2"/>')
    # The axis line runs to $0.30 but the last label is $0.25, so the tick text
    # cannot collide with the legend column.
    for tick in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
        add(f'<line x1="{x(tick):.1f}" y1="{y(0):.1f}" x2="{x(tick):.1f}" y2="{y(0) + 5:.1f}" '
            f'stroke="{INK}" stroke-width="1"/>')
        add(f'<text x="{x(tick):.1f}" y="{y(0) + 20:.1f}" font-size="11" fill="{MUTED}" '
            f'text-anchor="middle">${tick:.2f}</text>')
    add(f'<text x="{(L + W - R) / 2:.1f}" y="{H - 18}" font-size="12" fill="{INK}" text-anchor="middle">'
        "beta -- dollars per minute of latency saved</text>")

    # one step function per shape
    for shape in SHAPES:
        color, dy = COLORS[shape], OFFSET[shape]
        d: list[str] = []
        prev_k: int | None = None
        for lo, hi, k in spans[shape]:
            if lo >= XMAX:
                break
            x0, x1 = x(lo), x(min(hi, XMAX))
            if prev_k is None:
                d.append(f"M {x0:.1f} {y(k) + dy:.1f}")
            else:
                d.append(f"L {x0:.1f} {y(k) + dy:.1f}")  # the vertical riser
            d.append(f"L {x1:.1f} {y(k) + dy:.1f}")
            prev_k = k
        add(f'<path d="{" ".join(d)}" fill="none" stroke="{color}" stroke-width="2.4" '
            f'stroke-linejoin="round" stroke-linecap="round"/>')
        # a dot at each breakpoint: these are exact, not sampled
        for lo, _, k in spans[shape]:
            if 0 < lo < XMAX:
                add(f'<circle cx="{x(lo):.1f}" cy="{y(k) + dy:.1f}" r="3" fill="{color}"/>')

    # legend, with each shape's terminal answer spelled out
    ly = T + 6
    add(f'<text x="{W - R + 14}" y="{ly}" font-size="12" font-weight="600" fill="{INK}">shape</text>')
    for shape in SHAPES:
        ly += 26
        last_lo, _, last_k = spans[shape][-1]
        add(f'<line x1="{W - R + 14}" y1="{ly - 4}" x2="{W - R + 40}" y2="{ly - 4}" '
            f'stroke="{COLORS[shape]}" stroke-width="2.4" stroke-linecap="round"/>')
        add(f'<text x="{W - R + 48}" y="{ly}" font-size="12" fill="{INK}">{shape}</text>')
        tail = f"{last_k} above ${last_lo:.2f}" if last_lo > 0 else f"{last_k} at every beta"
        add(f'<text x="{W - R + 48}" y="{ly + 14}" font-size="10.5" fill="{MUTED}">{tail}</text>')
        ly += 14

    # Placed under the legend, not near the axis, so nothing overlaps the ticks.
    ly += 22
    for line in (
        "The attended persona sits at",
        "$1.00/min -- far off-scale right.",
        "Every shape has saturated by",
        "$0.25, so no curve moves again.",
    ):
        ly += 14
        add(f'<text x="{W - R + 14}" y="{ly}" font-size="10.5" fill="{MUTED}">{line}</text>')

    add("</svg>")
    return "\n".join(p)


def main() -> None:
    spans = intervals()
    out = pathlib.Path(__file__).parent / "beta-staircase.svg"
    out.write_text(build(spans) + "\n")
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    for shape, s in spans.items():
        print(f"  {shape:<8} {len(s)} interval(s), k -> {[k for _, _, k in s]}")


if __name__ == "__main__":
    main()
