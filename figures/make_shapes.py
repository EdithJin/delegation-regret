"""Emit figures/shapes.svg -- the four dependency shapes a scenario can have.

    python -m figures.make_shapes

Structure only. Nothing here is priced, so nothing here changes when the cost
constants are calibrated -- which is why this is the figure the README carries
and `make_beta_staircase.py` is not.

Layout is computed from the DAG objects themselves (longest-path layering), so
the picture cannot disagree with what the generator produces.
"""

from __future__ import annotations

import pathlib

from generator.dag import DAG, sample_dag

N = 6
SHAPES = ("wide", "chain", "diamond", "mixed")

# Structural claims only: each follows from the edges, not from any cost model.
BLURB = {
    "wide": "nothing waits",
    "chain": "each waits on the last",
    "diamond": "parallel middle",
    "mixed": "partial parallelism",
}
IMPLIES = {
    "wide": "fan-out's best case",
    "chain": "fan-out buys nothing",
    "diamond": "the interesting case",
    "mixed": "neither extreme",
}

PANEL_W, PANEL_H = 186, 214
PAD_L, PAD_T = 22, 96
W = PAD_L * 2 + PANEL_W * len(SHAPES)
H = PAD_T + PANEL_H + 58

DX, DY, R = 27, 29, 9.5

INK = "#1f2328"
MUTED = "#6e7781"
CARD = "#fbfbfa"
PANEL = "#ffffff"
EDGE = "#8c959f"
FILL = {"wide": "#1f6feb", "chain": "#bf3989", "diamond": "#bc4c00", "mixed": "#0f7b6c"}


def layers(dag: DAG) -> dict[str, int]:
    """Longest path from any source. Nodes in one layer are mutually reachable
    from nothing in that layer, so they can genuinely run at the same time."""
    depth: dict[str, int] = {}
    for v in dag.topo_order:  # topo order guarantees predecessors are already done
        preds = dag.preds[v]
        depth[v] = 0 if not preds else 1 + max(depth[u] for u in preds)
    return depth


def positions(dag: DAG, ox: float, oy: float) -> dict[str, tuple[float, float]]:
    """Layer left-to-right, spread within a layer top-to-bottom, centre the whole
    drawing in its panel so panels of different aspect ratios still line up."""
    depth = layers(dag)
    cols: dict[int, list[str]] = {}
    for v in dag.topo_order:
        cols.setdefault(depth[v], []).append(v)

    span_x = (max(cols) ) * DX
    span_y = (max(len(c) for c in cols.values()) - 1) * DY
    x0 = ox + (PANEL_W - span_x) / 2
    y0 = oy + (PANEL_H - span_y) / 2

    pos: dict[str, tuple[float, float]] = {}
    for d, members in cols.items():
        offset = (span_y - (len(members) - 1) * DY) / 2
        for i, v in enumerate(members):
            pos[v] = (x0 + d * DX, y0 + offset + i * DY)
    return pos


def arrow(x1: float, y1: float, x2: float, y2: float) -> list[str]:
    """Line clipped to both node boundaries, plus an explicit arrowhead polygon.

    Drawn as a polygon rather than a <marker> because GitHub sanitizes inline SVG
    and marker support is not worth betting the figure on.
    """
    dx, dy = x2 - x1, y2 - y1
    dist = (dx * dx + dy * dy) ** 0.5
    if dist < 1e-9:
        return []
    ux, uy = dx / dist, dy / dist
    sx, sy = x1 + ux * (R + 1.5), y1 + uy * (R + 1.5)
    ex, ey = x2 - ux * (R + 3.0), y2 - uy * (R + 3.0)
    head, half = 6.0, 3.0
    bx, by = ex - ux * head, ey - uy * head
    px, py = -uy * half, ux * half
    return [
        f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{EDGE}" stroke-width="1.4"/>',
        f'<polygon points="{ex:.1f},{ey:.1f} {bx + px:.1f},{by + py:.1f} {bx - px:.1f},{by - py:.1f}" fill="{EDGE}"/>',
    ]


def build() -> str:
    p: list[str] = []
    add = p.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">')
    add(f'<rect width="{W}" height="{H}" rx="6" fill="{CARD}"/>')

    add(f'<text x="{PAD_L}" y="34" font-size="15" font-weight="600" fill="{INK}">'
        "What a scenario is: the same work, four dependency structures</text>")
    add(f'<text x="{PAD_L}" y="56" font-size="12" fill="{MUTED}">'
        "Six subtasks of equal size in every case. Only the edges differ -- and the edges are what should change "
        "whether fanning out</text>")
    add(f'<text x="{PAD_L}" y="73" font-size="12" fill="{MUTED}">'
        "is the right call. An arrow means the target consumes an artifact the source produces, so it cannot start "
        "first.</text>")

    for i, shape in enumerate(SHAPES):
        dag = sample_dag(shape, N, sizes=(3,), seed=0)
        ox = PAD_L + i * PANEL_W
        add(f'<rect x="{ox + 6}" y="{PAD_T}" width="{PANEL_W - 12}" height="{PANEL_H}" rx="6" '
            f'fill="{PANEL}" stroke="#e6e8eb" stroke-width="1"/>')

        pos = positions(dag, ox, PAD_T)
        for u, v in sorted(dag.edges):
            add("".join(arrow(*pos[u], *pos[v])))
        for v in dag.topo_order:
            cx, cy = pos[v]
            add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{R}" fill="{FILL[shape]}"/>')
            add(f'<text x="{cx:.1f}" y="{cy + 3.4:.1f}" font-size="9.5" fill="#ffffff" '
                f'text-anchor="middle">{v[1:]}</text>')

        # Three short centred lines: a panel is only 186px wide, so a single
        # caption line overruns into its neighbour.
        cx = ox + PANEL_W / 2
        n_edges = len(dag.edges)
        add(f'<text x="{cx:.1f}" y="{PAD_T + PANEL_H + 22}" font-size="13" font-weight="600" '
            f'fill="{INK}" text-anchor="middle">{shape}</text>')
        add(f'<text x="{cx:.1f}" y="{PAD_T + PANEL_H + 38}" font-size="11" fill="{MUTED}" '
            f'text-anchor="middle">{n_edges} dependenc{"y" if n_edges == 1 else "ies"} '
            f'-- {BLURB[shape]}</text>')
        add(f'<text x="{cx:.1f}" y="{PAD_T + PANEL_H + 53}" font-size="11" fill="{FILL[shape]}" '
            f'text-anchor="middle">{IMPLIES[shape]}</text>')

    add("</svg>")
    return "\n".join(p)


def main() -> None:
    out = pathlib.Path(__file__).parent / "shapes.svg"
    out.write_text(build() + "\n")
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    for shape in SHAPES:
        dag = sample_dag(shape, N, sizes=(3,), seed=0)
        depth = layers(dag)
        print(f"  {shape:<8} {len(dag.edges)} edges, {max(depth.values()) + 1} layer(s)")


if __name__ == "__main__":
    main()
