"""Dependency DAGs: the hidden ground-truth structure of a scenario.

Design doc: Phase1-DelegationBench-Design.md section 4, Stage 1.

A scenario's DAG is what the generator knows and the agent must discover. The
*shape* is the experimental variable that makes the correct delegation decision
vary: wide-independent work rewards fan-out, a strict chain punishes it, and a
diamond sits in between. Everything downstream (oracle, cost model, scoring)
reads structure only through this module.

Nodes carry a nominal `size` rather than tokens or minutes. Calibration
(Stage 3) maps size -> (tokens, minutes) per model, so the same DAG can be
re-priced per price vector without regenerating anything. Keeping the structure
free of prices is what makes section 5.1's cross-model invariance check cheap.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import cached_property

__all__ = ["Node", "DAG", "wide_independent", "chain", "diamond", "mixed", "sample_dag", "SHAPES"]


@dataclass(frozen=True)
class Node:
    """One executable subtask.

    `size` is nominal work units -- calibration turns it into tokens and minutes
    -- and it is the ONLY property that reaches the oracle. That is deliberate
    and load-bearing: `block_dollars` prices a block on TOTAL SIZE UNITS and not
    on which nodes compose it, which is what lets a handful of calibration runs
    price thousands of plans instead of running them.

    A `family` field naming which template instantiated the node used to sit
    here, and it went unread the moment the transformation payload that used it
    was replaced by bug injection. Reintroducing one is not free labelling: it
    would mean nodes of equal size cost different amounts, which is precisely
    the units-not-identity assumption the cost model rests on. If heterogeneous
    subtasks are wanted, the honest route is `calibrate.curve_disagreement` over
    varied block COMPOSITIONS, which measures whether that assumption survives
    rather than quietly voiding it.
    """

    id: str
    size: int = 1


@dataclass(frozen=True)
class DAG:
    """An acyclic dependency graph over subtasks.

    An edge (u, v) means v consumes an artifact u produces, so v cannot start
    until u has finished. These dependencies are real, not decorative: running
    v before u produces a broken artifact that the verifier catches.
    """

    nodes: tuple[Node, ...]
    edges: frozenset[tuple[str, str]]
    shape: str = "custom"

    def __post_init__(self) -> None:
        ids = [n.id for n in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate node ids")
        known = set(ids)
        for u, v in self.edges:
            if u not in known or v not in known:
                raise ValueError(f"edge ({u}, {v}) references an unknown node")
            if u == v:
                raise ValueError(f"self-edge on {u}")
        self.topo_order  # raises if cyclic

    # -- basic accessors -------------------------------------------------

    @cached_property
    def ids(self) -> tuple[str, ...]:
        return tuple(n.id for n in self.nodes)

    @cached_property
    def by_id(self) -> dict[str, Node]:
        return {n.id: n for n in self.nodes}

    @cached_property
    def preds(self) -> dict[str, frozenset[str]]:
        out: dict[str, set[str]] = {i: set() for i in self.ids}
        for u, v in self.edges:
            out[v].add(u)
        return {k: frozenset(v) for k, v in out.items()}

    @cached_property
    def succs(self) -> dict[str, frozenset[str]]:
        out: dict[str, set[str]] = {i: set() for i in self.ids}
        for u, v in self.edges:
            out[u].add(v)
        return {k: frozenset(v) for k, v in out.items()}

    @cached_property
    def topo_order(self) -> tuple[str, ...]:
        """Kahn's algorithm. Raises on a cycle -- a cyclic DAG is a generator bug."""
        indeg = {i: len(self.preds[i]) for i in self.ids}
        ready = [i for i in self.ids if indeg[i] == 0]  # keep insertion order: deterministic
        order: list[str] = []
        while ready:
            u = ready.pop(0)
            order.append(u)
            for v in sorted(self.succs[u]):
                indeg[v] -= 1
                if indeg[v] == 0:
                    ready.append(v)
        if len(order) != len(self.ids):
            raise ValueError("dependency graph is cyclic")
        return tuple(order)

    @cached_property
    def reachable(self) -> dict[str, frozenset[str]]:
        """Transitive closure: reachable[u] = every node u can reach (excluding u).

        Used by the oracle to test whether a candidate block partition induces a
        cyclic quotient graph, and by the generator to check that "independent"
        nodes really are independent before assigning disjoint file footprints.
        """
        out: dict[str, set[str]] = {i: set() for i in self.ids}
        for u in reversed(self.topo_order):
            for v in self.succs[u]:
                out[u].add(v)
                out[u] |= out[v]
        return {k: frozenset(v) for k, v in out.items()}

    def is_independent(self, a: str, b: str) -> bool:
        return b not in self.reachable[a] and a not in self.reachable[b]

    @property
    def n(self) -> int:
        return len(self.nodes)


# -- shape families -------------------------------------------------------
#
# Four families, chosen because each makes a different plan optimal. Section 6
# prospectively fixes the class mix and reports per-class as primary, so these labels
# are load-bearing: they are the aggregation unit, not documentation.


def _nodes(n: int, sizes: list[int], rng: random.Random) -> tuple[Node, ...]:
    return tuple(Node(id=f"n{i}", size=rng.choice(sizes)) for i in range(n))


def wide_independent(n: int, sizes=(1,), rng=None) -> DAG:
    """n independent nodes. Fan-out's best case; always-serial's worst."""
    rng = rng or random.Random(0)
    return DAG(_nodes(n, list(sizes), rng), frozenset(), "wide")


def chain(n: int, sizes=(1,), rng=None) -> DAG:
    """A strict chain n0 -> n1 -> ... Fan-out buys nothing and pays spawn cost.

    This is also the shape the context-drag crossover class uses (section 5.1):
    at large n the serial context grows until delegating the tail wins anyway.
    """
    rng = rng or random.Random(0)
    nodes = _nodes(n, list(sizes), rng)
    return DAG(nodes, frozenset((f"n{i}", f"n{i+1}") for i in range(n - 1)), "chain")


def diamond(width: int, sizes=(1,), rng=None) -> DAG:
    """source -> `width` parallel middles -> sink.

    The interesting case: the parallel middle rewards fan-out, but the lead must
    finish the source before it can brief anyone and must absorb every result
    before it can start the sink -- so the serial lead bookends bound the gain.
    """
    rng = rng or random.Random(0)
    nodes = _nodes(width + 2, list(sizes), rng)
    src, sink = "n0", f"n{width + 1}"
    mids = [f"n{i}" for i in range(1, width + 1)]
    edges = {(src, m) for m in mids} | {(m, sink) for m in mids}
    return DAG(nodes, frozenset(edges), "diamond")


def mixed(n: int, layers: int = 3, density: float = 0.4, sizes=(1,), rng=None) -> DAG:
    """Layered random DAG: edges only from an earlier layer to a later one.

    Acyclic by construction, and the layer count controls how much genuine
    parallelism exists. This is the class that keeps the benchmark from being
    three hand-picked shapes.
    """
    rng = rng or random.Random(0)
    nodes = _nodes(n, list(sizes), rng)
    assign: dict[str, int] = {f"n{i}": rng.randrange(layers) for i in range(n)}
    # guarantee a non-empty first layer so the DAG has at least one root
    assign["n0"] = 0
    edges = set()
    for u in [f"n{i}" for i in range(n)]:
        for v in [f"n{i}" for i in range(n)]:
            if assign[u] < assign[v] and rng.random() < density:
                edges.add((u, v))
    return DAG(nodes, frozenset(edges), "mixed")


SHAPES = {"wide": wide_independent, "chain": chain, "diamond": diamond, "mixed": mixed}


def sample_dag(
    shape: str,
    n: int,
    sizes: list[int] | tuple[int, ...] = (1,),
    seed: int = 0,
) -> DAG:
    """Sample one DAG. Seeded so every scenario is reproducible from its id.

    `diamond` interprets n as total node count (width = n - 2), so callers can
    ask for "an 8-node scenario" uniformly across shapes -- node count is what
    drives cost, and section 10's budget is priced in nodes.
    """
    rng = random.Random(seed)
    kw = dict(sizes=tuple(sizes), rng=rng)
    if shape == "diamond":
        if n < 3:
            raise ValueError("a diamond needs at least 3 nodes")
        return diamond(n - 2, **kw)
    if shape not in SHAPES:
        raise ValueError(f"unknown shape {shape!r}; expected one of {sorted(SHAPES)}")
    return SHAPES[shape](n, **kw)
