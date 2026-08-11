"""Plan space, feasibility, and the event-driven cost/latency model.

Design doc: Phase1-DelegationBench-Design.md section 4, Stage 5.

A PLAN assigns every subtask to somebody: one (possibly empty) block executed
inline by the lead, and zero or more blocks each handed to one subagent. The
oracle enumerates every legal assignment and prices it.

Two corrections to the doc are implemented here, both surfaced on Aug 9 and both
pushing the doc's oracle toward more spawning than reality warrants:

1. FEASIBILITY. The doc contracts every block, including the inline one, and
   requires the resulting quotient graph to be acyclic. That wrongly rejects the
   natural diamond plan (lead does source and sink, subagents do the parallel
   middle), because contracting the inline block asserts the lead executes its
   nodes contiguously -- and the lead is precisely the participant for whom that
   is false. A subagent is atomic: one spawn in, one result out, nothing
   extractable from the middle. The lead is present throughout and hands work
   over as it goes. So: contract SUBAGENT blocks only; inline nodes stay as
   individual vertices; require that graph acyclic. Strictly more permissive,
   and it still rejects every genuine circular wait.

2. LATENCY. The doc computes "longest path in the quotient graph plus the lead's
   own serial timeline". Those two quantities overlap in time, so adding them
   double-counts -- the all-inline plan comes out at 2x its true latency. Worse,
   the corrected feasibility rule makes the quotient cyclic for legal plans, so
   "longest path" is undefined for them. Replaced by an event-driven simulation
   with the lead as a single serial resource.

Consequence of one-level delegation worth stating: subagents cannot talk to each
other, so every inter-block artifact routes through the lead. A subagent-to-
subagent dependency therefore costs an absorption AND a briefing on the lead's
serial timeline. That is not an extra assumption -- it is what "delegation depth
is fixed at one level" (section 7) means once you write down the schedule.

Cost constants here are PLACEHOLDERS pending the Aug 13 calibration run. Nothing
in this module hard-codes them; they all live in CostModel.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

from .dag import DAG

__all__ = ["CostModel", "Plan", "enumerate_plans", "evaluate", "PlanResult", "pareto_front", "is_tier_a", "optimal_k_intervals"]


# ---------------------------------------------------------------- cost model


@dataclass(frozen=True)
class CostModel:
    """Every price and duration the oracle uses. Calibrated, never assumed.

    Units are pinned here once: cost is BILLED DOLLARS, latency is MINUTES, so
    beta is dollars per minute of latency saved and alpha is identically 1.
    Section 6 reports implied beta in exactly these units.
    """

    # per unit of node size
    minutes_per_unit: float = 1.0
    dollars_per_unit: float = 0.10

    # the lead's own overhead
    explore_minutes: float = 1.0
    explore_dollars: float = 0.05

    # spawn overhead, decomposed per section 5.1: fixed + briefing + absorption
    spawn_fixed_dollars: float = 0.02
    brief_minutes: float = 0.5
    brief_dollars_per_node: float = 0.01
    absorb_minutes: float = 0.5
    absorb_dollars_per_node: float = 0.01

    # context reuse within one block: nodes after the first are discounted.
    # Placeholder for the fitted delta curve (Stage 5) -- which is parameterized
    # on block CONTEXT SIZE, not node count, so this scalar is a stand-in only.
    delta: float = 0.8

    concurrency_cap: int = 4
    beta: float = 0.0  # dollars per minute; 0 = the "background" persona


# --------------------------------------------------------------- plan space


@dataclass(frozen=True)
class Plan:
    """inline = nodes the lead runs itself. blocks = one frozenset per subagent."""

    inline: frozenset[str]
    blocks: tuple[frozenset[str], ...]

    @property
    def k(self) -> int:
        return len(self.blocks)


def _set_partitions(items: list[str]):
    """Every way to split `items` into non-empty groups. B(n) of them."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for sub in _set_partitions(rest):
        for i in range(len(sub)):
            yield sub[:i] + [[first] + sub[i]] + sub[i + 1 :]
        yield [[first]] + sub


def _is_feasible(dag: DAG, plan: Plan) -> bool:
    """Contract subagent blocks only; inline nodes stay separate; require acyclic.

    A cycle means some subagent must hand the lead an artifact partway through
    its own errand, which one spawn cannot do.
    """
    owner: dict[str, str] = {v: v for v in plan.inline}
    for i, b in enumerate(plan.blocks):
        for v in b:
            owner[v] = f"B{i}"

    verts = set(owner.values())
    adj: dict[str, set[str]] = {v: set() for v in verts}
    for u, v in dag.edges:
        if owner[u] != owner[v]:
            adj[owner[u]].add(owner[v])

    # Kahn
    indeg = {v: 0 for v in verts}
    for u in adj:
        for w in adj[u]:
            indeg[w] += 1
    ready = [v for v in verts if indeg[v] == 0]
    seen = 0
    while ready:
        u = ready.pop()
        seen += 1
        for w in adj[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                ready.append(w)
    return seen == len(verts)


def enumerate_plans(dag: DAG) -> list[Plan]:
    """Every feasible plan. Enumerated once per DAG and cached by the caller.

    Feasibility depends only on the DAG, never on prices, so this set is reused
    across the whole epistemic grid (section 4: enumerate-once-then-vectorize).
    """
    out: list[Plan] = []
    for part in _set_partitions(list(dag.ids)):
        groups = [frozenset(g) for g in part]
        # the inline block may be empty (a lead that purely orchestrates)
        for inline_idx in [None] + list(range(len(groups))):
            if inline_idx is None:
                plan = Plan(frozenset(), tuple(groups))
            else:
                plan = Plan(groups[inline_idx], tuple(g for i, g in enumerate(groups) if i != inline_idx))
            if plan.k > dag.n:  # cannot exceed one subagent per node
                continue
            if _is_feasible(dag, plan):
                out.append(plan)
    return out


# ------------------------------------------------------------ cost & latency


@dataclass(frozen=True)
class PlanResult:
    plan: Plan
    cost: float  # billed dollars
    latency: float  # minutes
    order: tuple[int, ...]  # latency-minimizing launch order over blocks

    def objective(self, beta: float) -> float:
        return self.cost + beta * self.latency


def _block_minutes(dag: DAG, block: frozenset[str], cm: CostModel) -> float:
    """One agent runs its nodes one at a time, regardless of dependencies."""
    return sum(dag.by_id[v].size for v in block) * cm.minutes_per_unit


def _block_dollars(dag: DAG, block: frozenset[str], cm: CostModel) -> float:
    """First node pays full freight; later nodes reuse the accumulated context."""
    order = [v for v in dag.topo_order if v in block]
    total = 0.0
    for i, v in enumerate(order):
        c = dag.by_id[v].size * cm.dollars_per_unit
        total += c if i == 0 else c * cm.delta
    return total


def _simulate(dag: DAG, plan: Plan, cm: CostModel, order: tuple[int, ...]) -> float:
    """Event-driven schedule with the lead as one serial resource.

    The lead does exactly one thing at a time: explore, emit a briefing, run an
    inline node, or absorb a result. A subagent starts only once its inputs exist
    AND the lead has finished briefing it AND a concurrency slot is free.
    Everything a subagent produces becomes usable only after the lead absorbs it.
    """
    owner_block = {v: i for i, b in enumerate(plan.blocks) for v in b}

    avail: dict[str, float] = {}  # when each node's artifact is usable by the lead
    rank = {b: i for i, b in enumerate(order)}  # launch preference, not a hard sequence
    unbriefed = set(range(plan.k))
    launched: dict[int, float] = {}
    sub_finish: dict[int, float] = {}
    absorbed: set[int] = set()
    inline_done: set[str] = set()
    lead = cm.explore_minutes

    def preds_ready(nodes) -> float | None:
        """Latest time all external predecessors are usable, or None if not yet."""
        t = 0.0
        for v in nodes:
            for u in dag.preds[v]:
                if u in nodes:
                    continue
                if u not in avail:
                    return None
                t = max(t, avail[u])
        return t

    def slot_free_at(now: float) -> float:
        in_flight = sorted(f for i, f in sub_finish.items() if launched[i] <= now < f)
        if len(in_flight) < cm.concurrency_cap:
            return now
        return in_flight[len(in_flight) - cm.concurrency_cap]

    guard = 0
    while unbriefed or (len(absorbed) < plan.k) or (len(inline_done) < len(plan.inline)):
        guard += 1
        if guard > 10_000:
            raise RuntimeError("scheduler failed to converge -- infeasible plan reached _simulate")

        # (start, kind_priority, tie_break, kind, key). Briefings first so workers
        # start as early as possible, then absorptions (cheap, and they unblock
        # dependents), then the lead's own inline work.
        cands: list[tuple[float, int, int, str, object]] = []

        for b in sorted(unbriefed, key=lambda x: rank[x]):
            r = preds_ready(plan.blocks[b])
            if r is not None:
                t = max(lead, r)
                cands.append((max(t, slot_free_at(t)), 0, rank[b], "brief", b))

        for b in range(plan.k):
            if b in sub_finish and b not in absorbed:
                cands.append((max(lead, sub_finish[b]), 1, rank[b], "absorb", b))

        for i, v in enumerate(dag.topo_order):
            if v in plan.inline and v not in inline_done:
                r = preds_ready({v})
                if r is not None:
                    cands.append((max(lead, r), 2, i, "inline", v))

        if not cands:
            raise RuntimeError("no action available -- infeasible plan reached _simulate")

        start = min(c[0] for c in cands)
        _, _, _, kind, key = min((c for c in cands if c[0] == start), key=lambda c: (c[1], c[2]))

        if kind == "brief":
            b = key
            lead = start + cm.brief_minutes
            launched[b] = lead
            sub_finish[b] = lead + _block_minutes(dag, plan.blocks[b], cm)
            unbriefed.discard(b)
        elif kind == "absorb":
            b = key
            lead = start + cm.absorb_minutes
            for v in plan.blocks[b]:
                avail[v] = lead
            absorbed.add(b)
        else:
            v = key
            lead = start + dag.by_id[v].size * cm.minutes_per_unit
            avail[v] = lead
            inline_done.add(v)

    return lead


def evaluate(dag: DAG, plan: Plan, cm: CostModel) -> PlanResult:
    """Price one plan. Cost is launch-order independent; latency is not, so the
    latency-minimizing order is chosen and reported alongside."""
    cost = cm.explore_dollars + _block_dollars(dag, plan.inline, cm)
    for b in plan.blocks:
        cost += (
            cm.spawn_fixed_dollars
            + cm.brief_dollars_per_node * len(b)
            + cm.absorb_dollars_per_node * len(b)
            + _block_dollars(dag, b, cm)
        )

    orders = permutations(range(plan.k)) if plan.k <= 6 else [tuple(range(plan.k))]
    best_t, best_o = float("inf"), tuple(range(plan.k))
    for o in orders:
        t = _simulate(dag, plan, cm, o)
        if t < best_t:
            best_t, best_o = t, o
    return PlanResult(plan, cost, best_t, best_o)


def pareto_front(results: list[PlanResult]) -> list[PlanResult]:
    """Plans not beaten on both cost and latency."""
    out = []
    for r in results:
        if not any(
            (o.cost <= r.cost and o.latency <= r.latency) and (o.cost < r.cost or o.latency < r.latency)
            for o in results
        ):
            out.append(r)
    return out


def is_tier_a(results: list[PlanResult], tol: float = 1e-9) -> bool:
    """Tier A: one (cost, latency) OUTCOME dominates everything, so the same plan
    is optimal at every persona (section 5.1).

    Tested on distinct (C, L) VALUES, not on the number of plans on the front.
    Symmetric plans -- swap which of two identical nodes goes to which subagent --
    produce many front entries with one outcome: a wide scenario with 5 equal
    nodes has 81 front entries and 7 distinct values. Counting entries would
    reject essentially every wide-independent scenario from Tier A, which is
    precisely where Tier A is supposed to live.

    `tol` should be set from section 5.1's materiality floor, not left at 1e-9:
    these are floats derived from measured constants that carry CIs, and exact
    equality is not a meaningful test.
    """
    vals = {(round(r.cost / tol), round(r.latency / tol)) for r in pareto_front(results)}
    return len(vals) == 1


def optimal_k_intervals(results: list[PlanResult]) -> list[tuple[float, float, int]]:
    """For which beta is each spawn count k optimal? Returns (lo, hi, k), sorted.

    This is section 6's implied-beta estimator, and it replaces the doc's
    C*(k), L*(k) construction, which is not well defined: "the best partition
    with exactly k blocks" has no meaning without a beta, and taking cost from
    the cheapest k-plan and latency from the fastest k-plan invents a point no
    real plan achieves. That fictional point is weakly better than the true
    per-k optimum at EVERY beta, so every k looks more rationalizable than it
    is -- which inflates the beta-intervals and makes an over-spawning agent
    look consistent with some latency price. Wrong direction for this report.

    Done correctly: f_k(beta) = min over plans with exactly k blocks of
    (cost + beta*latency) -- a concave piecewise-linear function of beta. k is
    optimal wherever f_k is the global minimum. Exact, no grid, computed from
    the already-cached plan set. A k that is optimal on no interval is a spawn
    count no latency price rationalizes.
    """
    lines = [(r.cost, r.latency, r.plan.k) for r in results]
    bps = {0.0}
    for i in range(len(lines)):
        ci, li, _ = lines[i]
        for j in range(i + 1, len(lines)):
            cj, lj, _ = lines[j]
            if abs(li - lj) > 1e-12:
                b = (ci - cj) / (lj - li)
                if b > 1e-9:
                    bps.add(b)
    edges = sorted(bps) + [max(bps) * 2 + 1.0]
    out: list[tuple[float, float, int]] = []
    for lo, hi in zip(edges, edges[1:]):
        mid = (lo + hi) / 2
        k = min(lines, key=lambda L: L[0] + mid * L[1])[2]
        if out and out[-1][2] == k:
            out[-1] = (out[-1][0], hi, k)
        else:
            out.append((lo, hi, k))
    return out
