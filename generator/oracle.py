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

Cost constants here are PLACEHOLDERS pending calibration. Nothing
in this module hard-codes them; they all live in CostModel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations

from .dag import DAG

__all__ = [
    "CostModel",
    "Plan",
    "enumerate_plans",
    "evaluate",
    "PlanResult",
    "pareto_front",
    "is_tier_a",
    "optimal_k_intervals",
    "all_inline",
    "max_fanout",
    "is_feasible",
    "linear_curve",
    "block_dollars",
    "block_minutes",
    "sub_block_dollars",
    "sub_block_minutes",
    "throughput_penalty",
    "Measured",
    "MeasuredInt",
    "MeasuredCurve",
    "provenance",
    "unmeasured",
    "OUTCOME_TOL",
]


# Metadata every cost constant carries: where it came from, and what breaks if
# it is wrong. Attached to the VALUE rather than kept in a parallel table keyed
# by name -- a table drifts, because someone adds a constant and forgets the
# entry, or renames a field and the entry silently orphans. Here the status
# cannot be separated from the number it describes.
#
# Each class subclasses the builtin it stands in for, so the value still behaves
# as a plain float, int, or tuple in every arithmetic expression and no call site
# had to change. (No __slots__: int and tuple are variable-layout and reject
# them, so all three carry an ordinary __dict__.)
#
# Status is PER CONSTANT, not per model, because calibration is incremental: the
# block curves come from one run and spawn overhead from another. A single
# "is this model calibrated" flag cannot say that block cost is measured while
# absorption is still a guess.

def _stamp(obj, status, how, sensitivity, source):
    obj.status = status  # "placeholder" | "measured" | "pinned" | "stated"
    obj.how = how  # the measurement protocol, or why it is not measured
    obj.sensitivity = sensitivity  # what a wrong value does to the answer
    obj.source = source  # which run measured it, under which price sheet
    return obj


class Measured(float):
    def __new__(cls, value, *, status="placeholder", how="", sensitivity="", source=""):
        return _stamp(super().__new__(cls, value), status, how, sensitivity, source)


class MeasuredInt(int):
    def __new__(cls, value, *, status="placeholder", how="", sensitivity="", source=""):
        return _stamp(super().__new__(cls, value), status, how, sensitivity, source)


class MeasuredCurve(tuple):
    def __new__(cls, value, *, status="placeholder", how="", sensitivity="", source=""):
        return _stamp(super().__new__(cls, tuple(value)), status, how, sensitivity, source)


_PROVENANCED = (Measured, MeasuredInt, MeasuredCurve)


def provenance(cm: "CostModel") -> tuple[tuple[str, str, str, str], ...]:
    """(name, status, how, source) for every constant, read off the fields.

    Derived rather than maintained. There is no second list to keep in sync.
    """
    import dataclasses

    out = []
    for f in dataclasses.fields(cm):
        v = getattr(cm, f.name)
        if isinstance(v, _PROVENANCED):
            out.append((f.name, v.status, v.how, v.source))
        else:
            # A bare value someone passed in. Unknown provenance is worse than a
            # placeholder, because nothing records that it needs checking.
            out.append((f.name, "unknown", "set directly, provenance not recorded", ""))
    return tuple(out)


def unmeasured(cm: "CostModel") -> tuple[str, ...]:
    """Constants that are still guesses. Empty means every number is earned."""
    return tuple(n for n, s, _, _ in provenance(cm) if s in ("placeholder", "unknown"))


# Tolerance for "these two plans have the same outcome". Gates the Tier-A test
# and the tie-break in `optimal_k_intervals`, so a wrong value silently changes
# which scenarios count as persona-independent and which spawn count an
# over-spawning agent looks consistent with.
#
# The default is exact float equality, which tests nothing: these are floats
# derived from measured constants that carry confidence intervals. The value that
# means something is the MATERIALITY FLOOR -- the smallest objective difference
# the calibration can actually resolve -- and it cannot be known before the
# constants are measured, which is why the default is a placeholder rather than a
# guess dressed as a number.
#
# `harness.calibration` derives the floor from a completed calibration, PER
# AXIS: a dollar floor from the curve's cross-run disagreement and a minute
# floor from the timing model's residual (`CalibrationResult.floors`). Pass
# them explicitly -- `is_tier_a(results, tol=<dollars>, tol_latency=<minutes>)`
# and `optimal_k_intervals(results, tol=<dollars>, minute_tol=<minutes>)` --
# rather than mutating this constant, so one calibration does not silently
# redefine "equal" for every other run in the process. `scoring.regret` wires
# them through `score(..., floors=(dollars, minutes))`.
OUTCOME_TOL = 1e-9


# ---------------------------------------------------------- the block curves


def linear_curve(per_unit: float, at_units: int = 100) -> tuple[tuple[int, float], ...]:
    """A single-point curve with no reuse and no drag: cost is exactly linear.

    This is the honest placeholder. It asserts nothing about how a block's cost
    scales, which is the one thing calibration exists to find out.
    """
    return ((at_units, per_unit * at_units),)


def _interpolate(curve: tuple[tuple[int, float], ...], units: float) -> float:
    """Read a measured curve at `units`, linear between points.

    Below the first measured point the curve is scaled from the origin: a block
    smaller than anything measured is priced pro rata rather than extrapolated
    backwards off a segment slope, which could go negative. Above the last point
    the final segment's slope continues, which prices a plan outside the
    measured range -- keep the calibration's largest block at or above the
    largest scenario.
    """
    if units <= 0:
        return 0.0
    if not curve:
        raise ValueError("empty block curve")
    pts = tuple(sorted(curve))
    u0, v0 = pts[0]
    if units <= u0:
        return v0 * units / u0
    for (ua, va), (ub, vb) in zip(pts, pts[1:]):
        if units <= ub:
            return va + (vb - va) * (units - ua) / (ub - ua)
    if len(pts) == 1:
        return v0 * units / u0
    (ua, va), (ub, vb) = pts[-2], pts[-1]
    return vb + (vb - va) * (units - ub) / (ub - ua)


# ---------------------------------------------------------------- cost model


@dataclass(frozen=True)
class CostModel:
    """Every price and duration the oracle uses. Calibrated, never assumed.

    Units are pinned here once: cost is BILLED DOLLARS, latency is MINUTES, so
    beta is dollars per minute of latency saved and alpha is identically 1.
    Section 6 reports implied beta in exactly these units.
    """

    # Block curves: (cumulative size units in one block) -> value. These carry
    # the whole size-dependence of the model. A single agent working a block
    # gains from context REUSE (later subtasks need less setup) and loses to
    # context DRAG (the whole conversation re-billed every turn) -- two opposing
    # forces on one axis, and which wins is an empirical fact, not a parameter.
    block_dollars_curve: tuple = MeasuredCurve(
        ((100, 10.0),),
        status="placeholder",
        how=(
            "Serial calibration run, spawning disabled, node order FORCED per run. Segment "
            "the proxy call log at each subtask completion; plot cumulative billed dollars "
            "NET OF THE ORIENTATION PREFIX (everything before the first write -- that spend "
            "is explore_dollars' to charge, once) against cumulative size units. Average >=3 "
            "distinct orderings x 2 repeats and report the spread -- disagreement falsifies "
            "the units-not-identity assumption. Coverage must reach the largest total "
            "scenario size or the all-inline plan is priced by extrapolation. Pin the 1-hour "
            "cache TTL first: at the 5-minute default, prefix entries expire during the gaps "
            "while tests run, reads silently become writes at 1.25x, and the curve bends "
            "superlinear for a reason unrelated to context."
        ),
        sensitivity=(
            "CRITICAL. Its shape decides whether spawning can ever be CHEAPER rather than "
            "merely faster. The linear placeholder asserts neither reuse nor drag, so under it "
            "the beta=0 oracle says 'spawn nothing' everywhere -- placeholder, not result."
        ),
    )
    block_minutes_curve: tuple = MeasuredCurve(
        ((100, 100.0),),
        status="placeholder",
        how=(
            "Same segmentation, summing analytic per-call minutes from the fitted timing model "
            "rather than wall clock, which is confounded by rate limits and provider load."
        ),
        sensitivity="CRITICAL. The latency half; sets where fan-out stops paying.",
    )

    # What a SUBAGENT is billed to work a block, net of its first call (which
    # `spawn_fixed_dollars` already carries). A separate curve from the lead's,
    # because the two roles pay different context economics: the serial lead
    # works inside one warm, cached conversation, while every subagent pays
    # fresh cache-writes for its seeded context and re-bills its own growing
    # conversation from zero. Until Aug 25 spawned blocks were priced off the
    # lead's serial curve, and the estimation gate caught that composition
    # under-pricing real delegation runs by 38-53% with the plan ranking
    # inverted -- the pro-spawn direction. The placeholders are numerically
    # identical to the lead-curve placeholders, so an uncalibrated model prices
    # both roles the same and asserts nothing about the difference.
    sub_block_dollars_curve: tuple = MeasuredCurve(
        ((100, 10.0),),
        status="placeholder",
        how=(
            "From the fan-out and bundled calibration arms: each subagent's own calls, summed, "
            "minus its first call, keyed on the size units of the nodes attribution says it "
            "worked. Direct per-block totals -- no regression, no decomposition."
        ),
        sensitivity=(
            "CRITICAL, and one-sided: reusing the lead's serial curve here under-priced "
            "delegation ~2x on the first preflight and inverted the executed-plan ranking."
        ),
    )
    sub_block_minutes_curve: tuple = MeasuredCurve(
        ((100, 100.0),),
        status="placeholder",
        how=(
            "Same per-subagent segmentation, in analytic minutes from the fitted timing model, "
            "minus the first call's minutes."
        ),
        sensitivity="CRITICAL. Sets how long a spawned block actually runs in the schedule.",
    )

    # the lead's own overhead
    explore_minutes: float = Measured(
        1.0,
        status="placeholder",
        how="Analytic minutes from run start to the first write_file or spawn_subagent call.",
        sensitivity="NONE for plan selection: identical in every plan, so it cancels. Verified by inflating\n            it 100x -- every plan shifted by exactly the same amount, no winner changed.",
    )
    explore_dollars: float = Measured(
        0.05,
        status="placeholder",
        how="Tokens billed from run start to the first write_file or spawn_subagent call.",
        sensitivity="NONE for plan selection, same reason. Do not spend measurement effort here.",
    )


    # spawn overhead, decomposed per section 5.1: fixed + briefing + absorption
    spawn_fixed_dollars: float = Measured(
        0.02,
        status="placeholder",
        how="Spawn a subagent with a trivial instruction ('read this file, report its line\n            count'). Its total minus the negligible work is the floor: system prompt, tool\n            definitions, one round trip.",
        sensitivity="HIGH. With briefing and absorption this IS the price of delegating.",
    )
    # Briefing and absorption are AFFINE in block size on BOTH axes: a fixed part
    # per subagent plus a part that scales with how many nodes the instruction
    # covers. That shape is not a choice -- it is what the fitted timing model
    # (`minutes = a + b*input_tokens + output_tokens/throughput`) says, since a
    # briefing's tokens grow with the nodes it describes. Pricing the dollars per
    # node while charging the minutes flat billed the SAME EMITTED TOKENS two
    # different ways, and the flat form under-charged wide blocks -- which made
    # bundling several nodes into one subagent look faster than it is.
    brief_minutes: float = Measured(
        0.25,
        status="placeholder",
        how="The fixed half: intercept of briefing duration against instruction length, from the\n            fitted timing model. Measure with `brief_minutes_per_node` in one regression, not\n            separately -- they are the intercept and slope of one line.",
        sensitivity="HIGH, and structurally so: briefings serialise on the lead, so this constant and its\n            per-node partner set the width at which fan-out stops buying time.",
    )
    brief_minutes_per_node: float = Measured(
        0.25,
        status="placeholder",
        how="The slope: extra briefing duration per node the instruction covers. Same regression\n            as `brief_minutes`, and the same trace rows as `brief_dollars_per_node` -- one\n            quantity of emitted tokens read in minutes rather than dollars.",
        sensitivity="HIGH. It is what stops a subagent holding five nodes from being briefed as fast as\n            one holding a single node. Setting it to 0 restores the flat form, which\n            under-charges wide blocks and flatters bundling.",
    )
    brief_dollars_per_node: float = Measured(
        0.01,
        status="placeholder",
        how="The instruction text in a spawn_subagent call IS output tokens the lead emitted.\n            One regression row per spawn: x = nodes the spawn covered (from attribution),\n            y = the spawn's instruction-length share of the turn's billed output tokens.\n            Never whole-turn dollars -- context prefill grows turn over turn and swamps\n            the briefing signal.",
        sensitivity="HIGH. Scales with block size, so it prices wide fan-out specifically.",
    )
    absorb_minutes: float = Measured(
        0.25,
        status="placeholder",
        how="The fixed half, same boundary and same regression shape as briefing.",
        sensitivity="HIGH. Absorptions serialise and queue at the end of a wide fan-out.",
    )
    absorb_minutes_per_node: float = Measured(
        0.25,
        status="placeholder",
        how="The slope: extra absorption duration per node in the returning block. Paired with\n            `absorb_dollars_per_node` -- the same returned summary, in minutes.",
        sensitivity="HIGH. A block of five nodes returns a longer summary than a block of one, and the\n            lead reads them one at a time.",
    )
    absorb_dollars_per_node: float = Measured(
        0.01,
        status="placeholder",
        how="Watch the lead's input-token count per turn during a fan-out run; it steps up when a\n            result lands. Multiply that step by the lead turns remaining.",
        sensitivity="HIGH -- and the flat per-node form UNDER-MODELS the real cost, because a returned\n            summary is re-billed on every subsequent lead turn. The error flatters fan-out.",
    )


    # Per-stream slowdown when k subagents share one account's rate limits. The
    # design has carried this as a KNOWN GAP: a single throughput figure
    # overstates fan-out's latency advantage, the same directional bias as
    # pricing spawns at zero latency, and it bites hardest exactly where the
    # headline finding lives.
    #
    # Carried as a curve keyed on IN-FLIGHT COUNT, read the same way the block
    # curves are read: (concurrency, multiplier on a block's duration). The
    # placeholder is flat 1.0 at every measured point, which asserts no
    # degradation -- the honest stand-in, and the one that keeps the current
    # numbers unchanged until the measurement exists.
    throughput_curve: tuple = MeasuredCurve(
        ((1, 1.0), (4, 1.0), (8, 1.0)),
        status="placeholder",
        how=(
            "Run the same block at concurrency 1, 4, and 8 and divide each duration by the "
            "concurrency-1 duration. Measure on ONE account, since the shared quota is the "
            "mechanism. Report the multiplier, not the raw duration, so it composes with a "
            "block curve measured separately."
        ),
        sensitivity=(
            "HIGH on wide shapes and none on chains. Flat 1.0 asserts that four subagents "
            "each run as fast as one would alone, which is the direction that flatters "
            "fan-out -- so the placeholder biases toward the hypothesis under test."
        ),
    )

    concurrency_cap: int = MeasuredInt(
        4,
        status="pinned",
        how="NOT MEASURED. A published harness constant; must equal harness.tools.MAX_CONCURRENCY.",
        sensitivity=(
            "An oracle with more workers than the agent can use penalises the agent for a "
            "constraint it never faced."
        ),
    )
    beta: float = Measured(
        0.0,
        status="stated",
        how="NOT MEASURED. The persona under test: background is 0, attended is anchored to "
        "blocked developer time.",
        sensitivity="It IS the question, not an input to be estimated.",
    )

    @property
    def is_calibrated(self) -> bool:
        """True only when no constant is still a guess."""
        return not unmeasured(self)

    def calibrate(self, source: str, **values) -> "CostModel":
        """Return a copy with the named constants replaced and marked measured.

        `source` records the run and the price sheet's as-of date. An undated
        dollar is not a unit, and this is what stops one being printed as though
        it were. Constants not named here keep their placeholder status, so a
        partially calibrated model reports itself as exactly that.
        """
        import dataclasses

        known = {f.name for f in dataclasses.fields(self)}
        patched = {}
        for name, value in values.items():
            if name not in known:
                raise ValueError(
                    f"{name!r} is not a field of CostModel; known constants are "
                    + ", ".join(sorted(known))
                )
            old = getattr(self, name)
            if not isinstance(old, _PROVENANCED):
                raise ValueError(
                    f"{name!r} carries no provenance, so calibrating it would record a "
                    "measurement against a value of unknown origin"
                )
            cls = type(old)
            patched[name] = cls(
                value, status="measured", how=old.how, sensitivity=old.sensitivity, source=source
            )
        return dataclasses.replace(self, **patched)


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


def is_feasible(dag: DAG, plan: Plan) -> bool:
    """Public name for the feasibility rule. See `_is_feasible`.

    Exposed because callers outside the enumerator now construct plans -- the
    calibration driver builds a bundled fan-out, and a plan that induces a
    circular wait would be rejected by `enumerate_plans` but accepted by a
    hand-built one. A rule the enumerator enforces and nobody else can check is
    a rule with a hole in it.
    """
    return _is_feasible(dag, plan)


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


def _units(dag: DAG, block: frozenset[str]) -> int:
    return sum(dag.by_id[v].size for v in block)


def throughput_penalty(cm: CostModel, in_flight: int) -> float:
    """Duration multiplier for a subagent running alongside `in_flight - 1` others.

    Read off `throughput_curve`. Below the first measured point it is 1.0 rather
    than interpolated from the origin -- a block running alone cannot be faster
    than the concurrency-1 measurement, and `_interpolate` scales pro rata below
    its first point, which here would report a multiplier near zero.
    """
    if in_flight <= 1:
        return 1.0
    pts = tuple(sorted(cm.throughput_curve))
    if in_flight <= pts[0][0]:
        return pts[0][1]
    return _interpolate(cm.throughput_curve, in_flight)


def block_minutes(dag: DAG, block: frozenset[str], cm: CostModel) -> float:
    """One agent runs its nodes one at a time, regardless of dependencies."""
    return _interpolate(cm.block_minutes_curve, _units(dag, block))


def block_dollars(dag: DAG, block: frozenset[str], cm: CostModel) -> float:
    """What the LEAD is billed for working this block in its own context.

    Read straight off the measured curve, keyed on total size units. WHICH nodes
    they are does not enter: the modelling assumption is that block cost depends
    on how much work is in the block, not on which subtasks compose it. That is
    one testable approximation in place of two unmeasurable parameters, and it
    is checked by calibrating a second node ordering and comparing the curves.
    """
    return _interpolate(cm.block_dollars_curve, _units(dag, block))


def sub_block_dollars(dag: DAG, block: frozenset[str], cm: CostModel) -> float:
    """What a SUBAGENT is billed to work this block, beyond its first call.

    Never the lead's curve: the estimation gate showed that substitution
    under-prices real delegation runs by ~2x, because a subagent pays fresh-
    context costs the warm serial lead never sees.
    """
    return _interpolate(cm.sub_block_dollars_curve, _units(dag, block))


def sub_block_minutes(dag: DAG, block: frozenset[str], cm: CostModel) -> float:
    """A subagent's working duration for this block, beyond its first call."""
    return _interpolate(cm.sub_block_minutes_curve, _units(dag, block))


_block_minutes = block_minutes
_block_dollars = block_dollars


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
    inline_units = 0  # size units the lead has already worked through itself
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
            lead = start + cm.brief_minutes + cm.brief_minutes_per_node * len(plan.blocks[b])
            launched[b] = lead
            # How many others are already running when this one starts. The
            # penalty is applied at LAUNCH rather than recomputed as the mix
            # changes: a subagent that starts into a crowded account is slowed
            # for its whole errand, and re-deriving the multiplier mid-flight
            # would model a scheduler that reallocates quota, which no provider
            # offers.
            concurrent = 1 + sum(
                1 for i, f in sub_finish.items() if launched[i] <= lead < f
            )
            sub_finish[b] = lead + sub_block_minutes(dag, plan.blocks[b], cm) * throughput_penalty(
                cm, concurrent
            )
            unbriefed.discard(b)
        elif kind == "absorb":
            b = key
            lead = start + cm.absorb_minutes + cm.absorb_minutes_per_node * len(plan.blocks[b])
            for v in plan.blocks[b]:
                avail[v] = lead
            absorbed.add(b)
        else:
            v = key
            # The lead's inline nodes share ONE context, so they are a block like
            # any other -- but the lead executes them individually, interleaved
            # with briefings and absorptions, so each needs its own duration.
            # Take the MARGINAL value: what adding this node costs on top of the
            # inline work already done. Summed over the block this telescopes to
            # exactly block_minutes(inline), so latency and cost stay consistent.
            before = inline_units
            inline_units += dag.by_id[v].size
            lead = start + (
                _interpolate(cm.block_minutes_curve, inline_units)
                - _interpolate(cm.block_minutes_curve, before)
            )
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
            + sub_block_dollars(dag, b, cm)
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


def is_tier_a(
    results: list[PlanResult], tol: float = OUTCOME_TOL, tol_latency: float | None = None
) -> bool:
    """Tier A: one (cost, latency) OUTCOME dominates everything, so the same plan
    is optimal at every persona (section 5.1).

    Tested on distinct (C, L) VALUES, not on the number of plans on the front.
    Symmetric plans -- swap which of two identical nodes goes to which subagent --
    produce many front entries with one outcome: a wide scenario with 5 equal
    nodes has 81 front entries and 7 distinct values. Counting entries would
    reject essentially every wide-independent scenario from Tier A, which is
    precisely where Tier A is supposed to live.

    `tol` gates the COST axis in dollars and `tol_latency` the LATENCY axis in
    minutes -- two units, two tolerances, both set from the calibration's
    materiality floors rather than left at 1e-9: these are floats derived from
    measured constants that carry CIs, and exact equality is not a meaningful
    test. `tol_latency` defaults to `tol` for callers predating the split.
    """
    tl = tol if tol_latency is None else tol_latency
    vals = {(round(r.cost / tol), round(r.latency / tl)) for r in pareto_front(results)}
    return len(vals) == 1


def optimal_k_intervals(
    results: list[PlanResult], tol: float = OUTCOME_TOL, minute_tol: float = 0.0
) -> list[tuple[float, float, int]]:
    """For which beta is each spawn count k optimal? Returns (lo, hi, k), sorted.

    `tol` (dollars) and `minute_tol` (minutes) come from the calibration's
    materiality floors; the tie-break tolerance at a probe beta is
    `tol + probe * minute_tol`, matching the objective's own units at that
    beta. The defaults reproduce the old exact-equality behaviour.

    The intervals partition [0, inf): the first starts at 0.0 and the last ends
    at `math.inf`, because past the largest breakpoint the argmin cannot change
    again. That upper edge is reported as infinity rather than as a large finite
    number so no caller can mistake a padding value for a real breakpoint and
    print a ceiling this analysis never derived.

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
    edges = sorted(bps) + [math.inf]
    out: list[tuple[float, float, int]] = []
    for lo, hi in zip(edges, edges[1:]):
        # Any interior point identifies the interval's argmin; the boundaries are
        # ties by construction. On the unbounded tail, lo + 1 is interior.
        probe = lo + 1.0 if hi == math.inf else (lo + hi) / 2
        # Ties are rampant and must not be broken by enumeration order. Swapping
        # two interchangeable nodes between two subagents yields a different plan
        # with an identical outcome, and on wide shapes dozens of plans reach the
        # same optimum at several DIFFERENT spawn counts -- so a bare min() picks
        # whichever the enumerator happened to emit first, and the reported k
        # oscillates with beta instead of climbing.
        #
        # Break ties toward the SMALLEST k. At equal cost and equal latency fewer
        # subagents is strictly preferable -- less failure surface, fewer moving
        # parts -- and for the implied-beta estimator it is the conservative
        # choice: reporting the largest tied k would let an over-spawning agent
        # look rationalizable across more of the beta axis, which biases the
        # estimator toward the finding this project is trying to test.
        scored = [(L[0] + probe * L[1], L[2]) for L in lines]
        floor = min(s for s, _ in scored)
        k = min(kk for s, kk in scored if s <= floor + tol + probe * minute_tol)
        if out and out[-1][2] == k:
            out[-1] = (out[-1][0], hi, k)
        else:
            out.append((lo, hi, k))
    return out


# ------------------------------------------------- the two degenerate policies
#
# Section 6 reports both as reference lines on every headline figure, and their
# spread is regret's denominator. They are ordinary members of the enumerated
# plan set -- no execution required to price them, which is why the denominator
# costs no API runs.


def all_inline(dag: DAG) -> Plan:
    """Do everything yourself. The COST test's baseline."""
    return Plan(frozenset(dag.ids), ())


def max_fanout(dag: DAG) -> Plan:
    """One subagent per node. Never cheaper; sometimes faster."""
    return Plan(frozenset(), tuple(frozenset({v}) for v in dag.ids))
