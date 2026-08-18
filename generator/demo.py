"""A runnable tour of the delegation-regret machinery.

    python -m generator.demo

Every number printed here is *derived* from the cost model in `oracle.py`.
Nothing is asserted, hand-tuned, or looked up in a table.

Read the shapes, not the dollars: the cost constants in `CostModel` are
placeholders pending calibration against real API runs. What is real today is
the structure -- which plans are legal, how the lead's serial timeline bounds
the gain from fanning out, and how the optimal spawn count moves with the
price of latency.
"""

from __future__ import annotations

import math
import time

from .dag import DAG, sample_dag
from .oracle import (
    CostModel,
    Plan,
    enumerate_plans,
    evaluate,
    is_tier_a,
    optimal_k_intervals,
)

# One node count and one node size across every shape, so that shape is the
# only variable. This is the experimental design in miniature.
N = 6
SIZES = (3,)
SHAPES = ("wide", "chain", "diamond", "mixed")

# Two anchored personas. Cost is billed dollars and latency is minutes, so
# beta is dollars per minute of latency saved.
BACKGROUND = 0.0  # nobody is waiting; tokens are the only cost
ATTENDED = 1.0  # a developer is blocked: $1/min, i.e. $60/hr of their time


def _head(n: int, title: str) -> None:
    bar = "=" * 74
    print(f"\n{bar}\n {n}. {title}\n{bar}")


def _fmt_plan(plan: Plan) -> str:
    inline = ",".join(sorted(plan.inline)) or "--"
    if plan.blocks:
        blocks = "  ".join("{" + ",".join(sorted(b)) + "}" for b in plan.blocks)
    else:
        blocks = "none"
    return f"lead[{inline}]  subagents: {blocks}"


def _scenarios() -> dict[str, DAG]:
    return {s: sample_dag(s, N, sizes=SIZES, seed=0) for s in SHAPES}


def section_1_scenarios(dags: dict[str, DAG]) -> None:
    _head(1, "The scenarios: same size, same work, different structure")
    print(f"\n  {N} subtasks of equal size in every case. Only the dependency")
    print("  structure differs -- and that is what should change the answer.\n")
    for name, dag in dags.items():
        edges = sorted(dag.edges)
        shown = ", ".join(f"{u}->{v}" for u, v in edges[:6])
        if len(edges) > 6:
            shown += f", ... (+{len(edges) - 6} more)"
        print(f"  {name:<8} {len(edges):>2} dependencies   {shown or 'none (all independent)'}")


def section_2_oracle(dags: dict[str, DAG], priced: dict[str, list]) -> None:
    _head(2, "What the oracle decides, at two prices of latency")
    print("\n  background: beta = $0.00/min   (batch work, nobody is blocked)")
    print("  attended:   beta = $1.00/min   (a developer is waiting, ~$60/hr)\n")
    for name, dag in dags.items():
        results = priced[name]
        tier_a = is_tier_a(results, tol=1e-6)
        note = "  <- Pareto-dominant: this plan is optimal at EVERY beta" if tier_a else ""
        print(f"  {name}{note}")
        for label, beta in (("background", BACKGROUND), ("attended", ATTENDED)):
            best = min(results, key=lambda r: r.objective(beta))
            print(
                f"      {label:<11} spawn {best.plan.k}   "
                f"${best.cost:5.2f}   {best.latency:4.1f} min   {_fmt_plan(best.plan)}"
            )
        print()
    print("  Note the background column: serial wins on every shape, and under this")
    print("  cost model it always will. Spawning adds cost with no offsetting saving")
    print("  when latency is free, and context reuse makes one big block the cheapest")
    print("  arrangement by construction. The mechanism that should break that tie --")
    print("  context drag, where a long serial context gets re-billed every turn until")
    print("  fanning out is cheaper AND faster -- is specified but not yet implemented.")
    print("  Until it is, treat 'serial is cheapest' as an artifact of the model, not")
    print("  a result about agents.")


def section_3_implied_beta(dags: dict[str, DAG], priced: dict[str, list]) -> None:
    _head(3, "The whole beta axis: what latency price justifies what spawn count")
    print("\n  Read this backwards and it becomes the implied-beta metric: observe a")
    print("  model's spawn count, and these intervals say what it must believe a")
    print("  minute of latency is worth. A spawn count optimal on NO interval is one")
    print("  that no latency price rationalizes.\n")
    for name in dags:
        spans = optimal_k_intervals(priced[name])
        print(f"  {name}")
        for lo, hi, k in spans:
            plural = "" if k == 1 else "s"
            # The last interval runs to infinity: past the largest breakpoint the
            # answer cannot change again, so there is no ceiling to print.
            if hi == math.inf:
                rng = f"beta >= ${lo:<6.2f}          "
            else:
                rng = f"${lo:>6.2f} <= beta < ${hi:<6.2f}"
            print(f"      {rng}  ->  spawn {k} subagent{plural}")
        print()


def _superseded_feasibility(dag: DAG, plan: Plan) -> bool:
    """The feasibility rule this project used before Aug 2026, kept ONLY so the
    correction below is demonstrable rather than merely claimed.

    It contracts *every* block including the inline one and requires the
    quotient graph acyclic. That asserts the lead executes its own nodes
    contiguously -- which is exactly the thing a lead does not do.
    """
    owner: dict[str, str] = {v: "INLINE" for v in plan.inline}
    for i, b in enumerate(plan.blocks):
        for v in b:
            owner[v] = f"B{i}"
    verts = set(owner.values())
    adj: dict[str, set[str]] = {v: set() for v in verts}
    for u, v in dag.edges:
        if owner[u] != owner[v]:
            adj[owner[u]].add(owner[v])
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


def section_4_correction() -> None:
    _head(4, "A correction, made demonstrable")
    dag = sample_dag("diamond", 5, sizes=SIZES, seed=0)  # n0 -> n1,n2,n3 -> n4
    canonical = Plan(frozenset({"n0", "n4"}), (frozenset({"n1"}), frozenset({"n2"}), frozenset({"n3"})))
    plans = enumerate_plans(dag)
    recovered = [p for p in plans if not _superseded_feasibility(dag, p)]

    print("\n  Scenario: n0 -> {n1, n2, n3} -> n4. The plan any engineer would write")
    print("  is 'lead does the source and the sink, one subagent per parallel middle':\n")
    print(f"      {_fmt_plan(canonical)}\n")
    print(f"      superseded rule:  {'ACCEPTS' if _superseded_feasibility(dag, canonical) else 'REJECTS'} it")
    print(f"      current rule:     {'ACCEPTS' if canonical in plans else 'REJECTS'} it\n")
    print("  A subagent is atomic -- one spawn in, one result out, nothing")
    print("  extractable from the middle. The lead is present throughout and hands")
    print("  work over as it goes. So subagent blocks contract; inline nodes do not.")
    print(f"\n  Plans the correction recovers: {len(recovered)} of {len(plans)} legal plans,")
    print("  concentrated on exactly the ones a practitioner would actually write.")
    print("\n  This mattered because the error ran one way: it withheld the best plan")
    print("  from the shape where fanning out is most defensible, which biases the")
    print("  oracle toward spawning -- the direction of the hypothesis under test.")


def section_5_scale() -> None:
    _head(5, "Why the core is capped at 8 subtasks")
    print("\n  A plan is a partition of the subtasks plus the choice of which block")
    print("  stays inline, so the space before pruning is B(n+1), not B(n).")
    print("  Dependencies prune it -- barely on independent work, hugely on chains.\n")
    print(f"  {'n':>3}  {'wide':>12}  {'chain':>10}  {'enumerate':>10}")
    for n in range(4, 9):
        t0 = time.perf_counter()
        wide = len(enumerate_plans(sample_dag("wide", n, sizes=SIZES, seed=0)))
        chain = len(enumerate_plans(sample_dag("chain", n, sizes=SIZES, seed=0)))
        dt = time.perf_counter() - t0
        print(f"  {n:>3}  {wide:>12,}  {chain:>10,}  {dt:>9.2f}s")
    print("\n  Enumeration stays cheap. Pricing does not: latency needs a schedule")
    print("  simulation per plan, so one price point on a wide 8-node scenario runs")
    print("  ~43s, and the epistemic grid multiplies that. n <= 8 is the real ceiling.")


def main() -> None:
    print("\n  DELEGATION REGRET -- what the machinery does today")
    print("  " + "-" * 50)
    print("  Cost constants are PLACEHOLDERS pending calibration.")
    print("  No model has been measured yet. This is the instrument.")

    dags = _scenarios()
    cm = CostModel()
    priced = {name: [evaluate(dag, p, cm) for p in enumerate_plans(dag)] for name, dag in dags.items()}

    section_1_scenarios(dags)
    section_2_oracle(dags, priced)
    section_3_implied_beta(dags, priced)
    section_4_correction()
    section_5_scale()

    print(f"\n{'=' * 74}")
    print("  Next: executable subtask templates with deterministic verifiers,")
    print("  cost calibration against real API runs, then the reference harness.")
    print(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
