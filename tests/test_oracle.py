"""Invariants of the plan space, the schedule simulation, and the beta estimator.

Three specification errors were found while implementing the solver, and every
one of them biased the oracle toward more spawning -- the direction of the
hypothesis under test. Each is pinned here by a test that fails if the error
comes back:

    correction 1 (feasibility)  TestFeasibility
    correction 2 (latency)      TestLatencyIsNotDoubleCounted
    correction 3 (implied beta) TestOptimalKIntervals

The rest guard the properties those corrections were derived from.
"""

from __future__ import annotations

import dataclasses
import math
import unittest
from itertools import permutations

from generator.dag import DAG, Node, sample_dag, SHAPES
from generator.oracle import (
    CostModel,
    OUTCOME_TOL,
    Plan,
    PlanResult,
    all_inline,
    block_minutes,
    enumerate_plans,
    evaluate,
    is_tier_a,
    max_fanout,
    optimal_k_intervals,
    pareto_front,
)

CM = CostModel()
SIZES = (3,)


def bell(n: int) -> int:
    """B(n) via the Bell triangle. Independent of the code under test."""
    row = [1]
    for _ in range(n):
        nxt = [row[-1]]
        for x in row:
            nxt.append(nxt[-1] + x)
        row = nxt
    return row[0]


def work_minutes(dag: DAG, cm: CostModel = CM) -> float:
    """Total serial minutes: every node, one agent, one context."""
    return block_minutes(dag, frozenset(dag.ids), cm)


class TestPlanSpace(unittest.TestCase):
    def test_wide_plan_count_is_bell_n_plus_1(self) -> None:
        # A plan is a partition of the subtasks PLUS the choice of which block
        # stays inline, so the unpruned space is B(n+1), not B(n). Independent
        # work has no precedence to prune, so wide hits the bound exactly.
        for n in range(3, 9):
            self.assertEqual(len(enumerate_plans(sample_dag("wide", n, sizes=SIZES))), bell(n + 1), f"n={n}")

    def test_precedence_prunes_chains_hard(self) -> None:
        for n in range(4, 9):
            chain = len(enumerate_plans(sample_dag("chain", n, sizes=SIZES)))
            self.assertLess(chain, bell(n + 1))

    def test_every_plan_is_a_partition_of_the_subtasks(self) -> None:
        for shape in SHAPES:
            dag = sample_dag(shape, 6, sizes=SIZES)
            for plan in enumerate_plans(dag):
                covered: list[str] = list(plan.inline)
                for b in plan.blocks:
                    covered.extend(b)
                self.assertEqual(sorted(covered), sorted(dag.ids))  # no gaps, no double assignment
                self.assertEqual(plan.k, len(plan.blocks))

    def test_degenerate_policies_are_always_legal(self) -> None:
        # Both reference lines must exist on every shape or regret has no
        # denominator to normalize against.
        for shape in SHAPES:
            dag = sample_dag(shape, 6, sizes=SIZES)
            plans = enumerate_plans(dag)
            self.assertIn(all_inline(dag), plans, f"{shape}: all-serial missing")
            self.assertIn(max_fanout(dag), plans, f"{shape}: max-fanout missing")

    def test_purely_orchestrating_lead_is_representable(self) -> None:
        dag = sample_dag("wide", 4, sizes=SIZES)
        self.assertTrue(any(p.inline == frozenset() for p in enumerate_plans(dag)))


class TestFeasibility(unittest.TestCase):
    """Correction 1. Subagent blocks contract; inline nodes do not."""

    def setUp(self) -> None:
        self.dag = sample_dag("diamond", 5, sizes=SIZES)  # n0 -> n1,n2,n3 -> n4
        self.plans = enumerate_plans(self.dag)

    def test_canonical_diamond_plan_is_accepted(self) -> None:
        # The plan any practitioner would write: lead takes source and sink, one
        # subagent per parallel middle. Rejecting this withheld the best plan
        # from the shape where fanning out is most defensible.
        canonical = Plan(
            frozenset({"n0", "n4"}),
            (frozenset({"n1"}), frozenset({"n2"}), frozenset({"n3"})),
        )
        self.assertIn(canonical, self.plans)

    def test_circular_wait_through_one_subagent_is_rejected(self) -> None:
        # One subagent holding both the source and the sink would have to hand
        # the lead an artifact partway through its own errand. One spawn cannot.
        bad = Plan(frozenset({"n1", "n2", "n3"}), (frozenset({"n0", "n4"}),))
        self.assertNotIn(bad, self.plans)

    def test_split_middles_across_two_subagents_stays_legal(self) -> None:
        ok = Plan(frozenset({"n0", "n4"}), (frozenset({"n1", "n2"}), frozenset({"n3"})))
        self.assertIn(ok, self.plans)

    def test_current_rule_is_strictly_more_permissive_than_contracting_inline(self) -> None:
        # The superseded rule contracted every block including the inline one,
        # which asserts the lead runs its nodes contiguously -- the one
        # participant for whom that is false. Every plan it accepted is still
        # accepted, and it wrongly rejected some the current rule keeps.
        accepted = set(self.plans)
        superseded = {p for p in self.plans if _contract_everything(self.dag, p)}
        self.assertTrue(superseded < accepted)  # proper subset
        self.assertGreater(len(accepted - superseded), 0)

    def test_no_accepted_plan_has_a_circular_wait(self) -> None:
        # Permissiveness must not cost soundness: contract the subagent blocks of
        # every accepted plan and confirm the quotient graph really is acyclic.
        for shape in SHAPES:
            dag = sample_dag(shape, 6, sizes=SIZES)
            for plan in enumerate_plans(dag):
                self.assertTrue(_quotient_acyclic(dag, plan), f"{shape}: {plan}")


def _contract_everything(dag: DAG, plan: Plan) -> bool:
    """The superseded rule: contract the inline block too."""
    owner = {v: "INLINE" for v in plan.inline}
    for i, b in enumerate(plan.blocks):
        for v in b:
            owner[v] = f"B{i}"
    return _acyclic({owner[u]: None for u in owner}, [(owner[u], owner[v]) for u, v in dag.edges])


def _quotient_acyclic(dag: DAG, plan: Plan) -> bool:
    owner = {v: v for v in plan.inline}
    for i, b in enumerate(plan.blocks):
        for v in b:
            owner[v] = f"B{i}"
    return _acyclic({owner[u]: None for u in owner}, [(owner[u], owner[v]) for u, v in dag.edges])


def _acyclic(verts: dict, edges: list) -> bool:
    """DFS cycle check written independently of the solver's Kahn implementation."""
    adj: dict = {v: set() for v in verts}
    for u, v in edges:
        if u != v:
            adj[u].add(v)
    WHITE, GREY, BLACK = 0, 1, 2
    color = {v: WHITE for v in verts}

    def visit(u) -> bool:
        color[u] = GREY
        for w in adj[u]:
            if color[w] == GREY or (color[w] == WHITE and not visit(w)):
                return False
        color[u] = BLACK
        return True

    return all(color[v] != WHITE or visit(v) for v in verts)


class TestLatencyIsNotDoubleCounted(unittest.TestCase):
    """Correction 2. The lead's own timeline and the quotient path overlap in
    time, so adding them inflated the all-serial plan by 2x -- which made every
    spawning plan look better than it is."""

    def test_all_inline_latency_is_explore_plus_the_work(self) -> None:
        # The exact regression: one agent, one node at a time, nothing else on
        # the timeline. Any double-count shows up here first.
        for shape in SHAPES:
            for n in (3, 5, 6):
                dag = sample_dag(shape, n, sizes=SIZES)
                r = evaluate(dag, all_inline(dag), CM)
                self.assertAlmostEqual(r.latency, CM.explore_minutes + work_minutes(dag), places=9)

    def test_single_subagent_pays_exactly_one_brief_and_one_absorb(self) -> None:
        dag = sample_dag("chain", 6, sizes=SIZES)
        n = dag.n
        r = evaluate(dag, Plan(frozenset(), (frozenset(dag.ids),)), CM)
        # One briefing and one absorption, each AFFINE in the block's node count.
        expected = (
            CM.explore_minutes
            + CM.brief_minutes + CM.brief_minutes_per_node * n
            + work_minutes(dag)
            + CM.absorb_minutes + CM.absorb_minutes_per_node * n
        )
        self.assertAlmostEqual(r.latency, expected, places=9)

    def test_briefing_and_absorption_scale_with_block_size(self) -> None:
        """Regression. Dollars were charged per node and minutes flat per block,
        though both price the SAME EMITTED TOKENS -- a briefing covering three
        nodes is a longer instruction than one covering a single node, in money
        and in the time the lead spends emitting it. The flat form under-charged
        wide blocks, which made bundling several nodes into one subagent look
        faster than it is. The fitted timing model
        (`minutes = a + b*input_tokens + ...`) says the shape is affine, so a
        fixed part plus a per-node part is what the model already implies."""
        dag = sample_dag("wide", 4, sizes=SIZES)
        ids = sorted(dag.ids)

        one = evaluate(dag, Plan(frozenset(ids[1:]), (frozenset({ids[0]}),)), CM)
        three = evaluate(dag, Plan(frozenset(ids[3:]), (frozenset(ids[:3]),)), CM)

        # Same total work either way; the difference is entirely lead-side
        # briefing and absorption on a 1-node block versus a 3-node block.
        gap = (CM.brief_minutes_per_node + CM.absorb_minutes_per_node) * 2
        self.assertGreater(CM.brief_minutes_per_node, 0.0, "flat briefing time is the bug")
        self.assertAlmostEqual(three.latency - one.latency, gap, places=9)

    def test_flat_briefing_time_is_recoverable_but_not_the_default(self) -> None:
        # Zeroing the per-node terms restores the superseded flat behaviour, so
        # the correction is a strict generalisation rather than a replacement.
        dag = sample_dag("wide", 4, sizes=SIZES)
        flat = CostModel(brief_minutes_per_node=0.0, absorb_minutes_per_node=0.0)
        ids = sorted(dag.ids)
        one = evaluate(dag, Plan(frozenset(ids[1:]), (frozenset({ids[0]}),)), flat)
        three = evaluate(dag, Plan(frozenset(ids[3:]), (frozenset(ids[:3]),)), flat)
        self.assertAlmostEqual(three.latency, one.latency, places=9)

    def test_fanout_schedule_matches_a_hand_derived_timeline(self) -> None:
        # 3 independent size-3 nodes, one subagent each. Briefings serialize on
        # the lead (1.0, 1.5, 2.0), each subagent runs 3 min from its briefing,
        # absorptions serialize behind them: 4.5 -> 5.0 -> 5.5 -> 6.0.
        dag = sample_dag("wide", 3, sizes=SIZES)
        r = evaluate(dag, max_fanout(dag), CM)
        self.assertAlmostEqual(r.latency, 6.0, places=9)
        self.assertAlmostEqual(r.cost, CM.explore_dollars + 3 * (0.02 + 0.01 + 0.01 + 0.30), places=9)

    def test_spawning_is_never_free_in_time(self) -> None:
        # If briefings and absorptions cost nothing on the lead's timeline, then
        # maximal fan-out is latency-free by construction and the oracle spawns
        # everywhere. On a chain, fanning out can only add time.
        dag = sample_dag("chain", 5, sizes=SIZES)
        serial = evaluate(dag, all_inline(dag), CM)
        for plan in enumerate_plans(dag):
            if plan.k:
                self.assertGreater(evaluate(dag, plan, CM).latency, serial.latency - 1e-9)

    def test_no_plan_finishes_before_its_critical_path(self) -> None:
        for shape in SHAPES:
            dag = sample_dag(shape, 6, sizes=SIZES)
            biggest = max(dag.nodes, key=lambda n: n.size)
            floor = CM.explore_minutes + block_minutes(dag, frozenset({biggest.id}), CM)
            for plan in enumerate_plans(dag):
                r = evaluate(dag, plan, CM)
                self.assertTrue(math.isfinite(r.latency))
                self.assertGreaterEqual(r.latency, floor - 1e-9)

    def test_every_legal_plan_schedules_without_deadlock(self) -> None:
        # _simulate raises rather than returning a wrong number if it stalls, so
        # this asserts feasibility and the scheduler agree on every shape.
        for shape in SHAPES:
            dag = sample_dag(shape, 6, sizes=SIZES)
            for plan in enumerate_plans(dag):
                evaluate(dag, plan, CM)

    def test_cost_is_launch_order_independent_but_latency_is_not(self) -> None:
        dag = sample_dag("mixed", 6, sizes=SIZES, seed=0)
        varies = False
        for plan in enumerate_plans(dag):
            if plan.k < 2 or plan.k > 4:
                continue
            from generator.oracle import _simulate

            lats = {round(_simulate(dag, plan, CM, o), 9) for o in permutations(range(plan.k))}
            best = evaluate(dag, plan, CM).latency
            self.assertAlmostEqual(best, min(lats), places=9)  # the reported latency is the best order
            varies = varies or len(lats) > 1
        self.assertTrue(varies, "no plan's latency depended on launch order -- check the simulation")

    def test_raising_the_concurrency_cap_never_hurts(self) -> None:
        dag = sample_dag("wide", 6, sizes=SIZES)
        plan = max_fanout(dag)
        lats = [
            evaluate(dag, plan, CostModel(concurrency_cap=c)).latency for c in (1, 2, 4, 8)
        ]
        for a, b in zip(lats, lats[1:]):
            self.assertLessEqual(b, a + 1e-9)
        self.assertLess(lats[-1], lats[0])  # the cap actually binds at c=1


class TestParetoAndTierA(unittest.TestCase):
    def test_chain_is_tier_a(self) -> None:
        # No latency price justifies spawning on a chain, so one outcome
        # dominates the whole front and the persona is irrelevant.
        dag = sample_dag("chain", 6, sizes=SIZES)
        results = [evaluate(dag, p, CM) for p in enumerate_plans(dag)]
        self.assertTrue(is_tier_a(results, tol=1e-6))

    def test_wide_is_not_tier_a(self) -> None:
        dag = sample_dag("wide", 6, sizes=SIZES)
        results = [evaluate(dag, p, CM) for p in enumerate_plans(dag)]
        self.assertFalse(is_tier_a(results, tol=1e-6))

    def test_tier_a_tests_outcomes_not_plan_count(self) -> None:
        # Symmetric plans -- swap which of two identical nodes goes to which
        # subagent -- put many plans on the front with one (cost, latency)
        # value. Counting front entries would reject every wide scenario.
        dag = sample_dag("wide", 5, sizes=SIZES)
        results = [evaluate(dag, p, CM) for p in enumerate_plans(dag)]
        front = pareto_front(results)
        values = {(round(r.cost, 6), round(r.latency, 6)) for r in front}
        self.assertGreater(len(front), len(values))

    def test_front_members_are_mutually_non_dominated(self) -> None:
        dag = sample_dag("diamond", 6, sizes=SIZES)
        front = pareto_front([evaluate(dag, p, CM) for p in enumerate_plans(dag)])
        for a in front:
            for b in front:
                if a is b:
                    continue
                strictly_better = (b.cost <= a.cost and b.latency <= a.latency) and (
                    b.cost < a.cost or b.latency < a.latency
                )
                self.assertFalse(strictly_better)


class TestOptimalKIntervals(unittest.TestCase):
    """Correction 3. Taking cost from the cheapest k-plan and latency from the
    fastest k-plan invents a point no real plan achieves, and that fictional
    point is weakly better than the true per-k optimum at every beta -- so every
    spawn count looks rationalizable. The intervals must come from real plans."""

    def _results(self, shape: str, n: int = 6) -> tuple[DAG, list[PlanResult]]:
        dag = sample_dag(shape, n, sizes=SIZES)
        return dag, [evaluate(dag, p, CM) for p in enumerate_plans(dag)]

    def test_intervals_partition_the_beta_axis(self) -> None:
        for shape in SHAPES:
            _, results = self._results(shape)
            spans = optimal_k_intervals(results)
            self.assertEqual(spans[0][0], 0.0, f"{shape}: must start at beta=0")
            self.assertEqual(spans[-1][1], math.inf, f"{shape}: must be open above")
            for (lo1, hi1, _), (lo2, _, _) in zip(spans, spans[1:]):
                self.assertLess(lo1, hi1)
                self.assertEqual(hi1, lo2, f"{shape}: gap or overlap at {hi1}")

    def test_adjacent_intervals_report_different_k(self) -> None:
        for shape in SHAPES:
            _, results = self._results(shape)
            ks = [k for _, _, k in optimal_k_intervals(results)]
            self.assertEqual(len(ks), len(set(ks)) if len(ks) == len(set(ks)) else len(ks))
            for a, b in zip(ks, ks[1:]):
                self.assertNotEqual(a, b, f"{shape}: unmerged adjacent interval")

    def test_reported_k_is_the_true_argmin_inside_each_interval(self) -> None:
        # Brute force against the intervals: no grid search, no tolerance games.
        for shape in SHAPES:
            _, results = self._results(shape)
            for lo, hi, k in optimal_k_intervals(results):
                for probe in _probes(lo, hi):
                    floor = min(r.objective(probe) for r in results)
                    # Ties across different k are normal -- swapping two
                    # interchangeable nodes between subagents changes the plan
                    # and not the outcome. The estimator resolves them toward the
                    # smallest k, so that is what must be reported.
                    tied = {r.plan.k for r in results if r.objective(probe) <= floor + OUTCOME_TOL}
                    self.assertEqual(
                        min(tied), k, f"{shape}: beta={probe} reports k={k}, smallest tied argmin is {min(tied)}"
                    )

    def test_every_reported_k_is_achieved_by_a_real_plan(self) -> None:
        # The fictional-point failure mode: the objective at the reported k must
        # be attained by a plan that actually exists with exactly that many
        # subagents, not by a per-k composite of two different plans.
        for shape in SHAPES:
            _, results = self._results(shape)
            for lo, hi, k in optimal_k_intervals(results):
                probe = _probes(lo, hi)[0]
                best = min(r.objective(probe) for r in results)
                attained = min(r.objective(probe) for r in results if r.plan.k == k)
                self.assertAlmostEqual(best, attained, places=9, msg=f"{shape}: k={k}")

    def test_chain_never_rationalizes_spawning(self) -> None:
        _, results = self._results("chain")
        self.assertEqual(optimal_k_intervals(results), [(0.0, math.inf, 0)])

    def test_wide_climbs_with_beta_then_saturates_at_the_cap(self) -> None:
        _, results = self._results("wide")
        spans = optimal_k_intervals(results)
        ks = [k for _, _, k in spans]
        self.assertEqual(ks, sorted(ks), "optimal spawn count must be monotone in beta")
        self.assertEqual(ks[0], 0)
        self.assertEqual(ks[-1], CM.concurrency_cap)  # no reason to brief a worker that cannot run

    def test_serial_is_optimal_at_beta_zero_on_every_shape(self) -> None:
        # True under the current cost model and NOT a result about agents: with
        # latency free and context reuse discounting one big block, serial wins
        # by construction. Context drag is what should break this tie, and it is
        # not implemented -- if this test ever fails, that landed.
        for shape in SHAPES:
            _, results = self._results(shape)
            self.assertEqual(min(results, key=lambda r: r.objective(0.0)).plan.k, 0, shape)


def _probes(lo: float, hi: float) -> list[float]:
    """Interior points of [lo, hi). Boundaries are ties by construction."""
    if hi == math.inf:
        return [lo + 1.0, lo + 1000.0]
    return [(lo + hi) / 2, lo + 0.75 * (hi - lo)]


if __name__ == "__main__":
    unittest.main()


class TestProvenance(unittest.TestCase):
    """The cost model says which of its numbers are earned. That claim has to
    hold structurally, or the record is decoration."""

    def test_every_pricing_field_carries_its_own_provenance(self) -> None:
        # The point of putting status on the VALUE rather than in a table keyed
        # by name: you cannot add a constant and forget to record where it came
        # from, because a bare float reports itself as "unknown" -- which is
        # worse than "placeholder", since nothing flags it for checking.
        from generator.oracle import provenance

        for name, status, _, _ in provenance(CostModel()):
            self.assertNotEqual(status, "unknown", f"{name} has no recorded provenance")

    def test_a_bare_value_reports_unknown_rather_than_passing_silently(self) -> None:
        from generator.oracle import provenance

        cm = dataclasses.replace(CostModel(), spawn_fixed_dollars=0.99)
        statuses = {n: s for n, s, _, _ in provenance(cm)}
        self.assertEqual(statuses["spawn_fixed_dollars"], "unknown")

    def test_default_model_is_uncalibrated_and_names_its_guesses(self) -> None:
        from generator.oracle import unmeasured

        cm = CostModel()
        self.assertFalse(cm.is_calibrated)
        self.assertTrue(unmeasured(cm))
        for name in unmeasured(cm):
            self.assertTrue(hasattr(cm, name))

    def test_calibration_is_incremental_and_keeps_the_protocol(self) -> None:
        # Calibration happens run by run: block curves from one, spawn overhead
        # from another. A per-model flag could not express the middle state.
        from generator.oracle import unmeasured

        cm = CostModel()
        before = len(unmeasured(cm))
        after = cm.calibrate("run-abc | prices 2026-08-23", spawn_fixed_dollars=0.031)
        self.assertEqual(len(unmeasured(after)), before - 1)
        self.assertEqual(after.spawn_fixed_dollars.status, "measured")
        self.assertEqual(after.spawn_fixed_dollars.source, "run-abc | prices 2026-08-23")
        self.assertEqual(float(after.spawn_fixed_dollars), 0.031)
        # the measurement protocol survives the measurement
        self.assertEqual(after.spawn_fixed_dollars.how, cm.spawn_fixed_dollars.how)
        self.assertFalse(after.is_calibrated, "one constant measured is not a calibrated model")

    def test_calibrating_an_unknown_name_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            CostModel().calibrate("run-abc", not_a_constant=1.0)

    def test_measured_values_still_behave_as_numbers(self) -> None:
        cm = CostModel()
        self.assertIsInstance(cm.spawn_fixed_dollars * 3, float)
        self.assertIsInstance(cm.concurrency_cap + 1, int)
        self.assertEqual(len(cm.block_dollars_curve), 1)

    def test_pinned_concurrency_cap_matches_the_harness(self) -> None:
        # Provenance calls this "pinned ... must equal harness.tools.MAX_CONCURRENCY".
        # An oracle allowed more workers than the agent can use would penalize the
        # agent for a constraint it never faced.
        from harness.tools import MAX_CONCURRENCY

        self.assertEqual(CostModel().concurrency_cap, MAX_CONCURRENCY)
