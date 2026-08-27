"""What regret guarantees, and what it refuses to compute.

The dangerous failure here is not a wrong number, it is a plausible number
produced from an unsound run. So most of these tests are about the exclusions:
a scenario whose baseline failed, whose baseline did not execute the plan it was
handed, or whose agent edited the suites, must not be scored at all.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator.dag import sample_dag
from generator.oracle import CostModel, all_inline, enumerate_plans, evaluate, max_fanout
from harness.calibrate import PriceSheet, TimingModel
from harness.trace import LEAD, ModelCall, SpawnRecord, Trace, subagent_actor
from scoring.regret import (
    ScoreCard,
    implied_beta,
    intersect_beta,
    oracle_plan_health,
    quality_by_executor,
    score,
    stakes,
)

CM = CostModel()
PRICE = PriceSheet(
    model="test",
    as_of="2026-08-23",
    input_per_mtok=3.0,
    output_per_mtok=15.0,
    cache_read_per_mtok=0.3,
    cache_write_per_mtok=3.75,
)
# Flat one minute per call: every latency below is countable by hand.
TIMING = TimingModel(a_minutes=1.0, b_minutes_per_input_token=0.0, output_tokens_per_minute=1e12)


def trace(
    scenario_id="s",
    nodes=("n0", "n1", "n2"),
    *,
    calls=3,
    spawns=0,
    passed=True,
    attribution=None,
    tampered=(),
    notes=(),
) -> Trace:
    t = Trace(scenario_id=scenario_id, model="m")
    for i in range(calls):
        t.calls.append(ModelCall(actor=LEAD, index=i, input_tokens=1000, output_tokens=100))
    for i in range(spawns):
        t.spawns.append(SpawnRecord(index=i, batch=0, instruction="", files=()))
        t.calls.append(
            ModelCall(actor=subagent_actor(i), index=0, input_tokens=1000, output_tokens=100)
        )
    t.verdicts = {n: passed for n in nodes}
    t.node_attribution = attribution or {n: LEAD for n in nodes}
    t.tampered_tests = tuple(tampered)
    t.notes = list(notes)
    return t


class TestStakes(unittest.TestCase):
    """The denominator comes off the plan table, with no runs."""

    def test_both_degenerate_policies_are_priced_without_execution(self) -> None:
        dag = sample_dag("wide", 4, sizes=(3,))
        results = [evaluate(dag, p, CM) for p in enumerate_plans(dag)]
        st = stakes(dag, CM, beta=1.0, results=results)
        by_plan = {r.plan: r for r in results}
        self.assertAlmostEqual(
            st.all_inline_objective, by_plan[all_inline(dag)].objective(1.0)
        )
        self.assertAlmostEqual(
            st.max_fanout_objective, by_plan[max_fanout(dag)].objective(1.0)
        )

    def test_the_oracle_is_the_best_plan_and_the_spread_is_non_negative(self) -> None:
        for shape in ("wide", "chain", "diamond"):
            dag = sample_dag(shape, 5, sizes=(3,))
            for beta in (0.0, 0.02, 1.0):
                st = stakes(dag, CM, beta)
                self.assertLessEqual(st.oracle_objective, st.worst_degenerate + 1e-12, shape)
                self.assertGreaterEqual(st.spread, -1e-12, shape)

    def test_wide_has_more_at_stake_than_chain(self) -> None:
        # The reason the denominator exists at all: the same overspend means
        # very different things on the two shapes.
        beta = 1.0
        wide = stakes(sample_dag("wide", 6, sizes=(3,)), CM, beta)
        chain = stakes(sample_dag("chain", 6, sizes=(3,)), CM, beta)
        self.assertGreater(
            wide.spread / wide.oracle_objective, chain.spread / chain.oracle_objective
        )


class TestRegret(unittest.TestCase):
    def setUp(self) -> None:
        self.dag = sample_dag("wide", 3, sizes=(2,))
        self.results = [evaluate(self.dag, p, CM) for p in enumerate_plans(self.dag)]

    def _score(self, agent, baseline, beta=1.0) -> ScoreCard:
        return score(agent, baseline, self.dag, CM, beta, PRICE, TIMING, self.results)

    def test_an_agent_matching_the_baseline_has_zero_regret(self) -> None:
        a, b = trace(calls=4), trace(calls=4)
        card = self._score(a, b)
        self.assertFalse(card.excluded, card.reason)
        self.assertAlmostEqual(card.regret, 0.0)

    def test_regret_is_the_gap_over_the_spread(self) -> None:
        a, b = trace(calls=6), trace(calls=4)
        card = self._score(a, b)
        expected = (
            a.objective(PRICE, TIMING, 1.0) - b.objective(PRICE, TIMING, 1.0)
        ) / card.stakes.spread
        self.assertAlmostEqual(card.regret, expected)
        self.assertGreater(card.regret, 0.0)

    def test_negative_regret_is_reported_not_clipped(self) -> None:
        # An agent beating the executed oracle plan means the decomposition or
        # the cost model is exploitable. That is a result and it gets reported
        # as one; suppressing it would hide the soft-dependency approximation
        # the payload knowingly carries.
        card = self._score(trace(calls=2), trace(calls=6))
        self.assertLess(card.regret, 0.0)

    def test_beating_all_inline_is_a_first_class_flag(self) -> None:
        cheap = self._score(trace(calls=1), trace(calls=1))
        self.assertTrue(cheap.beat_all_inline)
        dear = self._score(trace(calls=400), trace(calls=1))
        self.assertFalse(dear.beat_all_inline)

    def test_beat_all_inline_prefers_the_measured_run_over_the_model_price(self) -> None:
        # The estimation gate showed the model prices all-inline ~13% low,
        # which makes always-serial artificially hard to beat -- a bias toward
        # the "agents cannot beat serial" headline. When an executed inline
        # run exists, its measured objective is the comparison, full stop.
        from scoring.regret import ScoreCard, Stakes

        st = Stakes(oracle_objective=1.0, all_inline_objective=10.0,
                    max_fanout_objective=12.0, oracle_k=0)
        base = dict(scenario_id="s", beta=0.0, agent_objective=5.0, stakes=st)
        self.assertTrue(ScoreCard(**base).beat_all_inline)  # model says 10
        measured = ScoreCard(**base, measured_all_inline=4.9)  # reality says 4.9
        self.assertFalse(measured.beat_all_inline)

    def test_model_best_refuses_to_call_a_near_tie(self) -> None:
        # cal-opus-v2's gate verdict: the model orders extremes correctly and
        # fumbles only near-ties. So inside the noise floor the SIMPLEST plan
        # is executed; outside it, the model's cheapest still wins.
        from generator.oracle import Plan, PlanResult
        from scoring.regret import model_best

        simple = PlanResult(Plan(frozenset({"a", "b"}), ()), cost=1.02, latency=1.0, order=())
        fancy = PlanResult(
            Plan(frozenset(), (frozenset({"a"}), frozenset({"b"}))),
            cost=1.00, latency=1.0, order=(0, 1),
        )
        within = model_best([simple, fancy], 0.0, floors=(0.05, None))
        self.assertEqual(within.plan.k, 0)  # $0.02 apart, floor $0.05: near-tie
        outside = model_best([simple, fancy], 0.0, floors=(0.001, None))
        self.assertEqual(outside.plan.k, 2)  # gap exceeds the floor: model calls it
        no_floors = model_best([simple, fancy], 0.0)
        self.assertEqual(no_floors.plan.k, 2)  # old exact-argmin behaviour


class TestExclusions(unittest.TestCase):
    """What must never be scored. Each of these would produce a plausible number."""

    def setUp(self) -> None:
        self.dag = sample_dag("wide", 3, sizes=(2,))
        self.results = [evaluate(self.dag, p, CM) for p in enumerate_plans(self.dag)]

    def _score(self, agent, baseline) -> ScoreCard:
        return score(agent, baseline, self.dag, CM, 1.0, PRICE, TIMING, self.results)

    def test_a_failed_agent_run_is_excluded(self) -> None:
        # A run that failed half its tasks and spent nothing is not "efficient".
        card = self._score(trace(passed=False, calls=1), trace())
        self.assertTrue(card.excluded)
        self.assertIsNone(card.regret)
        self.assertIn("did not verify", card.reason)

    def test_a_failed_baseline_leaves_the_scenario_with_no_valid_baseline(self) -> None:
        # Section 6.6's procedural consequence, and the one that would quietly
        # break the metric: regret would charge the agent for declining to do
        # something that does not actually work.
        card = self._score(trace(), trace(passed=False))
        self.assertTrue(card.excluded)
        self.assertIn("no valid baseline", card.reason)

    def test_a_baseline_that_ignored_its_plan_is_excluded(self) -> None:
        card = self._score(trace(), trace(notes=["PLAN NOT FOLLOWED: asked for ..."]))
        self.assertTrue(card.excluded)
        self.assertIn("did not execute the requested plan", card.reason)

    def test_a_tampering_run_is_excluded_even_though_it_verified(self) -> None:
        # verify() restores the suites, so a tampering run can still pass. It is
        # still not a measurement of anything.
        card = self._score(trace(tampered=("n0",)), trace())
        self.assertTrue(card.excluded)
        self.assertIn("edited generated suites", card.reason)

    def test_a_missing_baseline_is_excluded_rather_than_defaulted(self) -> None:
        card = self._score(trace(), None)
        self.assertTrue(card.excluded)
        self.assertIn("no baseline", card.reason)

    def test_an_ungraded_run_never_counts_as_a_cheap_success(self) -> None:
        blank = Trace(scenario_id="s", model="m")
        self.assertFalse(blank.succeeded)
        self.assertTrue(self._score(blank, trace()).excluded)

    def test_an_excluded_card_still_carries_the_stakes_and_the_observed_k(self) -> None:
        # Excluded scenarios are reported, never silently dropped -- a quietly
        # shrinking sample is worse than a visible failure count.
        card = self._score(trace(passed=False, spawns=2), trace())
        self.assertIsNotNone(card.stakes)
        self.assertEqual(card.agent_k, 2)


class TestImpliedBeta(unittest.TestCase):
    def setUp(self) -> None:
        self.dag = sample_dag("wide", 6, sizes=(3,))
        self.results = [evaluate(self.dag, p, CM) for p in enumerate_plans(self.dag)]

    def test_every_optimal_k_is_rationalizable_at_its_own_interval(self) -> None:
        from generator.oracle import optimal_k_intervals

        for lo, hi, k in optimal_k_intervals(self.results):
            span = implied_beta(self.dag, CM, k, self.results)
            self.assertIsNotNone(span, k)
            self.assertLessEqual(span[0], lo + 1e-9)
            self.assertGreaterEqual(span[1], hi - 1e-9)

    def test_a_spawn_count_no_price_justifies_returns_none(self) -> None:
        # On a chain, fanning out buys nothing at any latency price. A model
        # that spawned anyway is not irrational at some beta -- it is
        # unrationalizable, and that is the finding.
        chain = sample_dag("chain", 6, sizes=(3,))
        results = [evaluate(chain, p, CM) for p in enumerate_plans(chain)]
        self.assertIsNone(implied_beta(chain, CM, 5, results))
        self.assertIsNotNone(implied_beta(chain, CM, 0, results))

    def test_intersection_across_scenarios_narrows(self) -> None:
        self.assertEqual(intersect_beta([(0.0, 1.0), (0.5, 2.0)]), (0.5, 1.0))
        self.assertEqual(intersect_beta([(0.1, math.inf)]), (0.1, math.inf))

    def test_an_empty_intersection_is_a_result_not_an_error(self) -> None:
        # No single latency price explains the model's choices: its behaviour is
        # intransitive across scenarios. Reported as None, never as a point
        # estimate, because beta-rationalizability is the thing under test.
        self.assertIsNone(intersect_beta([(0.0, 0.1), (0.5, 2.0)]))
        self.assertIsNone(intersect_beta([(0.0, 1.0), None]))
        self.assertIsNone(intersect_beta([]))


class TestCurrencyGuard(unittest.TestCase):
    """The numerator is measured; the denominator is computed. If the cost model
    is not calibrated they are not in the same units, and the ratio is a clean
    number that means nothing. Section 6.4's "an error in the denominator merely
    scales regret" is sound but assumes one currency."""

    def setUp(self) -> None:
        self.dag = sample_dag("wide", 3, sizes=(2,))
        self.results = [evaluate(self.dag, p, CM) for p in enumerate_plans(self.dag)]

    def test_an_uncalibrated_cost_model_marks_the_card_not_comparable(self) -> None:
        card = score(trace(), trace(), self.dag, CM, 1.0, PRICE, TIMING, self.results)
        self.assertFalse(CM.is_calibrated)
        self.assertFalse(card.excluded)  # still computed, for structure
        self.assertFalse(card.comparable)
        self.assertTrue(any("NOT COMPARABLE" in w for w in card.warnings))

    def test_a_calibrated_model_carries_no_warning(self) -> None:
        cm = CM
        for name in __import__("generator.oracle", fromlist=["x"]).unmeasured(cm):
            if name == "beta":
                continue
            cm = cm.calibrate("test-run 2026-08-23", **{name: getattr(cm, name)})
        card = score(trace(), trace(), self.dag, cm, 1.0, PRICE, TIMING, self.results)
        self.assertTrue(cm.is_calibrated)
        self.assertEqual(card.warnings, ())
        self.assertTrue(card.comparable)

    def test_the_warning_survives_onto_excluded_cards_too(self) -> None:
        card = score(trace(passed=False), trace(), self.dag, CM, 1.0, PRICE, TIMING, self.results)
        self.assertTrue(card.excluded)
        self.assertTrue(card.warnings)


class TestQualityChecks(unittest.TestCase):
    def test_pass_rate_splits_by_who_did_the_work(self) -> None:
        # Answers "maybe delegation produces worse work" from data already
        # collected.
        t = Trace(scenario_id="s", model="m")
        t.node_attribution = {"n0": LEAD, "n1": subagent_actor(0), "n2": subagent_actor(1)}
        t.verdicts = {"n0": True, "n1": True, "n2": False}
        q = quality_by_executor([t])
        self.assertEqual((q.lead_passed, q.lead_total), (1, 1))
        self.assertEqual((q.subagent_passed, q.subagent_total), (1, 2))
        self.assertAlmostEqual(q.rate("subagent"), 0.5)

    def test_unattributed_nodes_count_against_neither_executor(self) -> None:
        t = Trace(scenario_id="s", model="m")
        t.node_attribution = {"n0": LEAD}
        t.verdicts = {"n0": True, "n1": False}  # n1 was never written
        q = quality_by_executor([t])
        self.assertEqual(q.lead_total + q.subagent_total, 1)

    def test_oracle_plans_failing_more_often_invalidates_the_metric(self) -> None:
        agents = [trace(passed=True) for _ in range(10)]
        oracles = [trace(passed=i < 5) for i in range(10)]
        health = oracle_plan_health(agents, oracles)
        self.assertAlmostEqual(health.oracle_pass_rate, 0.5)
        self.assertIn("INVALID", health.verdict)

    def test_matching_pass_rates_mean_regret_is_pure_economics(self) -> None:
        agents = [trace(passed=True) for _ in range(5)]
        oracles = [trace(passed=True) for _ in range(5)]
        self.assertIn("pure economics", oracle_plan_health(agents, oracles).verdict)


class TestEndToEnd(unittest.TestCase):
    """Stages 1 through 6, in one pass, with no API key.

    Every other test here checks one joint. This one checks that the joints
    connect: sample a DAG, build the payload, price the plan table, run a model
    against the workspace, run the oracle plan as the baseline, grade both
    independently, and divide. If the pipeline has a gap, it shows up here as an
    exception or an exclusion rather than as a number nobody can reproduce.
    """

    def test_a_scenario_produces_a_regret_number(self) -> None:
        import tempfile

        from generator.scenario import build_scenario
        from generator.templates import module_path
        from harness.client import Reply, ScriptedClient, ToolRequest
        from harness.runner import run_agent, run_plan

        dag = sample_dag("wide", 3, sizes=(1,), seed=4)
        scn = build_scenario(dag, "e2e", seed=4)
        results = [evaluate(dag, p, CM) for p in enumerate_plans(dag)]
        beta = 1.0
        oracle = min(results, key=lambda r: (r.objective(beta), r.plan.k))

        def tool(name, args, id="t"):
            return ToolRequest(id=id, name=name, arguments=args)

        def rep(*calls):
            return Reply(tool_calls=tuple(calls), input_tokens=800, output_tokens=120, total_s=1.0)

        def fix(node):
            return tool("write_file", {"path": module_path(node), "content": scn.reference[node]},
                        id=f"w{node}")

        done = tool("finish", {"summary": "done"}, id="f")

        # The agent fans out one subagent per node -- the over-spawning
        # behaviour the benchmark exists to price.
        agent_script = {
            "lead": [
                rep(*[tool("spawn_subagent", {"instruction": f"fix {n}", "files": []}, id=f"s{n}")
                      for n in dag.ids]),
                rep(done),
            ]
        }
        for n in dag.ids:
            agent_script[f"fix {n}"] = [rep(fix(n)), rep(done)]

        # The baseline follows the oracle plan: inline nodes itself, one spawn
        # per block.
        base_script = {"lead": [rep(
            *[fix(n) for n in sorted(oracle.plan.inline)],
            *[tool("spawn_subagent", {"instruction": f"block {i}", "files": []}, id=f"b{i}")
              for i in range(oracle.plan.k)],
        ), rep(done)]}
        for i, block in enumerate(oracle.plan.blocks):
            base_script[f"block {i}"] = [rep(*[fix(n) for n in sorted(block)]), rep(done)]

        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            agent = run_agent(scn, ScriptedClient(script=agent_script), a)
            baseline = run_plan(scn, oracle.plan, ScriptedClient(script=base_script), b,
                                order=oracle.order)

        self.assertTrue(agent.succeeded, agent.verdicts)
        self.assertTrue(baseline.succeeded, baseline.verdicts)
        self.assertFalse([n for n in baseline.notes if n.startswith("PLAN NOT FOLLOWED")],
                         baseline.notes)

        card = score(agent, baseline, dag, CM, beta, PRICE, TIMING, results)
        self.assertFalse(card.excluded, card.reason)
        self.assertIsInstance(card.regret, float)
        self.assertEqual(card.agent_k, len(dag.ids))
        self.assertEqual(card.oracle_k, oracle.plan.k)

        # The two quality checks run off the same traces, no extra work.
        q = quality_by_executor([agent, baseline])
        self.assertEqual(q.lead_total + q.subagent_total, 2 * len(dag.ids))
        self.assertIn("pure economics", oracle_plan_health([agent], [baseline]).verdict)


if __name__ == "__main__":
    unittest.main()
