"""The committee audit, exercised without spending anything.

The audit is the machinery behind the headline X% -- the fraction of the plan
table the calculation may safely discard -- so its math is pinned offline: the
predicted-outcome collapse, the 2-eps band, the prospectively specified scorecard rules, and the
compliance gating that keeps an excluded run from quietly becoming a survivor.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator.dag import sample_dag
from generator.oracle import Plan, PlanResult, enumerate_plans
from generator.scenario import build_scenario
from harness.audit import (
    AuditResult,
    allocation_shape_classes,
    allocation_shape_key,
    audit_summary,
    committee_band,
    distinct_outcomes,
    eps_from_calibration,
    kendall_tau,
    outcome_key,
    run_audit,
)
from harness.calibrate import PriceSheet, TimingModel
from harness.calibration import CalibrationResult
from harness.trace import LEAD, ModelCall, Trace

PRICE = PriceSheet("fake", "2026-08-23", 3.0, 15.0, 0.3, 3.75)
TIMING = TimingModel(0.001, 0.0, 1e12)
FLOORS = (0.05, 0.05)


def pr(cost, latency, k=0, tag="a") -> PlanResult:
    blocks = tuple(frozenset({f"{tag}{i}"}) for i in range(k))
    inline = frozenset({f"{tag}-inline"}) if k == 0 else frozenset()
    return PlanResult(Plan(inline, blocks), cost=cost, latency=latency, order=tuple(range(k)))


class TestPureMachinery(unittest.TestCase):
    def test_wide4_has_52_labeled_plans_and_12_structural_shapes(self) -> None:
        dag = sample_dag("wide", 4, sizes=(15,), seed=11)
        plans = enumerate_plans(dag)
        classes = allocation_shape_classes(dag, plans)

        self.assertEqual(len(plans), 52)
        self.assertEqual(len(classes), 12)
        self.assertEqual(
            sorted(len(members) for members in classes.values()),
            [1, 1, 1, 3, 4, 4, 4, 4, 6, 6, 6, 12],
        )
        self.assertEqual(sum(len(members) for members in classes.values()), 52)

    def test_shape_key_refuses_nonwide_or_heterogeneous_dags(self) -> None:
        chain = sample_dag("chain", 3, sizes=(1,), seed=7)
        heterogeneous = sample_dag("wide", 3, sizes=(1, 2), seed=7)
        self.assertIsNone(allocation_shape_key(chain, enumerate_plans(chain)[0]))
        self.assertIsNone(
            allocation_shape_key(heterogeneous, enumerate_plans(heterogeneous)[0])
        )

    def test_same_predicted_bucket_selects_the_simplest_plan(self) -> None:
        # Two plans the model cannot distinguish at floor resolution: one
        # representative, and it is the simpler plan scoring would execute.
        a = pr(1.00, 2.00, k=2, tag="x")
        b = pr(1.01, 2.01, k=0, tag="y")  # within the 0.05 floors of a
        reps = distinct_outcomes([a, b], FLOORS)
        self.assertEqual(len(reps), 1)
        self.assertEqual(reps[0].plan.k, 0)
        self.assertEqual(outcome_key(a, FLOORS), outcome_key(b, FLOORS))

    def test_distinct_predicted_outcomes_stay_distinct(self) -> None:
        reps = distinct_outcomes([pr(1.0, 1.0), pr(2.0, 1.0), pr(1.0, 3.0)], FLOORS)
        self.assertEqual(len(reps), 3)

    def test_the_band_always_contains_the_predicted_minimum_and_grows_with_eps(self) -> None:
        reps = [pr(1.0, 1.0), pr(1.2, 1.2), pr(3.0, 3.0)]
        tight = committee_band(reps, 1.0, 0.0)
        wide = committee_band(reps, 1.0, 0.5)
        widest = committee_band(reps, 1.0, 1.0)
        self.assertIn(min(reps, key=lambda r: r.objective(1.0)), tight)
        self.assertTrue(set(id(r) for r in tight) <= set(id(r) for r in wide))
        self.assertEqual(len(tight), 1)
        self.assertEqual(len(wide), 2)    # (1+2*0.5)*2.0 = 4.0 excludes the 6.0 plan
        self.assertEqual(len(widest), 3)  # (1+2*1.0)*2.0 = 6.0 reaches it, inclusive

    def test_kendall_tau_signs_and_small_n_refusal(self) -> None:
        self.assertAlmostEqual(kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(kendall_tau([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)
        self.assertIsNone(kendall_tau([1, 2], [2, 1]))

    def test_eps_is_read_off_the_estimation_gate_never_invented(self) -> None:
        calib = CalibrationResult(
            source="s",
            diagnostics={"composition": {"checked": True, "arms": [
                {"rel_gap_dollars": -0.13, "rel_gap_minutes": -0.12},
                {"rel_gap_dollars": -0.18, "rel_gap_minutes": -0.41},
            ]}},
        )
        self.assertAlmostEqual(eps_from_calibration(calib), 0.41)
        self.assertIsNone(eps_from_calibration(CalibrationResult(source="s")))


def scenario():
    return build_scenario(sample_dag("wide", 3, sizes=(1,), seed=7), "aud3", seed=7)


def fake_runner(comply_spawns=True, cost_of=None):
    """A runner returning synthetic-but-shaped traces: measured cost grows with
    the plan's spawn count unless overridden, so serial is the true champion."""
    calls = {"n": 0}

    def runner(scn, plan, client, root, **kwargs):
        calls["n"] += 1
        t = Trace(scenario_id=scn.id, model="m", condition=kwargs.get("condition", ""))
        tokens = (cost_of(plan) if cost_of else 1000 * (1 + plan.k))
        t.calls.append(ModelCall(actor=LEAD, index=0, input_tokens=tokens,
                                 output_tokens=100, total_s=1.0, t_request=0.0))
        t.verdicts = {v: True for v in scn.dag.ids}
        t.finished = True
        if plan.k and not comply_spawns:
            t.notes.append("PLAN NOT FOLLOWED: ran everything inline")
        return t

    runner.calls = calls
    return runner


class TestRunAudit(unittest.TestCase):
    def _go(self, **kw):
        from generator.oracle import CostModel

        scn = scenario()
        with tempfile.TemporaryDirectory() as out:
            result = run_audit(scn, None, PRICE, TIMING, CostModel(), out,
                               beta=1.0, eps=kw.pop("eps", None), floors=(0.01, 0.01),
                               runner=kw.pop("runner", fake_runner()), **kw)
            reloaded = AuditResult.load(Path(out) / "audit.json")
        return result, reloaded

    def test_dry_run_prices_the_table_and_spends_nothing(self) -> None:
        runner = fake_runner()
        result, reloaded = self._go(dry_run=True, runner=runner)
        self.assertEqual(runner.calls["n"], 0)
        self.assertTrue(result.rows)
        self.assertTrue(all(r.measured_objective is None for r in result.rows))
        self.assertEqual(reloaded.n_outcomes, result.n_outcomes)
        self.assertEqual(reloaded.n_shape_classes, result.n_shape_classes)
        self.assertEqual(
            sum(row["labeled_plan_count"] for row in reloaded.shape_classes),
            result.n_plans,
        )
        self.assertIn("DRY RUN", result.report())

    def test_the_scorecard_applies_the_prespecified_rules(self) -> None:
        # Serial measures cheapest (fake runner: cost grows with k). Under the
        # placeholder model serial is also predicted best, so r=1 and
        # x = 1 - 1/N exactly -- the rule, not a curve fit.
        result, _ = self._go()
        self.assertTrue(result.complete)
        champ = next(r for r in result.rows if r.tag == result.champion_tag)
        self.assertEqual(champ.k, 0)
        self.assertEqual(result.champion_predicted_rank, 1)
        self.assertAlmostEqual(result.safe_filter_x, 1.0 - 1.0 / result.n_outcomes)
        self.assertIsNotNone(result.tau)

    def test_noncompliant_runs_are_excluded_and_the_audit_says_incomplete(self) -> None:
        result, _ = self._go(runner=fake_runner(comply_spawns=False))
        excluded = [r for r in result.rows if r.excluded]
        self.assertTrue(excluded)
        self.assertTrue(all(r.k > 0 for r in excluded))
        self.assertTrue(all("assigned plan" in r.reason for r in excluded))
        self.assertFalse(result.complete)
        self.assertTrue(any("INCOMPLETE" in n for n in result.notes))
        # the champion is judged over measured rows only, and says so
        self.assertEqual(
            next(r.k for r in result.rows if r.tag == result.champion_tag), 0)

    def test_band_only_refuses_a_safe_filter_rate(self) -> None:
        # The X% is only meaningful if the discarded region was checked.
        result, _ = self._go(band_only=True, eps=0.1)
        self.assertIsNone(result.safe_filter_x)
        self.assertTrue(any("band-only" in n for n in result.notes))


class TestAuditSummary(unittest.TestCase):
    def test_x_is_the_worst_case_never_the_mean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for name, r, n in (("s1", 1, 10), ("s2", 3, 10)):
                a = AuditResult(scenario_id=name, beta=1.0, eps=0.3, floors=(0.01, 0.01),
                                n_plans=50, n_outcomes=n, n_band=4,
                                champion_tag="plan000-k0", champion_predicted_rank=r,
                                safe_filter_x=1.0 - r / n, champion_in_band=True,
                                tau=0.8, complete=True, spend_dollars=1.0)
                a.write(Path(tmp) / name)
            s = audit_summary(sorted(Path(tmp).glob("*/audit.json")))
        self.assertEqual(s["champion_retention"], "2/2")
        self.assertAlmostEqual(s["safe_filter_X"], 0.7)  # min(0.9, 0.7)
        self.assertEqual(s["per_scenario"][0]["scenario"], "s2")  # worst first

    def test_an_incomplete_audit_blocks_a_certified_x(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            good = AuditResult(scenario_id="ok", beta=1.0, eps=0.3, floors=(0.01, 0.01),
                               n_plans=10, n_outcomes=5, n_band=2,
                               champion_tag="plan000-k0", champion_predicted_rank=1,
                               safe_filter_x=0.8, champion_in_band=True,
                               tau=0.9, complete=True, spend_dollars=1.0)
            bad = AuditResult(scenario_id="holey", beta=1.0, eps=0.3, floors=(0.01, 0.01),
                              n_plans=10, n_outcomes=5, n_band=2,
                              champion_tag="plan001-k1", champion_predicted_rank=2,
                              safe_filter_x=0.6, champion_in_band=True,
                              tau=0.5, complete=False, spend_dollars=1.0)
            good.write(Path(tmp) / "ok")
            bad.write(Path(tmp) / "holey")
            s = audit_summary(sorted(Path(tmp).glob("*/audit.json")))
        self.assertIsNone(s["safe_filter_X"])
        self.assertTrue(s["caveats"])


if __name__ == "__main__":
    unittest.main()
