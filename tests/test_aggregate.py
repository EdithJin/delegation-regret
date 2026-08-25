"""Aggregation, the manifest, and the two method checks.

The failure mode these guard against is a number that looks like a result and is
not: a pooled mean whose weights nobody chose, an interval narrowed by counting
correlated repeats as independent, or a discovery-cost effect produced entirely by
comparing two different scenario sets.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator.dag import sample_dag
from generator.manifest import (
    ANCHOR,
    CORE,
    CORE_SEED_MAX,
    HELDOUT_SEED_MIN,
    Manifest,
    ScenarioSpec,
    heldout,
    load,
    write,
)
from generator.oracle import CostModel, enumerate_plans, evaluate, max_fanout
from scoring.aggregate import aggregate, beats_all_inline_rate, cluster_bootstrap
from scoring.regret import ScoreCard, Stakes
from scoring.validate import OrderingReport, anti_oracle, check_ordering, discovery_cost

CM = CostModel()


def card(scenario_id, regret, *, beta=1.0, excluded=False, reason="", k=2,
         all_inline=10.0, agent=5.0, warnings=()):
    return ScoreCard(
        scenario_id=scenario_id,
        beta=beta,
        regret=None if excluded else regret,
        agent_objective=agent,
        baseline_objective=agent,
        stakes=Stakes(oracle_objective=1.0, all_inline_objective=all_inline,
                      max_fanout_objective=4.0, oracle_k=1),
        agent_k=k,
        oracle_k=1,
        excluded=excluded,
        reason=reason,
        warnings=warnings,
        agent_passed=not excluded,
        baseline_passed=not excluded,
    )


class TestManifest(unittest.TestCase):
    """The mix is the one thing that cannot be regenerated from a seed."""

    def test_the_core_mix_is_equal_across_shapes(self) -> None:
        counts = {s: len(v) for s, v in CORE.by_shape().items()}
        self.assertEqual(len(set(counts.values())), 1, counts)
        self.assertEqual(set(counts), {"wide", "chain", "diamond", "mixed"})

    def test_the_fingerprint_changes_when_the_spec_list_does(self) -> None:
        # This is what makes pre-registration checkable: a mix widened after
        # seeing an unflattering aggregate leaves a trace.
        widened = Manifest(name="core", specs=CORE.specs + (ScenarioSpec("wide", 4, 3, 99),))
        self.assertNotEqual(widened.fingerprint, CORE.fingerprint)

    def test_heldout_seeds_cannot_collide_with_core(self) -> None:
        self.assertLessEqual(max(s.seed for s in CORE.specs), CORE_SEED_MAX)
        self.assertGreaterEqual(min(s.seed for s in heldout(20).specs), HELDOUT_SEED_MIN)

    def test_successive_heldout_refreshes_are_disjoint(self) -> None:
        first, second = heldout(6), heldout(6, offset=6)
        self.assertFalse({s.seed for s in first.specs} & {s.seed for s in second.specs})

    def test_specs_regenerate_the_same_scenario_every_time(self) -> None:
        spec = ANCHOR.specs[0]
        a, b = spec.build(), spec.build()
        self.assertEqual(a.modules, b.modules)
        self.assertEqual(a.tests, b.tests)
        self.assertEqual(a.id, spec.id)

    def test_a_tampered_manifest_file_refuses_to_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write(ANCHOR, Path(tmp) / "m.json")
            raw = json.loads(path.read_text())
            raw["specs"].append({"shape": "wide", "n": 5, "size": 1, "seed": 7})
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError) as caught:
                load(path)
            self.assertIn("fingerprint mismatch", str(caught.exception))

    def test_a_bad_spec_is_rejected_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            ScenarioSpec("triangle", 4, 1, 0)
        with self.assertRaises(ValueError):
            ScenarioSpec("diamond", 2, 1, 0)


class TestBootstrap(unittest.TestCase):
    def test_repeats_within_a_scenario_do_not_narrow_the_interval(self) -> None:
        # The rule this enforces: the sample size that matters is the number of
        # SCENARIOS. Counting each repeat as independent would shrink the
        # interval the more repeats you add, so more data would look like more
        # certainty while saying nothing new about the population.
        two_scenarios = [[0.1] * 20, [0.9] * 20]
        wide = cluster_bootstrap(two_scenarios, resamples=500, seed=1)
        flat = cluster_bootstrap([[0.1], [0.9]], resamples=500, seed=1)
        self.assertEqual(wide, flat)

    def test_more_scenarios_do_narrow_it(self) -> None:
        few = cluster_bootstrap([[0.1], [0.9]], resamples=800, seed=2)
        many = cluster_bootstrap([[0.5]] * 40, resamples=800, seed=2)
        self.assertLess(many[1] - many[0], few[1] - few[0])

    def test_one_cluster_yields_no_interval(self) -> None:
        self.assertIsNone(cluster_bootstrap([[0.4]], resamples=100))
        self.assertIsNone(cluster_bootstrap([], resamples=100))

    def test_the_interval_is_reproducible(self) -> None:
        clusters = [[0.2, 0.3], [0.8], [0.5, 0.4, 0.6]]
        self.assertEqual(
            cluster_bootstrap(clusters, resamples=400, seed=7),
            cluster_bootstrap(clusters, resamples=400, seed=7),
        )


class TestAggregate(unittest.TestCase):
    def setUp(self) -> None:
        self.shape_of = lambda sid: sid.split("-")[0]

    def test_per_class_summaries_are_produced_per_shape(self) -> None:
        cards = [
            card("wide-1", 0.2), card("wide-2", 0.4),
            card("chain-1", 0.0), card("chain-2", 0.1),
        ]
        agg = aggregate(cards, self.shape_of, resamples=200)
        self.assertEqual([c.shape for c in agg.per_class], ["chain", "wide"])
        wide = next(c for c in agg.per_class if c.shape == "wide")
        self.assertAlmostEqual(wide.mean_regret, 0.3)

    def test_pooling_averages_class_means_not_scenarios(self) -> None:
        # Nine wide scenarios and one chain: pooling scenarios would give wide
        # nine times the weight the manifest gave it.
        cards = [card(f"wide-{i}", 1.0) for i in range(9)] + [card("chain-1", 0.0)]
        agg = aggregate(cards, self.shape_of, resamples=200)
        self.assertAlmostEqual(agg.pooled_regret, 0.5)  # not 0.9

    def test_the_pooled_line_says_the_mix_is_not_a_property_of_the_metric(self) -> None:
        agg = aggregate([card("wide-1", 0.2), card("chain-1", 0.4)], self.shape_of,
                        resamples=200)
        self.assertIn("unweighted mean", str(agg))
        self.assertIn("the mix is the manifest", str(agg))

    def test_exclusions_are_counted_and_their_reasons_kept(self) -> None:
        cards = [
            card("wide-1", 0.2),
            card("wide-2", None, excluded=True, reason="oracle plan failed to verify"),
            card("wide-3", None, excluded=True, reason="oracle plan failed to verify"),
        ]
        agg = aggregate(cards, self.shape_of, resamples=200)
        wide = agg.per_class[0]
        self.assertEqual(wide.n_excluded, 2)
        self.assertEqual(wide.exclusion_reasons["oracle plan failed to verify"], 2)
        self.assertEqual(agg.n_excluded, 2)

    def test_negative_regret_scenarios_are_counted_separately(self) -> None:
        cards = [card("chain-1", -0.3), card("chain-2", 0.4)]
        agg = aggregate(cards, self.shape_of, resamples=200)
        # The signature of the soft-dependency approximation mattering, which the
        # payload docstring says to watch for rather than assume away.
        self.assertEqual(agg.per_class[0].negative_regret_scenarios, 1)

    def test_beats_all_inline_is_reported_as_a_count(self) -> None:
        cards = [card("wide-1", 0.1, agent=5.0, all_inline=10.0),
                 card("wide-2", 0.1, agent=20.0, all_inline=10.0)]
        self.assertEqual(beats_all_inline_rate(cards), (1, 2))

    def test_an_uncomparable_card_makes_the_aggregate_uncomparable(self) -> None:
        agg = aggregate([card("wide-1", 0.2, warnings=("NOT COMPARABLE: ...",))],
                        self.shape_of, resamples=200)
        self.assertFalse(agg.comparable)
        self.assertIn("NOT COMPARABLE", str(agg))

    def test_mixing_betas_in_one_aggregate_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            aggregate([card("wide-1", 0.1, beta=0.0), card("wide-2", 0.1, beta=1.0)],
                      self.shape_of)


class TestAntiOracle(unittest.TestCase):
    """The only check in the repo that can falsify the method."""

    def setUp(self) -> None:
        self.dag = sample_dag("wide", 4, sizes=(3,))
        self.results = [evaluate(self.dag, p, CM) for p in enumerate_plans(self.dag)]

    def test_the_anti_oracle_is_the_true_worst_not_max_fanout(self) -> None:
        beta = 1.0
        worst = anti_oracle(self.dag, CM, beta, self.results)
        by_plan = {r.plan: r for r in self.results}
        fanout = by_plan[max_fanout(self.dag)]
        self.assertGreaterEqual(worst.objective(beta), fanout.objective(beta))
        for r in self.results:
            self.assertLessEqual(r.objective(beta), worst.objective(beta) + 1e-12)

    def test_agreement_is_judged_on_sign_not_magnitude(self) -> None:
        beta = 1.0
        best = min(self.results, key=lambda r: r.objective(beta))
        worst = anti_oracle(self.dag, CM, beta, self.results)
        # Measured gap far smaller than predicted, but the same direction: the
        # cost model's job here is to RANK, and it did.
        ok = check_ordering("s", beta, best, worst, measured_oracle=1.0, measured_anti=1.01)
        self.assertTrue(ok.agreed)
        bad = check_ordering("s", beta, best, worst, measured_oracle=2.0, measured_anti=1.0)
        self.assertFalse(bad.agreed)

    def test_an_unverified_run_is_inconclusive_not_a_failure(self) -> None:
        beta = 1.0
        best = min(self.results, key=lambda r: r.objective(beta))
        worst = anti_oracle(self.dag, CM, beta, self.results)
        check = check_ordering("s", beta, best, worst, 1.0, 2.0, anti_ok=False)
        self.assertIsNone(check.agreed)
        self.assertIn("anti-oracle run did not verify", check.reason)

    def test_a_failed_ordering_is_reported_as_invalidating(self) -> None:
        beta = 1.0
        best = min(self.results, key=lambda r: r.objective(beta))
        worst = anti_oracle(self.dag, CM, beta, self.results)
        report = OrderingReport([
            check_ordering("a", beta, best, worst, 1.0, 2.0),
            check_ordering("b", beta, best, worst, 2.0, 1.0),
        ])
        self.assertIn("ORDERING FAILED", report.verdict)
        self.assertIn("invalidates the derived-ground-truth argument", report.verdict)

    def test_all_inconclusive_says_so_rather_than_claiming_support(self) -> None:
        beta = 1.0
        best = min(self.results, key=lambda r: r.objective(beta))
        worst = anti_oracle(self.dag, CM, beta, self.results)
        report = OrderingReport([check_ordering("a", beta, best, worst, None, None)])
        self.assertIn("INCONCLUSIVE", report.verdict)


class TestDiscoveryCost(unittest.TestCase):
    def test_the_split_uses_only_scenarios_both_arms_scored(self) -> None:
        # Comparing arm means over different scenario sets could produce the whole
        # effect from the shapes' differing spreads alone.
        hidden = [card("wide-1", 0.6), card("wide-2", 0.4), card("chain-9", 0.9)]
        disclosed = [card("wide-1", 0.3), card("wide-2", 0.1),
                     card("chain-9", None, excluded=True, reason="failed")]
        split = discovery_cost(hidden, disclosed)
        self.assertEqual(split.n_hidden, 2)
        self.assertAlmostEqual(split.hidden_regret, 0.5)
        self.assertAlmostEqual(split.disclosed_regret, 0.2)
        self.assertAlmostEqual(split.discovery_share, 0.6)

    def test_no_shared_scenarios_yields_no_split(self) -> None:
        split = discovery_cost([card("wide-1", 0.5)], [card("chain-1", 0.2)])
        self.assertIsNone(split.hidden_regret)
        self.assertIn("not enough scored runs", str(split))

    def test_non_positive_hidden_regret_leaves_the_share_undefined(self) -> None:
        # A ratio over a zero or negative denominator is not "large", it is
        # meaningless -- the same class of error as an uncalibrated denominator.
        split = discovery_cost([card("wide-1", -0.2)], [card("wide-1", -0.4)])
        self.assertIsNone(split.discovery_share)
        self.assertIn("undefined", str(split))


if __name__ == "__main__":
    unittest.main()
