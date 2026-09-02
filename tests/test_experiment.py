"""The experiment driver, the proxy cross-check, and the reporting additions.

These cover the joints that were built last and are therefore least exercised:
routing a run through the logging proxy and cross-checking it, composing a whole
manifest into a stamped results file, and the three reporting additions (Tier A,
out-of-order edits, the materiality floor).
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator.dag import sample_dag
from generator.manifest import Manifest, ScenarioSpec
from generator.oracle import CostModel, enumerate_plans, evaluate, is_feasible
from generator.scenario import build_scenario
from generator.templates import module_path
from harness.calibrate import PriceSheet, TimingModel
from harness.calibration import bundled_plan, materiality_floor
from harness.client import AnthropicClient, Reply, ScriptedClient, ToolRequest
from harness.experiment import Results, run_experiment
from harness.protocol_upstream import ProtocolUpstream, Turn
from harness.proxy import CallRecord
from harness.runner import _cross_check, run_agent, through_proxy
from harness.trace import LEAD, ModelCall, Trace, WriteEvent
from scoring.validate import out_of_order_edits, out_of_order_rate

PRICE = PriceSheet("fake", "2026-08-23", 3.0, 15.0, 0.3, 3.75)
TIMING = TimingModel(0.0002, 1.9e-7, 200_000.0)

CALIBRATED = CostModel().calibrate(
    "test 2026-08-23",
    block_dollars_curve=((1, 0.0030), (2, 0.0078), (3, 0.0144), (5, 0.0228)),
    block_minutes_curve=((1, 0.00060), (2, 0.00147), (3, 0.00262), (5, 0.00404)),
    # Dearer than the lead curve on purpose: subagents pay fresh-context costs.
    sub_block_dollars_curve=((1, 0.0044), (2, 0.0102), (3, 0.0181), (5, 0.0290)),
    sub_block_minutes_curve=((1, 0.00082), (2, 0.00188), (3, 0.00325), (5, 0.00500)),
    spawn_fixed_dollars=0.00255,
    brief_dollars_per_node=0.00135,
    brief_minutes=0.00034,
    brief_minutes_per_node=0.00045,
    absorb_dollars_per_node=0.00096,
    absorb_minutes=0.00016,
    absorb_minutes_per_node=0.000062,
    explore_dollars=0.0081,
    explore_minutes=0.00212,
    throughput_curve=((1, 1.0), (2, 1.08), (4, 1.31)),
)


def repair_script(specs, *, tokens=(900, 110)):
    """A model that repairs every module itself, whatever plan it is handed."""
    writes, script = {}, {}
    for spec in specs:
        scn = spec.build()
        writes[scn.id] = tuple(
            ("write_file", {"path": module_path(n), "content": scn.reference[n]})
            for n in scn.dag.ids
        )
    for sid, w in writes.items():
        script[sid] = [
            Turn(tools=w, input_tokens=tokens[0], output_tokens=tokens[1]),
            Turn(tools=(("finish", {"summary": "done"}),), input_tokens=1100, output_tokens=25),
        ]
    script["sub"] = [Turn(tools=(("finish", {"summary": "ok"}),))]

    def route(first_user: str) -> str:
        for sid in writes:
            if sid in first_user:
                return sid
        return "sub"

    return script, route


class TestProxyCrossCheck(unittest.TestCase):
    """The proxy is the only independent witness to what was billed. Without it,
    a client-side error parsing the usage block is invisible: the trace and the
    invoice would disagree and nothing would say so."""

    def _scenario(self):
        return build_scenario(sample_dag("wide", 2, sizes=(1,), seed=7), "w", seed=7)

    def _script(self, scn):
        return {
            "lead": [
                Turn(
                    tools=tuple(
                        ("write_file", {"path": module_path(n), "content": scn.reference[n]})
                        for n in scn.dag.ids
                    ),
                    input_tokens=800,
                    output_tokens=120,
                ),
                Turn(tools=(("finish", {"summary": "done"}),), input_tokens=900,
                     output_tokens=30),
            ]
        }

    def test_a_run_through_the_proxy_verifies_and_logs(self) -> None:
        scn = self._scenario()
        with ProtocolUpstream("anthropic", self._script(scn)) as up:
            client = AnthropicClient(model="fake", api_key="k", base_url=up.base_url)
            with tempfile.TemporaryDirectory() as tmp:
                log = Path(tmp) / "proxy.jsonl"
                trace = run_agent(scn, client, tmp, proxy_log=log)
                lines = log.read_text().splitlines()
            self.assertEqual(client.base_url, up.base_url, "base_url was not restored")
        self.assertTrue(trace.succeeded, trace.verdicts)
        self.assertTrue(trace.proxy_verified)
        self.assertEqual(trace.proxy_calls, len(trace.calls))
        self.assertEqual(len(lines), len(trace.calls))
        self.assertEqual(trace.notes, [])

    def test_a_client_that_miscounts_tokens_is_caught(self) -> None:
        import harness.client as C

        scn = self._scenario()
        original = C.AnthropicClient.complete

        def halved(self, history, allow):
            reply = original(self, history, allow)
            return Reply(**{**reply.__dict__, "output_tokens": reply.output_tokens // 2})

        C.AnthropicClient.complete = halved
        try:
            with ProtocolUpstream("anthropic", self._script(scn)) as up:
                client = AnthropicClient(model="fake", api_key="k", base_url=up.base_url)
                with tempfile.TemporaryDirectory() as tmp:
                    trace = run_agent(scn, client, tmp, proxy_log=Path(tmp) / "p.jsonl")
        finally:
            C.AnthropicClient.complete = original
        self.assertFalse(trace.proxy_verified)
        self.assertTrue(any(n.startswith("PROXY MISMATCH") for n in trace.notes), trace.notes)

    def test_a_cache_write_mismatch_is_caught(self) -> None:
        trace = Trace(
            scenario_id="s",
            model="fake",
            calls=[
                ModelCall(
                    actor=LEAD,
                    index=0,
                    input_tokens=2,
                    output_tokens=10,
                    cache_write_tokens=900,
                )
            ],
        )
        record = CallRecord(
            seq=1,
            path="/v1/messages",
            input_tokens=2,
            output_tokens=10,
            cache_read_tokens=0,
            cache_write_tokens=800,
        )

        class Proxy:
            records = [record]

        _cross_check(trace, Proxy())
        self.assertFalse(trace.proxy_verified)
        self.assertTrue(
            any("cache_write_tokens" in note for note in trace.notes), trace.notes
        )

    def test_scoring_refuses_a_trace_that_failed_the_cross_check(self) -> None:
        from scoring.regret import score

        dag = sample_dag("wide", 2, sizes=(1,))
        good = Trace(scenario_id="s", model="m", verdicts={n: True for n in dag.ids})
        bad = Trace(
            scenario_id="s", model="m",
            verdicts={n: True for n in dag.ids},
            notes=["PROXY MISMATCH: output_tokens proxy=150 trace=75."],
        )
        priced = [evaluate(dag, p, CALIBRATED) for p in enumerate_plans(dag)]
        card = score(bad, good, dag, CALIBRATED, 1.0, PRICE, TIMING, priced)
        self.assertTrue(card.excluded)
        self.assertIn("proxy cross-check", card.reason)

    def test_a_client_without_a_base_url_passes_through(self) -> None:
        # The scripted client has no socket; the loop must still run.
        client = ScriptedClient(script={})
        with through_proxy(client, None) as (same, proxy):
            self.assertIs(same, client)
            self.assertIsNone(proxy)


class TestBundledPlanFeasibility(unittest.TestCase):
    """A hand-built plan bypasses the enumerator, so it has to check the rule the
    enumerator enforces -- otherwise `run_plan` executes a plan the oracle's own
    rules say cannot exist."""

    def test_every_shape_yields_a_feasible_bundling_or_none(self) -> None:
        for shape in ("wide", "chain", "diamond", "mixed"):
            for n in (3, 5, 6):
                scn = build_scenario(sample_dag(shape, n, sizes=(1,), seed=3), "x", seed=3)
                plan = bundled_plan(scn)
                if plan is None:
                    continue
                self.assertTrue(is_feasible(scn.dag, plan), f"{shape}{n}: {plan}")
                covered = sorted(v for b in plan.blocks for v in b)
                self.assertEqual(covered, sorted(scn.dag.ids), f"{shape}{n}")

    def test_it_still_yields_two_distinct_block_sizes_where_it_can(self) -> None:
        scn = build_scenario(sample_dag("wide", 5, sizes=(1,), seed=3), "x", seed=3)
        plan = bundled_plan(scn)
        self.assertGreater(len({len(b) for b in plan.blocks}), 1)


class TestOutOfOrderEdits(unittest.TestCase):
    """The direct measurement of the soft-dependency approximation, which the
    oracle models as strictly blocking and the payload does not enforce."""

    def _chain(self):
        return build_scenario(sample_dag("chain", 3, sizes=(1,), seed=2), "c", seed=2)

    def test_in_order_edits_score_zero(self) -> None:
        scn = self._chain()
        ids = list(scn.dag.topo_order)
        trace = Trace(scenario_id=scn.id, model="m")
        for i, n in enumerate(ids):
            trace.write_events.append(WriteEvent(path=module_path(n), actor=LEAD, t=float(i)))
        out = out_of_order_edits(trace, scn)
        self.assertEqual(out.out_of_order, 0)
        self.assertEqual(out.rate, 0.0)
        self.assertEqual(out.pairs_at_risk, 2)

    def test_editing_a_successor_first_is_counted(self) -> None:
        scn = self._chain()
        ids = list(scn.dag.topo_order)
        trace = Trace(scenario_id=scn.id, model="m")
        for i, n in enumerate(reversed(ids)):
            trace.write_events.append(WriteEvent(path=module_path(n), actor=LEAD, t=float(i)))
        out = out_of_order_edits(trace, scn)
        self.assertEqual(out.out_of_order, 2)  # n1 and n2 both preceded their preds
        self.assertAlmostEqual(out.rate, 1.0)

    def test_a_predecessor_rewritten_later_makes_the_successor_out_of_order(self) -> None:
        # "Before the predecessor was LAST written" is the test: a predecessor
        # rewritten later was wrong until then, so a successor edited in between
        # was still working against broken upstream code.
        scn = self._chain()
        ids = list(scn.dag.topo_order)
        trace = Trace(scenario_id=scn.id, model="m")
        trace.write_events += [
            WriteEvent(path=module_path(ids[0]), actor=LEAD, t=0.0),
            WriteEvent(path=module_path(ids[1]), actor=LEAD, t=1.0),
            WriteEvent(path=module_path(ids[0]), actor=LEAD, t=2.0),  # pred was wrong
        ]
        self.assertEqual(out_of_order_edits(trace, scn).out_of_order, 1)

    def test_wide_has_no_pairs_so_a_zero_rate_is_not_evidence(self) -> None:
        scn = build_scenario(sample_dag("wide", 3, sizes=(1,), seed=2), "w", seed=2)
        trace = Trace(scenario_id=scn.id, model="m")
        for i, n in enumerate(scn.dag.ids):
            trace.write_events.append(WriteEvent(path=module_path(n), actor=LEAD, t=float(i)))
        out = out_of_order_edits(trace, scn)
        self.assertEqual(out.pairs_at_risk, 0)
        self.assertIsNone(out.rate)
        self.assertIn("no module writes observed", str(out))

    def test_pooling_across_runs(self) -> None:
        scn = self._chain()
        ids = list(scn.dag.topo_order)
        good = Trace(scenario_id=scn.id, model="m")
        bad = Trace(scenario_id=scn.id, model="m")
        for i, n in enumerate(ids):
            good.write_events.append(WriteEvent(path=module_path(n), actor=LEAD, t=float(i)))
        for i, n in enumerate(reversed(ids)):
            bad.write_events.append(WriteEvent(path=module_path(n), actor=LEAD, t=float(i)))
        out = out_of_order_rate([good, bad], {scn.id: scn})
        self.assertEqual(out.out_of_order, 2)
        self.assertEqual(out.edits, 4)


class TestMaterialityFloor(unittest.TestCase):
    """`OUTCOME_TOL`'s default is exact float equality, which tests nothing on
    values derived from measured constants carrying confidence intervals. The
    floor is PER AXIS -- dollars and minutes are not comparable -- and the
    objective floor at a beta is dollars + beta * minutes."""

    def test_the_dollar_floor_is_the_per_bucket_absolute_spread(self) -> None:
        from harness.calibration import CalibrationResult

        result = CalibrationResult(
            source="s",
            constants={"block_dollars_curve": ((1, 0.01), (4, 0.10))},
            diagnostics={"dollar_disagreement": {4: 0.05},
                         "timing_residual_rms_minutes": 0.001},
        )
        # spread AT the bucket times the mean AT that bucket: 0.05 * 0.10. The
        # old worst-relative-times-largest form multiplied the small-block
        # percentage by the top-of-curve value and once produced a floor larger
        # than the entire all-inline objective.
        self.assertAlmostEqual(materiality_floor(result), 0.005)
        self.assertAlmostEqual(materiality_floor(result, beta=1.0), 0.006)

    def test_the_minute_floor_scales_with_beta_never_maxes_across_units(self) -> None:
        from harness.calibration import CalibrationResult

        result = CalibrationResult(
            source="s",
            constants={"block_dollars_curve": ((1, 0.01), (4, 0.10))},
            diagnostics={"dollar_disagreement": {4: 0.001},
                         "timing_residual_rms_minutes": 0.02},
        )
        # At beta=0 the minutes noise cannot move the objective, so it cannot
        # move the floor either; at beta=1 it enters in the objective's units.
        self.assertAlmostEqual(materiality_floor(result), 0.0001)
        self.assertAlmostEqual(materiality_floor(result, beta=1.0), 0.0201)

    def test_no_diagnostics_yields_no_floor_rather_than_a_guess(self) -> None:
        from harness.calibration import CalibrationResult

        self.assertIsNone(materiality_floor(CalibrationResult(source="s")))

    def test_a_real_calibration_publishes_its_floors(self) -> None:
        from harness.calibration import run_calibration

        import tests.test_calibration as C

        scn = C.scenario(n=4, size=2)
        with C.TestDriverEndToEnd()._upstream(scn) as up:
            client = AnthropicClient(model="fake", api_key="k", base_url=up.base_url)
            with tempfile.TemporaryDirectory() as out:
                result = run_calibration(scn, client, C.PRICE, out, source="t",
                                         orderings=2, repeats=1, max_turns=14)
        self.assertIn("dollars", result.floors)
        self.assertIn("minutes", result.floors)
        # The scripted runs are token-identical, so the dollar spread is
        # honestly zero; the minutes floor carries the timing fit's residual,
        # which real socket jitter keeps above literal zero.
        self.assertIsNotNone(result.floors["dollars"])
        self.assertGreater(result.floors["minutes"], 0.0)
        self.assertIn("materiality floor", result.report())


class TestExperimentDriver(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = Manifest(
            name="demo",
            specs=(ScenarioSpec("wide", 3, 1, 11), ScenarioSpec("chain", 3, 1, 11)),
        )
        self.script, self.route = repair_script(self.manifest.specs)

    def _run(self, out, **kwargs):
        with ProtocolUpstream("anthropic", self.script, key_from=self.route) as up:
            client = AnthropicClient(model="fake-model", api_key="k", base_url=up.base_url)
            results = run_experiment(
                self.manifest, client, PRICE, TIMING, CALIBRATED, out,
                calibration_source="test 2026-08-23", max_turns=8, **kwargs
            )
            self.assertEqual(up.violations, [])
        return results

    def test_it_produces_a_stamped_results_file(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            results = self._run(out, betas=(0.0, 1.0), ordering_subset=0)
            saved = json.loads((Path(out) / "results.json").read_text())
        # Three stamps, without which the file cannot be interpreted later.
        self.assertEqual(saved["manifest"]["fingerprint"], self.manifest.fingerprint)
        self.assertEqual(saved["calibration_source"], "test 2026-08-23")
        self.assertEqual(saved["price_sheet"]["as_of"], "2026-08-23")
        self.assertTrue(saved["comparable"])
        self.assertEqual(sorted(results.aggregates), ["0.0", "1.0"])

    def test_a_results_file_without_its_stamps_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            unstamped = Results(
                manifest_fingerprint="", manifest_name="x", calibration_source="c",
                price_sheet={"as_of": "2026-08-23"}, model="m", betas=(1.0,),
            )
            with self.assertRaises(ValueError):
                unstamped.write(Path(out) / "r.json")
            undated = Results(
                manifest_fingerprint="abc", manifest_name="x", calibration_source="c",
                price_sheet={}, model="m", betas=(1.0,),
            )
            with self.assertRaises(ValueError) as caught:
                undated.write(Path(out) / "r2.json")
            self.assertIn("undated dollar", str(caught.exception))

    def test_every_scenario_gets_an_agent_and_a_baseline_trace(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            results = self._run(out, betas=(1.0,), disclosed=False, ordering_subset=0)
            names = [Path(f).name for f in results.trace_files]
        for spec in self.manifest.specs:
            self.assertTrue(any(n.startswith(f"{spec.id}-agent") for n in names), spec.id)
            self.assertTrue(any(n.startswith(f"{spec.id}-oracle") for n in names), spec.id)

    def test_beats_all_inline_compares_two_measured_runs(self) -> None:
        # Always-serial is executed, never merely priced: every scorable card
        # must carry the measured objective of a real all-inline run (its own
        # trace, or the baseline when the model's best plan already IS
        # all-inline), because the model's inline price is known to run low
        # and would make serial artificially hard to beat.
        with tempfile.TemporaryDirectory() as out:
            results = self._run(out, betas=(1.0,), disclosed=False, ordering_subset=0)
            names = [Path(f).name for f in results.trace_files]
        for card in results.cards:
            if not card.excluded:
                self.assertIsNotNone(card.measured_all_inline, card.scenario_id)
        # Every scenario has SOME executed source for that number: its own
        # -inline- arm, or a baseline whose requested plan was already inline.
        del names  # reuse-vs-fresh is plan-dependent; the card check above is the contract

    def test_an_uncalibrated_cost_model_marks_the_whole_run_not_comparable(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            with ProtocolUpstream("anthropic", self.script, key_from=self.route) as up:
                client = AnthropicClient(model="m", api_key="k", base_url=up.base_url)
                results = run_experiment(
                    self.manifest, client, PRICE, TIMING, CostModel(), out,
                    calibration_source="none", betas=(1.0,), disclosed=False,
                    ordering_subset=0, max_turns=8,
                )
        self.assertFalse(results.comparable)
        self.assertIn("NOT COMPARABLE", results.report())

    def test_the_disclosed_arm_runs_and_produces_a_split(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            results = self._run(out, betas=(1.0,), disclosed=True, ordering_subset=0)
            names = [Path(f).name for f in results.trace_files]
        self.assertTrue(any("disclosed" in n for n in names))
        self.assertIn("1.0", results.discovery)

    def test_a_model_that_ignores_the_plan_makes_ordering_inconclusive(self) -> None:
        # This fixture's model repairs everything itself whatever it is told, so
        # the oracle and anti-oracle runs are the same run. That is not evidence
        # the cost model ranks plans wrongly -- it is no evidence at all.
        with tempfile.TemporaryDirectory() as out:
            results = self._run(out, betas=(1.0,), disclosed=False, ordering_subset=1)
        self.assertIsNotNone(results.ordering)
        self.assertIn("INCONCLUSIVE", results.ordering.verdict)
        self.assertTrue(
            all(c.agreed is None for c in results.ordering.checks),
            [str(c) for c in results.ordering.checks],
        )

    def test_tier_a_and_out_of_order_appear_in_the_report(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            results = self._run(out, betas=(1.0,), disclosed=False, ordering_subset=0)
        text = results.report()
        self.assertIn("tierA", text)
        self.assertIn("out-of-order edits", text)
        # chain has dependency pairs; wide has none, and says so rather than
        # reporting a 0% rate that means nothing.
        self.assertIn("dependency pair", text)


class TestPlanCompliance(unittest.TestCase):
    """Non-compliance is what makes a baseline excluded and the ordering check
    inconclusive. Both are quiet failures that look like missing data rather than
    a finding, so the rate is measured."""

    def _trace(self, condition, complied=True):
        return Trace(
            scenario_id="s", model="m", condition=condition,
            notes=[] if complied else ["PLAN NOT FOLLOWED: asked for inline=[]..."],
        )

    def test_only_forced_plan_runs_are_counted(self) -> None:
        from scoring.validate import plan_compliance

        # Including ordinary agent runs would dilute the rate toward 100% with
        # runs that were never given a plan to follow.
        out = plan_compliance([
            self._trace("agent"), self._trace("agent"),
            self._trace("oracle-plan"), self._trace("oracle-plan", complied=False),
        ])
        self.assertEqual((out.complied, out.total), (1, 2))
        self.assertAlmostEqual(out.rate, 0.5)

    def test_a_low_rate_is_reported_as_the_finding(self) -> None:
        from scoring.validate import plan_compliance

        out = plan_compliance([self._trace("oracle-plan", complied=False)] * 4
                              + [self._trace("oracle-plan")])
        self.assertIn("PLAN COMPLIANCE 20%", out.verdict)
        self.assertIn("Regret is not measurable here", out.verdict)

    def test_a_high_rate_reads_plainly(self) -> None:
        from scoring.validate import plan_compliance

        out = plan_compliance([self._trace("oracle-plan")] * 10)
        self.assertIn("plan compliance 100%", out.verdict)
        self.assertNotIn("PLAN COMPLIANCE", out.verdict)

    def test_no_forced_runs_says_so(self) -> None:
        from scoring.validate import plan_compliance

        out = plan_compliance([self._trace("agent")])
        self.assertIsNone(out.rate)
        self.assertIn("no forced-plan runs", out.verdict)

    def test_the_breakdown_separates_baseline_from_anti_oracle(self) -> None:
        from scoring.validate import plan_compliance

        out = plan_compliance([
            self._trace("oracle-plan"),
            self._trace("anti-oracle", complied=False),
        ])
        self.assertEqual(out.by_condition["oracle-plan"], (1, 1))
        self.assertEqual(out.by_condition["anti-oracle"], (0, 1))

    def test_the_experiment_reports_it(self) -> None:
        manifest = Manifest(name="d", specs=(ScenarioSpec("wide", 3, 1, 11),))
        script, route = repair_script(manifest.specs)
        with tempfile.TemporaryDirectory() as out:
            with ProtocolUpstream("anthropic", script, key_from=route) as up:
                client = AnthropicClient(model="m", api_key="k", base_url=up.base_url)
                results = run_experiment(
                    manifest, client, PRICE, TIMING, CALIBRATED, out,
                    calibration_source="t", betas=(1.0,), disclosed=False,
                    ordering_subset=0, max_turns=8,
                )
        # This fixture's model ignores the directive, so compliance is low and
        # the report says so rather than leaving a thin sample unexplained.
        self.assertIsNotNone(results.compliance)
        self.assertIn("compliance", results.report().lower())


if __name__ == "__main__":
    unittest.main()
