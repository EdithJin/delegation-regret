"""The calibration driver, exercised without spending anything.

Calibration is the one step whose output every other number depends on, and the
one that costs real money to repeat. So the extraction has to be right the first
time, which means it has to be tested against runs whose correct answer is known
in advance.

`ProtocolUpstream` makes that possible: it serves scripted turns with exact token
counts, so the constants the extraction *should* recover are arithmetic we can do
by hand. If the driver mis-segments a run or fits the wrong slope, it shows up
here rather than in a $200 afternoon.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator.dag import sample_dag
from generator.oracle import CostModel, all_inline, unmeasured
from generator.scenario import build_scenario
from generator.templates import module_path
from harness.calibrate import PriceSheet, TimingModel, fit_timing_model
from harness.calibration import (
    CalibrationResult,
    block_curves,
    boundaries_from_trace,
    explore_overhead,
    extract,
    replay,
    run_calibration,
    spawn_overhead,
)
from harness.client import AnthropicClient
from harness.protocol_upstream import ProtocolUpstream, Turn
from harness.trace import LEAD, ModelCall, Trace, WriteEvent

PRICE = PriceSheet(
    model="fake-model",
    as_of="2026-08-23",
    input_per_mtok=3.0,
    output_per_mtok=15.0,
    cache_read_per_mtok=0.3,
    cache_write_per_mtok=3.75,
)


def scenario(n: int = 3, size: int = 1, seed: int = 11):
    return build_scenario(sample_dag("wide", n, sizes=(size,), seed=seed), f"cal{n}", seed=seed)


class TestSegmentation(unittest.TestCase):
    """The boundaries are the whole measurement; a wrong one silently reshapes
    the curve rather than raising."""

    def test_boundaries_follow_completion_order_and_accumulate_size(self) -> None:
        scn = scenario(n=3, size=2)
        ids = list(scn.dag.ids)
        trace = Trace(scenario_id=scn.id, model="m")
        # Written out of node order on purpose: the curve's x-axis is cumulative
        # work in COMPLETION order, not node index.
        for t, node in ((0.5, ids[2]), (1.5, ids[0]), (2.5, ids[1])):
            trace.write_events.append(WriteEvent(path=module_path(node), actor=LEAD, t=t))
        bounds = boundaries_from_trace(trace, scn)
        self.assertEqual([b.units for b in bounds], [2, 4, 6])
        self.assertEqual([b.t_done for b in bounds], [0.5, 1.5, 2.5])

    def test_a_rewrite_moves_the_boundary_later(self) -> None:
        # A second write means the first was wrong, so the node cost everything
        # up to the final one.
        scn = scenario(n=1, size=3)
        node = scn.dag.ids[0]
        trace = Trace(scenario_id=scn.id, model="m")
        trace.write_events += [
            WriteEvent(path=module_path(node), actor=LEAD, t=1.0),
            WriteEvent(path=module_path(node), actor=LEAD, t=9.0),
        ]
        bounds = boundaries_from_trace(trace, scn)
        self.assertEqual([(b.units, b.t_done) for b in bounds], [(3, 9.0)])

    def test_subagent_writes_are_ignored_in_a_serial_calibration(self) -> None:
        # If a subagent wrote, the forced-serial plan was not followed and this
        # segmentation would be measuring a different thing entirely.
        scn = scenario(n=2)
        ids = list(scn.dag.ids)
        trace = Trace(scenario_id=scn.id, model="m")
        trace.write_events += [
            WriteEvent(path=module_path(ids[0]), actor=LEAD, t=1.0),
            WriteEvent(path=module_path(ids[1]), actor="subagent:0", t=2.0),
        ]
        self.assertEqual(len(boundaries_from_trace(trace, scn)), 1)

    def test_test_file_writes_do_not_create_boundaries(self) -> None:
        scn = scenario(n=1)
        trace = Trace(scenario_id=scn.id, model="m")
        trace.write_events.append(WriteEvent(path="tests/test_n0.py", actor=LEAD, t=1.0))
        self.assertEqual(boundaries_from_trace(trace, scn), [])


class TestCurveExtraction(unittest.TestCase):
    def _serial_trace(self, scn, per_call_in=1000, per_call_out=100) -> Trace:
        """One synthetic serial run: one call per node, one write after each."""
        trace = Trace(scenario_id=scn.id, model="m")
        for i, node in enumerate(scn.dag.ids):
            trace.calls.append(
                ModelCall(actor=LEAD, index=i, input_tokens=per_call_in,
                          output_tokens=per_call_out, total_s=6.0, t_request=float(i))
            )
            trace.write_events.append(
                WriteEvent(path=module_path(node), actor=LEAD, t=float(i) + 0.5)
            )
        return trace

    def test_the_curve_is_cumulative_and_monotonic(self) -> None:
        scn = scenario(n=4, size=1)
        trace = self._serial_trace(scn)
        timing = TimingModel(a_minutes=1.0, b_minutes_per_input_token=0.0,
                             output_tokens_per_minute=1e12)
        curves = block_curves([trace], scn, PRICE, timing)
        dollars = curves["block_dollars_curve"]
        self.assertEqual([u for u, _ in dollars], [1, 2, 3, 4])
        values = [v for _, v in dollars]
        self.assertEqual(values, sorted(values))
        # One call per unit, each 1000 in + 100 out at the sheet above.
        per_call = (1000 * 3.0 + 100 * 15.0) / 1_000_000
        self.assertAlmostEqual(values[0], per_call, places=9)
        self.assertAlmostEqual(values[-1], 4 * per_call, places=9)

    def test_repeats_are_averaged_and_disagreement_is_reported(self) -> None:
        scn = scenario(n=3, size=1)
        cheap = self._serial_trace(scn, per_call_in=1000)
        dear = self._serial_trace(scn, per_call_in=3000)
        timing = TimingModel(1.0, 0.0, 1e12)
        curves = block_curves([cheap, dear], scn, PRICE, timing)
        self.assertEqual(curves["n_curves"], 2)
        self.assertTrue(curves["dollar_disagreement"])
        # Disagreement is the spread as a fraction of the mean, and it IS the
        # test of the units-not-identity assumption.
        self.assertGreater(max(curves["dollar_disagreement"].values()), 0.2)

    def test_a_run_that_completed_nothing_raises_rather_than_fitting_noise(self) -> None:
        scn = scenario(n=2)
        with self.assertRaises(ValueError):
            block_curves([Trace(scenario_id=scn.id, model="m")], scn, PRICE,
                         TimingModel(1.0, 0.0, 1e12))


class TestOverheadExtraction(unittest.TestCase):
    def _fanout_trace(self) -> Trace:
        """A lead that spawns twice -- one 1-node block, one 2-node block -- so
        the briefing slope is identifiable."""
        trace = Trace(scenario_id="s", model="m")
        from harness.trace import SpawnRecord

        trace.calls.append(
            ModelCall(actor=LEAD, index=0, input_tokens=500, output_tokens=50,
                      tools_invoked=("list_files",), t_request=0.0)
        )
        trace.calls.append(
            ModelCall(actor=LEAD, index=1, input_tokens=600, output_tokens=200,
                      tools_invoked=("spawn_subagent",), t_request=1.0)
        )
        trace.calls.append(
            ModelCall(actor=LEAD, index=2, input_tokens=900, output_tokens=400,
                      tools_invoked=("spawn_subagent", "spawn_subagent"), t_request=2.0)
        )
        trace.calls.append(
            ModelCall(actor=LEAD, index=3, input_tokens=1500, output_tokens=30,
                      tools_invoked=("finish",), t_request=3.0)
        )
        for i in range(3):
            trace.spawns.append(SpawnRecord(index=i, batch=0 if i == 0 else 1,
                                            instruction=f"do {i}", files=()))
            trace.calls.append(
                ModelCall(actor=f"subagent:{i}", index=0, input_tokens=400,
                          output_tokens=60, t_request=4.0 + i)
            )
        return trace

    def test_spawn_fixed_is_the_subagents_first_call(self) -> None:
        trace = self._fanout_trace()
        out = spawn_overhead([trace], PRICE, TimingModel(1.0, 0.0, 1e12))
        expected = (400 * 3.0 + 60 * 15.0) / 1_000_000
        self.assertEqual(out["n_subagents"], 3)
        self.assertAlmostEqual(out["spawn_fixed_dollars"], expected, places=9)

    def test_briefing_is_fitted_as_a_slope_not_a_constant(self) -> None:
        # C.1 requires the affine form and requires intercept and slope to come
        # from ONE regression; fitting them apart invites a slope that
        # contradicts its own intercept.
        out = spawn_overhead([self._fanout_trace()], PRICE, TimingModel(1.0, 0.001, 1e12))
        self.assertIsNotNone(out["brief_dollars_per_node"])
        self.assertIsNotNone(out["brief_minutes_per_node"])
        self.assertGreater(out["brief_dollars_per_node"], 0.0)

    def test_one_block_size_leaves_the_slope_unidentified(self) -> None:
        # The failure mode a fan-out run falls into when every spawn covers
        # exactly one node. Reporting a number here would be fitting a line to a
        # single point, so it is refused with a reason.
        trace = Trace(scenario_id="s", model="m")
        for i in range(3):
            trace.calls.append(
                ModelCall(actor=LEAD, index=i, input_tokens=500, output_tokens=100,
                          tools_invoked=("spawn_subagent",), t_request=float(i))
            )
        out = spawn_overhead([trace], PRICE, TimingModel(1.0, 0.0, 1e12))
        self.assertIsNone(out["brief_dollars_per_node"])
        self.assertIn("unidentifiable", out["brief_dollars_why"])

    def test_absorption_is_read_from_the_context_step(self) -> None:
        out = spawn_overhead([self._fanout_trace()], PRICE, TimingModel(1.0, 0.001, 1e12))
        self.assertIsNotNone(out["absorb_dollars_per_node"])
        self.assertGreater(out["absorb_dollars_per_node"], 0.0)

    def test_explore_stops_at_the_first_action(self) -> None:
        out = explore_overhead(self._fanout_trace(), PRICE, TimingModel(1.0, 0.0, 1e12))
        # Calls 0 (list_files) and 1 (the first spawn) inclusive.
        self.assertEqual(out["n_calls"], 2)
        self.assertGreater(out["explore_dollars"], 0.0)

    def test_a_lead_that_never_acted_yields_no_explore_constant(self) -> None:
        trace = Trace(scenario_id="s", model="m")
        trace.calls.append(ModelCall(actor=LEAD, index=0, input_tokens=1, output_tokens=1))
        out = explore_overhead(trace, PRICE, TimingModel(1.0, 0.0, 1e12))
        self.assertIsNone(out["explore_dollars"])
        self.assertIn("never acted", out["why"])


class TestDriverEndToEnd(unittest.TestCase):
    """Real runs through the real client, against a validating fake provider.

    The fixture is deliberately realistic rather than minimal, because the
    identifiability constraints are real: token counts must vary, input and
    output must vary INDEPENDENTLY, and the fan-out runs must brief more than one
    block size. A minimal fixture passes while silently exercising none of it.
    """

    def _fixture(self, scn):
        from harness.calibration import bundled_plan

        ids = list(scn.dag.ids)
        write = lambda n: ("write_file", {"path": module_path(n), "content": scn.reference[n]})
        finish = ("finish", {"summary": "done"})
        blocks = [sorted(b) for b in bundled_plan(scn).blocks]

        script = {
            # Serial: one node per turn, context growing, output varying.
            "serial": [
                *[
                    Turn(tools=(write(n),), input_tokens=700 + 400 * i, output_tokens=60 + 40 * i)
                    for i, n in enumerate(ids)
                ],
                Turn(tools=(finish,), input_tokens=2500, output_tokens=25),
            ],
            # Fan-out: one brief per node. Output scales with nodes briefed.
            "fanout": [
                Turn(
                    tools=tuple(
                        ("spawn_subagent", {"instruction": f"repair {n}", "files": []})
                        for n in ids
                    ),
                    input_tokens=900,
                    output_tokens=90 * len(ids),
                ),
                Turn(tools=(finish,), input_tokens=2000, output_tokens=40),
            ],
            # Bundled: fewer, larger blocks -> the second x value for the slope.
            "bundled": [
                Turn(
                    tools=tuple(
                        ("spawn_subagent", {"instruction": "repair " + ",".join(b), "files": []})
                        for b in blocks
                    ),
                    input_tokens=900,
                    output_tokens=90 * len(blocks),
                ),
                Turn(tools=(finish,), input_tokens=2000, output_tokens=40),
            ],
        }
        for b in blocks:
            script["repair " + ",".join(b)] = [
                Turn(tools=tuple(write(n) for n in b), input_tokens=500, output_tokens=70),
                Turn(tools=(finish,), input_tokens=800, output_tokens=30),
            ]
        for n in ids:
            script[f"repair {n}"] = [
                Turn(tools=(write(n),), input_tokens=500, output_tokens=70),
                Turn(tools=(finish,), input_tokens=800, output_tokens=30),
            ]

        def route(first_user: str) -> str:
            # Forced-serial and forced-fanout leads open with the SAME task list
            # and differ only in the appended plan directive, so the default
            # first-line routing cannot tell them apart.
            if "Do not spawn any subagent" in first_user:
                return "serial"
            if f"Issue exactly {len(ids)} " in first_user:
                return "fanout"
            if f"Issue exactly {len(blocks)} " in first_user:
                return "bundled"
            return first_user.splitlines()[0].strip()

        return script, route

    def _upstream(self, scn):
        script, route = self._fixture(scn)
        return ProtocolUpstream(
            "anthropic",
            script,
            key_from=route,
            # Latency has to depend on input AND output separately, or the timing
            # model cannot separate prefill from generation.
            base_latency_s=0.002,
            seconds_per_output_token=0.00012,
            seconds_per_input_token=0.000006,
        )

    def test_a_calibration_run_measures_every_constant_and_persists_it(self) -> None:
        scn = scenario(n=4, size=2)
        with self._upstream(scn) as up:
            client = AnthropicClient(model="fake-model", api_key="k", base_url=up.base_url)
            with tempfile.TemporaryDirectory() as out:
                result = run_calibration(
                    scn, client, PRICE, out, source="test-run 2026-08-23",
                    orderings=2, repeats=1, max_turns=14,
                )
                saved = json.loads((Path(out) / "calibration.json").read_text())
                traces = sorted(Path(out).glob("trace-*.json"))
                # Every run is on disk before anything is fitted, so a failed
                # extraction never costs the runs.
                self.assertEqual(len(traces), 4)  # 2 serial + fanout + bundled
                again = replay(out, scn, PRICE, source="test-run 2026-08-23")
            self.assertEqual(up.violations, [])

        self.assertEqual(saved["source"], "test-run 2026-08-23")
        self.assertEqual(saved["price_sheet"]["as_of"], "2026-08-23")
        self.assertEqual(result.skipped, {}, result.skipped)
        self.assertEqual(result.diagnostics["n_serial_runs"], 2)
        # Reproducible from the saved traces alone -- that is what makes the
        # calibration checkable rather than merely recorded.
        self.assertEqual(result.constants, again.constants)

    def test_every_measured_constant_is_positive_and_stamped(self) -> None:
        scn = scenario(n=4, size=2)
        with self._upstream(scn) as up:
            client = AnthropicClient(model="fake-model", api_key="k", base_url=up.base_url)
            with tempfile.TemporaryDirectory() as out:
                result = run_calibration(scn, client, PRICE, out,
                                         source="test-run 2026-08-23",
                                         orderings=2, repeats=1, max_turns=14)
            self.assertEqual(up.violations, [])

        cm = result.to_cost_model()
        self.assertEqual(unmeasured(cm), (), unmeasured(cm))
        self.assertTrue(cm.is_calibrated)
        for name in result.constants:
            value = getattr(cm, name)
            self.assertEqual(value.status, "measured", name)
            self.assertEqual(value.source, "test-run 2026-08-23", name)
            scalars = [value] if isinstance(value, float) else [v for _, v in value]
            for scalar in scalars:
                self.assertGreater(scalar, 0.0, f"{name} came out non-positive")

    def test_a_partial_calibration_reports_itself_as_partial(self) -> None:
        # Constant output length -> no timing model -> the dollar half survives
        # and the minutes half is reported as skipped, with the reason.
        scn = scenario(n=3, size=1)
        flat = {
            "lead": [
                *[
                    Turn(tools=(("write_file", {"path": module_path(n),
                                                "content": scn.reference[n]}),),
                         input_tokens=800, output_tokens=100)
                    for n in scn.dag.ids
                ],
                Turn(tools=(("finish", {"summary": "done"}),)),
            ]
        }
        with ProtocolUpstream("anthropic", flat) as up:
            client = AnthropicClient(model="fake-model", api_key="k", base_url=up.base_url)
            with tempfile.TemporaryDirectory() as out:
                result = run_calibration(scn, client, PRICE, out, source="flat 2026-08-23",
                                         orderings=1, repeats=1, max_turns=10)
        self.assertIn("block_dollars_curve", result.constants)
        self.assertIn("block_minutes_curve", result.skipped)
        self.assertIn("timing model", result.skipped["block_minutes_curve"])
        self.assertFalse(result.to_cost_model().is_calibrated)

    def test_the_report_flags_wide_disagreement_rather_than_burying_it(self) -> None:
        result = CalibrationResult(
            source="s",
            diagnostics={"dollar_disagreement": {5: 0.4}},
            constants={"spawn_fixed_dollars": 0.01},
        )
        text = result.report()
        self.assertIn("WIDE", text)
        self.assertIn("40.0%", text)


class TestHeldOutCheck(unittest.TestCase):
    """The only number that says whether the curve generalises rather than
    merely fitting the runs it came from."""

    def test_a_linear_curve_predicts_its_own_top_end(self) -> None:
        from harness.calibration import holdout_check

        out = holdout_check(((1, 0.10), (2, 0.20), (3, 0.30), (4, 0.40)))
        self.assertTrue(out["checked"])
        self.assertAlmostEqual(out["predicted"], 0.40, places=9)
        self.assertLess(out["rel_error"], 1e-9)

    def test_a_curve_that_bends_late_is_caught(self) -> None:
        # The top end is where the all-inline plan is priced, so a curve that
        # only bends there is exactly the one worth catching.
        from harness.calibration import holdout_check

        out = holdout_check(((1, 0.10), (2, 0.20), (3, 0.30), (4, 0.90)))
        self.assertGreater(out["rel_error"], 0.5)

    def test_too_few_points_is_reported_not_faked(self) -> None:
        from harness.calibration import holdout_check

        out = holdout_check(((1, 0.1), (2, 0.2)))
        self.assertFalse(out["checked"])
        self.assertIn("need 3+", out["why"])

    def test_the_report_flags_a_curve_that_cannot_predict_itself(self) -> None:
        result = CalibrationResult(
            source="s",
            constants={"spawn_fixed_dollars": 0.01},
            diagnostics={"dollar_holdout": {
                "checked": True, "units": 8.0, "predicted": 0.10,
                "measured": 0.30, "rel_error": 0.667, "abs_error": 0.2,
            }},
        )
        text = result.report()
        self.assertIn("Extrapolation is doing the work", text)


class TestSignGuards(unittest.TestCase):
    """A wrong-signed constant is worse than an honest placeholder, because the
    sign is what the constant MEANS. Both guards refuse rather than clamp:
    clamping to zero asserts the effect is free, which is the same unmeasured
    claim pointing the other way."""

    def _records(self, rows):
        from harness.proxy import CallRecord

        return [
            CallRecord(seq=i, path="", input_tokens=i_t, output_tokens=o_t, total_s=s)
            for i, (i_t, o_t, s) in enumerate(rows)
        ]

    def test_a_negative_prefill_coefficient_is_refused(self) -> None:
        # A bigger context cannot be faster to START. Left alone it flips the
        # sign of context drag in the latency column, and the absorption minutes
        # derived from it come out negative too -- every absorbed result would
        # shorten the run.
        # Input and output must vary INDEPENDENTLY here, or the design matrix is
        # singular and the earlier "not identifiable" guard fires first -- also
        # correct, but a different failure.
        rows = [(1000, 100, 5.0), (5000, 100, 3.0), (1000, 300, 9.0), (5000, 300, 7.0)]
        with self.assertRaises(ValueError) as caught:
            fit_timing_model(self._records(rows))
        self.assertIn("negative cost per input token", str(caught.exception))

    def test_perfectly_collinear_calls_are_refused_too(self) -> None:
        # A log where input is a fixed multiple of output cannot separate prefill
        # from generation at all, and says so rather than returning one of the
        # infinitely many fits.
        rows = [(1000, 100, 6.0), (2000, 200, 4.0), (3000, 300, 2.0)]
        with self.assertRaises(ValueError) as caught:
            fit_timing_model(self._records(rows))
        self.assertIn("not identifiable", str(caught.exception))

    def test_a_negative_briefing_slope_is_refused(self) -> None:
        # Briefing more nodes cannot cost less; the instruction is strictly
        # longer. A negative slope would make wide fan-out look cheaper the wider
        # it got, which is the direction of the hypothesis under test.
        from harness.calibration import _affine

        out = _affine("brief", [(1, 0.010), (3, 0.002)], [(1, 0.010), (3, 0.002)])
        self.assertIsNone(out["brief_dollars_per_node"])
        self.assertIn("negative slope", out["brief_dollars_why"])

    def test_a_positive_slope_passes_through(self) -> None:
        from harness.calibration import _affine

        out = _affine("brief", [(1, 0.002), (3, 0.006)], [(1, 0.1), (3, 0.3)])
        self.assertAlmostEqual(out["brief_dollars_per_node"], 0.002)
        self.assertAlmostEqual(out["brief_minutes_per_node"], 0.1)


class TestBundledRunMakesTheSlopeIdentifiable(unittest.TestCase):
    def test_bundled_plan_briefs_two_distinct_block_sizes(self) -> None:
        from harness.calibration import bundled_plan

        scn = scenario(n=5)
        plan = bundled_plan(scn)
        sizes = sorted(len(b) for b in plan.blocks)
        self.assertGreater(len(set(sizes)), 1, sizes)
        covered = sorted(n for b in plan.blocks for n in b)
        self.assertEqual(covered, sorted(scn.dag.ids))
        self.assertEqual(plan.inline, frozenset())

    def test_too_small_a_dag_yields_no_bundled_plan(self) -> None:
        from harness.calibration import bundled_plan

        self.assertIsNone(bundled_plan(scenario(n=2)))


class TestCalibrationMakesRegretComparable(unittest.TestCase):
    """The payoff, and the reason calibration is on the critical path rather than
    being a precision improvement."""

    def test_the_same_traces_score_differently_before_and_after(self) -> None:
        from harness.trace import LEAD as L
        from harness.trace import ModelCall as MC
        from harness.trace import Trace as T
        from scoring.regret import score

        dag = sample_dag("wide", 3, sizes=(2,))
        from generator.oracle import enumerate_plans, evaluate

        timing = TimingModel(0.0002, 1.9e-7, 200_000.0)

        def tr(n):
            t = T(scenario_id="s", model="m")
            for i in range(n):
                t.calls.append(MC(actor=L, index=i, input_tokens=900, output_tokens=90))
            t.verdicts = {v: True for v in dag.ids}
            t.node_attribution = {v: L for v in dag.ids}
            return t

        placeholder = CostModel()
        calibrated = placeholder.calibrate(
            "test 2026-08-23",
            block_dollars_curve=((1, 0.0030), (2, 0.0078), (3, 0.0144), (5, 0.0228)),
            block_minutes_curve=((1, 0.00060), (2, 0.00147), (3, 0.00262), (5, 0.00404)),
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
        cards = {}
        for label, cm in (("placeholder", placeholder), ("calibrated", calibrated)):
            results = [evaluate(dag, p, cm) for p in enumerate_plans(dag)]
            cards[label] = score(tr(6), tr(4), dag, cm, 1.0, PRICE, timing, results)

        self.assertFalse(cards["placeholder"].comparable)
        self.assertTrue(cards["calibrated"].comparable)
        self.assertEqual(cards["calibrated"].warnings, ())
        # The placeholder denominator is orders of magnitude too large, so regret
        # against it looks near-perfect for reasons that have nothing to do with
        # the agent.
        self.assertLess(cards["placeholder"].regret, 0.05)
        self.assertGreater(cards["calibrated"].regret, 0.5)


if __name__ == "__main__":
    unittest.main()
