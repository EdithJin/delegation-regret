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

    def test_misaligned_buckets_average_monotone_not_by_coverage(self) -> None:
        # Forced orderings put runs on different cumulative-unit grids: a run
        # that starts with a size-5 node has no point at 3 units. Bucketing by
        # raw unit count then averages each bucket over whichever runs land on
        # it, and the first real preflight's mean curve came out NON-MONOTONE
        # (5 units priced below 3) purely as a coverage artifact. The average
        # must read every covering run via interpolation and never fall.
        from harness.calibrate import average_curves

        a = ((3, 0.30), (8, 0.80), (24, 2.40))   # started with the size-3 node
        b = ((5, 0.05), (8, 0.60), (24, 2.00))   # started with the size-5 node
        avg = dict(average_curves([a, b]))
        values = [v for _, v in sorted(avg.items())]
        self.assertEqual(values, sorted(values), values)
        # At 5 units run `a` contributes its interpolated measurement, not
        # nothing -- so run b's cheap draw cannot own the bucket outright.
        self.assertGreater(avg[5], 0.05)
        # Aligned buckets stay plain means.
        self.assertAlmostEqual(avg[8], 0.70)


class TestOverheadExtraction(unittest.TestCase):
    def _fanout_trace(self) -> Trace:
        """A lead that spawns three subagents across two turns; the middle one
        covers TWO nodes (per attribution) with a proportionally longer
        instruction, so the per-spawn briefing regression sees two distinct
        block sizes."""
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
        instructions = {
            0: "fix module a",
            1: "fix modules b and c, including their shared edge",  # 2 nodes
            2: "fix module d",
        }
        for i in range(3):
            trace.spawns.append(SpawnRecord(index=i, batch=0 if i == 0 else 1,
                                            instruction=instructions[i], files=()))
            trace.calls.append(
                ModelCall(actor=f"subagent:{i}", index=0, input_tokens=400,
                          output_tokens=60, t_request=4.0 + i)
            )
        # Who wrote what -- the ground truth the per-spawn x is read from.
        trace.node_attribution = {
            "a": "subagent:0", "b": "subagent:1", "c": "subagent:1", "d": "subagent:2",
        }
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
        from harness.trace import SpawnRecord

        trace = Trace(scenario_id="s", model="m")
        for i in range(3):
            trace.calls.append(
                ModelCall(actor=LEAD, index=i, input_tokens=500, output_tokens=100,
                          tools_invoked=("spawn_subagent",), t_request=float(i))
            )
            trace.spawns.append(SpawnRecord(index=i, batch=i, instruction=f"do {i}", files=()))
            trace.node_attribution[f"node{i}"] = f"subagent:{i}"
        out = spawn_overhead([trace], PRICE, TimingModel(1.0, 0.0, 1e12))
        self.assertIsNone(out["brief_dollars_per_node"])
        self.assertIn("unidentifiable", out["brief_dollars_why"])

    def test_absorption_is_read_from_the_context_step(self) -> None:
        out = spawn_overhead([self._fanout_trace()], PRICE, TimingModel(1.0, 0.001, 1e12))
        self.assertIsNotNone(out["absorb_dollars_per_node"])
        self.assertGreater(out["absorb_dollars_per_node"], 0.0)

    def test_explore_stops_before_the_first_action(self) -> None:
        out = explore_overhead(self._fanout_trace(), PRICE, TimingModel(1.0, 0.0, 1e12))
        # Call 0 (list_files) only. The acting turn's spend belongs to the work
        # it starts -- counting it here charged the same turn twice, once as
        # explore and once inside the briefing or the block curve.
        self.assertEqual(out["n_calls"], 1)
        self.assertGreater(out["explore_dollars"], 0.0)

    def test_sub_block_curve_is_the_subagents_own_bill_minus_its_first_call(self) -> None:
        # The estimation gate's finding turned into a measurement: a spawned
        # block is priced by what SUBAGENTS were billed, never by the lead's
        # warm-context serial curve. First calls are excluded -- they are
        # spawn_fixed_dollars' measurement, and charging them twice would
        # rebuild the double-count this curve exists to remove.
        from harness.calibration import sub_block_curves

        scn = scenario(n=3, size=2)
        ids = list(scn.dag.ids)
        trace = Trace(scenario_id=scn.id, model="m")
        # subagent:0 works one node (2 units): first call + one work call.
        trace.calls.append(ModelCall(actor="subagent:0", index=0,
                                     input_tokens=400, output_tokens=60, total_s=1.0))
        trace.calls.append(ModelCall(actor="subagent:0", index=1,
                                     input_tokens=1000, output_tokens=100, total_s=1.0))
        # subagent:1 works two nodes (4 units): first call + two work calls.
        trace.calls.append(ModelCall(actor="subagent:1", index=0,
                                     input_tokens=400, output_tokens=60, total_s=1.0))
        for i in (1, 2):
            trace.calls.append(ModelCall(actor="subagent:1", index=i,
                                         input_tokens=1500, output_tokens=120, total_s=1.0))
        trace.node_attribution = {ids[0]: "subagent:0", ids[1]: "subagent:1",
                                  ids[2]: "subagent:1"}
        out = sub_block_curves([trace], scn, PRICE, TimingModel(1.0, 0.0, 1e12))
        curve = dict(out["sub_block_dollars_curve"])
        self.assertEqual(sorted(curve), [2, 4])
        one_work_call = (1000 * 3.0 + 100 * 15.0) / 1e6
        two_work_calls = 2 * (1500 * 3.0 + 120 * 15.0) / 1e6
        self.assertAlmostEqual(curve[2], one_work_call, places=9)
        self.assertAlmostEqual(curve[4], two_work_calls, places=9)
        self.assertEqual(out["n_subagent_blocks"], 2)

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

        # Every lead opens with an orientation turn BEFORE acting: that prefix
        # is what explore_* measures, and the block curves subtract it -- a
        # fixture whose lead acts on turn one would leave the prefix path
        # silently unexercised.
        orient = Turn(tools=(("list_files", {}),), input_tokens=650, output_tokens=45)
        script = {
            # Serial: one node per turn, context growing, output varying.
            "serial": [
                orient,
                *[
                    Turn(tools=(write(n),), input_tokens=700 + 400 * i, output_tokens=60 + 40 * i)
                    for i, n in enumerate(ids)
                ],
                Turn(tools=(finish,), input_tokens=2500, output_tokens=25),
            ],
            # Fan-out: one brief per node. Output scales with nodes briefed.
            "fanout": [
                orient,
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
                orient,
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
        # The estimation gate ran: every executed plan got a composed
        # prediction compared against its measured total, on both axes.
        comp = result.diagnostics["composition"]
        self.assertTrue(comp["checked"], comp)
        self.assertEqual({a["plan"] for a in comp["arms"]},
                         {"all-inline", "max-fanout", "bundled"})
        for arm in comp["arms"]:
            self.assertIsNotNone(arm["rel_gap_dollars"], arm)
            self.assertIsNotNone(arm.get("rel_gap_minutes"), arm)
        self.assertIn(comp["cost_ranking_preserved"], (True, False))
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
        # Unfittable timing -> the dollar half survives and the minutes half is
        # reported as skipped, with the reason. Input is EXACTLY 8x output on
        # every call, so the design matrix is singular by construction: both
        # columns vary (the first-trace gate passes, as it should -- it is
        # necessary, not sufficient) but no jitter in real socket latency can
        # make collinear regressors identifiable. An earlier version used two
        # design points and constant output, and whether the fit failed then
        # depended on the timing noise of the machine running the tests.
        scn = scenario(n=3, size=1)
        tok = [(800, 100), (1600, 200), (800, 100)]
        flat = {
            "lead": [
                *[
                    Turn(tools=(("write_file", {"path": module_path(n),
                                                "content": scn.reference[n]}),),
                         input_tokens=tok[i][0], output_tokens=tok[i][1])
                    for i, n in enumerate(scn.dag.ids)
                ],
                Turn(tools=(("finish", {"summary": "done"}),),
                     input_tokens=1600, output_tokens=200),
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
        # A composed prediction from placeholder constants would price plans
        # with guesses; the estimation gate must refuse, with the reason.
        comp = result.diagnostics["composition"]
        self.assertFalse(comp["checked"])
        self.assertIn("placeholder", comp["why"])

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
    sign is what the constant MEANS. A bare clamp to zero would assert the
    effect is free -- the same unmeasured claim pointing the other way -- so a
    wrong sign is never clamped. It is refused, unless the log itself carries
    an independent measurement that settles the split: a negative prefill
    coefficient falls back to the TTFB slope against context (prefill observed
    directly, generation excluded by construction), and refusal remains for
    logs whose TTFB cannot identify a slope either. The fallback is
    model-agnostic and unreachable from any log whose primary fit succeeds."""

    def _records(self, rows, ttfb=None):
        from harness.proxy import CallRecord

        return [
            CallRecord(seq=i, path="", input_tokens=i_t, output_tokens=o_t, total_s=s,
                       ttfb_s=0.0 if ttfb is None else ttfb[i])
            for i, (i_t, o_t, s) in enumerate(rows)
        ]

    def test_a_negative_prefill_coefficient_is_refused_without_ttfb(self) -> None:
        # A bigger context cannot be faster to START. Left alone it flips the
        # sign of context drag in the latency column, and the absorption minutes
        # derived from it come out negative too -- every absorbed result would
        # shorten the run. With no TTFB in the log there is nothing to anchor
        # prefill to, so the original refusal stands.
        # Input and output must vary INDEPENDENTLY here, or the design matrix is
        # singular and the earlier "not identifiable" guard fires first -- also
        # correct, but a different failure.
        rows = [(1000, 100, 5.0), (5000, 100, 3.0), (1000, 300, 9.0), (5000, 300, 7.0)]
        with self.assertRaises(ValueError) as caught:
            fit_timing_model(self._records(rows))
        self.assertIn("negative cost per input token", str(caught.exception))

    def test_negative_prefill_with_flat_ttfb_anchors_to_zero(self) -> None:
        # The reasoning-model case measured live on 2026-08-28: thinking time
        # swells output in step with context, the total_s split goes negative,
        # but TTFB sits flat across context sizes -- prefill observably costs
        # ~nothing. The fit must pin prefill to the flat slope (0 after the
        # physical floor) and price all duration through output, thinking
        # included, instead of refusing.
        rows = [(1000, 100, 5.0), (5000, 100, 3.0), (1000, 300, 9.0), (5000, 300, 7.0)]
        tm = fit_timing_model(self._records(rows, ttfb=[0.5, 0.5, 0.5, 0.5]))
        self.assertEqual(tm.prefill_source, "ttfb-anchor")
        self.assertEqual(tm.b_minutes_per_input_token, 0.0)
        self.assertGreater(tm.output_tokens_per_minute, 0.0)
        # duration still fully accounted: throughput comes from the o=100 vs
        # o=300 contrast, (8-4)/200 min per token
        self.assertAlmostEqual(1.0 / tm.output_tokens_per_minute, (4.0 / 60.0) / 200.0)

    def test_negative_prefill_with_sloped_ttfb_uses_the_measured_slope(self) -> None:
        # When TTFB does grow with context, the anchor is that measured price,
        # not zero: 0.6s at 1000 context tokens, 3.0s at 5000, slope 0.0006
        # s/token = 1e-05 min/token.
        rows = [(1000, 100, 5.0), (5000, 100, 3.0), (1000, 300, 9.0), (5000, 300, 7.0)]
        tm = fit_timing_model(self._records(rows, ttfb=[0.6, 3.0, 0.6, 3.0]))
        self.assertEqual(tm.prefill_source, "ttfb-anchor")
        self.assertAlmostEqual(tm.b_minutes_per_input_token, 1e-05)

    def test_a_clean_primary_fit_never_touches_the_anchor(self) -> None:
        # The Claude-leg invariant: a log whose total_s fit succeeds keeps
        # byte-identical constants whether or not TTFB is present -- the
        # fallback is unreachable from the primary path.
        rows = [(1000, 100, 2.0), (5000, 100, 4.0), (1000, 300, 6.0), (5000, 300, 8.0)]
        plain = fit_timing_model(self._records(rows))
        with_ttfb = fit_timing_model(self._records(rows, ttfb=[0.5, 1.4, 0.5, 1.4]))
        self.assertEqual(plain.prefill_source, "total-s")
        self.assertEqual(plain, with_ttfb)

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


class TestTimingModelSeesContextNotBilledInput(unittest.TestCase):
    """The regressor is what the model READ, not what was billed as fresh input.

    Under the pinned cache breakpoints nearly the whole prompt bills as cache
    reads and writes, and billed `input_tokens` collapses to a near-constant
    residue -- the first real calibration carried input_tokens == 2 on every
    one of its 109 calls while the context varied by tens of thousands of
    tokens. The old fit regressed on billed input and was singular; where it
    did fit (the Haiku preflight), it priced 470K cache-read tokens as free.
    These tests fail if that regression returns."""

    A, B_FRESH, B_CACHED, THROUGHPUT = 0.01, 1e-6, 1e-7, 10_000.0

    def _record(self, i, fresh_in, cache_read, cache_write, out):
        from harness.proxy import CallRecord

        minutes = (self.A + self.B_FRESH * (fresh_in + cache_write)
                   + self.B_CACHED * cache_read + out / self.THROUGHPUT)
        return CallRecord(seq=i, path="", input_tokens=fresh_in, output_tokens=out,
                          cache_read_tokens=cache_read, cache_write_tokens=cache_write,
                          total_s=minutes * 60.0)

    def test_a_fully_cached_log_fits_where_the_old_regressor_was_singular(self) -> None:
        # Billed input constant at 2 on every call -- the exact shape of the
        # real overnight log -- with the context living in the cache columns.
        rows = [self._record(i, 2, cr, cw, out) for i, (cr, cw, out) in enumerate([
            (0, 1600, 800), (2400, 900, 300), (5200, 700, 1200),
            (9100, 500, 500), (14000, 400, 950),
        ])]
        tm = fit_timing_model(rows)
        self.assertAlmostEqual(tm.a_minutes, self.A, places=6)
        self.assertAlmostEqual(tm.b_minutes_per_input_token, self.B_FRESH, places=9)
        self.assertIsNotNone(tm.b_cached_minutes_per_token)
        self.assertAlmostEqual(tm.b_cached_minutes_per_token, self.B_CACHED, places=9)
        self.assertAlmostEqual(tm.output_tokens_per_minute, self.THROUGHPUT, places=2)
        # Cached prefill is the CHEAPER term -- that asymmetry is the reason
        # for the split fit, and the fallback below refuses its inversion.
        self.assertLess(tm.b_cached_minutes_per_token, tm.b_minutes_per_input_token)

    def test_an_uncached_log_reads_exactly_as_before(self) -> None:
        from harness.calibrate import call_minutes
        from harness.proxy import CallRecord

        rows = [self._record(i, f, 0, 0, o) for i, (f, o) in enumerate(
            [(1000, 100), (5000, 300), (1000, 300), (5000, 100), (3000, 200)])]
        tm = fit_timing_model(rows)
        self.assertIsNone(tm.b_cached_minutes_per_token)  # blended: nothing to split
        rec = CallRecord(seq=9, path="", input_tokens=2000, output_tokens=150, total_s=1.0)
        self.assertAlmostEqual(
            call_minutes(rec, tm),
            tm.a_minutes + tm.b_minutes_per_input_token * 2000 + 150 / tm.output_tokens_per_minute,
            places=12,
        )

    def test_cached_prefill_fitting_slower_than_fresh_falls_back_to_blended(self) -> None:
        # Data generated with the coefficients physically backwards: cache reads
        # ten times SLOWER than fresh prefill. The split fit recovers exactly
        # that, which is grounds to distrust the split, not to publish it.
        a, b_fresh, b_cached, thr = 0.01, 1e-7, 1e-6, 10_000.0
        from harness.proxy import CallRecord

        rows = []
        for i, (f, cr, o) in enumerate([(1000, 0, 800), (2000, 2400, 300),
                                        (5000, 5200, 1200), (1500, 9100, 500),
                                        (4200, 14000, 950)]):
            minutes = a + b_fresh * f + b_cached * cr + o / thr
            rows.append(CallRecord(seq=i, path="", input_tokens=f, output_tokens=o,
                                   cache_read_tokens=cr, total_s=minutes * 60.0))
        tm = fit_timing_model(rows)
        self.assertIsNone(tm.b_cached_minutes_per_token)  # fell back to blended
        self.assertGreaterEqual(tm.b_minutes_per_input_token, 0.0)

    def test_call_minutes_prices_cache_reads(self) -> None:
        from harness.calibrate import call_minutes
        from harness.proxy import CallRecord

        tm = TimingModel(0.0, 1e-6, 1e12, b_cached_minutes_per_token=1e-7)
        rec = CallRecord(seq=0, path="", input_tokens=10, output_tokens=0,
                         cache_read_tokens=1000, cache_write_tokens=90, total_s=1.0)
        # Fresh = 10 uncached + 90 written; cached = 1000 read.
        self.assertAlmostEqual(call_minutes(rec, tm), 1e-6 * 100 + 1e-7 * 1000, places=15)


class TestAbsorptionSeesTheCachedContextStep(unittest.TestCase):
    """The real fan-out log: billed input constant at 2, all context growth in
    the cache columns. The old extractor read a zero step off every such trace
    and reported 'no lead turn followed a spawn' -- wrong on both counts: the
    turns existed, and the step was there, in the fields it was not reading."""

    def _cached_fanout_trace(self, grow: int) -> Trace:
        from harness.trace import SpawnRecord

        trace = Trace(scenario_id="s", model="m")
        trace.calls.append(
            ModelCall(actor=LEAD, index=0, input_tokens=2, output_tokens=50,
                      cache_read_tokens=1000, cache_write_tokens=300,
                      tools_invoked=("spawn_subagent",), t_request=0.0)
        )
        trace.calls.append(
            ModelCall(actor=LEAD, index=1, input_tokens=2, output_tokens=30,
                      cache_read_tokens=1300 + grow, cache_write_tokens=200,
                      tools_invoked=("finish",), t_request=5.0)
        )
        trace.spawns.append(SpawnRecord(index=0, batch=0, instruction="do it", files=()))
        trace.calls.append(
            ModelCall(actor="subagent:0", index=0, input_tokens=400,
                      output_tokens=60, t_request=1.0)
        )
        return trace

    def test_the_step_is_read_from_the_cache_columns(self) -> None:
        tm = TimingModel(1.0, 0.001, 1e12, b_cached_minutes_per_token=0.0005)
        out = spawn_overhead([self._cached_fanout_trace(grow=4000)], PRICE, tm)
        # ctx before = 2 + 300 + 1000 = 1302; after = 2 + 200 + 5300 = 5502.
        self.assertIsNotNone(out["absorb_dollars_per_node"])
        self.assertAlmostEqual(out["absorb_dollars_per_node"], 4200 * 3.0 / 1e6, places=12)
        # Absorbed tokens are cache READS on every later turn, so the cached
        # coefficient prices their minutes, not the fresh one.
        self.assertAlmostEqual(out["absorb_minutes_per_node"], 4200 * 0.0005, places=9)

    def test_zero_growth_names_the_real_failure(self) -> None:
        out = spawn_overhead([self._cached_fanout_trace(grow=-500)], PRICE,
                             TimingModel(1.0, 0.0, 1e12))
        self.assertIsNone(out["absorb_dollars_per_node"])
        self.assertIn("post-spawn", out["absorb_why"])

    def test_no_post_spawn_turn_keeps_the_structural_message(self) -> None:
        trace = Trace(scenario_id="s", model="m")
        trace.calls.append(ModelCall(actor=LEAD, index=0, input_tokens=500, output_tokens=50,
                                     tools_invoked=("list_files",), t_request=0.0))
        trace.calls.append(ModelCall(actor=LEAD, index=1, input_tokens=600, output_tokens=200,
                                     tools_invoked=("spawn_subagent",), t_request=1.0))
        out = spawn_overhead([trace], PRICE, TimingModel(1.0, 0.0, 1e12))
        self.assertIn("no lead turn followed a spawn", out["absorb_why"])


class TestCalibrationAbortsBeforeSpendOnADegenerateFirstTrace(unittest.TestCase):
    """The first real calibration spent its whole overnight budget before the
    extraction reported an unfittable timing model. The gate reads the regressor
    columns off the first serial trace and stops the run right there."""

    def test_constant_columns_abort_after_one_trace(self) -> None:
        scn = scenario(n=3, size=1)
        flat = {
            "lead": [
                *[
                    Turn(tools=(("write_file", {"path": module_path(n),
                                                "content": scn.reference[n]}),),
                         input_tokens=800, output_tokens=100)
                    for n in scn.dag.ids
                ],
                Turn(tools=(("finish", {"summary": "done"}),),
                     input_tokens=800, output_tokens=100),
            ]
        }
        with ProtocolUpstream("anthropic", flat) as up:
            client = AnthropicClient(model="fake-model", api_key="k", base_url=up.base_url)
            with tempfile.TemporaryDirectory() as out:
                with self.assertRaises(ValueError) as caught:
                    run_calibration(scn, client, PRICE, out, source="flat 2026-08-23",
                                    orderings=2, repeats=2, max_turns=10)
                traces = sorted(Path(out).glob("trace-*.json"))
                # One trace on disk, seven runs never launched.
                self.assertEqual(len(traces), 1, traces)
        self.assertIn("constant", str(caught.exception))


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
