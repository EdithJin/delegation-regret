"""What the runner guarantees, checked without an API key or a dollar.

The runner is the piece of this repo that is easiest to get subtly wrong and
hardest to notice: one-level delegation, the concurrency cap, which spawns
overlap, and who wrote which file. Every one of those failures produces a trace
that looks perfectly plausible and a regret number that is quietly meaningless.

So the loop is driven by a scripted client and the guarantees are asserted
structurally. Nothing here tests a model; all of it tests the harness.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator.dag import sample_dag
from generator.oracle import Plan
from generator.scenario import build_scenario
from generator.templates import module_path, test_path
from harness.client import Reply, ScriptedClient, ToolRequest
from harness.runner import SUBAGENT_TOOLS, describe_plan, run_agent, run_plan
from harness.tools import MAX_CONCURRENCY
from harness.trace import LEAD, ModelCall, SpawnRecord, Trace, batch_makespan


def scenario(shape: str = "wide", n: int = 3, size: int = 1, seed: int = 5):
    return build_scenario(sample_dag(shape, n, sizes=(size,), seed=seed), f"{shape}{n}", seed=seed)


def call(name: str, arguments: dict, id: str = "t") -> ToolRequest:
    return ToolRequest(id=id, name=name, arguments=arguments)


def reply(*tools: ToolRequest, text: str = "", tokens: tuple[int, int] = (100, 20)) -> Reply:
    return Reply(
        text=text,
        tool_calls=tuple(tools),
        input_tokens=tokens[0],
        output_tokens=tokens[1],
        total_s=1.0,
        ttfb_s=0.5,
    )


def repair(scn, node_id: str) -> ToolRequest:
    """A tool call that writes the correct module for one node."""
    return call(
        "write_file",
        {"path": module_path(node_id), "content": scn.reference[node_id]},
        id=f"w-{node_id}",
    )


FINISH = call("finish", {"summary": "done"}, id="f")


class TestLeadOnlyRun(unittest.TestCase):
    """The k=0 path: the lead repairs everything itself."""

    def setUp(self) -> None:
        self.scn = scenario()
        self.client = ScriptedClient(
            script={"lead": [reply(*[repair(self.scn, n) for n in self.scn.dag.ids]), reply(FINISH)]}
        )

    def test_run_verifies_and_attributes_every_node_to_the_lead(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_agent(self.scn, self.client, tmp)
        self.assertTrue(trace.succeeded, trace.verdicts)
        self.assertTrue(trace.finished)
        self.assertEqual(trace.k, 0)
        self.assertEqual(set(trace.node_attribution.values()), {LEAD})
        self.assertEqual(sorted(trace.node_attribution), sorted(self.scn.dag.ids))

    def test_realized_plan_is_all_inline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_agent(self.scn, self.client, tmp)
        inline, blocks = trace.realized_plan()
        self.assertEqual(inline, frozenset(self.scn.dag.ids))
        self.assertEqual(blocks, ())


class TestDelegation(unittest.TestCase):
    """Spawning: one level deep, attributed, and batched."""

    def setUp(self) -> None:
        self.scn = scenario()
        ids = list(self.scn.dag.ids)
        # The lead spawns one subagent per node in ONE turn, then finishes.
        spawns = [
            call("spawn_subagent", {"instruction": f"fix {n}", "files": [module_path(n)]}, id=f"s-{n}")
            for n in ids
        ]
        script = {"lead": [reply(*spawns), reply(FINISH)]}
        for n in ids:
            script[f"fix {n}"] = [reply(repair(self.scn, n)), reply(FINISH)]
        self.client = ScriptedClient(script=script)
        self.ids = ids

    def test_subagents_do_the_work_and_are_attributed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_agent(self.scn, self.client, tmp)
        self.assertTrue(trace.succeeded, trace.verdicts)
        self.assertEqual(trace.k, len(self.ids))
        self.assertEqual(
            sorted(trace.node_attribution.values()),
            sorted(f"subagent:{i}" for i in range(len(self.ids))),
        )
        self.assertNotIn(LEAD, trace.node_attribution.values())

    def test_spawns_from_one_turn_share_a_batch(self) -> None:
        # The batch is what makes them concurrent. A lead that spawned serially
        # would produce distinct batches and pay for it in analytic latency.
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_agent(self.scn, self.client, tmp)
        self.assertEqual({s.batch for s in trace.spawns}, {0})

    def test_a_subagent_is_never_offered_the_spawn_tool(self) -> None:
        # One-level delegation, enforced by the allow-list rather than requested
        # in a prompt. This is the same property that makes two of the chain
        # plans infeasible in the oracle -- if the harness were more permissive,
        # the agent would be scored against a plan space it did not have.
        with tempfile.TemporaryDirectory() as tmp:
            run_agent(self.scn, self.client, tmp)
        subagent_offers = [allow for key, allow in self.client.seen if key.startswith("fix ")]
        self.assertTrue(subagent_offers)
        for allow in subagent_offers:
            self.assertNotIn("spawn_subagent", allow)
            self.assertEqual(allow, SUBAGENT_TOOLS)

    def test_a_subagent_that_tries_to_delegate_is_refused(self) -> None:
        scn = scenario()
        node = scn.dag.ids[0]
        client = ScriptedClient(
            script={
                "lead": [
                    reply(call("spawn_subagent", {"instruction": "go", "files": []}, id="s")),
                    reply(FINISH),
                ],
                "go": [
                    reply(call("spawn_subagent", {"instruction": "deeper", "files": []}, id="s2")),
                    reply(repair(scn, node)),
                    reply(FINISH),
                ],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_agent(scn, client, tmp)
        self.assertTrue(any("attempted to delegate" in n for n in trace.notes), trace.notes)
        self.assertEqual(trace.k, 1)  # the nested spawn created no second subagent

    def test_the_lead_sees_only_the_summary(self) -> None:
        scn = scenario(n=1)
        node = scn.dag.ids[0]
        client = ScriptedClient(
            script={
                "lead": [
                    reply(call("spawn_subagent", {"instruction": "one", "files": []}, id="s")),
                    reply(FINISH),
                ],
                "one": [
                    reply(repair(scn, node)),
                    reply(call("finish", {"summary": "repaired the module"}, id="f")),
                ],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_agent(scn, client, tmp)
        self.assertEqual(trace.spawns[0].summary, "repaired the module")
        self.assertTrue(trace.spawns[0].ok)


class TestGrading(unittest.TestCase):
    def test_an_unrepaired_run_does_not_succeed(self) -> None:
        scn = scenario()
        client = ScriptedClient(script={"lead": [reply(FINISH)]})
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_agent(scn, client, tmp)
        self.assertFalse(trace.succeeded)
        self.assertFalse(any(trace.verdicts.values()))

    def test_editing_a_suite_is_recorded_and_does_not_earn_a_pass(self) -> None:
        # The tamper guard, seen from the runner's side. Reading tampering has
        # to happen BEFORE verify(), which restores the suites and would erase
        # the evidence.
        scn = scenario()
        blank = "import unittest\n\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        pass\n"
        edits = [
            call("write_file", {"path": test_path(n), "content": blank}, id=f"e-{n}")
            for n in scn.dag.ids
        ]
        client = ScriptedClient(script={"lead": [reply(*edits), reply(FINISH)]})
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_agent(scn, client, tmp)
        self.assertEqual(sorted(trace.tampered_tests), sorted(scn.dag.ids))
        self.assertFalse(trace.succeeded)

    def test_a_run_that_never_finishes_is_flagged(self) -> None:
        scn = scenario()
        client = ScriptedClient(script={"lead": [reply(call("list_files", {}, id="l"))]})
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_agent(scn, client, tmp, max_turns=3)
        self.assertFalse(trace.finished)
        self.assertEqual(trace.turns, 3)
        self.assertTrue(any("never called finish" in n for n in trace.notes))


class TestForcedPlan(unittest.TestCase):
    """Section 6.1(b): execute a given plan for real, and check it was followed."""

    def setUp(self) -> None:
        self.scn = scenario(n=3)
        self.ids = list(self.scn.dag.ids)
        self.plan = Plan(frozenset({self.ids[0]}), (frozenset({self.ids[1], self.ids[2]}),))

    def _client(self, comply: bool) -> ScriptedClient:
        if comply:
            lead = reply(
                repair(self.scn, self.ids[0]),
                call(
                    "spawn_subagent",
                    {"instruction": "block", "files": []},
                    id="s",
                ),
            )
        else:
            lead = reply(*[repair(self.scn, n) for n in self.ids])  # did it all inline
        return ScriptedClient(
            script={
                "lead": [lead, reply(FINISH)],
                "block": [
                    reply(repair(self.scn, self.ids[1]), repair(self.scn, self.ids[2])),
                    reply(FINISH),
                ],
            }
        )

    def test_the_directive_names_the_nodes_on_both_sides(self) -> None:
        text = describe_plan(self.scn, self.plan)
        self.assertIn(self.ids[0], text)
        self.assertIn("Issue exactly 1 spawn_subagent call", text)
        self.assertIn(self.scn.subtasks[self.ids[1]].module, text)

    def test_pack_spawns_is_off_by_default_and_additive_when_on(self) -> None:
        # The baseline wording is the frozen instrument every existing run used;
        # the packing line is a separate condition, so it must be opt-in, appear
        # only when the plan actually spawns, and change the baseline text by
        # exactly one added line -- nothing rewritten.
        base = describe_plan(self.scn, self.plan)
        packed = describe_plan(self.scn, self.plan, pack_spawns=True)
        self.assertNotIn("ONE message", base)
        base_lines = [l for l in base.splitlines() if l]
        packed_lines = [l for l in packed.splitlines() if l]
        extra = [l for l in packed_lines if l not in base_lines]
        self.assertEqual(len(extra), 1, extra)
        self.assertIn("ONE message", extra[0])
        self.assertEqual([l for l in packed_lines if l in base_lines], base_lines)

        no_blocks = Plan(inline=frozenset(self.ids), blocks=())
        self.assertNotIn("ONE message", describe_plan(self.scn, no_blocks, pack_spawns=True))

    def test_a_compliant_run_carries_no_deviation_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_plan(self.scn, self.plan, self._client(True), tmp)
        self.assertTrue(trace.succeeded, trace.verdicts)
        self.assertFalse([n for n in trace.notes if n.startswith("PLAN NOT FOLLOWED")], trace.notes)
        inline, blocks = trace.realized_plan()
        self.assertEqual(inline, self.plan.inline)
        self.assertEqual(sorted(map(sorted, blocks)), sorted(map(sorted, self.plan.blocks)))

    def test_a_run_that_ignores_the_plan_is_flagged_not_repaired(self) -> None:
        # The baseline must be the plan it claims to be. A run that quietly did
        # something else would bias every regret number computed against it, so
        # it is recorded and Stage 6 excludes the scenario.
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_plan(self.scn, self.plan, self._client(False), tmp)
        self.assertTrue(any(n.startswith("PLAN NOT FOLLOWED") for n in trace.notes), trace.notes)

    def test_compliance_is_judged_on_attribution_not_on_what_was_said(self) -> None:
        # A lead that announces the plan in prose but does the work itself must
        # still be caught.
        client = ScriptedClient(
            script={
                "lead": [
                    reply(*[repair(self.scn, n) for n in self.ids], text="Delegating as instructed."),
                    reply(FINISH),
                ]
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_plan(self.scn, self.plan, client, tmp)
        self.assertTrue(any(n.startswith("PLAN NOT FOLLOWED") for n in trace.notes))


class TestAnalyticLatency(unittest.TestCase):
    """Latency is reconstructed, not summed and not clocked."""

    def setUp(self) -> None:
        from harness.calibrate import TimingModel

        # 1 minute per call, flat: makes every assertion below readable by hand.
        self.timing = TimingModel(
            a_minutes=1.0, b_minutes_per_input_token=0.0, output_tokens_per_minute=1e12
        )

    def _trace(self, spawn_batches: list[list[int]], lead_calls: int) -> Trace:
        """A synthetic trace: `spawn_batches[b]` gives each subagent's call count."""
        t = Trace(scenario_id="s", model="m")
        for i in range(lead_calls):
            t.calls.append(ModelCall(actor=LEAD, index=i, input_tokens=1, output_tokens=1))
        index = 0
        for batch, sizes in enumerate(spawn_batches):
            for n_calls in sizes:
                actor = f"subagent:{index}"
                t.spawns.append(SpawnRecord(index=index, batch=batch, instruction="", files=()))
                for j in range(n_calls):
                    t.calls.append(
                        ModelCall(actor=actor, index=j, input_tokens=1, output_tokens=1)
                    )
                index += 1
        return t

    def test_concurrent_subagents_cost_the_slowest_not_the_sum(self) -> None:
        # This is the whole mechanism by which delegation buys time. Summing
        # would price maximal fan-out identically to always-serial and delete
        # the finding the benchmark exists to produce.
        t = self._trace([[3, 3, 3]], lead_calls=2)
        self.assertAlmostEqual(t.analytic_minutes(self.timing), 2 + 3)

    def test_serialized_spawns_cost_more_than_batched_ones(self) -> None:
        batched = self._trace([[3, 3]], lead_calls=2)
        serial = self._trace([[3], [3]], lead_calls=2)
        self.assertLess(
            batched.analytic_minutes(self.timing), serial.analytic_minutes(self.timing)
        )

    def test_a_batch_wider_than_the_cap_queues(self) -> None:
        wide = self._trace([[2] * (MAX_CONCURRENCY + 1)], lead_calls=0)
        self.assertAlmostEqual(wide.analytic_minutes(self.timing), 4.0)  # two waves of 2 min

    def test_batch_makespan_packs_longest_first(self) -> None:
        self.assertAlmostEqual(batch_makespan([], cap=4), 0.0)
        self.assertAlmostEqual(batch_makespan([5.0, 1.0, 1.0], cap=4), 5.0)
        self.assertAlmostEqual(batch_makespan([3.0, 3.0, 3.0], cap=2), 6.0)

    def test_wall_clock_is_recorded_but_is_not_the_metric(self) -> None:
        scn = scenario()
        client = ScriptedClient(
            script={"lead": [reply(*[repair(scn, n) for n in scn.dag.ids]), reply(FINISH)]}
        )
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_agent(scn, client, tmp)
        self.assertGreater(trace.wall_seconds, 0.0)
        # The scripted client returns instantly, so wall clock and the analytic
        # figure are unrelated by construction -- which is the point.
        self.assertGreater(trace.analytic_minutes(self.timing), 0.0)


class TestBudgetGuard(unittest.TestCase):
    """A runaway loop against a paid endpoint costs money in proportion to how
    long nobody is watching, and the turn cap alone does not bound it: a lead
    that spawns four subagents per turn multiplies its own cap by five."""

    def test_the_call_ceiling_stops_the_run(self) -> None:
        from harness.runner import Budget

        scn = scenario()
        # A lead that never finishes: only the budget can stop it.
        client = ScriptedClient(script={"lead": [reply(call("list_files", {}, "l"))]})
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_agent(scn, client, tmp, max_turns=500, budget=Budget(max_calls=7))
        self.assertEqual(len(trace.calls), 7)
        self.assertTrue(any("call budget exhausted" in n for n in trace.notes), trace.notes)
        self.assertFalse(trace.finished)

    def test_the_ceiling_counts_subagents_too(self) -> None:
        # The whole point: a per-conversation cap would not bound a fan-out.
        from harness.runner import Budget

        scn = scenario(n=3)
        spawns = [
            call("spawn_subagent", {"instruction": f"fix {n}", "files": []}, f"s{n}")
            for n in scn.dag.ids
        ]
        script = {"lead": [reply(*spawns), reply(FINISH)],
                  "*": [reply(call("list_files", {}, "l"))]}
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_agent(scn, client_of(script), tmp, max_turns=50,
                              budget=Budget(max_calls=9))
        self.assertLessEqual(len(trace.calls), 9)
        self.assertTrue(any("budget exhausted" in n for n in trace.notes), trace.notes)

    def test_the_output_token_ceiling_also_stops_it(self) -> None:
        from harness.runner import Budget

        scn = scenario()
        client = ScriptedClient(
            script={"lead": [reply(call("list_files", {}, "l"), tokens=(10, 500))]}
        )
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_agent(scn, client, tmp, max_turns=500,
                              budget=Budget(max_output_tokens=1200))
        self.assertTrue(any("output-token budget" in n for n in trace.notes), trace.notes)
        self.assertLessEqual(sum(c.output_tokens for c in trace.calls), 1500)

    def test_an_aborted_run_is_a_failed_run(self) -> None:
        # Stopping mid-flight must not look like a cheap success.
        from harness.runner import Budget

        scn = scenario()
        client = ScriptedClient(script={"lead": [reply(call("list_files", {}, "l"))]})
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_agent(scn, client, tmp, budget=Budget(max_calls=2))
        self.assertFalse(trace.succeeded)


class TestRetry(unittest.TestCase):
    """A single rate limit used to end a conversation, and the run was then
    scored as a failure -- so a busy afternoon on the provider's side looked like
    a model that could not do the task. It looked like it ASYMMETRICALLY: wide
    fan-out makes more concurrent calls, draws more 429s, and would fail more
    often on the shape the headline finding rests on."""

    def setUp(self) -> None:
        from harness.runner import RetryPolicy

        self.fast = RetryPolicy(attempts=3, base_delay_s=0.0, max_delay_s=0.0)

    def test_a_rate_limit_is_retried_and_the_run_survives(self) -> None:
        import urllib.error

        scn = scenario()
        good = reply(*[repair(scn, n) for n in scn.dag.ids])
        flaky = _FlakyClient(
            script={"lead": [good, reply(FINISH)]},
            fail_on={1: urllib.error.HTTPError("u", 429, "slow down", {}, None)},
        )
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_agent(scn, flaky, tmp, retry=self.fast)
        self.assertTrue(trace.succeeded, trace.verdicts)
        self.assertTrue(any("retrying after HTTPError" in n for n in trace.notes), trace.notes)

    def test_retries_are_recorded_not_hidden(self) -> None:
        # A run that needed twenty retries is not the same measurement as one
        # that needed none, especially on the timing side.
        import urllib.error

        scn = scenario()
        flaky = _FlakyClient(
            script={"lead": [reply(*[repair(scn, n) for n in scn.dag.ids]), reply(FINISH)]},
            fail_on={1: urllib.error.HTTPError("u", 503, "overloaded", {}, None)},
        )
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_agent(scn, flaky, tmp, retry=self.fast)
        self.assertEqual(sum(1 for n in trace.notes if "retrying" in n), 1)

    def test_a_bad_request_is_not_retried(self) -> None:
        # A 400 from a malformed conversation will fail identically four more
        # times and hide the bug behind a delay.
        import urllib.error

        scn = scenario()
        flaky = _FlakyClient(
            script={"lead": [reply(FINISH)]},
            fail_on={1: urllib.error.HTTPError("u", 400, "bad shape", {}, None)},
            forever=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_agent(scn, flaky, tmp, retry=self.fast)
        self.assertFalse(any("retrying" in n for n in trace.notes), trace.notes)
        self.assertTrue(any("HTTPError" in n for n in trace.notes))

    def test_backoff_grows_and_is_capped(self) -> None:
        from harness.runner import RetryPolicy

        policy = RetryPolicy(attempts=6, base_delay_s=1.0, max_delay_s=8.0)
        self.assertEqual([policy.delay_for(i) for i in range(5)], [1.0, 2.0, 4.0, 8.0, 8.0])

    def test_transient_classification(self) -> None:
        import urllib.error

        from harness.runner import RetryPolicy

        self.assertTrue(RetryPolicy.is_transient(urllib.error.HTTPError("u", 429, "", {}, None)))
        self.assertTrue(RetryPolicy.is_transient(urllib.error.HTTPError("u", 503, "", {}, None)))
        self.assertTrue(RetryPolicy.is_transient(TimeoutError()))
        self.assertFalse(RetryPolicy.is_transient(urllib.error.HTTPError("u", 400, "", {}, None)))
        self.assertFalse(RetryPolicy.is_transient(ValueError("nope")))


def client_of(script):
    return ScriptedClient(script=script)


class _FlakyClient(ScriptedClient):
    """A scripted client that raises on chosen call numbers."""

    def __init__(self, *, script, fail_on, forever=False):
        super().__init__(script=script)
        self.fail_on = dict(fail_on)
        self.forever = forever
        self.n = 0

    def complete(self, history, allow):
        self.n += 1
        exc = self.fail_on.get(self.n)
        if exc is not None:
            if not self.forever:
                self.fail_on.pop(self.n)
            else:
                self.fail_on[self.n + 1] = exc
            raise exc
        return super().complete(history, allow)


class TestTraceRoundTrip(unittest.TestCase):
    def test_a_trace_survives_json(self) -> None:
        scn = scenario()
        client = ScriptedClient(
            script={"lead": [reply(*[repair(scn, n) for n in scn.dag.ids]), reply(FINISH)]}
        )
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_agent(scn, client, tmp)
            path = trace.write(Path(tmp) / "trace.json")
            self.assertEqual(json.loads(path.read_text())["schema_version"], 1)
            back = Trace.load(path)
        self.assertEqual(back.node_attribution, trace.node_attribution)
        self.assertEqual(back.verdicts, trace.verdicts)
        self.assertEqual(len(back.calls), len(trace.calls))
        self.assertEqual(back.k, trace.k)


if __name__ == "__main__":
    unittest.main()
