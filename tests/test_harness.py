"""What the measurement instrument guarantees, checked without spending anything.

Every reported dollar figure and latency is whatever the proxy wrote
down. A parser that silently misses the final usage block, or a relay that folds
generation time into prefill, does not produce an error -- it produces a clean
table of wrong numbers. So the instrument is tested against payloads whose
correct reading is known in advance, before it is pointed at anything billable.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import urllib.request
from pathlib import Path

from generator.templates import CheckResult
from harness.fake_upstream import (
    EXPECTED_ANTHROPIC,
    EXPECTED_OPENAI,
    EXPECTED_OPENAI_STREAM,
    FakeUpstream,
)
from harness.proxy import LoggingProxy, UsageSniffer
from harness.tools import TOOL_SPECS, Workspace, anthropic_tools, openai_tools, parse_arguments


class TestUsageSniffer(unittest.TestCase):
    def test_anthropic_stream_input_survives_the_output_delta(self) -> None:
        # message_start carries input tokens and message_delta carries the final
        # output count. Applying the delta's usage as a block would overwrite
        # input with nothing, and the cost half of every record would be empty.
        sniffer = UsageSniffer()
        sniffer.feed_event(
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 900, "cache_read_input_tokens": 400}},
            }
        )
        sniffer.feed_event({"type": "message_delta", "usage": {"output_tokens": 120}})
        self.assertEqual(sniffer.input_tokens, 900)
        self.assertEqual(sniffer.output_tokens, 120)
        self.assertEqual(sniffer.cache_read_tokens, 400)

    def test_sse_lines_split_across_reads_are_still_parsed(self) -> None:
        # SSE frames do not respect socket boundaries. Assuming whole lines
        # works on a fast localhost server and drops the final usage block on a
        # slow connection -- which is every real run.
        event = json.dumps({"type": "message_delta", "usage": {"output_tokens": 77}})
        raw = f"data: {event}\n\n".encode()
        sniffer = UsageSniffer()
        carry = b""
        for i in range(0, len(raw), 7):  # deliberately ugly chunk size
            carry = sniffer.feed_sse_chunk(raw[i : i + 7], carry)
        self.assertEqual(sniffer.output_tokens, 77)

    def test_anthropic_body_yields_usage_and_tool_calls(self) -> None:
        sniffer = UsageSniffer()
        sniffer.feed_body(
            json.dumps(
                {
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                    "content": [{"type": "tool_use", "name": "Task"}],
                    "stop_reason": "tool_use",
                }
            ).encode()
        )
        self.assertEqual((sniffer.input_tokens, sniffer.output_tokens), (10, 20))
        self.assertEqual(sniffer.tools_invoked, ["Task"])
        self.assertEqual(sniffer.stop_reason, "tool_use")

    def test_openai_body_is_normalized_onto_the_same_fields(self) -> None:
        # The open-weights leg has to land in the same log as the frontier legs
        # or the cross-model table is not comparable in the dimension the benchmark
        # claims to measure. "Normalized" includes the counting convention:
        # OpenAI's prompt_tokens is INCLUSIVE of cached tokens where Anthropic's
        # input_tokens is exclusive, so the sniffer records the fresh count
        # (30 - 25 = 5) -- the same subtraction OpenAIClient performs, without
        # which every cache-hitting run would bill cached tokens twice and fail
        # the trace/proxy cross-check. Cache writes pin to 0: OpenAI caching is
        # automatic and unbilled on write, so zero is the truth, not a gap.
        sniffer = UsageSniffer()
        sniffer.feed_body(
            json.dumps(
                {
                    "usage": {
                        "prompt_tokens": 30,
                        "completion_tokens": 40,
                        "prompt_tokens_details": {"cached_tokens": 25},
                    },
                    "choices": [
                        {
                            "message": {"tool_calls": [{"function": {"name": "read_file"}}]},
                            "finish_reason": "tool_calls",
                        }
                    ],
                }
            ).encode()
        )
        self.assertEqual((sniffer.input_tokens, sniffer.output_tokens), (5, 40))
        self.assertEqual(sniffer.cache_read_tokens, 25)
        self.assertEqual(sniffer.cache_write_tokens, 0)
        self.assertEqual(sniffer.tools_invoked, ["read_file"])

    def test_malformed_payloads_do_not_raise(self) -> None:
        sniffer = UsageSniffer()
        sniffer.feed_body(b"not json at all")
        sniffer.feed_sse_chunk(b"data: {broken\n\n")
        self.assertIsNone(sniffer.input_tokens)

    def test_a_path_pinned_openai_sniffer_reads_a_usage_only_chunk(self) -> None:
        # The include_usage finale is `"choices": []` from api.openai.com, and
        # some compatible servers omit the key entirely. Shape-sniffing reads
        # that frame as Anthropic-ish and drops the one chunk that carries the
        # money -- which is why the proxy pins the dialect from the request
        # path instead of guessing per frame.
        chunk = {"usage": {"prompt_tokens": 50, "completion_tokens": 9,
                           "prompt_tokens_details": {"cached_tokens": 20}}}
        pinned = UsageSniffer("openai")
        pinned.feed_event(dict(chunk))
        self.assertEqual((pinned.input_tokens, pinned.output_tokens), (30, 9))
        self.assertEqual(pinned.cache_read_tokens, 20)
        unpinned = UsageSniffer()
        unpinned.feed_event(dict(chunk))
        self.assertIsNone(unpinned.input_tokens)  # the failure the pin prevents

    def test_a_path_pinned_anthropic_sniffer_ignores_openai_shapes(self) -> None:
        # The complement: on /v1/messages the path decides too, so a payload
        # that happens to carry `choices` is not read as the other dialect.
        sniffer = UsageSniffer("anthropic")
        sniffer.feed_event({"choices": [], "usage": {"prompt_tokens": 50}})
        self.assertIsNone(sniffer.input_tokens)


class TestProxy(unittest.TestCase):
    def test_streaming_call_is_captured_exactly(self) -> None:
        with FakeUpstream(gap_s=0.03) as upstream, LoggingProxy(upstream.base_url) as proxy:
            request = urllib.request.Request(
                proxy.base_url + "/v1/messages",
                data=json.dumps({"stream": True, "model": "m", "tools": [{"name": "Task"}]}).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(request, timeout=30).read()
        record = proxy.records[-1]
        for key, value in EXPECTED_ANTHROPIC.items():
            self.assertEqual(getattr(record, key), value, key)
        self.assertEqual(record.tools_offered, ["Task"])
        self.assertTrue(record.billable)

    def test_the_relay_does_not_buffer_the_stream(self) -> None:
        # Regression test for a bug preflight caught on its first run: reading
        # the upstream with read(n) blocks until n bytes have accumulated, which
        # on a chunked response means waiting for the generation to finish. The
        # token counts stay perfectly correct and time-to-first-byte becomes
        # equal to the total, silently destroying the prefill term the cost
        # model is fitted on.
        with FakeUpstream(gap_s=0.05) as upstream, LoggingProxy(upstream.base_url) as proxy:
            request = urllib.request.Request(
                proxy.base_url + "/v1/messages",
                data=json.dumps({"stream": True}).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(request, timeout=30).read()
        record = proxy.records[-1]
        self.assertGreater(record.ttfb_s, 0.0)
        self.assertLess(record.ttfb_s, record.total_s * 0.8)

    def test_non_streaming_openai_call_is_captured(self) -> None:
        with FakeUpstream(gap_s=0.01) as upstream, LoggingProxy(upstream.base_url) as proxy:
            request = urllib.request.Request(
                proxy.base_url + "/v1/chat/completions",
                data=json.dumps({"model": "m"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(request, timeout=30).read()
        record = proxy.records[-1]
        for key, value in EXPECTED_OPENAI.items():
            self.assertEqual(getattr(record, key), value, key)

    def test_streaming_openai_call_is_captured_exactly(self) -> None:
        # The wire shape the OpenAI leg actually runs: chunked SSE with the
        # usage in a trailing include_usage frame. Same instrument guarantees
        # as the Anthropic stream -- exact tokens off the final frame, and a
        # first-byte timestamp that proves the relay did not buffer.
        with FakeUpstream(gap_s=0.03) as upstream, LoggingProxy(upstream.base_url) as proxy:
            request = urllib.request.Request(
                proxy.base_url + "/v1/chat/completions",
                data=json.dumps(
                    {"model": "m", "stream": True,
                     "stream_options": {"include_usage": True}}
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(request, timeout=30).read()
        record = proxy.records[-1]
        for key, value in EXPECTED_OPENAI_STREAM.items():
            self.assertEqual(getattr(record, key), value, key)
        self.assertTrue(record.billable)
        self.assertGreater(record.ttfb_s, 0.0)
        self.assertLess(record.ttfb_s, record.total_s * 0.8)

    def test_records_are_written_as_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "trace.jsonl"
            with FakeUpstream(gap_s=0.01) as upstream, LoggingProxy(
                upstream.base_url, log_path=log
            ) as proxy:
                request = urllib.request.Request(
                    proxy.base_url + "/v1/messages",
                    data=json.dumps({"stream": True}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(request, timeout=30).read()
            rows = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["input_tokens"], EXPECTED_ANTHROPIC["input_tokens"])


class TestToolSurface(unittest.TestCase):
    def test_both_wire_shapes_expose_the_same_tools(self) -> None:
        names = {n for n, _, _ in TOOL_SPECS}
        self.assertEqual({t["name"] for t in anthropic_tools()}, names)
        self.assertEqual({t["function"]["name"] for t in openai_tools()}, names)
        self.assertGreaterEqual(len(names), 6)

    def test_the_concurrency_cap_is_published_to_the_model(self) -> None:
        # Section 5.1: the oracle's action space is capped identically, so a cap
        # the model is never told about would penalize it for a constraint it
        # could not have known to respect.
        spawn = next(d for n, d, _ in TOOL_SPECS if n == "spawn_subagent")
        self.assertIn("4", spawn)
        self.assertIn("cannot", spawn)


class TestWorkspace(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "seed").mkdir()
        (self.root / "seed" / "base.py").write_text("VALUE = 41\n", encoding="utf-8")
        self.bench = Workspace(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_reads_and_writes_inside_the_root(self) -> None:
        self.assertIn("VALUE = 41", self.bench.read_file("seed/base.py"))
        self.bench.write_file("work/n0.py", "X = 1\n")
        self.assertEqual((self.root / "work" / "n0.py").read_text(), "X = 1\n")

    def test_escaping_the_workspace_is_refused(self) -> None:
        # Not a security posture -- the code is ours. A run that escaped would
        # contaminate every later scenario in the same session, and the damage
        # would surface days later as unexplained variance.
        for path in ("../outside.py", "seed/../../outside.py", "/etc/hosts"):
            with self.subTest(path=path):
                self.assertFalse(self.bench.invoke("write_file", {"path": path, "content": "x"}).ok)

    def test_run_python_executes_in_the_workspace(self) -> None:
        call = self.bench.invoke("run_python", {"code": "print(open('seed/base.py').read())"})
        self.assertTrue(call.ok)
        self.assertIn("VALUE = 41", call.result)

    def test_a_protocol_fumble_is_recorded_rather_than_repaired(self) -> None:
        # This is the signal the open-weights leg is judged on, so a wrong-shaped
        # call has to survive into the record instead of being papered over.
        self.assertFalse(self.bench.invoke("read_file", {"wrong_arg": "x"}).ok)
        self.assertFalse(self.bench.invoke("no_such_tool", {}).ok)
        self.assertEqual(self.bench.summary()["failed"], 2)

    def test_spawn_is_recorded_but_not_executed(self) -> None:
        call = self.bench.invoke("spawn_subagent", {"instruction": "do a thing"})
        self.assertTrue(call.ok)
        self.assertEqual(self.bench.summary()["spawns"], 1)

    def test_unparseable_arguments_are_reported(self) -> None:
        self.assertEqual(parse_arguments('{"path": "a.py"}'), {"path": "a.py"})
        self.assertEqual(parse_arguments(""), {})
        with self.assertRaises(ValueError):
            parse_arguments("{not json")
        with self.assertRaises(ValueError):
            parse_arguments("[1, 2]")


class TestSmokeCommandsStayWiredToTheGenerator(unittest.TestCase):
    """Regression, and the reason the other regressions here exist.

    `smoke.py` sits above the generator and below any paid run, so nothing
    imported it and nothing exercised it. The bug-injection rewrite renamed
    three things it depended on -- a template constant, `Scenario.seed`, and
    `CheckResult.reason` -- and the module stopped importing entirely while a
    green suite reported otherwise. A test that merely imports it would have
    caught the first; these reach the other two.
    """

    def test_the_module_imports(self) -> None:
        import harness.smoke  # noqa: F401

    def test_make_scenario_builds_a_verifiable_package(self) -> None:
        from harness.smoke import make_scenario

        scenario = make_scenario("diamond", 5, 1, 3)
        self.assertEqual(len(scenario.subtasks), 5)
        with tempfile.TemporaryDirectory() as tmp:
            scenario.materialize(tmp)
            scenario.write_reference(tmp)
            self.assertTrue(scenario.succeeded(tmp))

    def test_preflight_reports_on_the_scenario_it_builds(self) -> None:
        # Exercises the summary line that read the removed `Scenario.seed`.
        # The command's exit code depends on whether node and the claude CLI
        # are installed, which is a fact about this machine, not about the
        # code -- so the assertion is that it runs and grades the scenario row.
        import argparse

        from harness.smoke import PASS, cmd_preflight

        args = argparse.Namespace(shape="wide", n=3, size=1, seed=0, gap=0.01)
        rows: list = []
        import harness.smoke as smoke

        original = smoke.Report.add

        def capture(self, status, name, detail="") -> None:
            rows.append((status, name, detail))
            original(self, status, name, detail)

        smoke.Report.add = capture
        try:
            # The command is a console report; the assertions read `rows`.
            with contextlib.redirect_stdout(io.StringIO()):
                cmd_preflight(args)
        finally:
            smoke.Report.add = original

        scenario_rows = [r for r in rows if "scenario materializes" in r[1]]
        self.assertEqual(len(scenario_rows), 1)
        self.assertEqual(scenario_rows[0][0], PASS, scenario_rows[0][2])

    def test_failure_detail_reads_the_field_check_result_actually_has(self) -> None:
        from harness.smoke import failed_detail

        results = {
            "n0": CheckResult("n0", True),
            "n1": CheckResult("n1", False, "AssertionError: 7 != 5"),
        }
        detail = failed_detail(results)
        self.assertIn("n1", detail)
        self.assertIn("AssertionError", detail)
        self.assertNotIn("n0", detail)


if __name__ == "__main__":
    unittest.main()
