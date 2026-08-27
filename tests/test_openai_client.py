"""The OpenAI chat-completions leg: its pinned spec, its stream, its dispatch.

tests/test_client.py already drives both wire formats through the runner's
delegation round trip. What it does not pin is everything SPECIFIC to the
chat-completions leg -- the streaming request parameters actually sent, the
reassembly of a turn that arrives as deltas, the cached-token subtraction that
keeps the two providers' usage in one currency, the trace/proxy agreement that
subtraction exists to protect, and the model-to-client dispatch in harness.cli.
Each of those can drift silently while the round-trip tests stay green, so each
is pinned here against the validating fake, offline, keyless.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator.dag import sample_dag
from generator.scenario import build_scenario
from generator.templates import module_path
from harness.cli import (
    ANTHROPIC_API,
    HARNESS_SPEC,
    OPENAI_API,
    PRICE_SHEETS,
    build_client,
    endpoint_defaults,
    provider_for,
)
from harness.client import AnthropicClient, OpenAIClient
from harness.fake_upstream import EXPECTED_OPENAI_STREAM, FakeUpstream
from harness.protocol_upstream import ProtocolError, ProtocolUpstream, Turn, _validate_openai
from harness.proxy import LoggingProxy
from harness.runner import run_agent


def pinned_client(base_url: str) -> OpenAIClient:
    """The gpt-5.6-sol harness spec, as build_client would assemble it."""
    return OpenAIClient(model="gpt-5.6-sol", api_key="k", base_url=base_url,
                        max_tokens=16000, effort="high")


class TestPinnedSpecOnTheWire(unittest.TestCase):
    """The published constants of the OpenAI leg, checked on the wire.

    max_completion_tokens 16000 and reasoning_effort high are the leg's half of
    the Aug 24 harness-spec discipline; stream + stream_options.include_usage
    are not optional pins but the shape of the measurement itself -- without
    include_usage the final chunk carries no usage and every Reply reads zero.
    """

    def _sent(self, client: OpenAIClient, up: ProtocolUpstream) -> dict:
        client.complete(client.start("sys", "Read TASKS.md and go", "lead"), ("finish",))
        return up.requests[-1]

    def test_pinned_params_reach_the_provider(self) -> None:
        with ProtocolUpstream("openai", {"lead": [Turn(text="ok")]}) as up:
            sent = self._sent(pinned_client(up.base_url), up)
            self.assertEqual(up.violations, [])
        self.assertEqual(sent["max_completion_tokens"], 16000)
        self.assertEqual(sent["reasoning_effort"], "high")
        self.assertIs(sent["stream"], True)
        self.assertEqual(sent["stream_options"], {"include_usage": True})
        self.assertEqual(sent["tool_choice"], "auto")
        self.assertNotIn("max_tokens", sent)  # the legacy key gpt-* models 400
        self.assertNotIn("thinking", sent)  # an Anthropic-only parameter

    def test_the_system_prompt_opens_the_conversation(self) -> None:
        with ProtocolUpstream("openai", {"lead": [Turn(text="ok")]}) as up:
            sent = self._sent(pinned_client(up.base_url), up)
        self.assertEqual(sent["messages"][0], {"role": "system", "content": "sys"})

    def test_a_plumbing_client_sends_no_effort(self) -> None:
        # The defaults are the keyless-test configuration, mirroring the
        # Anthropic side: no reasoning_effort key -- but the stream itself is
        # not a pin that can be turned off, so it stays.
        with ProtocolUpstream("openai", {"lead": [Turn(text="ok")]}) as up:
            client = OpenAIClient(model="oss-fake", api_key="k", base_url=up.base_url)
            sent = self._sent(client, up)
        self.assertNotIn("reasoning_effort", sent)
        self.assertIs(sent["stream"], True)

    def test_tools_travel_in_the_function_envelope(self) -> None:
        with ProtocolUpstream("openai", {"lead": [Turn(text="ok")]}) as up:
            client = pinned_client(up.base_url)
            client.complete(client.start("s", "Read TASKS.md", "lead"),
                            ("read_file", "finish"))
            sent = up.requests[-1]
        for tool in sent["tools"]:
            self.assertEqual(tool["type"], "function")
            self.assertEqual(sorted(tool["function"]), ["description", "name", "parameters"])
        names = sorted(t["function"]["name"] for t in sent["tools"])
        self.assertEqual(names, ["finish", "read_file"])


class TestStreamedTurnAssembly(unittest.TestCase):
    """One assistant turn arrives as many deltas; the Reply must not show it."""

    def test_text_split_across_chunks_reassembles(self) -> None:
        with ProtocolUpstream("openai", {"lead": [Turn(text="delegation is not free")]}) as up:
            client = pinned_client(up.base_url)
            reply = client.complete(client.start("s", "Read TASKS.md", "lead"), ("finish",))
        self.assertEqual(reply.text, "delegation is not free")
        self.assertEqual(reply.stop_reason, "stop")
        self.assertEqual(reply.tool_calls, ())

    def test_a_multi_tool_call_turn_arrives_in_order(self) -> None:
        script = {"lead": [Turn(tools=(("read_file", {"path": "a.py"}),
                                       ("write_file", {"path": "b.py", "content": "x"})))]}
        with ProtocolUpstream("openai", script) as up:
            client = pinned_client(up.base_url)
            reply = client.complete(client.start("s", "Read TASKS.md", "lead"),
                                    ("read_file", "write_file", "finish"))
        self.assertEqual([c.name for c in reply.tool_calls], ["read_file", "write_file"])
        self.assertEqual(reply.tool_calls[0].arguments, {"path": "a.py"})
        self.assertEqual(reply.tool_calls[1].arguments, {"path": "b.py", "content": "x"})
        self.assertEqual(len({c.id for c in reply.tool_calls}), 2)
        self.assertEqual(reply.stop_reason, "tool_calls")

    def test_arguments_fragmented_mid_json_still_parse(self) -> None:
        # The canned FakeUpstream stream cuts an argument string inside a JSON
        # token. A client that parses per-chunk instead of concatenating first
        # sees garbage twice; a correct one sees one dict.
        with FakeUpstream(gap_s=0.0) as up:
            client = OpenAIClient(model="oss-fake", api_key="k", base_url=up.base_url)
            reply = client.complete(client.start("s", "go", "lead"), ("write_file",))
        self.assertEqual(len(reply.tool_calls), 1)
        self.assertEqual(reply.tool_calls[0].arguments,
                         {"path": "pkg/mod_n0.py", "content": "X = 1\n"})

    def test_ttfb_is_the_first_byte_not_the_last(self) -> None:
        # Served slowly on purpose: a client that buffered the stream (or read
        # with a blocking read(n)) would report ttfb equal to total, silently
        # destroying the prefill term -- invisible on a fast local server.
        with FakeUpstream(gap_s=0.05) as up:
            client = OpenAIClient(model="oss-fake", api_key="k", base_url=up.base_url)
            reply = client.complete(client.start("s", "go", "lead"), ("write_file",))
        self.assertGreater(reply.ttfb_s, 0.0)
        self.assertLess(reply.ttfb_s, reply.total_s * 0.8)


class TestUsageMapping(unittest.TestCase):
    """OpenAI usage lands on the same Reply fields, in the same currency."""

    def test_usage_comes_from_the_include_usage_finale(self) -> None:
        with FakeUpstream(gap_s=0.0) as up:
            client = OpenAIClient(model="oss-fake", api_key="k", base_url=up.base_url)
            reply = client.complete(client.start("s", "go", "lead"), ("write_file",))
        self.assertEqual(reply.input_tokens, EXPECTED_OPENAI_STREAM["input_tokens"])
        self.assertEqual(reply.output_tokens, EXPECTED_OPENAI_STREAM["output_tokens"])
        self.assertEqual(reply.cache_read_tokens, EXPECTED_OPENAI_STREAM["cache_read_tokens"])

    def test_the_write_column_is_a_true_zero(self) -> None:
        # OpenAI caching is automatic and unbilled on write: there is no write
        # count to read, and zero is the billed truth rather than a gap. The
        # cost model multiplies this by the sheet's $0.00 write rate.
        script = {"lead": [Turn(input_tokens=500, cache_read_tokens=400)]}
        with ProtocolUpstream("openai", script) as up:
            client = pinned_client(up.base_url)
            reply = client.complete(client.start("s", "Read TASKS.md", "lead"), ("finish",))
        self.assertEqual(reply.input_tokens, 500)  # 900 on the wire, 400 cached
        self.assertEqual(reply.cache_read_tokens, 400)
        self.assertEqual(reply.cache_write_tokens, 0)


class TestToolResultRoundTrip(unittest.TestCase):
    """The second turn is the test: history built from a streamed reply has to
    be accepted back, which is exactly where a reconstruction bug lives."""

    SCRIPT = {"lead": [Turn(tools=(("read_file", {"path": "a.py"}),
                                   ("write_file", {"path": "b.py", "content": "x"}))),
                       Turn(text="done")]}

    def test_the_replayed_turn_keeps_arguments_as_wire_strings(self) -> None:
        with ProtocolUpstream("openai", self.SCRIPT) as up:
            client = pinned_client(up.base_url)
            history = client.start("s", "Read TASKS.md", "lead")
            reply = client.complete(history, ("read_file", "write_file", "finish"))
            client.append_assistant(history, reply)
            client.append_tool_results(history, [(reply.tool_calls[0], "contents"),
                                                 (reply.tool_calls[1], "wrote 1 char")])
            client.complete(history, ("read_file", "write_file", "finish"))
            self.assertEqual(up.violations, [])
            sent = up.requests[-1]["messages"]
        assistant = sent[2]
        self.assertEqual(assistant["role"], "assistant")
        for call in assistant["tool_calls"]:
            # Verbatim wire strings, not re-serialized parses: a real endpoint
            # 400s object-typed arguments, and ProtocolUpstream now does too.
            self.assertIsInstance(call["function"]["arguments"], str)
        self.assertEqual(json.loads(assistant["tool_calls"][0]["function"]["arguments"]),
                         {"path": "a.py"})
        self.assertEqual([m["role"] for m in sent[3:5]], ["tool", "tool"])
        self.assertEqual(sent[3]["tool_call_id"], assistant["tool_calls"][0]["id"])
        self.assertEqual(sent[4]["tool_call_id"], assistant["tool_calls"][1]["id"])


class TestTraceAndProxyAgree(unittest.TestCase):
    """The cross-check the cached-token subtraction exists to protect: client
    and sniffer parse the same stream with different code, and a run whose two
    readings disagree is disqualified. Cache reads are nonzero in every script
    here, because with a zero cache the inclusive/exclusive conventions agree
    by accident and the test would prove nothing."""

    def test_every_token_field_matches_through_the_proxy(self) -> None:
        script = {"lead": [Turn(tools=(("read_file", {"path": "a.py"}),),
                                input_tokens=700, output_tokens=90, cache_read_tokens=300)]}
        with ProtocolUpstream("openai", script) as up, LoggingProxy(up.base_url) as proxy:
            client = pinned_client(proxy.base_url)
            reply = client.complete(client.start("s", "Read TASKS.md", "lead"),
                                    ("read_file", "finish"))
        record = proxy.records[-1]
        self.assertEqual(record.input_tokens, reply.input_tokens)
        self.assertEqual(record.input_tokens, 700)  # 1000 on the wire, 300 cached
        self.assertEqual(record.output_tokens, reply.output_tokens)
        self.assertEqual(record.cache_read_tokens, reply.cache_read_tokens)
        self.assertEqual(record.cache_write_tokens, reply.cache_write_tokens)
        self.assertEqual(record.stop_reason, reply.stop_reason)
        self.assertEqual(record.tools_invoked, ["read_file"])
        self.assertTrue(record.stream)
        self.assertTrue(record.billable)
        self.assertEqual(record.model, "gpt-5.6-sol")

    def test_a_cached_run_survives_the_runner_cross_check(self) -> None:
        # End to end: run_agent routes through the logging proxy on its own and
        # stamps PROXY MISMATCH onto the trace if the halves disagree. Before
        # the sniffer subtracted cached tokens, exactly this run failed.
        scn = build_scenario(sample_dag("wide", 1, sizes=(1,), seed=7), "w1", seed=7)
        node = scn.dag.ids[0]
        script = {"lead": [
            Turn(tools=(("write_file", {"path": module_path(node),
                                        "content": scn.reference[node]}),),
                 cache_read_tokens=350),
            Turn(tools=(("finish", {"summary": "done"}),), cache_read_tokens=350),
        ]}
        with ProtocolUpstream("openai", script) as up:
            client = pinned_client(up.base_url)
            with tempfile.TemporaryDirectory() as tmp:
                trace = run_agent(scn, client, tmp)
        self.assertEqual(up.violations, [])
        self.assertEqual(trace.notes, [])
        self.assertTrue(trace.proxy_verified)
        self.assertTrue(trace.succeeded)


class TestProviderDispatch(unittest.TestCase):
    """harness.cli owns the model-to-client mapping; probe.py and matrix.py
    construct through it too, so this is the one mapping to pin."""

    @staticmethod
    def _args(**over: object) -> SimpleNamespace:
        base = dict(base_url="http://127.0.0.1:1", api_key_env="DELEGATION_TEST_KEY",
                    max_tokens=16000, timeout=300.0)
        base.update(over)
        return SimpleNamespace(**base)

    def test_provider_for_splits_on_the_prefix(self) -> None:
        self.assertEqual(provider_for("gpt-5.6-sol"), "openai")
        self.assertEqual(provider_for("claude-opus-5"), "anthropic")
        self.assertEqual(provider_for("claude-haiku-4-5"), "anthropic")

    def test_a_gpt_model_gets_the_openai_client_with_its_spec(self) -> None:
        with mock.patch.dict("os.environ", {"DELEGATION_TEST_KEY": "sk-test"}):
            client = build_client(self._args(), PRICE_SHEETS["gpt-5.6-sol"])
        self.assertIsInstance(client, OpenAIClient)
        self.assertEqual(client.model, "gpt-5.6-sol")
        self.assertEqual(client.effort, "high")
        self.assertEqual(client.max_tokens, 16000)

    def test_the_list_sheet_prices_the_same_client(self) -> None:
        # "@list" restates rates; the client it builds must be identical.
        with mock.patch.dict("os.environ", {"DELEGATION_TEST_KEY": "sk-test"}):
            client = build_client(self._args(), PRICE_SHEETS["gpt-5.6-sol@list"])
        self.assertIsInstance(client, OpenAIClient)
        self.assertEqual(client.model, "gpt-5.6-sol")

    def test_a_claude_model_still_gets_the_anthropic_client(self) -> None:
        with mock.patch.dict("os.environ", {"DELEGATION_TEST_KEY": "sk-test"}):
            client = build_client(self._args(), PRICE_SHEETS["claude-opus-5"])
        self.assertIsInstance(client, AnthropicClient)
        self.assertEqual(client.thinking, {"type": "adaptive"})
        self.assertEqual(client.cache_ttl, "1h")

    def test_untouched_anthropic_defaults_reroute_for_gpt(self) -> None:
        # The CLI defaults are Anthropic-shaped. Left untouched with a gpt
        # model selected, they re-route -- typing --model gpt-5.6-sol must not
        # quietly post chat completions at api.anthropic.com.
        self.assertEqual(endpoint_defaults("gpt-5.6-sol", ANTHROPIC_API, "ANTHROPIC_API_KEY"),
                         (OPENAI_API, "OPENAI_API_KEY"))
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            client = build_client(
                self._args(base_url=ANTHROPIC_API, api_key_env="ANTHROPIC_API_KEY"),
                PRICE_SHEETS["gpt-5.6-sol"],
            )
        self.assertEqual(client.base_url, OPENAI_API)
        self.assertEqual(client.api_key, "sk-test")

    def test_explicit_flags_always_win(self) -> None:
        # Which is how tests (and open-source endpoints) point a gpt model at
        # any base URL, and how a differently named key variable stays honored.
        self.assertEqual(endpoint_defaults("gpt-5.6-sol", "http://localhost:8000", "MY_KEY"),
                         ("http://localhost:8000", "MY_KEY"))
        self.assertEqual(endpoint_defaults("claude-opus-5", ANTHROPIC_API, "ANTHROPIC_API_KEY"),
                         (ANTHROPIC_API, "ANTHROPIC_API_KEY"))

    def test_the_openai_spec_pins_what_it_should_and_nothing_else(self) -> None:
        # max_tokens 16000 matches the --max-tokens default every leg runs
        # under; thinking and cache_ttl are deliberately absent -- there is no
        # TTL to pin on automatic caching, which is the published
        # comparability caveat on the HARNESS_SPEC entry.
        self.assertEqual(HARNESS_SPEC["gpt-5.6-sol"], {"max_tokens": 16000, "effort": "high"})

    def test_the_promotional_sheet_is_dated_inside_its_window(self) -> None:
        promo, lst = PRICE_SHEETS["gpt-5.6-sol"], PRICE_SHEETS["gpt-5.6-sol@list"]
        self.assertEqual((promo.input_per_mtok, promo.output_per_mtok,
                          promo.cache_read_per_mtok), (4.00, 20.00, 0.40))
        self.assertEqual((lst.input_per_mtok, lst.output_per_mtok,
                          lst.cache_read_per_mtok), (5.00, 30.00, 0.50))
        self.assertEqual(promo.as_of, "2026-08-27")
        self.assertEqual(lst.as_of, "2026-08-27")


class TestTheFakeRejectsTheRealEndpoint400s(unittest.TestCase):
    """Mirror of the Anthropic parameter-rejection test: each check must be
    discriminating, so the accepted shape is asserted accepted at the end."""

    @staticmethod
    def _payload(**over: object) -> dict:
        base: dict = {
            "model": "gpt-5.6-sol",
            "messages": [{"role": "system", "content": "s"},
                         {"role": "user", "content": "go"}],
            "tools": [{"type": "function",
                       "function": {"name": "x", "description": "d", "parameters": {}}}],
            "max_completion_tokens": 16000,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        base.update(over)
        return base

    def test_parameter_400s(self) -> None:
        with self.assertRaises(ProtocolError):  # stream_options needs stream
            _validate_openai(self._payload(stream=False))
        with self.assertRaises(ProtocolError):  # gpt-* rejects the legacy key
            _validate_openai(self._payload(max_tokens=4096))
        with self.assertRaises(ProtocolError):  # zero truncates before the first token
            _validate_openai(self._payload(max_completion_tokens=0))
        with self.assertRaises(ProtocolError):  # an effort the API does not sell
            _validate_openai(self._payload(reasoning_effort="extreme"))
        with self.assertRaises(ProtocolError):  # model is required
            _validate_openai(self._payload(model=""))

    def test_replayed_tool_calls_must_carry_ids_and_string_arguments(self) -> None:
        def with_turn(call: dict) -> dict:
            return self._payload(messages=[
                {"role": "system", "content": "s"},
                {"role": "user", "content": "go"},
                {"role": "assistant", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call.get("id"), "content": "r"},
            ])

        with self.assertRaises(ProtocolError):  # parsed-object arguments
            _validate_openai(with_turn({"id": "c1", "type": "function",
                                        "function": {"name": "x", "arguments": {"p": 1}}}))
        with self.assertRaises(ProtocolError):  # a call with no id cannot be answered
            _validate_openai(with_turn({"type": "function",
                                        "function": {"name": "x", "arguments": "{}"}}))
        # And the correct replay is accepted, so the rejections discriminate.
        _validate_openai(with_turn({"id": "c1", "type": "function",
                                    "function": {"name": "x", "arguments": '{"p": 1}'}}))

    def test_the_accepted_shape_is_accepted(self) -> None:
        self.assertEqual(_validate_openai(self._payload(reasoning_effort="high")), "go")

    def test_forgetting_include_usage_yields_a_stream_with_no_usage(self) -> None:
        # Not a 400 on a real endpoint -- worse: a silent zero in every token
        # column. The fake reproduces that so the omission is testable.
        script = {"lead": [Turn(text="ok", input_tokens=900, output_tokens=120)]}
        with ProtocolUpstream("openai", script) as up:
            client = pinned_client(up.base_url)
            original = client.complete

            def stripped(history, allow):  # a client that forgot the option
                import harness.client as C

                real_post = C._post_sse

                def no_usage(url, payload, headers, timeout):
                    payload = dict(payload)
                    payload.pop("stream_options")
                    return real_post(url, payload, headers, timeout)

                C._post_sse = no_usage
                try:
                    return original(history, allow)
                finally:
                    C._post_sse = real_post

            reply = stripped(client.start("s", "Read TASKS.md", "lead"), ("finish",))
        self.assertEqual(up.violations, [])
        self.assertEqual(reply.text, "ok")  # the content still streamed
        self.assertEqual(reply.input_tokens, 0)  # but nothing was measured
        self.assertEqual(reply.output_tokens, 0)


if __name__ == "__main__":
    unittest.main()
