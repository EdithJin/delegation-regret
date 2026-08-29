"""The OpenAI responses leg: its pinned spec, its stream, its replay discipline.

tests/test_openai_client.py pins the chat-completions leg, and its dispatch
tests now assert that gpt-5.6-sol builds THIS client instead -- the move exists
because the live endpoint 400s function tools on chat completions unless
reasoning_effort is "none", and names /v1/responses as the API that takes tools
and reasoning together. What that file cannot cover is everything SPECIFIC to
this wire format: the flat tool shape, the reasoning parameter riding next to
tools in one accepted request, the turn arriving as typed events with a
terminal snapshot, the reasoning-item replay that statelessness makes the
client's job, the usage block whose output figure already contains the
reasoning tokens, and the proxy's third flavor. Each is pinned here against the
validating fake, offline, keyless.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator.dag import sample_dag
from generator.scenario import build_scenario
from generator.templates import module_path
from harness.cli import (
    ANTHROPIC_API,
    HARNESS_SPEC,
    OPENAI_API,
    client_for_model,
    endpoint_defaults,
)
from harness.client import OpenAIClient, OpenAIResponsesClient
from harness.protocol_upstream import (
    ProtocolError,
    ProtocolUpstream,
    Turn,
    _validate_responses,
)
from harness.proxy import LoggingProxy, UsageSniffer, _flavor_for_path
from harness.runner import run_agent


def pinned_client(base_url: str) -> OpenAIResponsesClient:
    """The gpt-5.6-sol harness spec, as build_client assembles it."""
    return OpenAIResponsesClient(model="gpt-5.6-sol", api_key="k", base_url=base_url,
                                 max_tokens=16000, effort="high")


def _first_user(request: dict) -> str:
    """The text that keys the fake's script: the first user message in `input`."""
    for item in request.get("input") or []:
        if isinstance(item, dict) and item.get("role") == "user":
            content = item.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(p.get("text", "") for p in content
                               if isinstance(p, dict) and p.get("type") == "input_text")
    return ""


class TestPinnedSpecOnTheWire(unittest.TestCase):
    """The published constants of the responses leg, checked on the wire.

    The defining assertion is the first one: tools and reasoning effort "high"
    travel in ONE request and the validating fake accepts it -- the pairing
    chat completions rejects is the entire reason this client exists.
    """

    def _sent(self, client: OpenAIResponsesClient, up: ProtocolUpstream) -> dict:
        client.complete(client.start("sys", "Read TASKS.md and go", "lead"), ("finish",))
        return up.requests[-1]

    def test_tools_and_reasoning_travel_together_and_are_accepted(self) -> None:
        with ProtocolUpstream("responses", {"lead": [Turn(text="ok")]}) as up:
            sent = self._sent(pinned_client(up.base_url), up)
            self.assertEqual(up.violations, [])
        self.assertEqual(sent["reasoning"], {"effort": "high"})
        self.assertTrue(sent["tools"])
        self.assertEqual(sent["max_output_tokens"], 16000)
        self.assertIs(sent["stream"], True)
        self.assertEqual(sent["tool_choice"], "auto")
        # Statelessness and its price, pinned: nothing stored server-side, and
        # reasoning items requested in replayable (encrypted) form.
        self.assertIs(sent["store"], False)
        self.assertEqual(sent["include"], ["reasoning.encrypted_content"])
        self.assertNotIn("previous_response_id", sent)
        # The chat-completions keys this API would 400.
        self.assertNotIn("messages", sent)
        self.assertNotIn("max_tokens", sent)
        self.assertNotIn("max_completion_tokens", sent)
        self.assertNotIn("reasoning_effort", sent)
        self.assertNotIn("stream_options", sent)

    def test_the_system_prompt_travels_as_instructions(self) -> None:
        with ProtocolUpstream("responses", {"lead": [Turn(text="ok")]}) as up:
            sent = self._sent(pinned_client(up.base_url), up)
        self.assertEqual(sent["instructions"], "sys")
        opener = sent["input"][0]
        self.assertEqual(opener["role"], "user")
        self.assertEqual(opener["content"],
                         [{"type": "input_text", "text": "Read TASKS.md and go"}])
        self.assertNotIn("_system", opener)  # the private key never reaches the wire

    def test_a_plumbing_client_sends_no_reasoning_but_stays_stateless(self) -> None:
        # effort None is the keyless-test default, mirroring the other legs:
        # no reasoning key. store/include are properties of the CLIENT, not of
        # a spec -- a reasoning-by-default model behind an effort-less config
        # would otherwise break on its second turn.
        with ProtocolUpstream("responses", {"lead": [Turn(text="ok")]}) as up:
            client = OpenAIResponsesClient(model="oss-fake", api_key="k",
                                           base_url=up.base_url)
            sent = self._sent(client, up)
        self.assertNotIn("reasoning", sent)
        self.assertIs(sent["store"], False)
        self.assertEqual(sent["include"], ["reasoning.encrypted_content"])

    def test_tools_travel_flat_not_in_the_function_envelope(self) -> None:
        with ProtocolUpstream("responses", {"lead": [Turn(text="ok")]}) as up:
            client = pinned_client(up.base_url)
            client.complete(client.start("s", "Read TASKS.md", "lead"),
                            ("read_file", "finish"))
            sent = up.requests[-1]
        for tool in sent["tools"]:
            self.assertEqual(tool["type"], "function")
            self.assertNotIn("function", tool)  # the chat envelope is a 400 here
            self.assertEqual(sorted(k for k in tool if k != "type"),
                             ["description", "name", "parameters"])
        self.assertEqual(sorted(t["name"] for t in sent["tools"]),
                         ["finish", "read_file"])


class TestStreamedTurnAssembly(unittest.TestCase):
    """One turn arrives as typed events; the Reply is read off the terminal
    snapshot, so it must match what the deltas spelled out."""

    def test_text_split_across_deltas_reassembles(self) -> None:
        with ProtocolUpstream("responses",
                              {"lead": [Turn(text="delegation is not free")]}) as up:
            client = pinned_client(up.base_url)
            reply = client.complete(client.start("s", "Read TASKS.md", "lead"), ("finish",))
        self.assertEqual(reply.text, "delegation is not free")
        self.assertEqual(reply.stop_reason, "stop")
        self.assertEqual(reply.tool_calls, ())

    def test_a_multi_tool_call_turn_arrives_in_order(self) -> None:
        script = {"lead": [Turn(tools=(("read_file", {"path": "a.py"}),
                                       ("write_file", {"path": "b.py", "content": "x"})))]}
        with ProtocolUpstream("responses", script) as up:
            client = pinned_client(up.base_url)
            reply = client.complete(client.start("s", "Read TASKS.md", "lead"),
                                    ("read_file", "write_file", "finish"))
        self.assertEqual([c.name for c in reply.tool_calls], ["read_file", "write_file"])
        self.assertEqual(reply.tool_calls[0].arguments, {"path": "a.py"})
        self.assertEqual(reply.tool_calls[1].arguments, {"path": "b.py", "content": "x"})
        self.assertEqual(len({c.id for c in reply.tool_calls}), 2)
        self.assertTrue(all(c.id for c in reply.tool_calls))
        self.assertEqual(reply.stop_reason, "tool_calls")

    def test_reasoning_items_never_leak_into_the_reply_text(self) -> None:
        # With effort high the stream opens on a reasoning item. Its content is
        # an opaque payload for replay, not prose; a client that concatenated
        # every item's content would hand the runner garbage as the summary.
        with ProtocolUpstream("responses", {"lead": [Turn(text="just this")]}) as up:
            reply = pinned_client(up.base_url).complete(
                pinned_client(up.base_url).start("s", "Read TASKS.md", "lead"), ("finish",)
            )
        self.assertEqual(reply.text, "just this")

    def test_ttfb_is_the_first_byte_not_the_last(self) -> None:
        # Served slowly on purpose: a client that buffered the stream (or read
        # with a blocking read(n)) would report ttfb equal to total, silently
        # destroying the prefill term -- invisible on a fast local server.
        with ProtocolUpstream("responses", {"lead": [Turn(text="hello world")]},
                              base_latency_s=0.02,
                              seconds_per_output_token=0.002) as up:
            client = pinned_client(up.base_url)
            reply = client.complete(client.start("s", "Read TASKS.md", "lead"), ("finish",))
        self.assertGreater(reply.ttfb_s, 0.0)
        self.assertLess(reply.ttfb_s, reply.total_s * 0.8)

    def test_a_stream_that_dies_before_the_snapshot_measures_nothing(self) -> None:
        # The turn is read off the terminal snapshot on purpose: a stream cut
        # before it yields zeros a test can see, never a plausible partial
        # turn scored as if the provider had vouched for it.
        with ProtocolUpstream("responses", {"lead": [Turn(text="ok")]}) as up:
            client = pinned_client(up.base_url)
            import harness.client as C

            real_post = C._post_sse

            def truncated(url, payload, headers, timeout):
                events, ttfb, total = real_post(url, payload, headers, timeout)
                kept = [e for e in events
                        if e.get("type") != "response.completed"]
                return kept, ttfb, total

            C._post_sse = truncated
            try:
                reply = client.complete(client.start("s", "Read TASKS.md", "lead"),
                                        ("finish",))
            finally:
                C._post_sse = real_post
        self.assertEqual(up.violations, [])
        self.assertEqual(reply.text, "")
        self.assertEqual(reply.input_tokens, 0)
        self.assertEqual(reply.output_tokens, 0)
        self.assertEqual(reply.raw, [])


class TestUsageMapping(unittest.TestCase):
    """Responses usage lands on the same Reply fields, in the same currency."""

    def test_reasoning_tokens_are_a_breakdown_not_an_addend(self) -> None:
        # On the wire: input_tokens 1000 (300 of them cached), output_tokens 90
        # of which 60 are reasoning. The billed figures are 700 fresh input and
        # 90 output -- a client that adds the reasoning share bills 150, one
        # that subtracts it bills 30, and both land on numbers no script holds.
        script = {"lead": [Turn(input_tokens=700, output_tokens=90,
                                cache_read_tokens=300, reasoning_tokens=60)]}
        with ProtocolUpstream("responses", script) as up:
            client = pinned_client(up.base_url)
            reply = client.complete(client.start("s", "Read TASKS.md", "lead"), ("finish",))
        self.assertEqual(reply.input_tokens, 700)
        self.assertEqual(reply.output_tokens, 90)
        self.assertEqual(reply.cache_read_tokens, 300)

    def test_the_write_column_is_a_true_zero(self) -> None:
        # Same automatic, unbilled caching as the chat leg: zero is the billed
        # truth for the write column, not a missing measurement.
        script = {"lead": [Turn(input_tokens=500, cache_read_tokens=400)]}
        with ProtocolUpstream("responses", script) as up:
            client = pinned_client(up.base_url)
            reply = client.complete(client.start("s", "Read TASKS.md", "lead"), ("finish",))
        self.assertEqual(reply.input_tokens, 500)
        self.assertEqual(reply.cache_read_tokens, 400)
        self.assertEqual(reply.cache_write_tokens, 0)


class TestToolResultRoundTrip(unittest.TestCase):
    """The second turn is the test, and on this wire format it tests MORE than
    id-matching: the previous turn's reasoning item has to travel back too, in
    encrypted form, or the replayed function calls are orphaned."""

    SCRIPT = {"lead": [Turn(tools=(("read_file", {"path": "a.py"}),
                                   ("write_file", {"path": "b.py", "content": "x"}))),
                       Turn(text="done")]}

    def _round_trip(self, up: ProtocolUpstream):
        client = pinned_client(up.base_url)
        history = client.start("s", "Read TASKS.md", "lead")
        reply = client.complete(history, ("read_file", "write_file", "finish"))
        client.append_assistant(history, reply)
        return client, history, reply

    def test_the_replayed_turn_carries_reasoning_and_wire_string_arguments(self) -> None:
        with ProtocolUpstream("responses", self.SCRIPT) as up:
            client, history, reply = self._round_trip(up)
            client.append_tool_results(history, [(reply.tool_calls[0], "contents"),
                                                 (reply.tool_calls[1], "wrote 1 char")])
            client.complete(history, ("read_file", "write_file", "finish"))
            self.assertEqual(up.violations, [])
            sent = up.requests[-1]["input"]
        kinds = [i.get("type") for i in sent]
        self.assertEqual(kinds, ["message", "reasoning", "function_call",
                                 "function_call", "function_call_output",
                                 "function_call_output"])
        self.assertTrue(sent[1]["encrypted_content"])  # replayable, verbatim
        for call in sent[2:4]:
            # Verbatim wire strings, not re-serialized parses -- the same
            # laundering rule the chat leg pins.
            self.assertIsInstance(call["arguments"], str)
        self.assertEqual(json.loads(sent[2]["arguments"]), {"path": "a.py"})
        self.assertEqual(sent[4]["call_id"], sent[2]["call_id"])
        self.assertEqual(sent[5]["call_id"], sent[3]["call_id"])
        self.assertEqual(sent[4]["output"], "contents")

    def test_dropping_the_reasoning_item_orphans_the_replay(self) -> None:
        # The bug this API makes newly possible: a client that keeps only the
        # "useful" items passes every single-turn test and 400s here.
        with ProtocolUpstream("responses", self.SCRIPT) as up:
            client, history, reply = self._round_trip(up)
            history[:] = [i for i in history if i.get("type") != "reasoning"]
            client.append_tool_results(history, [(reply.tool_calls[0], "contents"),
                                                 (reply.tool_calls[1], "wrote 1 char")])
            with self.assertRaises(urllib.error.HTTPError):
                client.complete(history, ("read_file", "write_file", "finish"))
            self.assertIn("reasoning", up.violations[-1])

    def test_stripping_the_encrypted_content_breaks_the_replay_too(self) -> None:
        # With store false nothing was kept server-side: an id-only reasoning
        # item cannot be resolved, so losing the payload is losing the turn.
        with ProtocolUpstream("responses", self.SCRIPT) as up:
            client, history, reply = self._round_trip(up)
            for item in history:
                if item.get("type") == "reasoning":
                    item.pop("encrypted_content", None)
            client.append_tool_results(history, [(reply.tool_calls[0], "contents"),
                                                 (reply.tool_calls[1], "wrote 1 char")])
            with self.assertRaises(urllib.error.HTTPError):
                client.complete(history, ("read_file", "write_file", "finish"))
            self.assertIn("encrypted_content", up.violations[-1])


class TestDelegationOverTheResponsesWire(unittest.TestCase):
    """The whole runner -- spawns and all -- over real HTTP against the
    validating fake, the same drill test_client.py runs the other two formats
    through. Statelessness is what is really under test: every conversation,
    lead and subagent alike, must replay from its own history only."""

    @staticmethod
    def _scenario(n: int = 2):
        return build_scenario(sample_dag("wide", n, sizes=(1,), seed=7), f"w{n}", seed=7)

    @staticmethod
    def _script(scn) -> dict:
        ids = list(scn.dag.ids)
        script = {
            "lead": [
                Turn(tools=tuple(
                    ("spawn_subagent", {"instruction": f"fix {n}", "files": []}) for n in ids
                )),
                # The second lead turn only happens if every summary travelled
                # back as a well-formed function_call_output AND the lead's own
                # reasoning item was replayed ahead of its spawn calls.
                Turn(tools=(("finish", {"summary": "all done"}),)),
            ]
        }
        for n in ids:
            script[f"fix {n}"] = [
                Turn(tools=(("write_file", {"path": module_path(n),
                                            "content": scn.reference[n]}),)),
                Turn(tools=(("finish", {"summary": f"repaired {n}"}),)),
            ]
        return script

    def test_delegation_completes_and_the_provider_never_objects(self) -> None:
        scn = self._scenario(n=2)
        with ProtocolUpstream("responses", self._script(scn)) as up:
            client = pinned_client(up.base_url)
            with tempfile.TemporaryDirectory() as tmp:
                trace = run_agent(scn, client, tmp)
        self.assertEqual(up.violations, [])
        self.assertEqual(trace.notes, [])
        self.assertTrue(trace.finished)
        self.assertTrue(trace.succeeded, trace.verdicts)
        self.assertEqual(trace.spawns[0].summary, "repaired n0")

    def test_a_subagent_conversation_never_carries_the_lead_history(self) -> None:
        # "A subagent is atomic" on this wire format too: no lead items, no
        # lead reasoning, no task list in the subagent's replayed input.
        scn = self._scenario(n=1)
        with ProtocolUpstream("responses", self._script(scn)) as up:
            client = pinned_client(up.base_url)
            with tempfile.TemporaryDirectory() as tmp:
                run_agent(scn, client, tmp)
            sub_requests = [r for r in up.requests
                            if _first_user(r).startswith("fix ")]
        self.assertTrue(sub_requests)
        for request in sub_requests:
            self.assertNotIn("TASKS.md", str(request["input"]))
            self.assertNotIn("spawn_subagent",
                             [t["name"] for t in request["tools"]])


class TestTraceAndProxyAgree(unittest.TestCase):
    """Client and sniffer parse the same stream with different code; a run
    whose two readings disagree is disqualified. Cache reads AND reasoning
    tokens are nonzero in every script here, because with either at zero the
    inclusive/exclusive conventions agree by accident."""

    def test_every_token_field_matches_through_the_proxy(self) -> None:
        script = {"lead": [Turn(tools=(("read_file", {"path": "a.py"}),),
                                input_tokens=700, output_tokens=90,
                                cache_read_tokens=300, reasoning_tokens=40)]}
        with ProtocolUpstream("responses", script) as up, LoggingProxy(up.base_url) as proxy:
            client = pinned_client(proxy.base_url)
            reply = client.complete(client.start("s", "Read TASKS.md", "lead"),
                                    ("read_file", "finish"))
        record = proxy.records[-1]
        self.assertEqual(record.input_tokens, reply.input_tokens)
        self.assertEqual(record.input_tokens, 700)  # 1000 on the wire, 300 cached
        self.assertEqual(record.output_tokens, reply.output_tokens)
        self.assertEqual(record.output_tokens, 90)  # 40 of them reasoning, still 90
        self.assertEqual(record.cache_read_tokens, reply.cache_read_tokens)
        self.assertEqual(record.cache_write_tokens, reply.cache_write_tokens)
        self.assertEqual(record.stop_reason, reply.stop_reason)
        self.assertEqual(record.tools_invoked, ["read_file"])
        self.assertTrue(record.stream)
        self.assertTrue(record.billable)
        self.assertEqual(record.model, "gpt-5.6-sol")

    def test_a_cached_reasoning_run_survives_the_runner_cross_check(self) -> None:
        # End to end: run_agent routes through the logging proxy on its own and
        # stamps PROXY MISMATCH onto the trace if the halves disagree -- the
        # check that catches a subtraction (or a reasoning-token mapping) done
        # in only one of the two parsers.
        scn = build_scenario(sample_dag("wide", 1, sizes=(1,), seed=7), "w1", seed=7)
        node = scn.dag.ids[0]
        script = {"lead": [
            Turn(tools=(("write_file", {"path": module_path(node),
                                        "content": scn.reference[node]}),),
                 cache_read_tokens=350, reasoning_tokens=40),
            Turn(tools=(("finish", {"summary": "done"}),),
                 cache_read_tokens=350, reasoning_tokens=40),
        ]}
        with ProtocolUpstream("responses", script) as up:
            client = pinned_client(up.base_url)
            with tempfile.TemporaryDirectory() as tmp:
                trace = run_agent(scn, client, tmp)
        self.assertEqual(up.violations, [])
        self.assertEqual(trace.notes, [])
        self.assertTrue(trace.proxy_verified)
        self.assertTrue(trace.succeeded)


class TestProviderDispatch(unittest.TestCase):
    """client_for_model is the ONE place a pinned model becomes a client, and
    gpt-* now defaults to the responses wire format there."""

    def test_gpt_models_route_to_the_responses_client_by_default(self) -> None:
        client = client_for_model("gpt-5.6-sol", "k", "http://127.0.0.1:1", 16000, 300.0)
        self.assertIsInstance(client, OpenAIResponsesClient)
        self.assertEqual(client.effort, "high")
        self.assertEqual(client.max_tokens, 16000)

    def test_the_chat_completions_client_stays_reachable_for_oss_endpoints(self) -> None:
        # The escape hatch the open-source leg will need: a spec pinning
        # {"api": "chat"} builds the untouched chat-completions client, with
        # the effort-"none" pin that wire format still requires for tools.
        spec = {"max_tokens": 16000, "effort": "none", "api": "chat"}
        with mock.patch.dict(HARNESS_SPEC, {"gpt-oss-test": spec}):
            client = client_for_model("gpt-oss-test", "k", "http://127.0.0.1:1",
                                      16000, 300.0)
        self.assertIsInstance(client, OpenAIClient)
        self.assertEqual(client.effort, "none")

    def test_endpoint_defaults_did_not_move(self) -> None:
        # The responses client posts to the same host chat completions did, so
        # the default re-routing is untouched by the API switch.
        self.assertEqual(endpoint_defaults("gpt-5.6-sol", ANTHROPIC_API, "ANTHROPIC_API_KEY"),
                         (OPENAI_API, "OPENAI_API_KEY"))
        self.assertEqual(endpoint_defaults("gpt-5.6-sol", "http://localhost:8000", "MY_KEY"),
                         ("http://localhost:8000", "MY_KEY"))


class TestTheFakeRejectsTheRealEndpoint400s(unittest.TestCase):
    """Mirror of the chat-completions rejection tests, against the SEPARATE
    responses validator. Each check must be discriminating, so the accepted
    shape -- tools and reasoning together -- is asserted accepted at the end."""

    @staticmethod
    def _payload(**over: object) -> dict:
        base: dict = {
            "model": "gpt-5.6-sol",
            "input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": "go"}]}],
            "instructions": "s",
            "tools": [{"type": "function", "name": "x", "description": "d",
                       "parameters": {}}],
            "max_output_tokens": 16000,
            "reasoning": {"effort": "high"},
            "stream": True,
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        base.update(over)
        return base

    def test_parameter_400s(self) -> None:
        with self.assertRaises(ProtocolError):  # the chat tool envelope
            _validate_responses(self._payload(tools=[
                {"type": "function",
                 "function": {"name": "x", "description": "d", "parameters": {}}}]))
        with self.assertRaises(ProtocolError):  # the chat conversation parameter
            _validate_responses(self._payload(messages=[]))
        with self.assertRaises(ProtocolError):  # the chat output cap
            _validate_responses(self._payload(max_completion_tokens=16000))
        with self.assertRaises(ProtocolError):  # the chat reasoning knob
            _validate_responses(self._payload(reasoning_effort="high"))
        with self.assertRaises(ProtocolError):  # the chat usage opt-in
            _validate_responses(self._payload(stream_options={"include_usage": True}))
        with self.assertRaises(ProtocolError):  # an effort the API does not sell
            _validate_responses(self._payload(reasoning={"effort": "extreme"}))
        with self.assertRaises(ProtocolError):  # below the API's floor of 16
            _validate_responses(self._payload(max_output_tokens=0))
        with self.assertRaises(ProtocolError):  # model is required
            _validate_responses(self._payload(model=""))
        with self.assertRaises(ProtocolError):  # nothing is stored to refer to
            _validate_responses(self._payload(previous_response_id="resp_1"))
        with self.assertRaises(ProtocolError):  # an include the API does not know
            _validate_responses(self._payload(include=["reasoning.plaintext"]))

    def _with_turn(self, *turn_items: dict, reasoning: object = "keep") -> dict:
        items: list = [{"type": "message", "role": "user",
                        "content": [{"type": "input_text", "text": "go"}]}]
        if reasoning == "keep":
            items.append({"type": "reasoning", "id": "rs_1", "summary": [],
                          "encrypted_content": "blob"})
        elif reasoning is not None:
            items.append(reasoning)
        items.extend(turn_items)
        return self._payload(input=items)

    def test_item_400s(self) -> None:
        call = {"type": "function_call", "id": "fc_1", "call_id": "c1",
                "name": "x", "arguments": "{}"}
        out = {"type": "function_call_output", "call_id": "c1", "output": "r"}
        with self.assertRaises(ProtocolError):  # a chat-shaped tool message
            _validate_responses(self._with_turn(
                call, {"role": "tool", "tool_call_id": "c1", "content": "r"}))
        with self.assertRaises(ProtocolError):  # parsed-object arguments (replay trap)
            _validate_responses(self._with_turn(
                dict(call, arguments={"p": 1}), out))
        with self.assertRaises(ProtocolError):  # a call with no call_id
            _validate_responses(self._with_turn(
                {k: v for k, v in call.items() if k != "call_id"}, out))
        with self.assertRaises(ProtocolError):  # an output answering nothing
            _validate_responses(self._with_turn(out, reasoning=None))
        with self.assertRaises(ProtocolError):  # an output for an unknown call
            _validate_responses(self._with_turn(
                call, dict(out, call_id="WRONG")))
        with self.assertRaises(ProtocolError):  # a call left unanswered
            _validate_responses(self._with_turn(call))
        with self.assertRaises(ProtocolError):  # reasoning without its payload
            _validate_responses(self._with_turn(
                call, out, reasoning={"type": "reasoning", "id": "rs_1", "summary": []}))
        with self.assertRaises(ProtocolError):  # a call with its reasoning dropped
            _validate_responses(self._with_turn(call, out, reasoning=None))
        with self.assertRaises(ProtocolError):  # reasoning dangling at the end
            _validate_responses(self._payload(input=[
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "go"}]},
                {"type": "reasoning", "id": "rs_1", "summary": [],
                 "encrypted_content": "blob"}]))
        # And the correct replay is accepted, so the rejections discriminate.
        self.assertEqual(_validate_responses(self._with_turn(call, out)), "go")

    def test_without_reasoning_a_bare_function_call_is_legal(self) -> None:
        # The reasoning-precedes-calls rule belongs to reasoning mode only; an
        # effort-less request replays calls with no reasoning items at all.
        call = {"type": "function_call", "id": "fc_1", "call_id": "c1",
                "name": "x", "arguments": "{}"}
        out = {"type": "function_call_output", "call_id": "c1", "output": "r"}
        payload = self._with_turn(call, out, reasoning=None)
        del payload["reasoning"]
        self.assertEqual(_validate_responses(payload), "go")

    def test_the_accepted_shape_is_accepted(self) -> None:
        self.assertEqual(_validate_responses(self._payload()), "go")


class TestSnifferSpeaksTheResponsesFlavor(unittest.TestCase):
    """The proxy's third dialect, verified on bare payloads before any request
    depends on it -- the same discipline the other two flavors got."""

    USAGE = {
        "input_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 300},
        "output_tokens": 90,
        "output_tokens_details": {"reasoning_tokens": 40},
        "total_tokens": 1090,
    }
    OUTPUT = [
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "blob"},
        {"type": "function_call", "id": "fc_1", "call_id": "c1",
         "name": "read_file", "arguments": '{"path": "a.py"}', "status": "completed"},
    ]

    def _response(self) -> dict:
        return {"id": "resp_1", "object": "response", "model": "gpt-5.6-sol",
                "status": "completed", "incomplete_details": None,
                "output": self.OUTPUT, "usage": self.USAGE}

    def _check(self, sniffer: UsageSniffer) -> None:
        self.assertEqual(sniffer.input_tokens, 700)  # 1000 on the wire, 300 cached
        self.assertEqual(sniffer.output_tokens, 90)  # reasoning included, not added
        self.assertEqual(sniffer.cache_read_tokens, 300)
        self.assertEqual(sniffer.cache_write_tokens, 0)
        self.assertEqual(sniffer.tools_invoked, ["read_file"])
        self.assertEqual(sniffer.stop_reason, "tool_calls")
        self.assertEqual(sniffer.model, "gpt-5.6-sol")

    def test_the_path_pins_the_flavor(self) -> None:
        self.assertEqual(_flavor_for_path("/v1/responses"), "responses")
        self.assertEqual(_flavor_for_path("/v1/chat/completions"), "openai")
        self.assertEqual(_flavor_for_path("/v1/messages"), "anthropic")

    def test_streaming_events_land_on_the_record_fields(self) -> None:
        sniffer = UsageSniffer("responses")
        sniffer.feed_event({"type": "response.created", "sequence_number": 0,
                            "response": {"status": "in_progress", "output": [],
                                         "usage": None}})
        sniffer.feed_event({"type": "response.output_item.added", "output_index": 1,
                            "item": {"type": "function_call", "id": "fc_1",
                                     "call_id": "c1", "name": "read_file",
                                     "arguments": ""}})
        sniffer.feed_event({"type": "response.completed", "response": self._response()})
        self._check(sniffer)

    def test_a_non_streaming_body_parses_identically(self) -> None:
        sniffer = UsageSniffer("responses")
        sniffer.feed_body(json.dumps(self._response()).encode())
        self._check(sniffer)


if __name__ == "__main__":
    unittest.main()
