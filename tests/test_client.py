"""Both wire formats, driven by the real clients through a validating fake.

`tests/test_runner.py` proves the LOOP is correct -- attribution, batching, the
concurrency cap, one-level delegation -- but it drives that loop with
`ScriptedClient`, which never builds a request. So none of it says whether a real
Anthropic or OpenAI endpoint would accept what the harness sends.

That gap matters most exactly where delegation lives. A single-turn exchange
cannot reveal a broken tool-result: there is no second turn to reject it. But the
spawn round trip is inherently multi-turn -- the lead emits `spawn_subagent`, the
subagent runs its own conversation, and its summary has to travel back as a
well-formed tool result before the lead can take another turn. If that packing is
wrong, delegation is the first thing that breaks and it breaks with a 400.

So these tests run the whole runner, spawns and all, over real HTTP against
`ProtocolUpstream`, which rejects any conversation a real endpoint would reject.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator.dag import sample_dag
from generator.scenario import build_scenario
from generator.templates import module_path
from harness.client import AnthropicClient, OpenAIClient
from harness.protocol_upstream import ProtocolError, ProtocolUpstream, Turn
from harness.runner import run_agent
from harness.trace import LEAD


def scenario(n: int = 2, seed: int = 7):
    return build_scenario(sample_dag("wide", n, sizes=(1,), seed=seed), f"w{n}", seed=seed)


def client_for(flavor: str, base_url: str):
    if flavor == "anthropic":
        return AnthropicClient(model="claude-fake", api_key="k", base_url=base_url)
    return OpenAIClient(model="oss-fake", api_key="k", base_url=base_url)


FLAVORS = ("anthropic", "openai")


class TestUsageParsing(unittest.TestCase):
    """Both providers report usage differently; the trace must not care."""

    def test_tokens_land_on_the_reply_identically(self) -> None:
        for flavor in FLAVORS:
            with self.subTest(flavor=flavor):
                script = {"lead": [Turn(text="hi", input_tokens=700, output_tokens=90,
                                        cache_read_tokens=300)]}
                with ProtocolUpstream(flavor, script) as up:
                    client = client_for(flavor, up.base_url)
                    history = client.start("sys", "Read TASKS.md and go", "lead")
                    reply = client.complete(history, ("read_file", "finish"))
                self.assertEqual(reply.input_tokens, 700, flavor)
                self.assertEqual(reply.output_tokens, 90, flavor)
                self.assertEqual(reply.cache_read_tokens, 300, flavor)

    def test_openai_prompt_tokens_are_split_not_double_counted(self) -> None:
        # OpenAI reports prompt_tokens INCLUSIVE of cached ones, and the two bill
        # at different rates. A client that forgets to subtract charges cached
        # tokens at the fresh rate and overstates every run's cost.
        script = {"lead": [Turn(input_tokens=500, cache_read_tokens=400)]}
        with ProtocolUpstream("openai", script) as up:
            client = client_for("openai", up.base_url)
            reply = client.complete(client.start("s", "Read TASKS.md", "lead"), ("finish",))
        self.assertEqual(reply.input_tokens, 500)
        self.assertEqual(reply.cache_read_tokens, 400)
        self.assertEqual(up.requests[0]["messages"][0]["role"], "system")

    def test_tool_calls_normalize_to_one_shape(self) -> None:
        for flavor in FLAVORS:
            with self.subTest(flavor=flavor):
                script = {"lead": [Turn(tools=(("read_file", {"path": "pkg/mod_n0.py"}),))]}
                with ProtocolUpstream(flavor, script) as up:
                    client = client_for(flavor, up.base_url)
                    reply = client.complete(client.start("s", "Read TASKS.md", "lead"), ("read_file",))
                self.assertEqual(len(reply.tool_calls), 1, flavor)
                self.assertEqual(reply.tool_calls[0].name, "read_file", flavor)
                self.assertEqual(reply.tool_calls[0].arguments, {"path": "pkg/mod_n0.py"}, flavor)
                self.assertTrue(reply.tool_calls[0].id, flavor)


class TestToolSurfaceOnTheWire(unittest.TestCase):
    def test_the_allow_list_is_what_actually_reaches_the_provider(self) -> None:
        # One-level delegation is only real if the subagent's request genuinely
        # omits the tool. Checked on the wire, not on the Python side.
        for flavor in FLAVORS:
            with self.subTest(flavor=flavor):
                with ProtocolUpstream(flavor, {"lead": [Turn(text="ok")]}) as up:
                    client = client_for(flavor, up.base_url)
                    client.complete(client.start("s", "Read TASKS.md", "lead"),
                                    ("read_file", "write_file", "finish"))
                    sent = up.requests[-1]["tools"]
                names = [
                    t["name"] if flavor == "anthropic" else t["function"]["name"] for t in sent
                ]
                self.assertEqual(sorted(names), ["finish", "read_file", "write_file"], flavor)
                self.assertNotIn("spawn_subagent", names, flavor)


class TestPinnedSpecOnTheWire(unittest.TestCase):
    """The Aug 24 harness-spec pins, checked on the wire.

    Three constants the report publishes -- adaptive thinking, effort, and the
    one-hour cache TTL -- plus the breakpoint discipline that makes the TTL
    mean anything: exactly two markers per request, system and conversation
    tail, with the tail marker moving forward each turn and never persisting
    into history (persisted markers would hit the four-per-request cap).
    """

    MARKER = {"type": "ephemeral", "ttl": "1h"}

    def _pinned(self, base_url: str) -> AnthropicClient:
        return AnthropicClient(model="claude-fake", api_key="k", base_url=base_url,
                               thinking={"type": "adaptive"}, effort="high",
                               cache_ttl="1h")

    def test_pinned_params_reach_the_provider(self) -> None:
        with ProtocolUpstream("anthropic", {"lead": [Turn(text="ok")]}) as up:
            client = self._pinned(up.base_url)
            client.complete(client.start("sys", "Read TASKS.md and go", "lead"), ("finish",))
            sent = up.requests[-1]
        self.assertEqual(sent["thinking"], {"type": "adaptive"})
        self.assertEqual(sent["output_config"], {"effort": "high"})
        self.assertEqual(up.violations, [])

    def test_two_breakpoints_system_and_tail(self) -> None:
        import json as _json

        with ProtocolUpstream("anthropic", {"lead": [Turn(text="ok")]}) as up:
            client = self._pinned(up.base_url)
            client.complete(client.start("sys", "Read TASKS.md and go", "lead"), ("finish",))
            sent = up.requests[-1]
        self.assertEqual(sent["system"][0]["cache_control"], self.MARKER)
        tail = sent["messages"][-1]["content"][-1]
        self.assertEqual(tail["cache_control"], self.MARKER)
        self.assertEqual(_json.dumps(sent).count('"cache_control"'), 2)

    def test_the_tail_marker_moves_and_never_persists(self) -> None:
        import json as _json

        script = {"lead": [Turn(tools=(("read_file", {"path": "pkg/mod_n0.py"}),)),
                           Turn(text="done")]}
        with ProtocolUpstream("anthropic", script) as up:
            client = self._pinned(up.base_url)
            history = client.start("sys", "Read TASKS.md and go", "lead")
            reply = client.complete(history, ("read_file", "finish"))
            client.append_assistant(history, reply)
            client.append_tool_results(history, [(reply.tool_calls[0], "contents")])
            client.complete(history, ("read_file", "finish"))
            second = up.requests[-1]
        # Still exactly two markers; the tail one now sits on the tool result,
        # and the opener went back to a plain string -- nothing persisted.
        self.assertEqual(_json.dumps(second).count('"cache_control"'), 2)
        self.assertEqual(second["messages"][-1]["content"][-1]["cache_control"], self.MARKER)
        self.assertIsInstance(second["messages"][0]["content"], str)
        self.assertEqual(up.violations, [])

    def test_a_plumbing_client_sends_no_pins(self) -> None:
        # The defaults are the keyless-test configuration: no thinking, no
        # output_config, string system -- the requests every other test sees.
        with ProtocolUpstream("anthropic", {"lead": [Turn(text="ok")]}) as up:
            client = client_for("anthropic", up.base_url)
            client.complete(client.start("sys", "Read TASKS.md and go", "lead"), ("finish",))
            sent = up.requests[-1]
        self.assertNotIn("thinking", sent)
        self.assertNotIn("output_config", sent)
        self.assertIsInstance(sent["system"], str)

    def test_the_fake_rejects_the_real_endpoint_400s(self) -> None:
        from harness.protocol_upstream import _validate_anthropic

        def payload(**over: object) -> dict:
            base: dict = {
                "model": "claude-fake",
                "system": "s",
                "tools": [{"name": "x", "description": "d", "input_schema": {}}],
                "messages": [{"role": "user", "content": "go"}],
            }
            base.update(over)
            return base

        with self.assertRaises(ProtocolError):  # budget_tokens is removed
            _validate_anthropic(payload(thinking={"type": "enabled", "budget_tokens": 2048}))
        with self.assertRaises(ProtocolError):  # haiku predates adaptive thinking
            _validate_anthropic(payload(model="claude-haiku-4-5",
                                        thinking={"type": "adaptive"}))
        with self.assertRaises(ProtocolError):  # haiku predates effort
            _validate_anthropic(payload(model="claude-haiku-4-5",
                                        output_config={"effort": "high"}))
        with self.assertRaises(ProtocolError):  # a TTL the API does not sell
            _validate_anthropic(payload(system=[{
                "type": "text", "text": "s",
                "cache_control": {"type": "ephemeral", "ttl": "2h"},
            }]))
        # And the accepted shape is accepted, so the rejections above are
        # discriminating rather than reflexive.
        _validate_anthropic(payload(thinking={"type": "adaptive"},
                                    output_config={"effort": "high"}))


class TestSpawnRoundTripOverTheWire(unittest.TestCase):
    """The question this file exists for: does a subagent's summary get back to
    the lead in a form the provider accepts?"""

    def _script(self, scn):
        ids = list(scn.dag.ids)
        script = {
            "lead": [
                Turn(tools=tuple(
                    ("spawn_subagent", {"instruction": f"fix {n}", "files": []}) for n in ids
                )),
                # The SECOND lead turn is the whole point: it can only happen if
                # the tool_results carrying each summary were packed correctly.
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
        for flavor in FLAVORS:
            with self.subTest(flavor=flavor):
                scn = scenario(n=2)
                with ProtocolUpstream(flavor, self._script(scn)) as up:
                    client = client_for(flavor, up.base_url)
                    with tempfile.TemporaryDirectory() as tmp:
                        trace = run_agent(scn, client, tmp)

                self.assertEqual(up.violations, [], f"{flavor}: {up.violations}")
                self.assertEqual(trace.notes, [], f"{flavor}: {trace.notes}")
                self.assertTrue(trace.finished, flavor)
                self.assertTrue(trace.succeeded, f"{flavor}: {trace.verdicts}")
                self.assertEqual(trace.k, len(scn.dag.ids), flavor)

    def test_the_summary_reaches_the_lead_as_a_matched_tool_result(self) -> None:
        # Not just "no error" -- the actual text has to be in the message the
        # lead's next request carries, keyed to the right call.
        for flavor in FLAVORS:
            with self.subTest(flavor=flavor):
                scn = scenario(n=1)
                with ProtocolUpstream(flavor, self._script(scn)) as up:
                    client = client_for(flavor, up.base_url)
                    with tempfile.TemporaryDirectory() as tmp:
                        trace = run_agent(scn, client, tmp)
                    lead_requests = [
                        r for r in up.requests
                        if _first_user(r, flavor).startswith("Read TASKS.md")
                    ]
                self.assertEqual(trace.spawns[0].summary, "repaired n0", flavor)
                self.assertGreaterEqual(len(lead_requests), 2, flavor)
                blob = str(lead_requests[-1]["messages"])
                self.assertIn("repaired n0", blob, flavor)

    def test_a_subagent_conversation_never_carries_the_lead_history(self) -> None:
        # "A subagent is atomic" on the wire: its request must not contain the
        # task list or anything the lead said. If it did, the harness would be
        # cheaper and better-informed than the oracle's model of it.
        for flavor in FLAVORS:
            with self.subTest(flavor=flavor):
                scn = scenario(n=1)
                with ProtocolUpstream(flavor, self._script(scn)) as up:
                    client = client_for(flavor, up.base_url)
                    with tempfile.TemporaryDirectory() as tmp:
                        run_agent(scn, client, tmp)
                    sub_requests = [
                        r for r in up.requests if _first_user(r, flavor).startswith("fix ")
                    ]
                self.assertTrue(sub_requests, flavor)
                for request in sub_requests:
                    blob = str(request["messages"])
                    self.assertNotIn("TASKS.md", blob, flavor)
                    self.assertNotIn("Test repair pass", blob, flavor)


class TestTheFakeActuallyRejects(unittest.TestCase):
    """A validating fake that validates nothing would be worse than none, so the
    guard is itself tested."""

    def test_an_unanswered_tool_use_is_refused(self) -> None:
        from harness.protocol_upstream import _validate_anthropic, _validate_openai

        with self.assertRaises(ProtocolError):
            _validate_anthropic(
                {
                    "system": "s",
                    "tools": [{"name": "x", "description": "d", "input_schema": {}}],
                    "messages": [
                        {"role": "user", "content": "go"},
                        {"role": "assistant", "content": [
                            {"type": "tool_use", "id": "a", "name": "x", "input": {}}
                        ]},
                    ],
                }
            )
        with self.assertRaises(ProtocolError):
            _validate_openai(
                {
                    "tools": [{"type": "function", "function": {}}],
                    "messages": [
                        {"role": "system", "content": "s"},
                        {"role": "user", "content": "go"},
                        {"role": "assistant", "tool_calls": [{"id": "c1"}]},
                    ],
                }
            )

    def test_a_mismatched_tool_result_id_is_refused(self) -> None:
        from harness.protocol_upstream import _validate_anthropic

        with self.assertRaises(ProtocolError):
            _validate_anthropic(
                {
                    "system": "s",
                    "tools": [{"name": "x", "description": "d", "input_schema": {}}],
                    "messages": [
                        {"role": "user", "content": "go"},
                        {"role": "assistant", "content": [
                            {"type": "tool_use", "id": "a", "name": "x", "input": {}}
                        ]},
                        {"role": "user", "content": [
                            {"type": "tool_result", "tool_use_id": "WRONG", "content": "r"}
                        ]},
                    ],
                }
            )

    def test_a_violation_surfaces_as_a_failed_turn_not_a_silent_pass(self) -> None:
        scn = scenario(n=1)
        with ProtocolUpstream("anthropic", {}) as up:  # empty script -> 400
            client = client_for("anthropic", up.base_url)
            with tempfile.TemporaryDirectory() as tmp:
                trace = run_agent(scn, client, tmp)
        self.assertTrue(trace.notes)
        self.assertFalse(trace.succeeded)


def _first_user(request: dict, flavor: str) -> str:
    for message in request.get("messages", []):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


if __name__ == "__main__":
    unittest.main()
