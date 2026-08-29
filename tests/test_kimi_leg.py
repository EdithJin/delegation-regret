"""The Kimi K3 open-weight leg: dispatch, pinned spec, visible-reasoning stream.

kimi-k3 speaks the same chat-completions dialect the OpenAI leg's client
already implements, so what needs pinning is only what is NEW about this leg:
the model-to-vendor dispatch (Moonshot's host and key, not OpenAI's), the
pinned operating point (effort "max" -- the top of Kimi's low/high/max roster,
matching the cross-leg rule that every model runs at its vendor's highest named
effort), the roster the fake enforces (there is no "none": K3 cannot stop
reasoning), and the stream shape unique to this model -- visible
reasoning_content deltas that the client must ignore without crashing and
without concatenating into the reply text. All offline, keyless, against the
validating fake; the real-endpoint preflight re-verifies each of these before
any calibrated spend.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.cli import (
    ANTHROPIC_API,
    HARNESS_SPEC,
    MOONSHOT_API,
    PRICE_SHEETS,
    client_for_model,
    endpoint_defaults,
    provider_for,
)
from harness.client import OpenAIClient
from harness.protocol_upstream import ProtocolError, ProtocolUpstream, Turn, _validate_openai


def kimi_client(base_url: str) -> OpenAIClient:
    """The chat client at the K3 leg's published constants."""
    return OpenAIClient(model="kimi-k3", api_key="k", base_url=base_url,
                        max_tokens=16000, effort="max")


class TestKimiDispatch(unittest.TestCase):
    """kimi-* is OpenAI-shaped wire on Moonshot's host with Moonshot's key."""

    def test_kimi_speaks_the_openai_wire(self) -> None:
        self.assertEqual(provider_for("kimi-k3"), "openai")

    def test_untouched_defaults_reroute_to_moonshot(self) -> None:
        url, env = endpoint_defaults("kimi-k3", ANTHROPIC_API, "ANTHROPIC_API_KEY")
        self.assertEqual(url, MOONSHOT_API)
        self.assertEqual(env, "KIMI_API_KEY")

    def test_explicit_overrides_always_win(self) -> None:
        url, env = endpoint_defaults("kimi-k3", "http://127.0.0.1:9", "MY_KEY")
        self.assertEqual((url, env), ("http://127.0.0.1:9", "MY_KEY"))

    def test_client_for_model_builds_the_chat_client_at_effort_max(self) -> None:
        client = client_for_model("kimi-k3", "k", "http://x", 16000, 30.0)
        self.assertIsInstance(client, OpenAIClient)
        self.assertEqual(client.effort, "max")

    def test_price_sheet_and_spec_are_pinned(self) -> None:
        sheet = PRICE_SHEETS["kimi-k3"]
        self.assertTrue(sheet.as_of)
        self.assertEqual(sheet.cache_write_per_mtok, 0.0)  # automatic caching, unbilled writes
        spec = HARNESS_SPEC["kimi-k3"]
        self.assertEqual(spec["effort"], "max")
        self.assertEqual(spec["api"], "chat")
        self.assertEqual(spec["max_tokens"], 16000)


class TestKimiWire(unittest.TestCase):
    """The K3 stream against the validating fake."""

    def test_pinned_params_reach_the_provider(self) -> None:
        with ProtocolUpstream("openai", {"lead": [Turn(text="ok")]}) as up:
            client = kimi_client(up.base_url)
            client.complete(client.start("sys", "Read TASKS.md and go", "lead"), ("finish",))
            sent = up.requests[-1]
            self.assertEqual(up.violations, [])
        self.assertEqual(sent["model"], "kimi-k3")
        self.assertEqual(sent["reasoning_effort"], "max")
        self.assertEqual(sent["max_completion_tokens"], 16000)

    def test_reasoning_content_deltas_never_reach_the_reply_text(self) -> None:
        # The fake streams reasoning_content deltas before content for every
        # kimi model. The reply must carry the answer alone: reasoning leaking
        # into text would corrupt every transcript and tool argument downstream.
        with ProtocolUpstream("openai", {"lead": [Turn(text="the answer")]}) as up:
            client = kimi_client(up.base_url)
            reply = client.complete(client.start("sys", "Read TASKS.md and go", "lead"), ("finish",))
            self.assertEqual(up.violations, [])
        self.assertEqual(reply.text, "the answer")
        self.assertNotIn("considering the workspace", reply.text)

    def test_tools_ride_with_reasoning_max(self) -> None:
        # The founding fact of this leg's wire choice: no gpt-style
        # tools-require-effort-none rule -- K3 reasons AND calls tools on
        # chat completions.
        with ProtocolUpstream("openai", {"lead": [Turn(tools=[("finish", "{}")])]}) as up:
            client = kimi_client(up.base_url)
            reply = client.complete(client.start("sys", "Read TASKS.md and go", "lead"), ("finish",))
            self.assertEqual(up.violations, [])
        self.assertEqual([t.name for t in reply.tool_calls], ["finish"])


class TestKimiRoster(unittest.TestCase):
    """The fake's kimi validation: low/high/max only, no 'none' to fall back to."""

    def _payload(self, effort):
        p = {
            "model": "kimi-k3",
            "stream": True,
            "messages": [{"role": "system", "content": "s"},
                         {"role": "user", "content": "u"}],
            "tools": [{"type": "function",
                       "function": {"name": "f", "parameters": {"type": "object"}}}],
        }
        if effort is not None:
            p["reasoning_effort"] = effort
        return p

    def test_the_pinned_effort_passes_with_tools(self) -> None:
        self.assertEqual(_validate_openai(self._payload("max")), "u")

    def test_omitting_effort_is_the_vendor_default(self) -> None:
        self.assertEqual(_validate_openai(self._payload(None)), "u")

    def test_none_is_rejected_by_the_fake_on_purpose(self) -> None:
        # Probed live 2026-08-28: the real endpoint ACCEPTS "none" and it
        # genuinely disables reasoning (the docs' always-on claim is wrong).
        # The fake still rejects it, deliberately: a stray gpt-style "none"
        # ported to this leg would silently flip its operating point on the
        # real wire -- the drift class that burned the gpt leg -- so it must
        # fail offline instead. If a reasoning-off K3 condition is ever
        # wanted, it enters as its own pinned spec, not as a passthrough.
        with self.assertRaises(ProtocolError) as caught:
            _validate_openai(self._payload("none"))
        self.assertIn("low/high/max", str(caught.exception))

    def test_gpt_roster_values_off_kimi_roster_are_rejected(self) -> None:
        with self.assertRaises(ProtocolError):
            _validate_openai(self._payload("medium"))


if __name__ == "__main__":
    unittest.main()
