"""Talking to a model, in whichever wire format it speaks.

Low-level design: low-level-design.md, "Stage 5.2".

The runner must not branch on provider. The matrix has an Anthropic leg and an
OpenAI-compatible open-weights leg, and any `if anthropic:` inside the agent loop
is a place the two legs can quietly diverge -- a different retry rule, a
different way of packing tool results -- after which the cross-model table is
not comparable and nothing in the output says so. So the loop is written once
against `Client`, and everything provider-shaped lives here.

A client owns the message history format, because that is the part that differs.
It hands back a `Reply` that is identical across providers.

USAGE IS READ OFF THE RESPONSE, NOT COUNTED HERE. Every token figure on a
`Reply` came from the provider's own `usage` block. `harness.proxy` reads the
same block independently on the wire, which makes the two a cross-check rather
than a duplication: if the trace and the proxy log disagree, one of them is
parsing wrong and the run is not trustworthy.

NON-STREAMING ON PURPOSE. The runner needs total duration and exact usage, both
of which a single response carries. Streaming would add a real time-to-first-byte
per call, but the fitted timing model regresses on `total_s`, and the proxy
already proves the streaming path is unbuffered where that matters. One less
moving part in the loop that spends money.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from .tools import TOOL_SPECS

__all__ = [
    "ToolRequest",
    "Reply",
    "Client",
    "AnthropicClient",
    "OpenAIClient",
    "ScriptedClient",
    "ANTHROPIC_VERSION",
]

ANTHROPIC_VERSION = "2023-06-01"


@dataclass(frozen=True)
class ToolRequest:
    """One tool the model asked for, normalized across providers."""

    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class Reply:
    """One assistant turn, in the same shape whatever spoke it."""

    text: str = ""
    tool_calls: tuple[ToolRequest, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    stop_reason: str = ""
    total_s: float = 0.0
    ttfb_s: float = 0.0
    raw: object = None  # the assistant message, verbatim, to append to history


class Client:
    """The interface the runner is written against.

    Subclasses own message construction. `allow` is the tool allow-list for this
    conversation -- a subagent is handed everything except `spawn_subagent`,
    which is how "delegation is one level deep" is enforced in the harness
    rather than merely requested in a prompt.

    Every subclass carries a `model` string, which lands on the trace. It is not
    declared here with a default: a dataclass subclass would inherit that value
    as a field default and then reject its own required fields for following a
    defaulted one.
    """

    def start(self, system: str, user: str, actor: str = "lead") -> list:
        """Open a conversation. `actor` labels which loop this is.

        Real clients ignore it -- the wire format does not care. `ScriptedClient`
        needs it to know which script to read, and threading it through here is
        better than having the test client guess from the prompt text.
        """
        raise NotImplementedError

    def complete(self, history: list, allow: tuple[str, ...]) -> Reply:
        raise NotImplementedError

    def append_assistant(self, history: list, reply: Reply) -> None:
        raise NotImplementedError

    def append_tool_results(self, history: list, results: list[tuple[ToolRequest, str]]) -> None:
        raise NotImplementedError


def _specs(allow: tuple[str, ...]) -> list[tuple[str, str, dict]]:
    return [t for t in TOOL_SPECS if t[0] in allow]


def _post(url: str, payload: dict, headers: dict, timeout: float) -> tuple[dict, float]:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, headers=headers)
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw), time.perf_counter() - started


# ------------------------------------------------------------------ Anthropic


def _tail_breakpoint(messages: list, marker: dict) -> list:
    """Copies of `messages` with `cache_control` on the final message's last block.

    Copies, never mutates: the marker belongs to THIS request. Persisted into
    history it would pile up one breakpoint per turn and hit the API's
    four-per-request cap within a few turns. Moving the single tail marker
    forward each turn is the documented incremental pattern -- reads match the
    longest previously cached prefix, so last turn's write is still read.
    """
    last = dict(messages[-1])
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = [{"type": "text", "text": content, "cache_control": marker}]
    elif isinstance(content, list) and content:
        blocks = list(content)
        blocks[-1] = {**blocks[-1], "cache_control": marker}
        last["content"] = blocks
    return messages[:-1] + [last]


@dataclass
class AnthropicClient(Client):
    """Anthropic Messages API. Point `base_url` at the logging proxy.

    The three optional fields after `timeout` are the pinned harness spec
    (TASKS-AND-OPEN-ISSUES section 2, Aug 24), threaded through
    `cli.HARNESS_SPEC` so each model's values live in one greppable place:

    * `thinking` -- {"type": "adaptive"} on the matrix models. The current
      generation accepts no other on-mode (budget_tokens is a 400 now), and
      claude-haiku-4-5, the plumbing model, accepts none at all, so it stays
      None there.
    * `effort` -- "high" on the matrix models. It is the API default, pinned
      explicitly because a silently inherited default is not a published
      constant.
    * `cache_ttl` -- "1h", matching the 2x write premium the price sheets
      carry. When set, each request holds exactly two cache breakpoints: the
      system block, and the final message (see `_tail_breakpoint`).

    All three default to None/off, which is the keyless-test configuration:
    the fake upstream then sees the same requests it always saw.
    """

    model: str
    api_key: str
    base_url: str
    max_tokens: int = 4096
    timeout: float = 300.0
    thinking: dict | None = None
    effort: str | None = None
    cache_ttl: str | None = None

    def start(self, system: str, user: str, actor: str = "lead") -> list:
        # The system prompt is a top-level parameter here, not a message, so it
        # travels with the history rather than inside it. Stored per call under
        # a private key the API never sees, because subagents run concurrently
        # and an attribute on the shared client would race.
        return [{"role": "user", "content": user, "_system": system}]

    @staticmethod
    def _split(history: list) -> tuple[str, list]:
        system = ""
        clean = []
        for message in history:
            if "_system" in message:
                system = message["_system"]
                message = {k: v for k, v in message.items() if k != "_system"}
            clean.append(message)
        return system, clean

    def complete(self, history: list, allow: tuple[str, ...]) -> Reply:
        system, messages = self._split(history)
        system_field: object = system
        if self.cache_ttl:
            marker = {"type": "ephemeral", "ttl": self.cache_ttl}
            system_field = [{"type": "text", "text": system, "cache_control": marker}]
            messages = _tail_breakpoint(messages, marker)
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_field,
            "messages": messages,
            "tools": [
                {"name": n, "description": d, "input_schema": s} for n, d, s in _specs(allow)
            ],
        }
        if self.thinking is not None:
            payload["thinking"] = self.thinking
        if self.effort is not None:
            payload["output_config"] = {"effort": self.effort}
        body, elapsed = _post(
            self.base_url.rstrip("/") + "/v1/messages",
            payload,
            {
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
            self.timeout,
        )
        usage = body.get("usage") or {}
        blocks = body.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        tools = tuple(
            ToolRequest(id=b.get("id", ""), name=b.get("name", ""), arguments=b.get("input") or {})
            for b in blocks
            if b.get("type") == "tool_use"
        )
        return Reply(
            text=text,
            tool_calls=tools,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
            cache_write_tokens=int(usage.get("cache_creation_input_tokens") or 0),
            stop_reason=body.get("stop_reason") or "",
            total_s=elapsed,
            ttfb_s=elapsed,
            raw={"role": "assistant", "content": blocks},
        )

    def append_assistant(self, history: list, reply: Reply) -> None:
        history.append(reply.raw)

    def append_tool_results(self, history: list, results: list[tuple[ToolRequest, str]]) -> None:
        # Anthropic wants every tool_result for one assistant turn in ONE user
        # message. Splitting them across messages is accepted by some versions
        # and rejected by others; one message is correct everywhere.
        history.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": call.id, "content": result}
                    for call, result in results
                ],
            }
        )


# --------------------------------------------------------------------- OpenAI


@dataclass
class OpenAIClient(Client):
    """Any OpenAI-compatible chat-completions endpoint, including local ones."""

    model: str
    api_key: str
    base_url: str
    timeout: float = 300.0

    def start(self, system: str, user: str, actor: str = "lead") -> list:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def complete(self, history: list, allow: tuple[str, ...]) -> Reply:
        payload = {
            "model": self.model,
            "messages": history,
            "tools": [
                {"type": "function", "function": {"name": n, "description": d, "parameters": s}}
                for n, d, s in _specs(allow)
            ],
            "tool_choice": "auto",
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body, elapsed = _post(
            self.base_url.rstrip("/") + "/v1/chat/completions", payload, headers, self.timeout
        )
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = body.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0)

        calls = []
        for c in message.get("tool_calls") or []:
            fn = c.get("function") or {}
            from .tools import parse_arguments

            try:
                arguments = parse_arguments(fn.get("arguments"))
            except ValueError:
                # Surfaced, not repaired: a model that cannot emit valid JSON
                # arguments is exactly what the open-weights gate is looking for,
                # and silently fixing it would hide the finding.
                arguments = {"__malformed__": str(fn.get("arguments"))[:200]}
            calls.append(ToolRequest(id=c.get("id", ""), name=fn.get("name", ""), arguments=arguments))

        # OpenAI reports prompt_tokens INCLUSIVE of cached ones; the cost model
        # bills the two at different rates, so the fresh count is the difference.
        prompt = int(usage.get("prompt_tokens") or 0)
        return Reply(
            text=message.get("content") or "",
            tool_calls=tuple(calls),
            input_tokens=max(prompt - cached, 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            cache_read_tokens=cached,
            stop_reason=choice.get("finish_reason") or "",
            total_s=elapsed,
            ttfb_s=elapsed,
            raw=message,
        )

    def append_assistant(self, history: list, reply: Reply) -> None:
        history.append(reply.raw)

    def append_tool_results(self, history: list, results: list[tuple[ToolRequest, str]]) -> None:
        for call, result in results:
            history.append({"role": "tool", "tool_call_id": call.id, "content": result})


# -------------------------------------------------------------------- testing


@dataclass
class ScriptedClient(Client):
    """A client that replays canned turns. No network, no key, no spend.

    The runner is the piece of this repo that is hardest to test and easiest to
    get subtly wrong -- concurrency, attribution, one-level delegation, the turn
    cap. Testing it against a real model would make the suite slow, flaky, and
    expensive, and would test the model rather than the harness. So the loop is
    driven by a script of `Reply`s and every structural guarantee is asserted
    offline.

    `script` maps a key to the replies to give in order. The lead is keyed
    "lead"; a subagent is keyed by the FIRST LINE of the instruction it was
    handed, so a test can give each delegated block its own behaviour. "*" is
    the fallback for any subagent the script does not name.

    Replies are consumed in order and the last one repeats, so a script does not
    have to predict exactly how many turns the loop will take.
    """

    model: str = "scripted"
    script: dict[str, list[Reply]] = field(default_factory=dict)
    on_missing: Callable[[str], Reply] | None = None
    seen: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    _cursor: dict[str, int] = field(default_factory=dict)
    _lock: object = field(default_factory=threading.Lock)

    def start(self, system: str, user: str, actor: str = "lead") -> list:
        key = "lead" if actor == "lead" else user.splitlines()[0].strip()
        return [
            {"role": "system", "content": system, "_key": key},
            {"role": "user", "content": user},
        ]

    def key_for(self, history: list) -> str:
        for message in history:
            if "_key" in message:
                return message["_key"]
        return "lead"

    def complete(self, history: list, allow: tuple[str, ...]) -> Reply:
        key = self.key_for(history)
        with self._lock:
            self.seen.append((key, allow))
            replies = self.script.get(key)
            if replies is None:
                replies = self.script.get("*")
            if replies is None:
                if self.on_missing is not None:
                    return self.on_missing(key)
                raise KeyError(
                    f"no scripted replies for {key!r}; scripted keys are "
                    + ", ".join(sorted(self.script))
                )
            i = self._cursor.get(key, 0)
            self._cursor[key] = i + 1
        return replies[min(i, len(replies) - 1)]

    def append_assistant(self, history: list, reply: Reply) -> None:
        history.append({"role": "assistant", "content": reply.text})

    def append_tool_results(self, history: list, results: list[tuple[ToolRequest, str]]) -> None:
        for call, result in results:
            history.append({"role": "tool", "tool_call_id": call.id, "content": result})
