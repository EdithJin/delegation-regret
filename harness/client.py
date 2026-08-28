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

NON-STREAMING ON THE ANTHROPIC LEG, STREAMING ON THE OPENAI ONE -- each on
purpose. The runner needs total duration and exact usage. On the Anthropic leg a
single response carries both, the fitted timing model regresses on `total_s`,
and the proxy already proves the streaming path is unbuffered where that
matters -- one less moving part in the loop that spends money. The
chat-completions leg streams instead: with `stream_options.include_usage` the
final chunk carries the same exact usage a buffered body would, the streaming
path is the one OpenAI-compatible open-source servers actually exercise, and it
hands the client a real time-to-first-byte -- where the Anthropic client can
only report `ttfb_s == total_s` and lean on the proxy for the split.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from .tools import TOOL_SPECS, parse_arguments

__all__ = [
    "ToolRequest",
    "Reply",
    "Client",
    "AnthropicClient",
    "OpenAIClient",
    "OpenAIResponsesClient",
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


def _post_sse(url: str, payload: dict, headers: dict, timeout: float) -> tuple[list[dict], float, float]:
    """POST expecting an SSE response; (decoded data payloads, ttfb_s, total_s).

    ttfb is stamped at the first response byte off the socket, which is why the
    read uses `read1` where available: `read(n)` blocks until it has all n bytes,
    which on a chunked stream means waiting for the generation to finish -- the
    token counts would stay perfectly correct and ttfb would silently equal the
    total (the same failure mode `harness.proxy` documents on its relay).

    SSE frames do not respect socket-read boundaries, so lines are cut at
    newlines with the incomplete tail carried into the next read -- assuming
    whole lines works on localhost and drops the final usage frame on a slow
    connection, which is every real run.
    """
    body = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, headers=headers)
    started = time.perf_counter()
    first: float | None = None
    events: list[dict] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        pull = getattr(response, "read1", None) or response.read
        buffer = b""
        while True:
            chunk = pull(8192)
            if not chunk:
                break
            if first is None:
                first = time.perf_counter()
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if not data or data == b"[DONE]":
                    continue
                try:
                    events.append(json.loads(data))
                except ValueError:
                    continue
    total = time.perf_counter() - started
    return events, (first - started) if first is not None else total, total


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
    """Any OpenAI-compatible chat-completions endpoint, including local ones.

    The wire format is `POST /v1/chat/completions` on purpose: it is the one
    dialect both api.openai.com and the open-source servers (vLLM, llama.cpp,
    Ollama, ...) speak, so one client covers the whole non-Anthropic column.

    Every request streams with `stream_options: {"include_usage": true}` -- see
    the module docstring for why this leg streams when the Anthropic one does
    not. The final chunk then carries the provider's own usage block, which is
    the only token source this repo accepts.

    `max_tokens` is sent as `max_completion_tokens`: current OpenAI reasoning
    models reject the legacy `max_tokens` key with a 400. `effort` is sent as
    `reasoning_effort`, this wire format's analogue of the Anthropic effort
    pin; None (the keyless-test default) sends neither key, mirroring the
    plumbing configuration on the Anthropic side.
    """

    model: str
    api_key: str
    base_url: str
    max_tokens: int = 4096
    timeout: float = 300.0
    effort: str | None = None

    def start(self, system: str, user: str, actor: str = "lead") -> list:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def complete(self, history: list, allow: tuple[str, ...]) -> Reply:
        payload = {
            "model": self.model,
            "messages": history,
            "max_completion_tokens": self.max_tokens,
            "tools": [
                {"type": "function", "function": {"name": n, "description": d, "parameters": s}}
                for n, d, s in _specs(allow)
            ],
            "tool_choice": "auto",
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.effort is not None:
            payload["reasoning_effort"] = self.effort
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        events, ttfb, total = _post_sse(
            self.base_url.rstrip("/") + "/v1/chat/completions", payload, headers, self.timeout
        )

        # One assistant turn arrives as many deltas: text in content fragments,
        # each tool call as a name-bearing opener followed by argument-string
        # fragments keyed by `index`, and -- because include_usage was requested
        # -- a final usage-only chunk after the finish_reason.
        text_parts: list[str] = []
        slots: dict[int, dict] = {}
        finish = ""
        usage: dict = {}
        for event in events:
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    text_parts.append(delta["content"])
                for c in delta.get("tool_calls") or []:
                    slot = slots.setdefault(
                        int(c.get("index") or 0), {"id": "", "name": "", "arguments": ""}
                    )
                    if c.get("id"):
                        slot["id"] = c["id"]
                    fn = c.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]

        calls = []
        ordered = [slots[i] for i in sorted(slots)]
        for slot in ordered:
            try:
                arguments = parse_arguments(slot["arguments"])
            except ValueError:
                # Surfaced, not repaired: a model that cannot emit valid JSON
                # arguments is exactly what the open-weights gate is looking for,
                # and silently fixing it would hide the finding.
                arguments = {"__malformed__": slot["arguments"][:200]}
            calls.append(ToolRequest(id=slot["id"], name=slot["name"], arguments=arguments))

        text = "".join(text_parts)
        # The assistant message to replay next turn, rebuilt in the shape the
        # endpoint demands back: `arguments` stays the VERBATIM wire string --
        # a real endpoint 400s a replayed tool call whose arguments arrive as a
        # parsed object, and re-serializing our parse would launder a malformed
        # emission into a valid-looking one.
        message: dict = {"role": "assistant", "content": text or None}
        if ordered:
            message["tool_calls"] = [
                {"id": s["id"], "type": "function",
                 "function": {"name": s["name"], "arguments": s["arguments"]}}
                for s in ordered
            ]

        # OpenAI reports prompt_tokens INCLUSIVE of cached ones; the cost model
        # bills the two at different rates, so the fresh count is the difference.
        details = usage.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0)
        prompt = int(usage.get("prompt_tokens") or 0)
        return Reply(
            text=text,
            tool_calls=tuple(calls),
            input_tokens=max(prompt - cached, 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            cache_read_tokens=cached,
            # There is no cache-write count to read: OpenAI prompt caching is
            # automatic, and writing to it is not a billed event with a token
            # figure. Zero is the true billed quantity for the write column,
            # not a missing measurement.
            cache_write_tokens=0,
            stop_reason=finish,
            total_s=total,
            ttfb_s=ttfb,
            raw=message,
        )

    def append_assistant(self, history: list, reply: Reply) -> None:
        history.append(reply.raw)

    def append_tool_results(self, history: list, results: list[tuple[ToolRequest, str]]) -> None:
        for call, result in results:
            history.append({"role": "tool", "tool_call_id": call.id, "content": result})


# ----------------------------------------------------------- OpenAI responses


@dataclass
class OpenAIResponsesClient(Client):
    """api.openai.com's `POST /v1/responses` wire format, streaming.

    This client exists because the chat-completions one CANNOT run the pinned
    spec: gpt-5.6 rejects function tools on /v1/chat/completions unless
    reasoning_effort is "none" (verified live 2026-08-27), and the error names
    this API as the one that takes tools and reasoning together. So gpt-5.6-sol
    dispatches here with effort "high" -- parity with the opus leg -- while
    `OpenAIClient` stays intact for the OpenAI-compatible open-source servers,
    which speak chat completions and nothing else.

    STATELESS ON PURPOSE. The API can hold the conversation server-side
    (`previous_response_id`), but the harness replays full history each turn and
    runs subagent conversations independently, so this client never uses it:
    every request carries `store: false` and the whole conversation in `input`.
    That makes reasoning replay OUR job -- the API emits reasoning items ahead
    of function calls and requires them back, verbatim, on the next turn, or it
    rejects the function_call as orphaned. With no server-side store those items
    are only replayable if they carry their content encrypted, which is what
    `include: ["reasoning.encrypted_content"]` requests. `raw` is therefore the
    turn's ENTIRE output-item list, replayed untouched; dropping the reasoning
    items (or their encrypted payloads) passes a single-turn test and 400s the
    first tool round trip -- the fake enforces the same rejection offline.

    The stream's delta events pace the bytes (and hand `_post_sse` a real first
    byte for ttfb); the turn itself is read off the terminal snapshot event
    (`response.completed` / `.incomplete` / `.failed`), which carries the
    response object -- output items, usage, status -- exactly as a buffered body
    would. Reading the provider's own final record beats re-assembling it from
    deltas: the counts must come from the provider (module docstring), and a
    stream that dies before the snapshot yields zeros a test can see rather
    than a plausible partial turn.

    Usage lands in the Anthropic-convention currency the whole repo uses:
    `input_tokens` arrives INCLUSIVE of cached tokens here (the same convention
    as chat completions' prompt_tokens, under a different field name), so the
    fresh count is the difference. `output_tokens` arrives INCLUSIVE of
    reasoning tokens and is exactly what bills at the output rate, so it maps
    across unchanged -- `output_tokens_details.reasoning_tokens` is a breakdown,
    never an addend. Cache writes are automatic and unbilled: zero, as on the
    chat leg.
    """

    model: str
    api_key: str
    base_url: str
    max_tokens: int = 4096
    timeout: float = 300.0
    effort: str | None = None

    def start(self, system: str, user: str, actor: str = "lead") -> list:
        # The system prompt is the top-level `instructions` parameter, not an
        # input item, so it rides with the history under a private key the API
        # never sees -- same pattern (and same concurrency reason) as the
        # Anthropic client.
        return [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user}],
                "_system": system,
            }
        ]

    @staticmethod
    def _split(history: list) -> tuple[str, list]:
        system = ""
        items = []
        for item in history:
            if "_system" in item:
                system = item["_system"]
                item = {k: v for k, v in item.items() if k != "_system"}
            items.append(item)
        return system, items

    def complete(self, history: list, allow: tuple[str, ...]) -> Reply:
        system, items = self._split(history)
        payload = {
            "model": self.model,
            "instructions": system,
            "input": items,
            "max_output_tokens": self.max_tokens,
            # Tools are FLAT on this wire format -- name, description, and
            # parameters at the top level. The chat-completions `function`
            # envelope is rejected here, and vice versa.
            "tools": [
                {"type": "function", "name": n, "description": d, "parameters": s}
                for n, d, s in _specs(allow)
            ],
            "tool_choice": "auto",
            "stream": True,
            # Statelessness and its price, together: store nothing server-side,
            # and ask for reasoning items in replayable (encrypted) form. Sent
            # unconditionally -- they are properties of this client, not of any
            # one model's spec, and a reasoning-by-default model behind an
            # effort-less config would otherwise break on its second turn.
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if self.effort is not None:
            payload["reasoning"] = {"effort": self.effort}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        events, ttfb, total = _post_sse(
            self.base_url.rstrip("/") + "/v1/responses", payload, headers, self.timeout
        )

        response: dict = {}
        for event in events:
            if event.get("type") in (
                "response.completed",
                "response.incomplete",
                "response.failed",
            ):
                response = event.get("response") or {}
        output = response.get("output") or []

        text_parts: list[str] = []
        calls: list[ToolRequest] = []
        for item in output:
            kind = item.get("type")
            if kind == "message":
                text_parts += [
                    part.get("text", "")
                    for part in item.get("content") or []
                    if part.get("type") == "output_text"
                ]
            elif kind == "function_call":
                wire = item.get("arguments") or ""
                try:
                    arguments = parse_arguments(wire)
                except ValueError:
                    # Surfaced, not repaired -- same rule as the chat leg.
                    arguments = {"__malformed__": wire[:200]}
                # `call_id` is the handle a function_call_output must answer;
                # the item's own `id` stays on the raw item for the replay.
                calls.append(
                    ToolRequest(
                        id=item.get("call_id", ""),
                        name=item.get("name", ""),
                        arguments=arguments,
                    )
                )

        # No finish_reason on this wire format: the response has a status, and
        # tool calls are just output items. Mapped onto the chat leg's
        # vocabulary so the two OpenAI legs land in one column downstream.
        status = response.get("status") or ""
        if status == "completed":
            stop = "tool_calls" if calls else "stop"
        elif status == "incomplete":
            stop = (response.get("incomplete_details") or {}).get("reason") or "incomplete"
        else:
            stop = status

        usage = response.get("usage") or {}
        details = usage.get("input_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0)
        inclusive = int(usage.get("input_tokens") or 0)
        return Reply(
            text="".join(text_parts),
            tool_calls=tuple(calls),
            input_tokens=max(inclusive - cached, 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_read_tokens=cached,
            cache_write_tokens=0,  # automatic and unbilled, as on the chat leg
            stop_reason=stop,
            total_s=total,
            ttfb_s=ttfb,
            # The whole item list, verbatim: reasoning items (with their
            # encrypted content), the message, and function_call items whose
            # `arguments` stay the wire string -- re-serializing a parse would
            # launder a malformed emission, and dropping the reasoning items
            # orphans every function call on the next request.
            raw=list(output),
        )

    def append_assistant(self, history: list, reply: Reply) -> None:
        history.extend(reply.raw or [])

    def append_tool_results(self, history: list, results: list[tuple[ToolRequest, str]]) -> None:
        # One function_call_output item per call, matched by call_id. The API
        # matches by id rather than by position, which the runner leans on: a
        # mixed turn executes plain tools before spawns, so results can arrive
        # out of emission order.
        for call, result in results:
            history.append(
                {"type": "function_call_output", "call_id": call.id, "output": result}
            )


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
