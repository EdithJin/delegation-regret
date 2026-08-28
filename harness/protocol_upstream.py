"""A fake provider that REFUSES malformed conversations.

`fake_upstream.FakeUpstream` serves canned bytes so the proxy's parser can be
checked. This one exists for the opposite direction: it checks what the harness
*sends*.

WHY A PERMISSIVE FAKE WOULD BE WORSE THAN NONE. The risky half of a client is not
reading a response -- a wrong field yields a zero and a test catches it. It is
BUILDING THE NEXT REQUEST. Both providers impose conversation invariants that a
canned server would happily ignore and a real endpoint rejects with a 400:

  * Anthropic requires every `tool_use` block in an assistant message to be
    answered by a `tool_result` in the very next user message, matched by id, and
    all of them in ONE user message.
  * OpenAI requires every entry in `tool_calls` to be answered by its own `tool`
    message carrying the matching `tool_call_id`.
  * The OpenAI responses API requires every `function_call` item to be answered
    by a `function_call_output` item carrying the matching `call_id` -- and,
    with reasoning on, requires each turn's `reasoning` items replayed ahead of
    its function calls, in replayable (encrypted) form when nothing was stored
    server-side. A client that drops them passes every single-turn test and
    400s its first tool round trip.

Break either and a single-turn test still passes -- there is no second turn to
reject. Delegation is precisely where the second turn matters: the lead emits a
`spawn_subagent` call, the subagent runs, and its summary has to travel back as a
well-formed tool result or the lead's next request is rejected. So the return
path cannot be verified by a fake that accepts anything.

This server therefore validates first and answers second, and a violation comes
back as an HTTP 400 with the reason, which surfaces in the trace as a failed turn
rather than as a silent shrug.

It is scripted the same way `ScriptedClient` is -- keyed on the first line of the
conversation's first user message -- so one test can drive a lead and several
subagents through the real wire formats.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__all__ = ["Turn", "ProtocolError", "ProtocolUpstream"]


class ProtocolError(Exception):
    """The harness sent something a real endpoint would reject."""


@dataclass(frozen=True)
class Turn:
    """One scripted assistant reply: some text and some tool calls.

    `output_tokens` is the BILLED output figure on every flavor. On the
    responses flavor `reasoning_tokens` names the subset of it that was spent
    reasoning (`output_tokens_details.reasoning_tokens` on the wire) -- a
    breakdown, never an addend, so a client that adds or subtracts it lands on
    a number no script contains.
    """

    text: str = ""
    tools: tuple[tuple[str, dict], ...] = ()  # (name, arguments)
    input_tokens: int = 900
    output_tokens: int = 120
    cache_read_tokens: int = 0
    reasoning_tokens: int = 0  # responses flavor only; subset of output_tokens


# ------------------------------------------------------------- validation

_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# The values this fake accepts for chat-completions reasoning_effort. Kept a
# little permissive on purpose (the harness only ever pins "high"); the exact
# roster a real gpt-5.6 endpoint accepts is a preflight item, not a constant
# this offline fake can certify.
_OPENAI_EFFORT_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh")


def _check_cache_control(block: dict, where: str) -> None:
    cc = block.get("cache_control")
    if cc is None:
        return
    if cc.get("type") != "ephemeral":
        raise ProtocolError(
            f"anthropic: cache_control in {where} has type {cc.get('type')!r}"
        )
    ttl = cc.get("ttl", "5m")
    if ttl not in ("5m", "1h"):
        raise ProtocolError(f"anthropic: cache_control ttl {ttl!r} is not a real TTL")


def _validate_anthropic(payload: dict) -> str:
    model = str(payload.get("model") or "")
    system = payload.get("system")
    if isinstance(system, list):
        if not system or not all(
            isinstance(b, dict) and b.get("type") == "text" and b.get("text") for b in system
        ):
            raise ProtocolError("anthropic: system blocks must be non-empty text blocks")
        for block in system:
            _check_cache_control(block, "system")
    elif not isinstance(system, str) or not system:
        raise ProtocolError("anthropic: system prompt missing")

    # The parameter 400s a real current-generation endpoint would raise. The
    # model-family checks matter most: the plumbing model (haiku) predates both
    # knobs, and sending it the matrix models' spec is exactly the mistake a
    # preflight would otherwise discover with real dollars.
    thinking = payload.get("thinking")
    if thinking is not None:
        if "budget_tokens" in thinking or thinking.get("type") == "enabled":
            raise ProtocolError(
                "anthropic: budget_tokens thinking is removed on current models"
            )
        if thinking.get("type") != "adaptive":
            raise ProtocolError(
                f"anthropic: unknown thinking type {thinking.get('type')!r}"
            )
        if "haiku" in model:
            raise ProtocolError(f"anthropic: {model} predates adaptive thinking")
    effort = (payload.get("output_config") or {}).get("effort")
    if effort is not None:
        if effort not in _EFFORT_LEVELS:
            raise ProtocolError(f"anthropic: unknown effort {effort!r}")
        if "haiku" in model:
            raise ProtocolError(f"anthropic: {model} predates the effort parameter")

    if not payload.get("tools"):
        raise ProtocolError("anthropic: no tools offered")
    for tool in payload["tools"]:
        if not {"name", "description", "input_schema"} <= set(tool):
            raise ProtocolError(f"anthropic: malformed tool spec {sorted(tool)}")

    messages = payload.get("messages") or []
    if not messages or messages[0].get("role") != "user":
        raise ProtocolError("anthropic: conversation must open with a user message")

    pending: set[str] = set()  # tool_use ids awaiting a result
    first_user = ""
    for i, message in enumerate(messages):
        role = message.get("role")
        expected = "user" if i % 2 == 0 else "assistant"
        if role != expected:
            raise ProtocolError(f"anthropic: message {i} is {role!r}, expected {expected!r}")
        content = message.get("content")
        if role == "user" and isinstance(content, str):
            first_user = first_user or content
            if pending:
                raise ProtocolError(f"anthropic: tool_use {sorted(pending)} never answered")
            continue
        if not isinstance(content, list) or not content:
            raise ProtocolError(f"anthropic: message {i} has empty content")
        for block in content:
            if isinstance(block, dict):
                _check_cache_control(block, f"message {i}")
        if role == "user":
            # A block-form opener (the cache-marked first request) still has to
            # key the script, so its text counts as the first user message.
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            first_user = first_user or text

        if role == "assistant":
            if pending:
                raise ProtocolError(f"anthropic: tool_use {sorted(pending)} never answered")
            pending = {b["id"] for b in content if b.get("type") == "tool_use"}
        else:
            answered = {b.get("tool_use_id") for b in content if b.get("type") == "tool_result"}
            stray = answered - pending
            if stray:
                raise ProtocolError(f"anthropic: tool_result for unknown id(s) {sorted(stray)}")
            missing = pending - answered
            if missing:
                raise ProtocolError(
                    f"anthropic: {sorted(missing)} left unanswered in the next user message"
                )
            pending = set()
    if pending:
        raise ProtocolError(f"anthropic: conversation ends with unanswered {sorted(pending)}")
    return first_user


def _validate_openai(payload: dict) -> str:
    messages = payload.get("messages") or []
    if not messages or messages[0].get("role") != "system":
        raise ProtocolError("openai: conversation must open with a system message")
    if not payload.get("tools"):
        raise ProtocolError("openai: no tools offered")
    for tool in payload["tools"]:
        if tool.get("type") != "function" or "function" not in tool:
            raise ProtocolError("openai: malformed tool spec")

    # The parameter 400s a real chat-completions endpoint would raise.
    model = str(payload.get("model") or "")
    if payload.get("stream_options") is not None and not payload.get("stream"):
        raise ProtocolError("openai: stream_options is only allowed when stream is true")
    if "max_tokens" in payload and model.startswith("gpt-"):
        raise ProtocolError(
            f"openai: {model} rejects legacy max_tokens; send max_completion_tokens"
        )
    mct = payload.get("max_completion_tokens")
    if mct is not None and (not isinstance(mct, int) or isinstance(mct, bool) or mct < 1):
        raise ProtocolError(f"openai: max_completion_tokens {mct!r} is not a positive integer")
    effort = payload.get("reasoning_effort")
    if effort is not None and effort not in _OPENAI_EFFORT_LEVELS:
        raise ProtocolError(f"openai: unknown reasoning_effort {effort!r}")

    first_user = ""
    pending: list[str] = []
    for i, message in enumerate(messages):
        role = message.get("role")
        if role == "user":
            first_user = first_user or (message.get("content") or "")
        if role == "tool":
            if not pending:
                raise ProtocolError(f"openai: tool message {i} answers nothing")
            if message.get("tool_call_id") != pending[0]:
                raise ProtocolError(
                    f"openai: tool message {i} has id {message.get('tool_call_id')!r}, "
                    f"expected {pending[0]!r} (results must follow in order)"
                )
            pending.pop(0)
            continue
        if pending:
            raise ProtocolError(f"openai: tool_calls {pending} never answered before a {role}")
        if role == "assistant":
            for c in message.get("tool_calls") or []:
                if not c.get("id"):
                    raise ProtocolError(f"openai: assistant tool_call in message {i} has no id")
                fn = c.get("function")
                if fn is not None and "arguments" in fn and not isinstance(fn["arguments"], str):
                    # The replay trap: arguments leave the endpoint as a JSON
                    # string, and must return as one. A client that appends its
                    # PARSED arguments to history passes every single-turn test
                    # and 400s on its second turn against the real thing.
                    raise ProtocolError(
                        f"openai: tool_call arguments in message {i} must be a JSON "
                        f"string, got {type(fn['arguments']).__name__}"
                    )
            pending = [c["id"] for c in (message.get("tool_calls") or [])]
    if pending:
        raise ProtocolError(f"openai: conversation ends with unanswered {pending}")
    if not model:
        raise ProtocolError("openai: model is required")
    return first_user


# What this fake accepts in the responses API's `include`. Only the entry the
# harness needs is recognized; the real roster is a preflight item.
_RESPONSES_INCLUDE = ("reasoning.encrypted_content",)


def _validate_responses(payload: dict) -> str:
    """The /v1/responses request validator -- SEPARATE from `_validate_openai`
    on purpose. The two dialects reject each other's parameters, and this one's
    defining acceptance is the pairing chat completions 400s on gpt-5.6: tools
    and reasoning TOGETHER. That pairing is never rejected here.
    """
    model = str(payload.get("model") or "")
    if not model:
        raise ProtocolError("responses: model is required")

    # The chat-completions parameters this API does not know. Each is the 400 a
    # half-ported client would hit on its very first request.
    if "messages" in payload:
        raise ProtocolError(
            "responses: unknown parameter 'messages'; the conversation travels in 'input'"
        )
    if "max_tokens" in payload or "max_completion_tokens" in payload:
        raise ProtocolError(
            "responses: unknown parameter; the output cap here is max_output_tokens"
        )
    if "reasoning_effort" in payload:
        raise ProtocolError(
            "responses: unknown parameter 'reasoning_effort'; send reasoning={'effort': ...}"
        )
    if "stream_options" in payload:
        raise ProtocolError(
            "responses: unknown parameter 'stream_options'; usage rides the terminal snapshot"
        )
    if payload.get("previous_response_id"):
        # The harness is stateless by design, and so is this fake: nothing was
        # ever stored for that id to name.
        raise ProtocolError(
            "responses: previous_response_id names a response this server never stored"
        )
    mot = payload.get("max_output_tokens")
    if mot is not None and (not isinstance(mot, int) or isinstance(mot, bool) or mot < 16):
        raise ProtocolError(
            f"responses: max_output_tokens {mot!r} is not an integer >= 16"
        )

    reasoning = payload.get("reasoning")
    effort = None
    if reasoning is not None:
        if not isinstance(reasoning, dict):
            raise ProtocolError("responses: reasoning must be an object")
        effort = reasoning.get("effort")
        if effort is not None and effort not in _OPENAI_EFFORT_LEVELS:
            raise ProtocolError(f"responses: unknown reasoning effort {effort!r}")

    if not payload.get("tools"):
        raise ProtocolError("responses: no tools offered")
    for tool in payload["tools"]:
        if tool.get("type") != "function":
            raise ProtocolError(f"responses: unknown tool type {tool.get('type')!r}")
        if "function" in tool:
            # The discriminator between the two OpenAI dialects: tools are FLAT
            # here, and the nested chat-completions envelope is a 400.
            raise ProtocolError(
                "responses: tools are flat (name at the top level); the "
                "chat-completions 'function' envelope is rejected"
            )
        if not tool.get("name"):
            raise ProtocolError("responses: tool spec has no name")
    for entry in payload.get("include") or []:
        if entry not in _RESPONSES_INCLUDE:
            raise ProtocolError(f"responses: unknown include entry {entry!r}")

    stateless = payload.get("store") is False
    reasoning_on = effort is not None and effort != "none"

    items = payload.get("input")
    if isinstance(items, str):
        return items  # a bare string is one user message
    if not isinstance(items, list) or not items:
        raise ProtocolError("responses: input must be a non-empty string or item list")

    first_user = ""
    pending: list[str] = []  # function_call call_ids awaiting their outputs
    reasoning_open = False  # a reasoning item still awaiting its following item
    reasoned_block = False  # the current assistant block opened with reasoning
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ProtocolError(f"responses: input item {i} is not an object")
        kind = item.get("type") or ("message" if "role" in item else "")
        if kind not in ("message", "reasoning", "function_call", "function_call_output"):
            raise ProtocolError(f"responses: unknown input item type {kind!r} at {i}")
        if pending and kind not in ("function_call", "function_call_output"):
            raise ProtocolError(
                f"responses: function_call(s) {pending} never answered before item {i}"
            )
        if reasoning_open and not (
            kind == "function_call"
            or (kind == "message" and item.get("role") == "assistant")
        ):
            raise ProtocolError(
                f"responses: reasoning item was provided without its required "
                f"following item (item {i} is {kind!r})"
            )

        if kind == "message":
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "system", "developer"):
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list) and content:
                    for part in content:
                        if not isinstance(part, dict) or part.get("type") != "input_text":
                            raise ProtocolError(
                                f"responses: {role} message {i} takes input_text parts"
                            )
                    text = "".join(p.get("text", "") for p in content)
                else:
                    raise ProtocolError(f"responses: message item {i} has empty content")
                if role == "user":
                    first_user = first_user or text
                reasoned_block = False
            elif role == "assistant":
                if isinstance(content, list) and content:
                    for part in content:
                        if not isinstance(part, dict) or part.get("type") not in (
                            "output_text",
                            "refusal",
                        ):
                            raise ProtocolError(
                                f"responses: assistant message {i} takes output parts"
                            )
                elif not (isinstance(content, str) and content):
                    raise ProtocolError(f"responses: message item {i} has empty content")
                reasoning_open = False
            else:
                raise ProtocolError(f"responses: message item {i} has role {role!r}")

        elif kind == "reasoning":
            if stateless and not item.get("encrypted_content"):
                # The replay trap this API adds on top of the chat one: with
                # store false nothing was kept server-side, so an id-only
                # reasoning item cannot be resolved. A client that requested
                # encrypted content and then dropped it lands here on turn two.
                raise ProtocolError(
                    f"responses: reasoning item at {i} has no encrypted_content; "
                    "with store false it cannot be resolved server-side"
                )
            reasoning_open = True
            reasoned_block = True

        elif kind == "function_call":
            if not item.get("call_id"):
                raise ProtocolError(f"responses: function_call at {i} has no call_id")
            if not item.get("name"):
                raise ProtocolError(f"responses: function_call at {i} has no name")
            arguments = item.get("arguments")
            if not isinstance(arguments, str):
                # Same replay trap as chat completions: arguments leave the
                # endpoint as a JSON string and must return as one.
                raise ProtocolError(
                    f"responses: function_call arguments at {i} must be a JSON "
                    f"string, got {type(arguments).__name__}"
                )
            if reasoning_on and not reasoned_block:
                raise ProtocolError(
                    f"responses: function_call {item['call_id']!r} was provided "
                    "without its required 'reasoning' item -- with reasoning on, "
                    "each turn's reasoning items must be replayed ahead of its calls"
                )
            pending.append(item["call_id"])
            reasoning_open = False

        else:  # function_call_output
            cid = item.get("call_id")
            if not pending:
                raise ProtocolError(f"responses: function_call_output at {i} answers nothing")
            if cid not in pending:
                raise ProtocolError(
                    f"responses: function_call_output at {i} has call_id {cid!r}; "
                    f"unanswered calls are {pending}"
                )
            if not isinstance(item.get("output"), str):
                raise ProtocolError(
                    f"responses: function_call_output at {i} must carry a string output"
                )
            # Matched by id, NOT by position: the runner answers a mixed turn's
            # plain tools before its spawns, so outputs may arrive out of
            # emission order and the real endpoint accepts that.
            pending.remove(cid)
            reasoned_block = False

    if pending:
        raise ProtocolError(f"responses: conversation ends with unanswered {pending}")
    if reasoning_open:
        raise ProtocolError(
            "responses: conversation ends with a reasoning item and no following item"
        )
    return first_user


# ---------------------------------------------------------------- responses


def _anthropic_body(turn: Turn, model: str) -> dict:
    content: list[dict] = []
    if turn.text:
        content.append({"type": "text", "text": turn.text})
    for i, (name, arguments) in enumerate(turn.tools):
        content.append({"type": "tool_use", "id": f"toolu_{i}", "name": name, "input": arguments})
    if not content:
        content.append({"type": "text", "text": "(no output)"})
    return {
        "id": "msg_p",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "tool_use" if turn.tools else "end_turn",
        "usage": {
            "input_tokens": turn.input_tokens,
            "output_tokens": turn.output_tokens,
            "cache_read_input_tokens": turn.cache_read_tokens,
            "cache_creation_input_tokens": 0,
        },
    }


def _openai_body(turn: Turn, model: str) -> dict:
    message: dict = {"role": "assistant", "content": turn.text or None}
    if turn.tools:
        message["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
            for i, (name, arguments) in enumerate(turn.tools)
        ]
    return {
        "id": "chatcmpl-p",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if turn.tools else "stop",
            }
        ],
        # prompt_tokens is INCLUSIVE of cached tokens here, which is the
        # difference the client has to undo. Building it that way on purpose so
        # a client that forgets is caught.
        "usage": {
            "prompt_tokens": turn.input_tokens + turn.cache_read_tokens,
            "completion_tokens": turn.output_tokens,
            "prompt_tokens_details": {"cached_tokens": turn.cache_read_tokens},
        },
    }


def _halves(text: str) -> list:
    """A string in two pieces, so reassembly across deltas is exercised."""
    mid = (len(text) + 1) // 2
    return [p for p in (text[:mid], text[mid:]) if p]


def _openai_frames(turn: Turn, model: str, include_usage: bool) -> list:
    """The scripted turn as streaming chunks, shaped the way the real endpoint
    streams them: a role opener, content fragments, each tool call as a
    name-bearing opener followed by argument fragments split MID-JSON, a
    finish_reason chunk, and a usage-only finale -- the finale ONLY when the
    request asked for stream_options.include_usage, because that is when the
    real endpoint sends one. A client that forgets the option gets a stream
    with no usage in it, and its Reply shows zeros a test can catch.
    """
    frames: list = [
        {
            "id": "chatcmpl-p",
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    ]
    for piece in _halves(turn.text):
        frames.append(
            {"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
        )
    for i, (name, arguments) in enumerate(turn.tools):
        frames.append(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": i,
                                    "id": f"call_{i}",
                                    "type": "function",
                                    "function": {"name": name, "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
        for piece in _halves(json.dumps(arguments)):
            frames.append(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [{"index": i, "function": {"arguments": piece}}]
                            },
                            "finish_reason": None,
                        }
                    ]
                }
            )
    frames.append(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls" if turn.tools else "stop",
                }
            ]
        }
    )
    if include_usage:
        frames.append(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": turn.input_tokens + turn.cache_read_tokens,
                    "completion_tokens": turn.output_tokens,
                    "prompt_tokens_details": {"cached_tokens": turn.cache_read_tokens},
                },
            }
        )
    return frames


# ------------------------------------------------------- the responses flavor


def _responses_usage_block(turn: Turn) -> dict:
    """input_tokens INCLUSIVE of cached (chat completions' convention under
    Anthropic's field name) and output_tokens INCLUSIVE of reasoning -- the
    billed figure -- with the reasoning share broken out beneath it. Built this
    way on purpose so a client that forgets the input subtraction, or treats
    reasoning tokens as an addend, lands on a number no script contains."""
    return {
        "input_tokens": turn.input_tokens + turn.cache_read_tokens,
        "input_tokens_details": {"cached_tokens": turn.cache_read_tokens},
        "output_tokens": turn.output_tokens,
        "output_tokens_details": {"reasoning_tokens": turn.reasoning_tokens},
        "total_tokens": turn.input_tokens + turn.cache_read_tokens + turn.output_tokens,
    }


def _responses_items(turn: Turn, reasoning: bool, encrypted: bool) -> list:
    """A scripted turn as output items: a reasoning item when the request runs
    with reasoning on -- in replayable form only when the request asked to
    `include` the encrypted content; forgetting that is not an error HERE, it
    is a 400 on the NEXT request, exactly the failure shape the real endpoint
    gives -- then the assistant message, then one function_call item per tool.
    Each function_call carries both its item `id` and the `call_id` an output
    must answer, because the two really are different handles on the wire."""
    items: list = []
    if reasoning:
        item: dict = {"type": "reasoning", "id": "rs_p", "summary": []}
        if encrypted:
            item["encrypted_content"] = "gAAAA-opaque-reasoning-payload"
        items.append(item)
    if turn.text or not turn.tools:
        items.append(
            {
                "type": "message",
                "id": "msg_p",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": turn.text or "(no output)",
                        "annotations": [],
                    }
                ],
            }
        )
    for i, (name, arguments) in enumerate(turn.tools):
        items.append(
            {
                "type": "function_call",
                "id": f"fc_{i}",
                "call_id": f"call_{i}",
                "name": name,
                "arguments": json.dumps(arguments),
                "status": "completed",
            }
        )
    return items


def _responses_body(turn: Turn, model: str, items: list) -> dict:
    return {
        "id": "resp_p",
        "object": "response",
        "model": model,
        "status": "completed",
        "incomplete_details": None,
        "output": items,
        "usage": _responses_usage_block(turn),
    }


def _responses_frames(turn: Turn, model: str, items: list) -> list:
    """The scripted turn as the responses API's streaming events, shaped the
    way the real endpoint streams them: typed events (each with a matching
    `event:` line and a sequence_number), item openers, text and argument
    deltas split MID-TOKEN, per-item `done` snapshots, and a terminal
    `response.completed` event carrying the whole response object -- which is
    the ONLY place usage appears. There is no usage-only frame and no [DONE]
    sentinel on this wire format; the stream simply ends after the snapshot.
    """
    skeleton = {
        "id": "resp_p",
        "object": "response",
        "model": model,
        "status": "in_progress",
        "output": [],
        "usage": None,
    }
    frames: list = [
        {"type": "response.created", "response": skeleton},
        {"type": "response.in_progress", "response": skeleton},
    ]
    for index, item in enumerate(items):
        if item["type"] == "reasoning":
            # The encrypted payload only lands on the `done` snapshot, so a
            # client that assembles from openers alone cannot replay the turn.
            opener = {k: v for k, v in item.items() if k != "encrypted_content"}
            frames.append(
                {"type": "response.output_item.added", "output_index": index, "item": opener}
            )
        elif item["type"] == "message":
            opener = dict(item, status="in_progress", content=[])
            frames.append(
                {"type": "response.output_item.added", "output_index": index, "item": opener}
            )
            frames.append(
                {
                    "type": "response.content_part.added",
                    "item_id": item["id"],
                    "output_index": index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                }
            )
            text = item["content"][0]["text"]
            for piece in _halves(text):
                frames.append(
                    {
                        "type": "response.output_text.delta",
                        "item_id": item["id"],
                        "output_index": index,
                        "content_index": 0,
                        "delta": piece,
                    }
                )
            frames.append(
                {
                    "type": "response.output_text.done",
                    "item_id": item["id"],
                    "output_index": index,
                    "content_index": 0,
                    "text": text,
                }
            )
            frames.append(
                {
                    "type": "response.content_part.done",
                    "item_id": item["id"],
                    "output_index": index,
                    "content_index": 0,
                    "part": item["content"][0],
                }
            )
        else:  # function_call
            opener = dict(item, arguments="", status="in_progress")
            frames.append(
                {"type": "response.output_item.added", "output_index": index, "item": opener}
            )
            for piece in _halves(item["arguments"]):
                frames.append(
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": item["id"],
                        "output_index": index,
                        "delta": piece,
                    }
                )
            frames.append(
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": item["id"],
                    "output_index": index,
                    "arguments": item["arguments"],
                }
            )
        frames.append(
            {"type": "response.output_item.done", "output_index": index, "item": item}
        )
    frames.append({"type": "response.completed", "response": _responses_body(turn, model, items)})
    for sequence, frame in enumerate(frames):
        frame["sequence_number"] = sequence
    return frames


def _responses_turn_index(items: object) -> int:
    """Which scripted turn a responses request asks for, derived statelessly
    like `_turn_for` does for the message flavors: each completed turn leaves
    exactly ONE contiguous group of function_call_output items in the replayed
    input (the runner answers a whole turn's calls in one append), so the group
    count is the number of assistant turns already taken."""
    if not isinstance(items, list):
        return 0
    groups = 0
    previous = False
    for item in items:
        is_output = isinstance(item, dict) and item.get("type") == "function_call_output"
        if is_output and not previous:
            groups += 1
        previous = is_output
    return groups


# ------------------------------------------------------------------- server


@dataclass
class ProtocolUpstream:
    """A validating fake for one provider, scripted per conversation."""

    flavor: str  # "anthropic" | "openai" | "responses"
    script: dict[str, list[Turn]] = field(default_factory=dict)
    model: str = "fake-model"
    host: str = "127.0.0.1"
    violations: list[str] = field(default_factory=list)
    requests: list[dict] = field(default_factory=list)
    # Seconds of simulated generation per output token. Zero by default, because
    # most tests do not care and sleeping makes them slow. Calibration tests DO
    # care: `fit_timing_model` regresses duration on token counts, so against a
    # server that answers instantly the fit sees noise and correctly refuses to
    # identify throughput. A realistic-shaped duration is what makes the timing
    # half of a calibration testable at all.
    seconds_per_output_token: float = 0.0
    # Prefill: duration that scales with the context being read, not with what is
    # generated. Needed INDEPENDENTLY of the output term, because
    # `fit_timing_model` has to separate the two and refuses when they are
    # collinear -- a fake whose latency depends only on output length cannot
    # exercise the prefill half of the model at all.
    seconds_per_input_token: float = 0.0
    base_latency_s: float = 0.0
    # Which script answers a request. Defaults to the first line of the first
    # user message. Overridden when two conversations open with the same line --
    # forced-serial and forced-fanout leads both start from the same task list
    # and differ only in the plan directive appended to it.
    key_from: object = None

    def __post_init__(self) -> None:
        if self.flavor not in ("anthropic", "openai", "responses"):
            raise ValueError(f"unknown flavor {self.flavor!r}")
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer((self.host, 0), self._handler())
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "ProtocolUpstream":
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self) -> "ProtocolUpstream":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _turn_for(self, key: str, payload: dict) -> Turn:
        """Which scripted turn answers this request.

        Derived from the request itself -- the number of assistant messages
        already in the history -- rather than from a per-key counter. A counter
        cannot serve the same conversation twice, and calibration deliberately
        replays one forced-serial run several times: the second replay would get
        whatever the first left on the cursor, which is the trailing `finish`, so
        it would write nothing and silently produce a run with no boundaries.
        Found exactly that way.

        Stateless also means thread-safe, which matters because fan-out runs
        several subagent conversations at once.
        """
        turns = self.script.get(key) or self.script.get("*")
        if turns is None:
            raise ProtocolError(f"no script for {key!r}; keys are {sorted(self.script)}")
        if self.flavor == "responses":
            index = _responses_turn_index(payload.get("input"))
        else:
            index = sum(1 for m in payload.get("messages", []) if m.get("role") == "assistant")
        return turns[min(index, len(turns) - 1)]

    def _handler(self):
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                with upstream._lock:
                    upstream.requests.append(payload)
                try:
                    if upstream.flavor == "anthropic":
                        first_user = _validate_anthropic(payload)
                    elif upstream.flavor == "responses":
                        first_user = _validate_responses(payload)
                    else:
                        first_user = _validate_openai(payload)
                    if callable(upstream.key_from):
                        key = upstream.key_from(first_user or "")
                    else:
                        key = (first_user or "").splitlines()[0].strip() or "lead"
                        if key.startswith("Read TASKS.md"):
                            key = "lead"
                    turn = upstream._turn_for(key, payload)
                except ProtocolError as exc:
                    with upstream._lock:
                        upstream.violations.append(str(exc))
                    return self._send(400, {"error": {"message": str(exc)}})
                # Prefill scales with the context being read, generation with
                # what is produced -- kept separate so the streamed path can
                # put them where a real endpoint does: prefill before the first
                # byte, generation spread across the chunks.
                prefill = (
                    upstream.base_latency_s
                    + upstream.seconds_per_input_token * turn.input_tokens
                )
                generation = upstream.seconds_per_output_token * turn.output_tokens
                if upstream.flavor == "responses":
                    effort = (payload.get("reasoning") or {}).get("effort")
                    items = _responses_items(
                        turn,
                        reasoning=effort is not None and effort != "none",
                        encrypted="reasoning.encrypted_content"
                        in (payload.get("include") or []),
                    )
                    if payload.get("stream"):
                        frames = _responses_frames(turn, upstream.model, items)
                        return self._send_sse(
                            frames, prefill, generation, named=True, sentinel=False
                        )
                    if prefill + generation:
                        time.sleep(prefill + generation)
                    return self._send(200, _responses_body(turn, upstream.model, items))
                if upstream.flavor == "openai" and payload.get("stream"):
                    include_usage = bool(
                        (payload.get("stream_options") or {}).get("include_usage")
                    )
                    frames = _openai_frames(turn, upstream.model, include_usage)
                    return self._send_sse(frames, prefill, generation)
                body = (
                    _anthropic_body(turn, upstream.model)
                    if upstream.flavor == "anthropic"
                    else _openai_body(turn, upstream.model)
                )
                if prefill + generation:
                    time.sleep(prefill + generation)
                self._send(200, body)

            def _send_sse(
                self,
                frames: list,
                prefill_s: float,
                generation_s: float,
                named: bool = False,
                sentinel: bool = True,
            ) -> None:
                # `named` frames carry the responses API's `event:` line (a
                # correct parser keys on the payload's `type`, never on it);
                # `sentinel` is the chat-completions [DONE] marker, which the
                # responses stream does not send -- it simply ends after the
                # terminal snapshot.
                if prefill_s:
                    time.sleep(prefill_s)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                pause = generation_s / max(len(frames), 1)
                for payload in frames:
                    if pause:
                        time.sleep(pause)
                    text = f"data: {json.dumps(payload)}\n\n"
                    if named:
                        text = f"event: {payload.get('type')}\n{text}"
                    frame = text.encode()
                    self.wfile.write(b"%X\r\n" % len(frame) + frame + b"\r\n")
                    self.wfile.flush()
                if sentinel:
                    done = b"data: [DONE]\n\n"
                    self.wfile.write(b"%X\r\n" % len(done) + done + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")

            def _send(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler
