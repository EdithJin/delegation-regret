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
    """One scripted assistant reply: some text and some tool calls."""

    text: str = ""
    tools: tuple[tuple[str, dict], ...] = ()  # (name, arguments)
    input_tokens: int = 900
    output_tokens: int = 120
    cache_read_tokens: int = 0


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
    # Verified against the live endpoint 2026-08-27 (wire_preflight.py case 4):
    # gpt-5.6 rejects function tools on chat completions unless reasoning_effort
    # is explicitly "none" -- the default is non-none, so OMITTING the field
    # with tools present also 400s. Reasoning-plus-tools needs /v1/responses.
    if payload.get("tools") and model.startswith("gpt-") and effort != "none":
        raise ProtocolError(
            f"openai: function tools with reasoning_effort are not supported for "
            f"{model} in /v1/chat/completions; use /v1/responses or set "
            "reasoning_effort to 'none'"
        )

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


# ------------------------------------------------------------------- server


@dataclass
class ProtocolUpstream:
    """A validating fake for one provider, scripted per conversation."""

    flavor: str  # "anthropic" | "openai"
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
        if self.flavor not in ("anthropic", "openai"):
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

            def _send_sse(self, frames: list, prefill_s: float, generation_s: float) -> None:
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
                    frame = f"data: {json.dumps(payload)}\n\n".encode()
                    self.wfile.write(b"%X\r\n" % len(frame) + frame + b"\r\n")
                    self.wfile.flush()
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
