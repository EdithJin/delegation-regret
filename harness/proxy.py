"""A logging forward proxy: the measurement instrument, not a debugging aid.

Design doc: Phase1-DelegationBench-Design.md section 7.

Cost and latency are not observations *about* this benchmark's results -- they
are the results. Regret is realized cost plus beta times realized latency, so
every number in the report is downstream of what this file records. That makes a
few things non-negotiable:

TOKENS COME FROM THE PROVIDER, NEVER FROM US. Counting tokens locally with a
tokenizer would be an estimate, and an estimate of the quantity under test is
not a measurement. Every count here is lifted from the provider's own `usage`
block, which is also what the invoice is computed from.

CACHE READS ARE RECORDED SEPARATELY. A cached input token is billed at a
fraction of a fresh one, so a run whose cache silently stopped working is not a
slightly noisy measurement -- it is a different price vector. Section 12's Aug 13
gate asserts `cache_read_input_tokens > 0` on every matrix model for exactly
this reason, and that assertion needs this field to exist.

TIMESTAMPS ARE TAKEN AT THE REQUEST, AND FIRST-BYTE IS SEPARATE FROM TOTAL.
The cost model needs `a + b * input_tokens + output_tokens / throughput`, which
cannot be fitted from a single duration. Time-to-first-byte is the prefill term
-- the one that makes a large lead context slower to *start* than a cold
subagent, which is what puts context drag in the latency column and what the
Tier-A crossover claim rests on.

The proxy speaks both the Anthropic Messages API and the OpenAI chat-completions
shape, streaming or not, because the third matrix leg is an open-weights model
behind an OpenAI-compatible endpoint and it has to land in the same log as
everything else or the cross-model table is not comparable.

Bytes are relayed as they arrive. The proxy must not buffer a streamed response
to inspect it, because doing so would fold the whole generation time into
time-to-first-byte and destroy the measurement it exists to take.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

__all__ = ["CallRecord", "UsageSniffer", "LoggingProxy", "ANTHROPIC_UPSTREAM"]

ANTHROPIC_UPSTREAM = "https://api.anthropic.com"

# Headers that describe this hop rather than the message, plus the two we must
# recompute. Forwarding these verbatim is how a proxy corrupts a stream.
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "accept-encoding",  # we relay raw bytes; let the upstream send us plain text
}


@dataclass
class CallRecord:
    """One model call, priced and timed. The unit every downstream number sums."""

    seq: int
    path: str
    model: str | None = None
    status: int = 0
    stream: bool = False
    t_request: float = 0.0  # wall clock at the moment the request went out
    ttfb_s: float = 0.0  # request sent -> first response byte (the prefill term)
    total_s: float = 0.0  # request sent -> last response byte
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    tools_offered: list[str] = field(default_factory=list)
    tools_invoked: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    error: str | None = None

    @property
    def billable(self) -> bool:
        """Did the provider tell us what this cost? A record that cannot answer
        this is not a measurement, and the smoke test's whole job is to find out
        whether the answer is yes before the matrix depends on it."""
        return self.input_tokens is not None and self.output_tokens is not None


class UsageSniffer:
    """Pulls usage, tool calls, and stop reason out of either API's wire format.

    Kept separate from the HTTP plumbing so it can be tested against recorded
    payloads without a socket -- which is how the parsing gets verified before
    any real money is spent on a request that might parse wrong.
    """

    def __init__(self) -> None:
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.cache_read_tokens: int | None = None
        self.cache_write_tokens: int | None = None
        self.tools_invoked: list[str] = []
        self.stop_reason: str | None = None
        self.model: str | None = None

    # -- provider-shaped readers ----------------------------------------

    def _anthropic_usage(self, usage: dict) -> None:
        # Anthropic reports input and output separately and may report either
        # one alone (message_start carries input, message_delta carries the
        # final output), so each field is set independently rather than as a
        # block -- overwriting input with None on the delta would lose it.
        if usage.get("input_tokens") is not None:
            self.input_tokens = usage["input_tokens"]
        if usage.get("output_tokens") is not None:
            self.output_tokens = usage["output_tokens"]
        if usage.get("cache_read_input_tokens") is not None:
            self.cache_read_tokens = usage["cache_read_input_tokens"]
        if usage.get("cache_creation_input_tokens") is not None:
            self.cache_write_tokens = usage["cache_creation_input_tokens"]

    def _openai_usage(self, usage: dict) -> None:
        if usage.get("prompt_tokens") is not None:
            self.input_tokens = usage["prompt_tokens"]
        if usage.get("completion_tokens") is not None:
            self.output_tokens = usage["completion_tokens"]
        details = usage.get("prompt_tokens_details") or {}
        if details.get("cached_tokens") is not None:
            self.cache_read_tokens = details["cached_tokens"]

    def _note_tool(self, name: str | None) -> None:
        if name and name not in self.tools_invoked:
            self.tools_invoked.append(name)

    # -- entry points ----------------------------------------------------

    def feed_event(self, payload: dict) -> None:
        """One decoded SSE `data:` payload, from either provider."""
        kind = payload.get("type")
        if kind == "message_start":
            message = payload.get("message") or {}
            self.model = message.get("model") or self.model
            self._anthropic_usage(message.get("usage") or {})
        elif kind == "content_block_start":
            block = payload.get("content_block") or {}
            if block.get("type") == "tool_use":
                self._note_tool(block.get("name"))
        elif kind == "message_delta":
            self._anthropic_usage(payload.get("usage") or {})
            self.stop_reason = (payload.get("delta") or {}).get("stop_reason") or self.stop_reason
        elif "choices" in payload:  # OpenAI streaming chunk
            self.model = payload.get("model") or self.model
            if payload.get("usage"):
                self._openai_usage(payload["usage"])
            for choice in payload["choices"]:
                delta = choice.get("delta") or {}
                for call in delta.get("tool_calls") or []:
                    self._note_tool((call.get("function") or {}).get("name"))
                if choice.get("finish_reason"):
                    self.stop_reason = choice["finish_reason"]

    def feed_body(self, body: bytes) -> None:
        """A complete non-streaming response body, from either provider."""
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(payload, dict):
            return
        self.model = payload.get("model") or self.model
        if "choices" in payload:  # OpenAI
            self._openai_usage(payload.get("usage") or {})
            for choice in payload["choices"]:
                message = choice.get("message") or {}
                for call in message.get("tool_calls") or []:
                    self._note_tool((call.get("function") or {}).get("name"))
                if choice.get("finish_reason"):
                    self.stop_reason = choice["finish_reason"]
            return
        self._anthropic_usage(payload.get("usage") or {})  # Anthropic
        for block in payload.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                self._note_tool(block.get("name"))
        self.stop_reason = payload.get("stop_reason") or self.stop_reason

    def feed_sse_chunk(self, chunk: bytes, carry: bytes = b"") -> bytes:
        """Feed raw streamed bytes; returns the incomplete tail to pass back in.

        SSE frames do not respect socket-read boundaries, so a line can arrive
        split across two chunks. Returning the remainder rather than assuming
        whole lines is the difference between reading the final usage block and
        silently missing it on a slow connection.
        """
        buffer = carry + chunk
        *lines, tail = buffer.split(b"\n")
        for line in lines:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if not data or data == b"[DONE]":
                continue
            try:
                self.feed_event(json.loads(data))
            except ValueError:
                continue
        return tail


def _tools_offered(body: bytes) -> tuple[list[str], str | None, bool]:
    """What the request asked for: tool names, model, and whether it streams."""
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return [], None, False
    if not isinstance(payload, dict):
        return [], None, False
    names = []
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        # Anthropic puts the name at the top level; OpenAI nests it under
        # `function`. Both shapes reach this proxy.
        name = tool.get("name") or (tool.get("function") or {}).get("name")
        if name:
            names.append(name)
    return names, payload.get("model"), bool(payload.get("stream"))


class LoggingProxy:
    """Runs a forwarding proxy on localhost and records every call through it.

    Point a client at `base_url` and it reaches `upstream` unchanged. Nothing is
    rewritten: the proxy must not alter the request, or the thing being measured
    is no longer the thing that would run in production.
    """

    def __init__(
        self,
        upstream: str = ANTHROPIC_UPSTREAM,
        log_path: str | Path | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        timeout: float = 900.0,
    ) -> None:
        self.upstream = upstream.rstrip("/")
        self.log_path = Path(log_path) if log_path else None
        self.timeout = timeout
        self.records: list[CallRecord] = []
        self._lock = threading.Lock()
        self._seq = 0
        self._server = ThreadingHTTPServer((host, port), self._make_handler())
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # -- lifecycle -------------------------------------------------------

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> LoggingProxy:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self) -> LoggingProxy:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- recording -------------------------------------------------------

    def _record(self, record: CallRecord) -> None:
        with self._lock:
            self.records.append(record)
            if self.log_path:
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    # -- the handler -----------------------------------------------------

    def _make_handler(self):
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:
                pass  # the JSONL record is the log; stderr noise would drown the run

            def do_GET(self) -> None:  # noqa: N802
                self._forward("GET")

            def do_POST(self) -> None:  # noqa: N802
                self._forward("POST")

            def _forward(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                tools, model, wants_stream = _tools_offered(body)
                record = CallRecord(
                    seq=proxy._next_seq(),
                    path=self.path,
                    model=model,
                    stream=wants_stream,
                    tools_offered=tools,
                )
                headers = {
                    k: v for k, v in self.headers.items() if k.lower() not in _HOP_BY_HOP
                }
                request = urllib.request.Request(
                    proxy.upstream + self.path, data=body or None, headers=headers, method=method
                )
                sniffer = UsageSniffer()
                record.t_request = time.time()
                started = time.perf_counter()
                try:
                    response = urllib.request.urlopen(request, timeout=proxy.timeout)
                except urllib.error.HTTPError as exc:
                    # An error response is still a response, and often still
                    # billed -- relay it and record it rather than swallowing it.
                    response = exc
                except Exception as exc:  # network failure: nothing to relay
                    record.error = f"{type(exc).__name__}: {exc}"
                    record.total_s = time.perf_counter() - started
                    proxy._record(record)
                    self.send_error(502, "upstream unreachable")
                    return

                record.status = response.status
                is_sse = "text/event-stream" in (response.headers.get("Content-Type") or "")
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() not in _HOP_BY_HOP:
                        self.send_header(key, value)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                first_byte: float | None = None
                carry = b""
                collected = bytearray()
                # read1 returns whatever has arrived; read(n) blocks until it has
                # all n bytes, which on a chunked stream means waiting for the
                # generation to finish. That difference is invisible in the token
                # counts and fatal to the latency measurement.
                pull = getattr(response, "read1", None) or response.read
                try:
                    while True:
                        chunk = pull(8192)
                        if not chunk:
                            break
                        if first_byte is None:
                            first_byte = time.perf_counter()
                        self.wfile.write(b"%X\r\n" % len(chunk) + chunk + b"\r\n")
                        self.wfile.flush()  # relay now; buffering would inflate TTFB
                        if is_sse:
                            carry = sniffer.feed_sse_chunk(chunk, carry)
                        else:
                            collected += chunk
                    self.wfile.write(b"0\r\n\r\n")
                except Exception as exc:
                    record.error = f"relay failed: {type(exc).__name__}: {exc}"
                finally:
                    response.close()

                if not is_sse:
                    sniffer.feed_body(bytes(collected))
                record.ttfb_s = (first_byte - started) if first_byte else 0.0
                record.total_s = time.perf_counter() - started
                record.input_tokens = sniffer.input_tokens
                record.output_tokens = sniffer.output_tokens
                record.cache_read_tokens = sniffer.cache_read_tokens
                record.cache_write_tokens = sniffer.cache_write_tokens
                record.tools_invoked = sniffer.tools_invoked
                record.stop_reason = sniffer.stop_reason
                record.model = record.model or sniffer.model
                proxy._record(record)

        return Handler
