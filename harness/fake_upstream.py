"""A canned provider, so the instrument can be verified before it is trusted.

The proxy's job is to capture exact token counts and honest timestamps. Checking
whether it does by pointing it at a real provider costs money, takes a minute,
and -- worst -- gives an ambiguous answer: if the usage block comes back empty
you cannot tell whether the parser is wrong or the request was.

So the parser is verified against responses whose usage is known in advance,
served slowly enough on purpose that a proxy which buffered a stream instead of
relaying it would show a time-to-first-byte equal to its total duration. That
failure is invisible against a fast local server and expensive to discover
later, because it does not corrupt the token counts -- only the latency half of
every measurement, which is the half nobody double-checks.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__all__ = ["FakeUpstream", "EXPECTED_ANTHROPIC", "EXPECTED_OPENAI"]

# What a correct parser must recover. The smoke test asserts against these
# rather than against "some number appeared".
EXPECTED_ANTHROPIC = {
    "input_tokens": 1234,
    "output_tokens": 57,
    "cache_read_tokens": 900,
    "cache_write_tokens": 0,
    "tools_invoked": ["Task"],
    "stop_reason": "tool_use",
}

EXPECTED_OPENAI = {
    "input_tokens": 800,
    "output_tokens": 40,
    "cache_read_tokens": 600,
    "tools_invoked": ["read_file"],
    "stop_reason": "tool_calls",
}

_ANTHROPIC_EVENTS = [
    (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_fake",
                "model": "claude-fake",
                "usage": {
                    "input_tokens": 1234,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 900,
                    "cache_creation_input_tokens": 0,
                },
            },
        },
    ),
    (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "tu_1", "name": "Task", "input": {}},
        },
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 57},
        },
    ),
    ("message_stop", {"type": "message_stop"}),
]

_OPENAI_BODY = {
    "id": "chatcmpl-fake",
    "model": "oss-fake",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "seed/base.py"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {
        "prompt_tokens": 800,
        "completion_tokens": 40,
        "prompt_tokens_details": {"cached_tokens": 600},
    },
}


class FakeUpstream:
    """Serves one canned Anthropic stream and one canned OpenAI response."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, gap_s: float = 0.05) -> None:
        self.gap_s = gap_s
        self._server = ThreadingHTTPServer((host, port), self._make_handler())
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> FakeUpstream:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self) -> FakeUpstream:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _make_handler(self):
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                if "chat/completions" in self.path:
                    self._json(_OPENAI_BODY)
                else:
                    self._sse()

            def _json(self, payload: dict) -> None:
                body = json.dumps(payload).encode()
                time.sleep(upstream.gap_s)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _sse(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for name, payload in _ANTHROPIC_EVENTS:
                    # The pause is the point: it separates first-byte from total,
                    # so a buffering proxy is caught rather than flattered.
                    time.sleep(upstream.gap_s)
                    frame = f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()
                    self.wfile.write(b"%X\r\n" % len(frame) + frame + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")

        return Handler
