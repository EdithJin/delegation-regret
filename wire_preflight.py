"""Wire-format preflight: the five live-endpoint probes, captured as code.

These five requests were first run ad hoc on 2026-08-27 while diagnosing why
every call of the GPT preflight 400'd; this file is their permanent, re-runnable
form (recovered verbatim from that session's record). Together they pin the one
wire fact the offline fakes could not know in advance:

    gpt-5.6 REJECTS function tools on /v1/chat/completions unless
    reasoning_effort is explicitly "none" (the default is non-none, so
    omitting the field with tools present also 400s). Tools WITH reasoning
    require the /v1/responses wire format.

That fact is also pinned offline (protocol_upstream's chat-completions
validation + tests) and in HARNESS_SPEC's effort pin (harness/cli.py). This
script is the live re-verification: run it whenever the model id, the pinned
spec, or the vendor's behavior is in doubt. Total cost well under one cent
(max_completion_tokens is tiny on every case).

    python3 wire_preflight.py                 # needs OPENAI_API_KEY; ~$0.002
    python3 wire_preflight.py --model gpt-x   # probe another chat-completions model

Exit 0 = every case behaved as recorded on 2026-08-27; exit 1 = the wire has
changed (re-pin the spec before trusting any run).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

TOOL = {"type": "function", "function": {"name": "f", "description": "a tool",
        "parameters": {"type": "object", "properties": {}}}}


def cases(model: str):
    base = {"model": model,
            "messages": [{"role": "user", "content": "say ok"}],
            "max_completion_tokens": 16}
    return [
        # (name, payload, expect_status, expect_in_error)
        ("minimal", base, 200, None),
        ("stream+usage", {**base, "stream": True,
                          "stream_options": {"include_usage": True}}, 200, None),
        ("reasoning-no-tools", {**base, "reasoning_effort": "high"}, 200, None),
        ("tools-default-effort", {**base, "tools": [TOOL], "tool_choice": "auto"},
         400, "reasoning_effort"),
        ("tools+effort-none", {**base, "reasoning_effort": "none",
                               "max_completion_tokens": 200,
                               "messages": [{"role": "user",
                                             "content": "call the tool f with no args"}],
                               "tools": [TOOL], "tool_choice": "auto"}, 200, None),
    ]


def probe(url: str, key: str, payload: dict) -> tuple[int, str]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp.read()
            return resp.status, ""
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="gpt-5.6-sol")
    p.add_argument("--base-url", default="https://api.openai.com")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    args = p.parse_args(argv)

    key = os.environ.get(args.api_key_env, "")
    if not key:
        raise SystemExit(f"{args.api_key_env} is not set; this preflight spends "
                         "~$0.002 against the real endpoint and has no keyless mode.")
    url = args.base_url.rstrip("/") + "/v1/chat/completions"

    failures = 0
    for name, payload, want_status, want_text in cases(args.model):
        status, err = probe(url, key, payload)
        ok = status == want_status and (want_text is None or want_text in err)
        failures += 0 if ok else 1
        detail = "" if ok else f"   <-- expected {want_status}" + (
            f" mentioning {want_text!r}" if want_text else "") + f"; error: {err[:160]}"
        print(f"  {name:<22} HTTP {status}  {'OK' if ok else 'CHANGED'}{detail}")

    print("\n  wire matches the 2026-08-27 record." if failures == 0 else
          f"\n  {failures} case(s) CHANGED: the endpoint's behavior moved -- re-pin "
          "HARNESS_SPEC/tests before trusting any run.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
