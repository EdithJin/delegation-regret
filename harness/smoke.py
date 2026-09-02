"""The week-1 smoke tests, as a command rather than a memory of an afternoon.

Design doc: Phase1-DelegationBench-Design.md, the week-1 smoke gate. Sprint
schedule: Sprint-Schedule.md.

Neither test produces a number that appears in the results. Each answers one
yes/no question whose answer changes what gets built next, and the entire value
is in learning it before the thing above it is built rather than after:

  preflight    Does the instrument work at all? Zero API calls, zero dollars.
  native       Does Claude Code delegate in headless mode, and does the proxy
               capture exact tokens and honest timestamps while it does?
               A miss drops the native teaser now, on evidence.
  openweights  Can the candidate open-weights model drive the full six-tool
               subagent schema over multiple turns without fumbling the
               protocol? A miss swaps the candidate once, then drops the leg --
               and because that leg is the binding constraint on scenario
               difficulty, the answer also sets the template library's ceiling.

Each command ends by printing the pre-committed triage action rather than a bare
pass/fail, because the decision under time pressure is what the gate exists to
force, and a verdict that leaves the decision open has not done its job.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from generator.dag import sample_dag
from generator.scenario import build_scenario

from .fake_upstream import EXPECTED_ANTHROPIC, EXPECTED_OPENAI, FakeUpstream
from .proxy import ANTHROPIC_UPSTREAM, CallRecord, LoggingProxy
from .tools import MAX_CONCURRENCY, TOOL_SPECS, Workspace, openai_tools, parse_arguments

MIN_NODE_MAJOR = 18

# Claude Code names its subagent tool `Task`; Agent Teams adds its own. Either
# appearing in a trace is the delegation feature having triggered.
DELEGATION_TOOLS = {"Task", "AgentTeam", "Agent"}

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


class Report:
    """A checklist that prints as it goes and remembers whether anything failed."""

    def __init__(self, title: str) -> None:
        print(f"\n{title}\n{'=' * len(title)}")
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))
        mark = {PASS: "  ok ", FAIL: "FAIL ", WARN: "warn "}[status]
        print(f"{mark} {name}" + (f"\n        {detail}" if detail else ""))

    @property
    def failed(self) -> bool:
        return any(status == FAIL for status, _, _ in self.rows)

    def verdict(self, on_pass: str, on_fail: str) -> int:
        print("\n" + ("-" * 60))
        print(("TRIAGE: " + (on_fail if self.failed else on_pass)).strip())
        print("-" * 60)
        return 1 if self.failed else 0


def make_scenario(shape: str, n: int, size: int, seed: int):
    dag = sample_dag(shape, n, sizes=(size,), seed=seed)
    return build_scenario(dag, f"{shape}{n}-s{seed}", seed=seed)


def failed_detail(results: dict) -> str:
    """One line naming the nodes that did not verify, and why.

    Extracted so it is unit-testable without a paid run. The field it reads off
    `CheckResult` is exactly the kind of thing a payload rewrite renames, and
    the only code path that touches it otherwise needs the `claude` CLI and a
    live API key -- which is how it stayed broken through one.
    """
    return "; ".join(f"{k}: {v.detail}" for k, v in results.items() if not v.passed)


def node_major() -> int | None:
    try:
        out = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, timeout=20
        ).stdout.strip()
        return int(out.lstrip("v").split(".")[0])
    except Exception:
        return None


# ------------------------------------------------------------------ preflight


def cmd_preflight(args: argparse.Namespace) -> int:
    """Verify the instrument against a canned provider. No network, no spend."""
    report = Report("Preflight — the instrument, checked before it is trusted")

    # 1. The scenario machinery.
    try:
        scenario = make_scenario(args.shape, args.n, args.size, args.seed)
        with tempfile.TemporaryDirectory() as tmp:
            scenario.materialize(tmp)
            scenario.write_reference(tmp)
            ok = scenario.succeeded(tmp)
        report.add(
            PASS if ok else FAIL,
            "scenario materializes and its answer key verifies",
            f"{scenario.id}: {len(scenario.subtasks)} subtasks, "
            f"{sum(t.size for t in scenario.subtasks.values())} injected defects",
        )
    except Exception as exc:
        report.add(FAIL, "scenario machinery", f"{type(exc).__name__}: {exc}")

    # 2 and 3. The proxy, against responses whose usage is known in advance.
    with FakeUpstream(gap_s=args.gap) as upstream, LoggingProxy(upstream.base_url) as proxy:
        for label, path, payload, expected in (
            ("Anthropic streaming", "/v1/messages", {"stream": True}, EXPECTED_ANTHROPIC),
            ("OpenAI non-streaming", "/v1/chat/completions", {}, EXPECTED_OPENAI),
        ):
            try:
                request = urllib.request.Request(
                    proxy.base_url + path,
                    data=json.dumps({**payload, "tools": [{"name": "probe"}]}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(request, timeout=30).read()
            except Exception as exc:
                report.add(FAIL, f"proxy relays {label}", f"{type(exc).__name__}: {exc}")
                continue

            record = proxy.records[-1]
            mismatches = [
                f"{key}: got {getattr(record, key)}, expected {value}"
                for key, value in expected.items()
                if getattr(record, key) != value
            ]
            report.add(
                PASS if not mismatches else FAIL,
                f"proxy captures exact usage and tool calls — {label}",
                "; ".join(mismatches)
                or f"in={record.input_tokens} out={record.output_tokens} "
                f"cache_read={record.cache_read_tokens} tools={record.tools_invoked}",
            )

            # A proxy that buffered the stream would report the whole generation
            # as prefill. That leaves token counts intact and silently destroys
            # the latency half of every measurement.
            if record.stream:
                streamed = record.ttfb_s > 0 and record.ttfb_s < record.total_s * 0.8
                report.add(
                    PASS if streamed else FAIL,
                    "proxy relays without buffering (first byte precedes the last)",
                    f"ttfb={record.ttfb_s * 1000:.0f}ms total={record.total_s * 1000:.0f}ms",
                )

    # 4. The tool surface both legs share.
    names = [n for n, _, _ in TOOL_SPECS]
    report.add(
        PASS if len(names) >= 6 else FAIL,
        f"subagent tool schema is complete ({len(names)} tools, cap {MAX_CONCURRENCY})",
        ", ".join(names),
    )

    # 5. Claude Code, which the native leg needs before it can be attempted.
    major = node_major()
    if major is None:
        report.add(FAIL, "node is on PATH", "no runnable `node` found")
    elif major < MIN_NODE_MAJOR:
        report.add(
            FAIL,
            f"node is >= {MIN_NODE_MAJOR} (Claude Code will not parse on older)",
            f"found v{major}. Fix: `nvm install 22 && nvm use 22`",
        )
    else:
        report.add(PASS, f"node v{major} satisfies Claude Code")

    if shutil.which("claude") is None:
        report.add(WARN, "claude CLI on PATH", "native leg unavailable until installed")
    else:
        try:
            version = subprocess.run(
                ["claude", "--version"], capture_output=True, text=True, timeout=60
            )
            ok = version.returncode == 0
            report.add(
                PASS if ok else FAIL,
                "claude CLI runs",
                version.stdout.strip() if ok else (version.stderr or "").strip()[:200],
            )
        except Exception as exc:
            report.add(FAIL, "claude CLI runs", f"{type(exc).__name__}: {exc}")

    return report.verdict(
        on_pass="instrument verified. The two paid smoke tests are now worth running.",
        on_fail="fix the failures above before spending anything — a paid run "
        "against a broken instrument produces numbers you cannot interpret.",
    )


# --------------------------------------------------------------------- native


def cmd_native(args: argparse.Namespace) -> int:
    """Test A: headless Claude Code behind the proxy, on a real scenario."""
    report = Report("Native — Claude Code, headless, behind the logging proxy")

    major = node_major()
    if major is None or major < MIN_NODE_MAJOR:
        report.add(FAIL, "node runtime", f"need >= v{MIN_NODE_MAJOR}, found v{major}")
        return report.verdict("", "install a newer node first: `nvm install 22 && nvm use 22`")

    scenario = make_scenario(args.shape, args.n, args.size, args.seed)
    workspace = Path(args.workspace or tempfile.mkdtemp(prefix="smoke-native-"))
    scenario.materialize(workspace, disclose_dag=args.disclose)
    log_path = workspace / "proxy.jsonl"
    print(f"workspace: {workspace}")
    print(f"scenario:  {scenario.id}, {len(scenario.subtasks)} subtasks, shape {args.shape}")

    prompt = (
        "Read TASKS.md and carry out every task it lists. Work in this directory only. "
        "The tasks have dependencies between them -- read the imports in each module to "
        "work out which. Repair only the modules the tasks name; do not edit any test. "
        "Stop when every task's suite passes."
    )
    command = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "acceptEdits",
    ]
    if args.model:
        command += ["--model", args.model]

    with LoggingProxy(args.upstream, log_path=log_path) as proxy:
        env = {**os.environ, "ANTHROPIC_BASE_URL": proxy.base_url}
        started = time.perf_counter()
        try:
            done = subprocess.run(
                command,
                cwd=workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            stdout, stderr, code = done.stdout, done.stderr, done.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr, code = f"timed out after {args.timeout}s", -1
        wall = time.perf_counter() - started
        records = list(proxy.records)

    (workspace / "claude-stream.jsonl").write_text(stdout, encoding="utf-8")

    # Q1: did the CLI run at all, and did it reach a model through our proxy?
    report.add(
        PASS if records else FAIL,
        "requests reached the model through the proxy",
        f"{len(records)} call(s) in {wall:.1f}s"
        + (f"; exit {code}: {stderr.strip()[:200]}" if code != 0 else ""),
    )
    if not records:
        return report.verdict("", "no traffic captured — check auth and ANTHROPIC_BASE_URL support.")

    # Q2: is every successful generation call billable and timed? This is the
    # measurement itself. Scoped to generation calls on evidence from the first
    # real run (Aug 24): headless Claude Code also emits auxiliary traffic -- a
    # count_tokens call, and utility-model calls that 404 against a bare API
    # key -- which bills nothing and carries no usage block by definition.
    # Failing on those would drop the teaser over traffic the matrix would
    # never price; what must never lack usage is a 200 from /v1/messages.
    def _generation(r: CallRecord) -> bool:
        return r.status == 200 and "/count_tokens" not in r.path

    generation = [r for r in records if _generation(r)]
    auxiliary = [r for r in records if not _generation(r)]
    unbillable = [r.seq for r in generation if not r.billable]
    report.add(
        PASS if generation and not unbillable else FAIL,
        "every successful generation call carries exact token counts",
        f"missing usage on call(s) {unbillable}"
        if unbillable
        else f"{len(generation)} generation call(s): "
        f"in={sum(r.input_tokens or 0 for r in generation)} "
        f"out={sum(r.output_tokens or 0 for r in generation)}"
        + (
            f"; {len(auxiliary)} auxiliary/failed call(s) excluded (count_tokens or non-200)"
            if auxiliary
            else ""
        ),
    )
    timed = [r for r in records if r.ttfb_s > 0 and r.total_s >= r.ttfb_s]
    report.add(
        PASS if len(timed) == len(records) else FAIL,
        "first-byte and total latency recorded separately on every call",
        f"{len(timed)}/{len(records)} well-formed; "
        f"median ttfb {sorted(r.ttfb_s for r in records)[len(records) // 2] * 1000:.0f}ms",
    )

    # The pre-calibration gate asserts this per model; surfacing it here is free.
    cached = sum(r.cache_read_tokens or 0 for r in records)
    report.add(
        PASS if cached > 0 else WARN,
        "prompt cache is being read",
        f"cache_read total {cached} tokens"
        + ("" if cached else " — a no-cache leg contaminates every constant measured after it"),
    )

    # Q3: did the delegation feature trigger? This is the whole native question.
    spawned = [r for r in records for name in r.tools_invoked if name in DELEGATION_TOOLS]
    from_stream = _delegation_from_stream(stdout)
    report.add(
        PASS if (spawned or from_stream) else WARN,
        "the delegation feature triggered in headless mode",
        f"{len(from_stream)} subagent call(s) in the CLI stream"
        if from_stream
        else "no subagent tool call observed — expected on a scenario this small, "
        "but a wide scenario that still never delegates is the finding",
    )

    tampered = scenario.tampered_tests(workspace)
    results = scenario.verify(workspace)
    passed = sum(1 for r in results.values() if r.passed)
    report.add(
        PASS if not tampered else WARN,
        "the run left the generated suites alone",
        "unmodified"
        if not tampered
        else f"edited the suite for {', '.join(tampered)} — restored before grading, "
        "but a model that rewrites its own tests is a finding, not a nuisance",
    )
    report.add(
        PASS if passed == len(results) else WARN,
        "the run produced verifiable artifacts",
        f"{passed}/{len(results)} subtasks verified"
        + ("" if passed == len(results) else "; " + failed_detail(results)[:300]),
    )

    print(f"\ntrace: {log_path}\nstream: {workspace / 'claude-stream.jsonl'}")
    return report.verdict(
        on_pass="native leg is viable. Execute the teaser in week 2, per the "
        "front-loaded-cost rule — do not hold it for week 3.",
        on_fail="drop the native teaser now, on evidence, per the smoke gate.",
    )


def _delegation_from_stream(stdout: str) -> list[str]:
    """Subagent tool calls as the CLI itself reported them.

    Read alongside the proxy rather than instead of it: the proxy is ground
    truth for what was billed, the CLI stream is ground truth for what the agent
    decided to do, and the native question is about the decision.
    """
    found: list[str] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        message = event.get("message") or {}
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                if block.get("name") in DELEGATION_TOOLS:
                    found.append(block.get("name"))
    return found


# ---------------------------------------------------------------- openweights


def cmd_openweights(args: argparse.Namespace) -> int:
    """Test B: the candidate open-weights model on the full subagent schema."""
    report = Report("Open-weights — multi-turn tool calling on the subagent schema")

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and not args.allow_anonymous:
        report.add(FAIL, f"credential in ${args.api_key_env}", "unset")
        return report.verdict("", f"export {args.api_key_env} and re-run.")

    scenario = make_scenario(args.shape, args.n, args.size, args.seed)
    workspace = Path(args.workspace or tempfile.mkdtemp(prefix="smoke-oss-"))
    scenario.materialize(workspace, disclose_dag=args.disclose)
    log_path = workspace / "proxy.jsonl"
    print(f"workspace: {workspace}\nmodel:     {args.model} at {args.base_url}")

    bench = Workspace(workspace)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a lead engineer working in a code workspace. Use the tools to "
                "inspect files and do the work. You may delegate self-contained pieces "
                f"to subagents with spawn_subagent (at most {MAX_CONCURRENCY} at once, "
                "and they cannot delegate further). Call finish when everything is done."
            ),
        },
        {
            "role": "user",
            "content": (
                "Read TASKS.md and carry out every task it lists. Work only in this "
                "workspace. Repair only the modules the tasks name; do not edit any test."
            ),
        },
    ]

    protocol_errors: list[str] = []
    turns = 0
    finished = False
    with LoggingProxy(args.base_url, log_path=log_path) as proxy:
        endpoint = proxy.base_url + "/v1/chat/completions"
        while turns < args.max_turns and not finished:
            turns += 1
            payload = {
                "model": args.model,
                "messages": messages,
                "tools": openai_tools(),
                "tool_choice": "auto",
            }
            try:
                request = urllib.request.Request(
                    endpoint,
                    data=json.dumps(payload).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                )
                body = json.loads(urllib.request.urlopen(request, timeout=args.timeout).read())
            except urllib.error.HTTPError as exc:
                protocol_errors.append(f"turn {turns}: HTTP {exc.code} {exc.read()[:200]!r}")
                break
            except Exception as exc:
                protocol_errors.append(f"turn {turns}: {type(exc).__name__}: {exc}")
                break

            try:
                message = body["choices"][0]["message"]
            except (KeyError, IndexError, TypeError):
                protocol_errors.append(f"turn {turns}: unparseable response shape")
                break
            messages.append(message)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                break  # the model answered in prose; nothing left to execute

            for call in tool_calls:
                name = (call.get("function") or {}).get("name") or "(unnamed)"
                try:
                    arguments = parse_arguments((call.get("function") or {}).get("arguments"))
                except ValueError as exc:
                    protocol_errors.append(f"turn {turns} {name}: {exc}")
                    arguments = {}
                executed = bench.invoke(name, arguments)
                if not executed.ok:
                    protocol_errors.append(f"turn {turns} {name}: {executed.result[:120]}")
                if name == "finish":
                    finished = True
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": executed.result[: args.max_tool_output],
                    }
                )
        records = list(proxy.records)

    summary = bench.summary()
    report.add(
        PASS if records else FAIL,
        "the endpoint answered through the proxy",
        f"{len(records)} call(s), {turns} turn(s)",
    )
    report.add(
        PASS if not protocol_errors else FAIL,
        "multi-turn tool calling is protocol-clean",
        "; ".join(protocol_errors[:4]) if protocol_errors else f"{summary['calls']} tool calls, 0 malformed",
    )
    report.add(
        PASS if turns >= 3 else WARN,
        "the model sustains a multi-turn loop",
        f"{turns} turn(s); tools used: {summary['by_tool'] or 'none'}",
    )
    report.add(
        PASS if all(r.billable for r in records) and records else WARN,
        "the endpoint reports usage the proxy can capture",
        f"in={sum(r.input_tokens or 0 for r in records)} "
        f"out={sum(r.output_tokens or 0 for r in records)}",
    )
    report.add(
        PASS if summary["spawns"] else WARN,
        "the model can emit a well-formed spawn",
        f"{summary['spawns']} spawn call(s)",
    )
    tampered = scenario.tampered_tests(workspace)
    results = scenario.verify(workspace)
    passed = sum(1 for r in results.values() if r.passed)
    report.add(
        PASS if not tampered else WARN,
        "the run left the generated suites alone",
        "unmodified" if not tampered else f"edited the suite for {', '.join(tampered)}",
    )
    report.add(
        PASS if passed else WARN,
        "subtasks are inside this model's capability (success ~ 1 is required)",
        f"{passed}/{len(results)} verified — this sets the template difficulty ceiling",
    )

    print(f"\ntrace: {log_path}")
    return report.verdict(
        on_pass="third leg is viable. Freeze the template library at a difficulty "
        "this model clears, per the Aug 14 rule.",
        on_fail="swap the candidate once. If the swap also fails, drop the leg and "
        "set the difficulty ceiling from the frontier models instead.",
    )


# ----------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness.smoke", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_scenario_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--shape", default="diamond", choices=("wide", "chain", "diamond", "mixed"))
        p.add_argument("--n", type=int, default=5)
        p.add_argument("--size", type=int, default=1)
        p.add_argument("--seed", type=int, default=3)
        p.add_argument("--disclose", action="store_true", help="render the DAG-disclosed surface")
        p.add_argument("--workspace", help="reuse a directory instead of a temp one")

    pre = sub.add_parser("preflight", help="verify the instrument; no API calls")
    add_scenario_args(pre)
    pre.add_argument("--gap", type=float, default=0.05, help="fake upstream inter-event delay")
    pre.set_defaults(func=cmd_preflight)

    nat = sub.add_parser("native", help="Test A: headless Claude Code behind the proxy")
    add_scenario_args(nat)
    nat.add_argument("--model", help="e.g. sonnet, opus, or a full model id")
    nat.add_argument("--upstream", default=ANTHROPIC_UPSTREAM)
    nat.add_argument("--timeout", type=float, default=900.0)
    nat.set_defaults(func=cmd_native)

    oss = sub.add_parser("openweights", help="Test B: the candidate open-weights model")
    add_scenario_args(oss)
    oss.add_argument("--base-url", required=True, help="OpenAI-compatible root, no /v1")
    oss.add_argument("--model", required=True)
    oss.add_argument("--api-key-env", default="OPENAI_API_KEY")
    oss.add_argument("--allow-anonymous", action="store_true", help="local server, no key")
    oss.add_argument("--max-turns", type=int, default=24)
    oss.add_argument("--max-tool-output", type=int, default=6000)
    oss.add_argument("--timeout", type=float, default=300.0)
    oss.set_defaults(func=cmd_openweights)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
