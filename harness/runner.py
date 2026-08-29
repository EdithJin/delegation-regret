"""Stage 5: point a model at a scenario, let it work, record everything.

Low-level design: low-level-design.md, "Stage 5".

Stage 4 computed what *should* happen. This finds out what does. It produces a
`Trace` and nothing else; Stage 6 turns traces into regret.

WHAT THE AGENT IS AND IS NOT TOLD. It gets `TASKS.md` and a directory. The
decomposition is disclosed -- the task list names the nodes -- because that is
the granularity contract that makes the oracle's plan space comparable at all.
The dependency structure is not, unless `disclose_dag=True`: which module imports
which is discoverable by reading imports or by running the suite and seeing where
failures originate, and that inference is part of the measured skill.

THE THREE THINGS THIS FILE HAS TO GET RIGHT
-------------------------------------------

**A subagent is atomic.** Fresh conversation, empty history, seeded with only the
instruction and the files it was handed. Its tools are the lead's minus
`spawn_subagent`, so delegation is one level deep by construction rather than by
request. The lead receives one string and cannot see the subagent's reasoning,
tool calls, or intermediate state. This is not a stylistic choice -- it is the
same property that makes two of the chain plans infeasible in Stage 4.3, and if
the harness were more permissive than the oracle the agent would be scored
against a plan space it did not have.

**Concurrency comes from one turn, not from many.** Subagents issued in a SINGLE
assistant turn run together on a pool capped at `MAX_CONCURRENCY`, matching the
oracle's cap exactly. A lead that spawns one, waits for the summary, then spawns
another has serialised its own fan-out; the trace records that faithfully via
`SpawnRecord.batch` and the latency reconstruction charges it accordingly. An
oracle with more workers than the agent can use would penalise the agent for a
constraint it never faced, so the two constants are pinned to each other by test.

**Attribution is captured at the write.** Each loop gets its own labelled
`Workspace` view over the same root, so `write_file` records who wrote what as it
happens. Section 5.4 is explicit that this cannot be recovered afterwards, and it
is what every quality check in Stage 6 depends on.

FORCED-PLAN MODE, AND WHY IT WORKS THE WAY IT DOES
--------------------------------------------------

Section 6.1(b) needs the oracle's plan *executed for real* -- the regret
numerator is measured against a run, not against the oracle's own prediction,
because otherwise a slightly-off constant surfaces as fake regret. The design
says "force the agent to do n0 itself and delegate n1 and n2" without saying how,
and there are two ways to read it.

The rejected reading is to have the runner issue the spawns itself. That would
guarantee compliance, but the lead would never emit the briefing tokens, so the
baseline would be cheaper than any real run of the same plan -- and it would be
cheaper by exactly the delegation overhead the benchmark exists to price. Regret
would be inflated against every agent, systematically, in the direction of the
hypothesis.

So `run_plan` runs the ordinary loop with the plan stated in the prompt, and then
CHECKS compliance against `trace.realized_plan()`, which is derived from
attribution rather than from anything the model claimed. Non-compliance is
recorded, not corrected: a baseline run that did not execute the requested plan
is not a baseline, and Stage 6 excludes the scenario rather than scoring against
it.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from generator.oracle import Plan
from generator.scenario import Scenario
from generator.templates import module_path

from .client import Client, Reply, ToolRequest
from .proxy import LoggingProxy
from .tools import MAX_CONCURRENCY, Workspace
from .trace import LEAD, ModelCall, SpawnRecord, Trace, WriteEvent, subagent_actor

__all__ = [
    "run_agent",
    "run_plan",
    "LEAD_SYSTEM",
    "SUBAGENT_SYSTEM",
    "describe_plan",
    "Budget",
    "BudgetExceeded",
    "RetryPolicy",
    "through_proxy",
]


@contextmanager
def through_proxy(client: Client, log_path=None):
    """Route a client's traffic through the logging proxy for the duration.

    WHY THIS IS NOT OPTIONAL POLISH. The proxy is the only INDEPENDENT witness to
    what was billed. `Reply` token counts come from the provider's usage block as
    parsed by the client; the proxy parses the same block off the wire with
    different code. Run both and a disagreement is visible; run only one and a
    parsing error is invisible -- the trace and the invoice would differ and
    nothing would say so. Every dollar figure downstream is that parse.

    It is also the only source of real time-to-first-byte, since the runner is
    non-streaming and reports `ttfb_s == total_s`.

    A client with no `base_url` (the scripted one) is passed through untouched, so
    tests do not need a socket to exercise the loop.
    """
    if not getattr(client, "base_url", None):
        yield client, None
        return
    with LoggingProxy(client.base_url, log_path=log_path) as proxy:
        original = client.base_url
        client.base_url = proxy.base_url
        try:
            yield client, proxy
        finally:
            client.base_url = original


def _cross_check(trace: Trace, proxy: LoggingProxy | None) -> None:
    """Compare what the client parsed against what the proxy saw.

    Counts, not identity: the proxy sees one HTTP exchange per model call, so the
    call count must match exactly, and the token totals must agree because both
    read the same usage blocks. A mismatch is recorded on the trace rather than
    raised -- the run happened and its artifacts are real -- but it disqualifies
    the numbers, and `scoring` treats a trace carrying this note as unusable.
    """
    if proxy is None:
        return
    records = [r for r in proxy.records if r.billable]
    trace.proxy_calls = len(proxy.records)
    if len(records) != len(trace.calls):
        trace.notes.append(
            f"PROXY MISMATCH: proxy logged {len(records)} billable call(s), trace has "
            f"{len(trace.calls)}. One of the two is losing calls; the token totals "
            "cannot be trusted."
        )
        return
    for field_name in ("input_tokens", "output_tokens", "cache_read_tokens"):
        seen = sum(getattr(r, field_name) or 0 for r in records)
        claimed = sum(getattr(c, field_name) for c in trace.calls)
        if seen != claimed:
            trace.notes.append(
                f"PROXY MISMATCH: {field_name} proxy={seen} trace={claimed}. The client "
                "and the proxy parsed the same usage blocks differently."
            )
    trace.proxy_verified = not any(n.startswith("PROXY MISMATCH") for n in trace.notes)


class BudgetExceeded(Exception):
    """A run hit its own spending ceiling and stopped."""


@dataclass
class Budget:
    """A hard ceiling on one run, in calls and in tokens.

    Not a nicety. A runaway loop against a paid endpoint is the one failure in
    this repo that costs money in proportion to how long nobody is watching, and
    the turn cap alone does not bound it: a lead that spawns four subagents per
    turn multiplies its own cap by five. So the ceiling is counted across every
    actor in the run and checked before each call.

    Hitting it is recorded as a note and ends the run cleanly, with whatever was
    graded so far -- an aborted run is a failed run, and Stage 6 excludes it.
    Silently continuing would be worse than stopping.
    """

    max_calls: int = 400
    max_output_tokens: int = 400_000
    calls: int = 0
    output_tokens: int = 0

    def check(self) -> None:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"call budget exhausted ({self.calls}/{self.max_calls})")
        if self.output_tokens >= self.max_output_tokens:
            raise BudgetExceeded(
                f"output-token budget exhausted "
                f"({self.output_tokens}/{self.max_output_tokens})"
            )

    def charge(self, reply: Reply) -> None:
        self.calls += 1
        self.output_tokens += reply.output_tokens


@dataclass(frozen=True)
class RetryPolicy:
    """What to do about a transient provider failure.

    A single rate-limit response used to end a conversation, and the run was then
    scored as a failure -- so a busy afternoon on the provider's side looked
    exactly like a model that could not do the task. Worse, it looked like it
    asymmetrically: wide fan-out makes more concurrent calls, so it draws more
    429s, so the shape the headline finding rests on would fail more often for a
    reason that has nothing to do with delegation.

    Retries are counted onto the trace, not hidden. A run that needed twenty of
    them is not the same measurement as one that needed none, and the timing side
    especially should be read with that in mind.
    """

    attempts: int = 4
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0

    def delay_for(self, attempt: int) -> float:
        return min(self.base_delay_s * (2 ** attempt), self.max_delay_s)

    @staticmethod
    def is_transient(exc: BaseException) -> bool:
        """Rate limits, overload, and gateway errors. Never a 4xx we caused."""
        import urllib.error

        if isinstance(exc, urllib.error.HTTPError):
            return exc.code in (408, 409, 425, 429, 500, 502, 503, 504)
        if isinstance(exc, urllib.error.URLError):
            return True
        return isinstance(exc, (TimeoutError, ConnectionError))

ALL_TOOLS = ("list_files", "read_file", "write_file", "run_python", "spawn_subagent", "finish")
SUBAGENT_TOOLS = tuple(t for t in ALL_TOOLS if t != "spawn_subagent")

LEAD_SYSTEM = (
    "You are a lead engineer working in a code workspace. Use the tools to inspect files "
    "and do the work. You may delegate self-contained pieces to subagents with "
    f"spawn_subagent: at most {MAX_CONCURRENCY} run at once, they share no context with "
    "you, they return a single summary, and they cannot delegate further. Spawning costs "
    "tokens and time, so delegate when the parallelism is worth more than the overhead. "
    "Repair only the modules the tasks name; never edit a test file. Call finish when "
    "every task is done."
)

SUBAGENT_SYSTEM = (
    "You are a subagent. You have been given one self-contained piece of work and the "
    "files you need. You cannot delegate and you cannot ask questions. Do the work with "
    "the tools, never edit a test file, then call finish with a one-paragraph summary of "
    "what you changed. That summary is the only thing your caller will see."
)

USER_PROMPT = (
    "Read TASKS.md and carry out every task it lists. Work only in this workspace.\n\n{tasks}"
)


def describe_plan(
    scenario: Scenario,
    plan: Plan,
    order: tuple[int, ...] | None = None,
    node_order: tuple[str, ...] | None = None,
    pack_spawns: bool = False,
) -> str:
    """The plan directive appended to a forced run's prompt.

    Written as an instruction rather than as machinery, for the reason in the
    module docstring: the lead has to emit the briefings itself or the baseline
    under-counts delegation overhead.

    `node_order` fixes the order the lead works its OWN nodes in. Calibration
    needs it: the block-curve check compares runs across node orderings, and an
    "ordering" the model chose for itself is the same ordering every time.
    """
    blocks = list(plan.blocks)
    if order:
        blocks = [blocks[i] for i in order]
    lines = [
        "",
        "## Required execution plan",
        "",
        "Follow this plan exactly. It is not a suggestion, and deviating invalidates "
        "this run.",
        "",
    ]
    inline = sorted(plan.inline)
    if node_order and inline:
        ordered = [n for n in node_order if n in plan.inline]
        ordered += [n for n in inline if n not in ordered]  # never drop a node
        lines.append(
            "- Do these yourself, without delegating, finishing them one at a time "
            f"in exactly this order: {', '.join(ordered)}"
        )
    else:
        lines.append(
            f"- Do these yourself, without delegating: {', '.join(inline) if inline else '(none)'}"
        )
    if blocks:
        lines.append(
            f"- Issue exactly {len(blocks)} spawn_subagent call(s), in this order, "
            "each covering exactly the nodes listed:"
        )
        for i, block in enumerate(blocks):
            names = ", ".join(sorted(block))
            files = ", ".join(
                f"{scenario.subtasks[n].module} and {scenario.subtasks[n].test_module}"
                for n in sorted(block)
            )
            lines.append(f"  {i + 1}. subagent for {names} — give it {files}")
        if pack_spawns:
            # The disambiguation condition (2026-08-28): the baseline wording
            # above never says WHEN to issue the calls, and models resolve that
            # ambiguity differently -- one packs them into a single turn (they
            # then run concurrently), another issues one per turn and waits.
            # This line removes the ambiguity; runs carrying it are a separate,
            # labeled condition, never comparable with the baseline wording.
            lines.append(
                "- Issue ALL of these spawn_subagent calls together in ONE "
                "message (a single assistant turn), so they run concurrently. "
                "Do not wait for any subagent's result before issuing the rest."
            )
    else:
        lines.append("- Do not spawn any subagent.")
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ the loops


def _run_loop(
    client: Client,
    workspace: Workspace,
    actor: str,
    system: str,
    user: str,
    allow: tuple[str, ...],
    max_turns: int,
    calls: list[ModelCall],
    spawns: list[SpawnRecord],
    notes: list[str],
    pool: ThreadPoolExecutor | None,
    scenario: Scenario | None,
    lock,
    clock,
    budget: Budget,
    retry: RetryPolicy,
) -> tuple[str, int, bool]:
    """One conversation. Returns (final summary, turns used, finished cleanly).

    Shared by the lead and every subagent -- they differ only in `allow`, the
    system prompt, and whether a pool was handed in. Writing them as one loop is
    deliberate: a subagent that behaved differently from the lead in some
    incidental way would put a confound between "work done inline" and "work
    done by a subagent", which is precisely the comparison Stage 6.5 makes.
    """
    history = client.start(system, user, actor)
    summary = ""
    finished = False
    turns = 0

    while turns < max_turns and not finished:
        turns += 1
        started_at = clock()
        try:
            with lock:
                budget.check()
            reply: Reply = _complete_with_retry(
                client, history, allow, actor, turns, retry, notes, lock
            )
            with lock:
                budget.charge(reply)
        except BudgetExceeded as exc:
            notes.append(f"{actor}: {exc}")
            break
        except Exception as exc:  # a dead call ends this loop, not the run
            notes.append(f"{actor}: turn {turns} failed: {type(exc).__name__}: {exc}")
            break

        with lock:
            calls.append(
                ModelCall(
                    actor=actor,
                    index=turns - 1,
                    t_request=started_at,
                    input_tokens=reply.input_tokens,
                    output_tokens=reply.output_tokens,
                    cache_read_tokens=reply.cache_read_tokens,
                    cache_write_tokens=reply.cache_write_tokens,
                    ttfb_s=reply.ttfb_s,
                    total_s=reply.total_s,
                    stop_reason=reply.stop_reason,
                    tools_invoked=tuple(c.name for c in reply.tool_calls),
                )
            )
        client.append_assistant(history, reply)

        if not reply.tool_calls:
            summary = reply.text or summary
            break  # answered in prose; nothing left to execute

        # Spawns in ONE assistant turn are the only thing that runs concurrently.
        batch = [c for c in reply.tool_calls if c.name == "spawn_subagent"]
        others = [c for c in reply.tool_calls if c.name != "spawn_subagent"]
        results: list[tuple[ToolRequest, str]] = []

        for call in others:
            executed = workspace.invoke(call.name, call.arguments)
            if call.name == "finish":
                finished = True
                summary = str(call.arguments.get("summary") or "") or summary
            results.append((call, executed.result))

        if batch:
            if pool is None:
                # A subagent asked to delegate. Refused here rather than in the
                # prompt, so one-level delegation is a property of the harness.
                for call in batch:
                    results.append((call, "spawn_subagent is not available to a subagent"))
                notes.append(f"{actor}: attempted to delegate; refused")
            else:
                results.extend(
                    _run_batch(client, workspace, batch, spawns, calls, notes, pool, scenario,
                               max_turns, lock, clock, budget, retry)
                )

        client.append_tool_results(history, results)

    return summary, turns, finished


def _complete_with_retry(
    client: Client,
    history: list,
    allow: tuple[str, ...],
    actor: str,
    turn: int,
    retry: RetryPolicy,
    notes: list[str],
    lock,
) -> Reply:
    """One model call, retried on transient failures only.

    A 400 from a malformed conversation must NOT be retried -- it will fail
    identically four more times and hide the bug behind a delay. Only rate
    limits, overload, and transport errors are retried.
    """
    last: BaseException | None = None
    for attempt in range(retry.attempts):
        try:
            return client.complete(history, allow)
        except Exception as exc:
            if not retry.is_transient(exc):
                raise
            last = exc
            with lock:
                notes.append(
                    f"{actor}: turn {turn} attempt {attempt + 1} "
                    f"retrying after {type(exc).__name__}: {str(exc)[:80]}"
                )
            if attempt + 1 < retry.attempts:
                time.sleep(retry.delay_for(attempt))
    raise last if last else RuntimeError("retry loop exited without a result")


def _run_batch(
    client: Client,
    workspace: Workspace,
    batch: list[ToolRequest],
    spawns: list[SpawnRecord],
    calls: list[ModelCall],
    notes: list[str],
    pool: ThreadPoolExecutor,
    scenario: Scenario | None,
    max_turns: int,
    lock,
    clock,
    budget: Budget,
    retry: RetryPolicy,
) -> list[tuple[ToolRequest, str]]:
    """Run one turn's spawns concurrently and hand back one string each."""
    with lock:
        batch_id = max((s.batch for s in spawns), default=-1) + 1
        base = len(spawns)
        seeded = []
        for offset, call in enumerate(batch):
            index = base + offset
            instruction = str(call.arguments.get("instruction") or "")
            files = tuple(str(f) for f in (call.arguments.get("files") or ()))
            spawns.append(
                SpawnRecord(index=index, batch=batch_id, instruction=instruction, files=files)
            )
            seeded.append((index, call, instruction, files))

    def one(item):
        index, call, instruction, files = item
        actor = subagent_actor(index)
        view = workspace.view(actor)
        seed = [instruction, ""]
        for path in files:
            body = view.read_file(path)
            seed += [f"### {path}", "```python", body, "```", ""]
        summary, turns, finished = _run_loop(
            client, view, actor, SUBAGENT_SYSTEM, "\n".join(seed), SUBAGENT_TOOLS,
            max_turns, calls, spawns, notes, None, scenario, lock, clock, budget, retry,
        )
        with lock:
            spawns[index] = SpawnRecord(
                index=index,
                batch=batch_id,
                instruction=instruction,
                files=files,
                summary=summary,
                ok=finished,
                turns=turns,
            )
            workspace.calls.extend(view.calls)
        return call, (summary or "(subagent returned no summary)")

    return list(pool.map(one, seeded))


# ------------------------------------------------------------------- entry


def _finalize(
    scenario: Scenario,
    workspace: Workspace,
    trace: Trace,
    started: float,
) -> Trace:
    """Grade, attribute, and close out. Order matters: read tampering BEFORE
    verifying, because `verify` restores the suites and would erase it."""
    trace.tampered_tests = scenario.tampered_tests(workspace.root)
    if trace.tampered_tests:
        trace.notes.append(
            "edited generated suites: " + ", ".join(trace.tampered_tests)
        )
    trace.contested_writes = workspace.contested_writes()
    trace.write_events = [WriteEvent(path=p, actor=a, t=t) for p, a, t in workspace.writes]

    written = workspace.attribution()
    trace.node_attribution = {
        node_id: written[module_path(node_id)]
        for node_id in scenario.dag.ids
        if module_path(node_id) in written
    }
    trace.verdicts = {k: v.passed for k, v in scenario.verify(workspace.root).items()}
    trace.tool_seconds = sum(c.duration_s for c in workspace.calls)
    trace.wall_seconds = time.perf_counter() - started
    return trace


def run_agent(
    scenario: Scenario,
    client: Client,
    root: str | Path,
    *,
    disclose_dag: bool = False,
    max_turns: int = 40,
    condition: str = "agent",
    repeat: int = 0,
    directive: str = "",
    budget: Budget | None = None,
    retry: RetryPolicy | None = None,
    proxy_log: str | Path | None = None,
) -> Trace:
    """Materialize the scenario, run the model against it, grade independently.

    `proxy_log` routes every call through the logging proxy and writes the raw
    exchange log there, then cross-checks the trace's token totals against it.
    Strongly recommended for any paid run and required for a calibration run --
    see `through_proxy`.
    """
    import threading

    root = Path(root)
    scenario.materialize(root, disclose_dag=disclose_dag)
    started = time.perf_counter()
    # One clock for the whole run, zeroed at its start, shared by calls and
    # writes. Calibration compares the two streams, so they must agree, and an
    # absolute epoch would leak machine state into a trace meant to be portable.
    clock = lambda: time.perf_counter() - started  # noqa: E731
    workspace = Workspace(root, actor=LEAD, clock=clock)
    trace = Trace(
        scenario_id=scenario.id, model=client.model, condition=condition, repeat=repeat
    )
    lock = threading.Lock()
    budget = budget or Budget()
    retry = retry or RetryPolicy()

    with through_proxy(client, proxy_log) as (client, proxy), ThreadPoolExecutor(
        max_workers=MAX_CONCURRENCY
    ) as pool:
        _, turns, finished = _run_loop(
            client,
            workspace,
            LEAD,
            LEAD_SYSTEM,
            USER_PROMPT.format(tasks=scenario.surface(disclose_dag)) + directive,
            ALL_TOOLS,
            max_turns,
            trace.calls,
            trace.spawns,
            trace.notes,
            pool,
            scenario,
            lock,
            clock,
            budget,
            retry,
        )
    _cross_check(trace, proxy)
    trace.turns = turns
    trace.finished = finished
    if not finished:
        trace.notes.append(f"lead never called finish; stopped after {turns} turn(s)")
    return _finalize(scenario, workspace, trace, started)


def run_plan(
    scenario: Scenario,
    plan: Plan,
    client: Client,
    root: str | Path,
    *,
    order: tuple[int, ...] | None = None,
    node_order: tuple[str, ...] | None = None,
    disclose_dag: bool = False,
    max_turns: int = 40,
    condition: str = "oracle-plan",
    repeat: int = 0,
    budget: Budget | None = None,
    retry: RetryPolicy | None = None,
    proxy_log: str | Path | None = None,
    pack_spawns: bool = False,
) -> Trace:
    """Execute a GIVEN plan for real, and check the run actually followed it.

    This is section 6.1(b)'s baseline. Compliance is judged against
    `trace.realized_plan()` -- derived from who wrote which module -- and never
    against what the model said. A non-compliant run is flagged, not repaired;
    Stage 6 then excludes the scenario rather than scoring against a baseline
    that is not the plan it claims to be.
    """
    trace = run_agent(
        scenario,
        client,
        root,
        disclose_dag=disclose_dag,
        max_turns=max_turns,
        condition=condition,
        repeat=repeat,
        directive=describe_plan(scenario, plan, order, node_order, pack_spawns=pack_spawns),
        budget=budget,
        retry=retry,
        proxy_log=proxy_log,
    )
    inline, blocks = trace.realized_plan()
    complied = inline == plan.inline and sorted(map(sorted, blocks)) == sorted(
        map(sorted, plan.blocks)
    )
    if not complied:
        trace.notes.append(
            f"PLAN NOT FOLLOWED: asked for inline={sorted(plan.inline)} "
            f"blocks={[sorted(b) for b in plan.blocks]}; "
            f"ran inline={sorted(inline)} blocks={[sorted(b) for b in blocks]}"
        )
    return trace
