"""What one run recorded, and the two numbers scoring reads off it.

Low-level design: low-level-design.md, "Stage 5.4".

A `Trace` is the whole observable result of pointing a model at one scenario:
every model call with its billed tokens, every subagent with when it was briefed
and what it was told, who produced each node's result, and what the independent
verifier said afterwards. Stage 6 reads nothing else.

TWO RULES THIS FILE EXISTS TO ENFORCE.

**Tokens come from the provider.** Every count here was lifted from a `usage`
block on a real response -- the same block the invoice is computed from and the
same one `harness.proxy` sniffs independently. Nothing is tokenized locally. An
estimate of the quantity under test is not a measurement, and the whole cost
half of the metric is downstream of this.

**Latency is analytic, not wall clock.** `analytic_minutes` reconstructs the
schedule from the call structure using the fitted timing model -- the same model
the oracle prices plans with, so both sides of regret are measured the same way.
Raw elapsed time is polluted by rate limits, queueing, and provider load; it is
not comparable across vendors or even across two runs of one scenario. Wall
clock is recorded as `wall_seconds` and used only as a sanity check.

WHY THE RECONSTRUCTION IS A SIMULATION RATHER THAN A SUM. Lead calls are serial
-- the lead is one conversation, one turn at a time. Subagents issued in the
SAME assistant turn run concurrently, capped, which is the entire mechanism by
which delegation buys time. Summing every call's minutes would price maximal
fan-out identically to always-serial and delete the finding. So `analytic_minutes`
walks the lead's timeline and, at each spawn batch, charges the batch's makespan
under the concurrency cap.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .calibrate import PriceSheet, TimingModel, call_minutes
from .proxy import CallRecord
from .tools import MAX_CONCURRENCY

__all__ = [
    "ModelCall",
    "WriteEvent",
    "SpawnRecord",
    "Trace",
    "LEAD",
    "subagent_actor",
    "batch_makespan",
]

LEAD = "lead"


def subagent_actor(index: int) -> str:
    return f"subagent:{index}"


@dataclass(frozen=True)
class ModelCall:
    """One API call, attributed to the loop that made it.

    `actor` is what makes the reconstruction possible: without it the trace is a
    flat list of calls with no way to know which of them overlapped in time.
    """

    actor: str
    index: int  # position within that actor's own conversation
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    ttfb_s: float = 0.0
    total_s: float = 0.0
    stop_reason: str = ""
    tools_invoked: tuple[str, ...] = ()
    # Seconds since the run began, at the moment the request went out. Required
    # by calibration, which segments a serial run's call log at each subtask
    # completion -- and that segmentation is impossible to recover afterwards,
    # so it is captured even though nothing in scoring reads it.
    t_request: float = 0.0

    def as_record(self) -> CallRecord:
        """Adapt to the shape `calibrate` already consumes.

        Calibration was written against the proxy's `CallRecord` and is tested
        offline against synthetic logs. Reusing that type rather than
        duplicating the pricing arithmetic keeps one definition of "what a call
        costs" in the repo.
        """
        return CallRecord(
            seq=self.index,
            path="",
            t_request=self.t_request,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
            ttfb_s=self.ttfb_s,
            total_s=self.total_s,
        )


@dataclass(frozen=True)
class WriteEvent:
    """A file written, by whom, and when, relative to the run's start.

    The other half of what calibration needs. A node's work is finished when its
    module was last written, so these are the boundaries that segment the call
    log -- and like attribution, the timing cannot be reconstructed later from a
    directory listing.
    """

    path: str
    actor: str
    t: float


@dataclass(frozen=True)
class SpawnRecord:
    """One delegation: what was asked, what came back, and when.

    `batch` groups spawns the lead issued in a SINGLE assistant turn. Those and
    only those overlap in time -- a lead that spawns one, waits for the summary,
    then spawns another has serialised its own fan-out and bought nothing. The
    field is therefore not bookkeeping; it is the difference between a plan that
    is fast and one that merely looks parallel.
    """

    index: int
    batch: int
    instruction: str
    files: tuple[str, ...]
    summary: str = ""
    ok: bool = True
    turns: int = 0

    @property
    def actor(self) -> str:
        return subagent_actor(self.index)


def batch_makespan(durations: list[float], cap: int = MAX_CONCURRENCY) -> float:
    """How long a batch of concurrent subagents takes, given only `cap` slots.

    List scheduling on `cap` identical machines, longest first. Exact for the
    common case (batch <= cap, so the makespan is just the slowest one) and the
    standard approximation beyond it -- and beyond it the harness itself is
    queueing, so an exact optimum would model a scheduler nobody implemented.
    """
    if not durations:
        return 0.0
    if cap <= 0:
        raise ValueError("concurrency cap must be positive")
    slots = [0.0] * cap
    for d in sorted(durations, reverse=True):
        i = min(range(cap), key=lambda j: slots[j])
        slots[i] += d
    return max(slots)


@dataclass
class Trace:
    """Everything one run produced. Stage 6 reads only this."""

    scenario_id: str
    model: str
    condition: str = "agent"  # "agent" | "oracle-plan" | "all-inline" | ...
    repeat: int = 0

    calls: list[ModelCall] = field(default_factory=list)
    spawns: list[SpawnRecord] = field(default_factory=list)
    node_attribution: dict[str, str] = field(default_factory=dict)
    verdicts: dict[str, bool] = field(default_factory=dict)

    write_events: list[WriteEvent] = field(default_factory=list)
    tampered_tests: tuple[str, ...] = ()
    contested_writes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    finished: bool = False
    turns: int = 0
    tool_seconds: float = 0.0
    wall_seconds: float = 0.0
    # Did an independent log of the same traffic agree with these token counts?
    # False means either no proxy was used or the two disagreed -- see the notes.
    proxy_verified: bool = False
    proxy_calls: int = 0
    notes: list[str] = field(default_factory=list)

    # -- shape -------------------------------------------------------------

    @property
    def k(self) -> int:
        """Subagents spawned. The observable the implied-beta estimator inverts."""
        return len(self.spawns)

    @property
    def succeeded(self) -> bool:
        """Every node verified. Regret is only comparable among runs that worked.

        A run with no verdicts has not been graded, and `False` is the right
        answer for it -- an ungraded run must never be scored as a successful
        cheap one.
        """
        return bool(self.verdicts) and all(self.verdicts.values())

    def realized_plan(self) -> tuple[frozenset[str], tuple[frozenset[str], ...]]:
        """(inline, blocks) as the run ACTUALLY happened, from attribution.

        This is what makes plan compliance checkable in 6.1(b): the requested
        plan is compared against this, not against what the model said it would
        do. A node nobody wrote is not silently dropped -- it lands nowhere, and
        `verdicts` will already have failed it.
        """
        inline = {n for n, who in self.node_attribution.items() if who == LEAD}
        blocks: dict[int, set[str]] = {}
        for node, who in self.node_attribution.items():
            if who.startswith("subagent:"):
                blocks.setdefault(int(who.split(":", 1)[1]), set()).add(node)
        return frozenset(inline), tuple(frozenset(b) for _, b in sorted(blocks.items()))

    # -- the two numbers ---------------------------------------------------

    def calls_as_records(self) -> list:
        """The call log in the shape `calibrate` consumes."""
        return [c.as_record() for c in self.calls]

    def dollars(self, price: PriceSheet) -> float:
        """Billed cost of every call, at a dated price sheet.

        The sheet is required rather than defaulted. An undated dollar is not a
        unit, and a run priced under one sheet does not compose with a run
        priced under another.
        """
        from .calibrate import call_dollars

        return sum(call_dollars(c.as_record(), price) for c in self.calls)

    def analytic_minutes(self, timing: TimingModel, cap: int = MAX_CONCURRENCY) -> float:
        """Reconstructed latency: the lead's serial timeline, with each spawn
        batch charged its makespan.

        Not the sum of call durations, and not the wall clock. See the module
        docstring for why each is wrong.

        The lead's calls are walked in order. When a call issued spawns, every
        subagent in that batch starts at the same moment, so the batch costs the
        slowest of them (or the `cap`-machine makespan when the batch is wider
        than the cap) -- and the lead is blocked for exactly that long, because
        a tool result is what unblocks its next turn.
        """
        per_actor: dict[str, list[ModelCall]] = {}
        for c in self.calls:
            per_actor.setdefault(c.actor, []).append(c)

        def actor_minutes(actor: str) -> float:
            return sum(call_minutes(c.as_record(), timing) for c in per_actor.get(actor, ()))

        batches: dict[int, list[float]] = {}
        for s in self.spawns:
            batches.setdefault(s.batch, []).append(actor_minutes(s.actor))

        total = actor_minutes(LEAD)
        for _, durations in sorted(batches.items()):
            total += batch_makespan(durations, cap)
        return total

    def objective(self, price: PriceSheet, timing: TimingModel, beta: float) -> float:
        """cost + beta * latency, in the oracle's units exactly."""
        return self.dollars(price) + beta * self.analytic_minutes(timing)

    # -- persistence -------------------------------------------------------

    def to_json(self) -> str:
        payload = asdict(self)
        payload["spawns"] = [asdict(s) for s in self.spawns]
        payload["write_events"] = [asdict(w) for w in self.write_events]
        payload["calls"] = [asdict(c) for c in self.calls]
        return json.dumps(payload, indent=2, sort_keys=True, default=list)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Trace":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["calls"] = [ModelCall(**{**c, "tools_invoked": tuple(c["tools_invoked"])}) for c in raw["calls"]]
        raw["spawns"] = [SpawnRecord(**{**s, "files": tuple(s["files"])}) for s in raw["spawns"]]
        raw["calls"] = raw["calls"]
        raw["write_events"] = [WriteEvent(**w) for w in raw.get("write_events", [])]
        raw["tampered_tests"] = tuple(raw["tampered_tests"])
        raw["contested_writes"] = {k: tuple(v) for k, v in raw["contested_writes"].items()}
        return cls(**raw)
