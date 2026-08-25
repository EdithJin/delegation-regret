"""Turning real runs into the oracle's constants, reproducibly and in code.

Low-level design: low-level-design.md, "Calibration".

`calibrate.py` holds the arithmetic -- curve building, the timing-model least
squares, the held-out check -- and is testable offline against synthetic logs.
This module is what actually *drives* the runs and extracts every constant from
them, so no number in the cost model is ever typed in by hand.

WHY IT HAS TO BE CODE AND NOT AN AFTERNOON. A constant obtained by reading a
dashboard is unreproducible, undated, and unattributable: six weeks later nobody
can say which run produced it, at which price sheet, or whether the extraction
was done the same way for the other constants. Worse, the extractions are not
independent -- the block curve and the timing model come from the same call log,
and the briefing slope is read from turns the same run produced. Doing them by
hand invites each to be read from a slightly different slice.

So: one command produces one `CalibrationResult`, which carries the derived
constants AND the raw traces they came from AND the diagnostics that say whether
to believe them. It writes all of that to disk. Re-running it on the saved traces
must reproduce the constants exactly, which is what `replay` is for.

THE TWO RUNS, per the design's Tier 1:

  forced-serial  ->  block_dollars_curve, block_minutes_curve, and the timing
                     model. One agent works every node in one context; segment
                     the call log at each node's completion and the cumulative
                     totals ARE the curve.
  forced-fanout  ->  spawn_fixed_dollars, brief_*, absorb_*, explore_*. One
                     subagent per node, so every overhead term appears once per
                     spawn and can be separated.
  throughput     ->  no run of its own. Every call carries a request time and a
                     duration, so how many were in flight at each call's start is
                     arithmetic over the traces already collected -- and the
                     serial and fan-out runs together span concurrency 1 to the
                     cap.
  forced-bundled ->  the briefing SLOPE. Not an extra idea; a necessity. Briefing
                     is affine in block size, and a run whose every spawn covers
                     exactly one node briefs only one block size -- so the slope
                     is a line through a single point. One run with a bundled
                     block gives the regression a second x value.

Both are ordinary `run_plan` calls against a real plan, which matters: the
constants are then measured on the same code path that scoring measures, rather
than on a bespoke harness that might differ.

A REQUIREMENT ON THE RUNS THEMSELVES, learned the hard way. The timing model
`minutes = a + b*input_tokens + output_tokens/throughput` needs output length to
VARY across calls, or throughput is unidentifiable and `fit_timing_model`
correctly refuses to invent it. A calibration scenario whose every turn emits
about the same number of tokens therefore yields no minutes curve at all. Pick a
scenario with a spread of node sizes, and check `diagnostics.timing_model` before
trusting anything on the latency axis. The dollar side is unaffected, so a failed
fit costs half a calibration rather than all of it.

WHAT IT REFUSES TO REPORT. A constant whose extraction did not have enough data
is left as a placeholder rather than being fitted to noise, and the reason is
recorded. `CostModel.calibrate` already refuses to stamp a value without a source
string, so a half-finished calibration reports itself as exactly that.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from generator.oracle import CostModel, Plan, all_inline, is_feasible, max_fanout
from generator.scenario import Scenario
from generator.templates import module_path

from .calibrate import (
    Boundary,
    PriceSheet,
    TimingModel,
    average_curves,
    call_dollars,
    cumulative_points,
    curve_disagreement,
    fit_timing_model,
    holdout_error,
)
from .client import Client
from .runner import Budget, RetryPolicy, run_plan
from .trace import LEAD, Trace

__all__ = [
    "boundaries_from_trace",
    "holdout_check",
    "materiality_floor",
    "block_curves",
    "spawn_overhead",
    "explore_overhead",
    "CalibrationResult",
    "run_calibration",
    "replay",
]


# ------------------------------------------------------- segmenting a run


def boundaries_from_trace(trace: Trace, scenario: Scenario) -> list[Boundary]:
    """When each node finished, and how much work was done by then.

    A node is finished when its module was LAST written -- a later rewrite means
    the earlier one was wrong, and the cumulative spend up to the final write is
    what that node actually cost. `units` is the running total of `size` in
    completion order, because the curve's x-axis is block size, not node index.

    Only the lead's writes count. This is the forced-serial run, so any write by
    a subagent means the plan was not followed and the segmentation would be
    measuring something else.
    """
    last: dict[str, float] = {}
    for event in trace.write_events:
        if event.actor != LEAD:
            continue
        for node_id in scenario.dag.ids:
            if event.path == module_path(node_id):
                last[node_id] = event.t
    ordered = sorted(last.items(), key=lambda kv: kv[1])
    out: list[Boundary] = []
    units = 0
    for node_id, t in ordered:
        units += scenario.dag.by_id[node_id].size
        out.append(Boundary(units=units, t_done=t))
    return out


def _minutes_points(
    trace: Trace, boundaries: list[Boundary], timing: TimingModel
) -> tuple[tuple[int, float], ...]:
    """The same segmentation, in analytic minutes rather than dollars.

    Deliberately not wall clock: the doc is explicit that the minutes curve must
    be summed from the fitted timing model, because raw elapsed time is
    confounded by rate limits and provider load, and the oracle prices plans in
    fitted minutes. Mixing the two would put the two halves of the cost model in
    different time units.
    """
    from .calibrate import call_minutes

    out = []
    for b in boundaries:
        out.append(
            (
                b.units,
                sum(
                    call_minutes(c.as_record(), timing)
                    for c in trace.calls
                    if c.t_request <= b.t_done
                ),
            )
        )
    return tuple(out)


def holdout_check(curve: tuple, price_label: str = "dollars") -> dict:
    """Predict the curve's largest point from the smaller ones, then compare.

    Everything else in this module produces a curve from the runs it was fitted
    on, which is not evidence that it generalises. This is the number that says
    whether the curve means anything: drop the largest measured block, predict it
    by interpolating the rest, and report the gap.

    LEAVE-ONE-OUT COSTS NOTHING. The design describes holding out a block size
    and running it separately, but a serial run's curve is already several nested
    measurements -- the point at 6 units contains the point at 4 -- so the
    largest point can simply be withheld from the fit. No extra spend, and it
    tests the part of the curve that matters most: the top end, where the
    all-inline plan is priced and where extrapolation would otherwise be
    silently doing the work.

    A large gap falsifies the units-not-identity assumption, and the honest
    output is the gap rather than the curve.
    """
    pts = tuple(sorted(curve))
    if len(pts) < 3:
        return {"axis": price_label, "checked": False,
                "why": f"need 3+ measured points to hold one out, have {len(pts)}"}
    units, measured = pts[-1]
    out = holdout_error(pts[:-1], units, measured)
    out["axis"] = price_label
    out["checked"] = True
    return out


def block_curves(
    traces: list[Trace], scenario: Scenario, price: PriceSheet, timing: TimingModel
) -> dict:
    """Both block curves plus the diagnostic that decides whether to trust them.

    Several traces are expected: the design calls for at least three node
    orderings and two repeats, because a curve fitted to one draw reports that
    draw's turn count as if it were a property of the model. `disagreement` is
    the spread across them as a fraction of the mean, and it IS THE TEST of the
    units-not-identity assumption -- if block cost really depends on total units
    and not on which nodes compose the block, the orderings agree.
    """
    dollar_curves, minute_curves = [], []
    for trace in traces:
        boundaries = boundaries_from_trace(trace, scenario)
        if not boundaries:
            continue
        dollar_curves.append(cumulative_points(trace.calls_as_records(), boundaries, price))
        minute_curves.append(_minutes_points(trace, boundaries, timing))
    if not dollar_curves:
        raise ValueError("no trace completed a node; nothing to segment")
    dollars = average_curves(dollar_curves)
    minutes = average_curves(minute_curves)
    return {
        "block_dollars_curve": dollars,
        "block_minutes_curve": minutes,
        "dollar_disagreement": curve_disagreement(dollar_curves),
        "minute_disagreement": curve_disagreement(minute_curves),
        "dollar_holdout": holdout_check(dollars, "dollars"),
        "minute_holdout": holdout_check(minutes, "minutes"),
        "n_curves": len(dollar_curves),
    }


# --------------------------------------------------- the overhead constants


def explore_overhead(trace: Trace, price: PriceSheet, timing: TimingModel) -> dict:
    """Everything the lead spent before it first wrote or delegated.

    Orientation and planning together, charged once. The design notes this is
    the one constant plan selection is insensitive to -- it is identical in every
    plan, so it cancels -- and says not to spend measurement effort here. It is
    extracted anyway because it costs nothing: the boundary is already in the
    trace.
    """
    from .calibrate import call_minutes

    first_action = None
    for call in sorted((c for c in trace.calls if c.actor == LEAD), key=lambda c: c.index):
        if {"write_file", "spawn_subagent"} & set(call.tools_invoked):
            first_action = call.index
            break
    if first_action is None:
        return {"explore_dollars": None, "explore_minutes": None, "why": "lead never acted"}
    before = [c for c in trace.calls if c.actor == LEAD and c.index <= first_action]
    return {
        "explore_dollars": sum(call_dollars(c.as_record(), price) for c in before),
        "explore_minutes": sum(call_minutes(c.as_record(), timing) for c in before),
        "n_calls": len(before),
    }


def spawn_overhead(traces: list[Trace], price: PriceSheet, timing: TimingModel) -> dict:
    """Separate the three delegation charges from a fan-out run.

    `spawn_fixed_dollars` -- a subagent's FIRST call is what it costs before doing
    any work: system prompt, tool definitions, one round trip. Averaged over
    every subagent in every trace.

    `brief_*` -- no inference needed. The instruction text in a `spawn_subagent`
    call IS output tokens the lead emitted, and `SpawnRecord.instruction` stores
    it verbatim. Regressing the lead's output tokens on the nodes briefed gives
    the intercept and the slope, which is the affine form the model wants. Fitted
    together for the reason C.1 gives: they are one line, and fitting them apart
    invites a slope that contradicts its own intercept.

    `absorb_*` -- the lead's input-token count STEPS UP when a result lands.
    Comparing the lead's input on the turn after a batch returns against the turn
    that issued it isolates the step. Divided by the nodes in the batch.
    """
    from .calibrate import call_minutes

    first_calls, brief_rows, absorb_rows = [], [], []
    for trace in traces:
        by_actor: dict[str, list] = {}
        for call in trace.calls:
            by_actor.setdefault(call.actor, []).append(call)
        for actor, calls in by_actor.items():
            if actor == LEAD or not calls:
                continue
            first = min(calls, key=lambda c: c.index)
            first_calls.append(call_dollars(first.as_record(), price))

        lead = sorted((c for c in trace.calls if c.actor == LEAD), key=lambda c: c.index)
        by_batch: dict[int, list] = {}
        for s in trace.spawns:
            by_batch.setdefault(s.batch, []).append(s)

        for call in lead:
            spawned = [n for n in call.tools_invoked if n == "spawn_subagent"]
            if not spawned:
                continue
            # This turn's output tokens paid for `nodes` worth of instruction.
            nodes = sum(1 for _ in spawned)
            brief_rows.append((nodes, call_dollars(call.as_record(), price),
                               call_minutes(call.as_record(), timing)))

        for i, call in enumerate(lead[:-1]):
            if "spawn_subagent" not in call.tools_invoked:
                continue
            step = lead[i + 1].input_tokens - call.input_tokens
            nodes = sum(1 for n in call.tools_invoked if n == "spawn_subagent")
            if step > 0 and nodes:
                absorb_rows.append((nodes, step))

    out: dict = {"n_subagents": len(first_calls), "n_brief_turns": len(brief_rows)}
    out["spawn_fixed_dollars"] = statistics.fmean(first_calls) if first_calls else None

    out.update(_affine("brief", [(n, d) for n, d, _ in brief_rows],
                       [(n, m) for n, _, m in brief_rows]))

    if absorb_rows:
        per_node = statistics.fmean(step / n for n, step in absorb_rows)
        out["absorb_dollars_per_node"] = per_node * price.input_per_mtok / 1_000_000.0
        out["absorb_minutes"] = timing.a_minutes
        out["absorb_minutes_per_node"] = per_node * timing.b_minutes_per_input_token
    else:
        out["absorb_dollars_per_node"] = None
        out["absorb_minutes"] = None
        out["absorb_minutes_per_node"] = None
        out["absorb_why"] = "no lead turn followed a spawn; cannot see the context step"
    return out


def _affine(prefix: str, dollar_rows: list, minute_rows: list) -> dict:
    """Least squares for `y = intercept + slope * nodes`, on both axes.

    Returns `None`s rather than a fit when the rows carry only one distinct node
    count: the slope is then unidentifiable, and reporting a number would be
    fitting a line to a single point. That is the failure mode a fan-out
    calibration run falls into if every spawn covers exactly one node, so the
    calibration plan deliberately includes a bundled block.
    """
    out: dict = {}
    for axis, rows in (("dollars", dollar_rows), ("minutes", minute_rows)):
        xs = [float(n) for n, _ in rows]
        ys = [float(v) for _, v in rows]
        suffix_i = f"{prefix}_{'dollars' if axis == 'dollars' else 'minutes'}"
        suffix_s = f"{prefix}_{'dollars_per_node' if axis == 'dollars' else 'minutes_per_node'}"
        if len(set(xs)) < 2:
            out[suffix_i] = None
            out[suffix_s] = None
            out[f"{prefix}_{axis}_why"] = (
                f"only {len(set(xs))} distinct block size(s) briefed; slope unidentifiable"
            )
            continue
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        denom = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
        if slope < 0:
            # A NEGATIVE SLOPE SAYS BRIEFING MORE NODES COSTS LESS, which is not
            # a thing -- the instruction is strictly longer. Refused for the same
            # reason `fit_timing_model` refuses a negative prefill coefficient:
            # the sign is what the constant MEANS, and a wrong-signed placeholder
            # is worse than an honest one. Here it would make wide fan-out look
            # cheaper the wider it got, which is the direction of the hypothesis
            # under test.
            #
            # It comes from too few distinct block sizes, or from block size
            # being confounded with something else in the run -- both fixed by a
            # calibration plan that brief several sizes with otherwise similar
            # turns, not by taking the absolute value.
            out[suffix_i] = None
            out[suffix_s] = None
            out[f"{prefix}_{axis}_why"] = (
                f"fitted a negative slope ({slope:.3g} per node) across "
                f"{len(set(xs))} block size(s); briefing more nodes cannot cost less, so "
                "block size is confounded in this run"
            )
            continue
        out[suffix_s] = slope
        out[suffix_i] = my - slope * mx
    # The cost side has no per-briefing fixed term of its own -- `spawn_fixed_dollars`
    # already carries it -- so the dollar intercept is folded there rather than
    # double-charged.
    out.pop(f"{prefix}_dollars", None)
    return out


def throughput_by_concurrency(traces: list[Trace], cap: int) -> dict:
    """How much slower a call runs while others are in flight.

    NO EXTRA RUNS NEEDED, which is the point. Every call carries `t_request` and
    `total_s`, so the number in flight at any call's start is a matter of
    arithmetic over intervals -- and the serial and fan-out runs together already
    span concurrency 1 through the cap. A separate concurrency sweep would spend
    money to observe something the existing traces contain.

    The quantity is SECONDS PER OUTPUT TOKEN, normalised to the concurrency-1
    figure. Per-token rather than per-call because a call that generated twice as
    much text is not evidence of contention, and normalised because the absolute
    rate belongs to the timing model -- this curve only carries the degradation.

    THIS IS THE ONE CONSTANT THAT CANNOT COME FROM THE ANALYTIC MODEL. Contention
    does not change token counts, so `analytic_minutes` is blind to it by
    construction. Observed duration is the only witness, which is also why it is
    the noisiest number in the calibration -- report the sample count with it.
    """
    per_level: dict[int, list[float]] = {}
    for trace in traces:
        spans = [
            (c.t_request, c.t_request + c.total_s, c.output_tokens)
            for c in trace.calls
            if c.total_s > 0 and c.output_tokens > 0
        ]
        for start, end, tokens in spans:
            in_flight = sum(1 for s2, e2, _ in spans if s2 <= start < e2)
            level = min(max(in_flight, 1), cap)
            per_level.setdefault(level, []).append((end - start) / tokens)
    if 1 not in per_level:
        return {"throughput_curve": None,
                "why": "no call ran alone; nothing to normalise the degradation against"}
    base = statistics.fmean(per_level[1])
    if base <= 0:
        return {"throughput_curve": None, "why": "concurrency-1 rate is not positive"}
    points = []
    for level in sorted(per_level):
        # Never report a multiplier below 1.0: a call cannot be FASTER for having
        # company. Below-1 draws are noise, and letting them through would make
        # fan-out look like it buys throughput.
        points.append((level, max(1.0, statistics.fmean(per_level[level]) / base)))
    return {
        "throughput_curve": tuple(points),
        "samples_per_level": {k: len(v) for k, v in sorted(per_level.items())},
    }


def materiality_floor(result: "CalibrationResult") -> float | None:
    """The smallest objective difference this calibration can actually resolve.

    `OUTCOME_TOL` decides when two plans "have the same outcome", which gates the
    Tier-A test and the tie-break in `optimal_k_intervals`. Its default is 1e-9 --
    exact float equality, which tests nothing, because these are floats derived
    from measured constants that carry confidence intervals. Its own provenance
    entry says it should come from the materiality floor and that the floor does
    not exist until the constants are measured.

    This is that floor. Two things bound how finely a difference can be resolved:
    the curve's own disagreement across node orderings (the spread in the dollar
    axis), and the timing model's residual (the spread in the minutes axis, in the
    same units the objective adds them). The floor is the larger, because a
    difference smaller than either is inside the noise of the measurement that
    produced it.

    Returns None when the diagnostics needed are absent -- better a missing floor
    than one derived from a calibration that did not measure the spread.
    """
    diag = result.diagnostics or {}
    spread = diag.get("dollar_disagreement") or {}
    curve = result.constants.get("block_dollars_curve")
    dollar_floor = None
    if spread and curve:
        # Relative spread times the largest measured value: the absolute dollar
        # amount the orderings disagree by at the top of the curve, which is
        # where the all-inline plan is priced.
        largest = max(v for _, v in curve)
        dollar_floor = max(spread.values()) * largest
    minute_floor = diag.get("timing_residual_rms_minutes")
    candidates = [f for f in (dollar_floor, minute_floor) if f is not None and f > 0]
    return max(candidates) if candidates else None


# ------------------------------------------------------------- the result


@dataclass
class CalibrationResult:
    """Every constant, every diagnostic, and the runs they came from."""

    source: str  # run id + price sheet date. An undated dollar is not a unit.
    price_sheet: dict = field(default_factory=dict)
    timing_model: dict = field(default_factory=dict)
    constants: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    trace_files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    outcome_tol: float | None = None
    skipped: dict = field(default_factory=dict)  # constant -> why it stayed a placeholder

    def to_cost_model(self, base: CostModel | None = None) -> CostModel:
        """Stamp the measured constants onto a CostModel.

        Only constants that actually came out of the data are stamped. Everything
        else keeps its placeholder status and its recorded reason, so
        `is_calibrated` stays False until nothing is guessing -- which is what
        lets a partially calibrated model report itself as exactly that.
        """
        cm = base or CostModel()
        usable = {k: v for k, v in self.constants.items() if v is not None}
        if not usable:
            return cm
        return cm.calibrate(self.source, **usable)

    def write(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "calibration.json"
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True, default=str),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationResult":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def report(self) -> str:
        lines = [f"calibration source: {self.source}", ""]
        lines += [f"  ! {n}" for n in self.notes]
        if self.notes:
            lines.append("")
        for name in sorted(self.constants):
            value = self.constants[name]
            lines.append(f"  {name:<28} {'--' if value is None else value}")
        if self.outcome_tol is not None:
            lines += [
                "",
                f"  materiality floor: {self.outcome_tol:.3g} -- differences smaller than "
                "this are inside the measurement's own noise. Pass as OUTCOME_TOL.",
            ]
        if self.skipped:
            lines += ["", "left as placeholders:"]
            lines += [f"  {k:<28} {v}" for k, v in sorted(self.skipped.items())]
        holdout = self.diagnostics.get("dollar_holdout") or {}
        if holdout.get("checked"):
            lines += [
                "",
                f"  held-out point at {holdout['units']:.0f} units: "
                f"predicted {holdout['predicted']:.5f}, measured {holdout['measured']:.5f} "
                f"({holdout['rel_error']:.1%} off)",
            ]
            if holdout["rel_error"] > 0.20:
                lines.append(
                    "  ^ the curve does not predict its own top end. Extrapolation is "
                    "doing the work, and the all-inline plan is priced there."
                )
        elif holdout:
            lines += ["", f"  held-out check skipped: {holdout.get('why')}"]
        if self.diagnostics.get("dollar_disagreement"):
            worst = max(self.diagnostics["dollar_disagreement"].values(), default=0.0)
            lines += ["", f"  curve disagreement across orderings: {worst:.1%} worst case"]
            if worst > 0.15:
                lines.append(
                    "  ^ WIDE. Block cost may depend on WHICH nodes are in the block, "
                    "not just how many -- the model's one approximation is in doubt."
                )
        return "\n".join(lines)


# ------------------------------------------------------------- the driver


def run_calibration(
    scenario: Scenario,
    client: Client,
    price: PriceSheet,
    out_dir: str | Path,
    *,
    source: str,
    orderings: int = 3,
    repeats: int = 2,
    max_turns: int = 60,
    budget: Budget | None = None,
    retry: RetryPolicy | None = None,
    require_proxy: bool = True,
) -> CalibrationResult:
    """Execute the calibration runs and extract every constant from them.

    `orderings * repeats` forced-serial runs plus one forced-fanout run. The
    serial repeats are not optional: a single run is one draw from a stochastic
    policy, and `curve_disagreement` across them is the only check on the
    model's one approximation.

    Every trace is written to `out_dir` before anything is fitted, so a failed
    extraction never costs the runs.

    EVERY RUN GOES THROUGH THE LOGGING PROXY, and `require_proxy` defaults to
    True for a reason specific to calibration. A client-side error in parsing the
    provider's usage block would be invisible in a single run, and here it would
    be baked into every constant -- and then into every dollar figure the report
    reports. The proxy parses the same blocks with different code, so a
    disagreement surfaces as a note on the trace. With `require_proxy` the
    disagreement aborts the extraction instead of being averaged into a curve.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workspaces = out_dir / "workspaces"
    workspaces.mkdir(exist_ok=True)

    serial_traces: list[Trace] = []
    files: list[str] = []
    notes: list[str] = []

    def guard(trace: Trace, tag: str) -> None:
        bad = [n for n in trace.notes if n.startswith("PROXY MISMATCH")]
        if bad and require_proxy:
            raise ValueError(
                f"calibration run {tag!r} failed the proxy cross-check: {bad[0]} "
                "Refusing to derive constants from token counts two parsers disagree "
                "about. The traces are on disk; fix the parser and re-run `extract`."
            )
        if bad:
            notes.append(f"{tag}: {bad[0]}")
        elif not trace.proxy_verified:
            notes.append(f"{tag}: no independent proxy log; token counts are unverified")
    for ordering in range(orderings):
        for repeat in range(repeats):
            tag = f"serial-o{ordering}-r{repeat}"
            trace = run_plan(
                scenario,
                all_inline(scenario.dag),
                client,
                workspaces / tag,
                condition="calibrate-serial",
                repeat=repeat,
                max_turns=max_turns,
                budget=budget,
                retry=retry,
                proxy_log=out_dir / f"proxy-{tag}.jsonl",
            )
            files.append(str(trace.write(out_dir / f"trace-{tag}.json")))
            guard(trace, tag)
            serial_traces.append(trace)

    fanout_traces: list[Trace] = []
    bundled = bundled_plan(scenario)
    if bundled is None:
        # Recorded rather than shrugged at: without a second block size the
        # briefing slope is a line through one point, and the caller should know
        # the calibration will come back partial rather than discover it later.
        notes.append(
            f"no legal bundled plan on shape {scenario.dag.shape!r}; the briefing slope "
            "will stay a placeholder. Calibrate on a wide scenario."
        )
    for tag, plan in (("fanout", max_fanout(scenario.dag)), ("bundled", bundled)):
        if plan is None:
            continue
        trace = run_plan(
            scenario,
            plan,
            client,
            workspaces / tag,
            condition=f"calibrate-{tag}",
            max_turns=max_turns,
            budget=budget,
            retry=retry,
            proxy_log=out_dir / f"proxy-{tag}.jsonl",
        )
        files.append(str(trace.write(out_dir / f"trace-{tag}.json")))
        guard(trace, tag)
        fanout_traces.append(trace)

    result = extract(serial_traces, fanout_traces, scenario, price, source=source)
    result.trace_files = files
    result.notes = notes
    result.write(out_dir)
    return result


def bundled_plan(scenario: Scenario) -> "Plan | None":
    """A fan-out with ONE bundled block, so the briefing slope is identifiable.

    Half the nodes go to a single subagent and the rest get one each. That gives
    the briefing regression two distinct block sizes from one run, which is the
    minimum for a slope to mean anything. Returns None when the DAG is too small
    to bundle, in which case the slope stays a placeholder and says so.

    FEASIBILITY IS CHECKED, because a hand-built plan bypasses the enumerator.
    On a chain, bundling non-adjacent nodes into one subagent induces a circular
    wait -- the subagent would have to hand out an intermediate result and wait
    for someone else before continuing, which one spawn cannot do. The enumerator
    rejects such plans; a plan built here would not have been, and `run_plan`
    would have executed an illegal one and produced a baseline the oracle's own
    rules say cannot exist.

    Bundle sizes are tried from largest down, so the first legal one gives the
    widest second x value for the briefing regression. Returns None when no
    bundling is legal -- on a strict chain none is -- in which case the slope
    stays a placeholder and says why.
    """
    ids = list(scenario.dag.ids)
    if len(ids) < 3:
        return None
    for cut in range(len(ids) - 1, 1, -1):
        plan = Plan(
            frozenset(), (frozenset(ids[:cut]),) + tuple(frozenset({v}) for v in ids[cut:])
        )
        if is_feasible(scenario.dag, plan):
            return plan
    return None


def extract(
    serial_traces: list[Trace],
    fanout_traces: "list[Trace] | Trace",
    scenario: Scenario,
    price: PriceSheet,
    *,
    source: str,
) -> CalibrationResult:
    """Derive every constant from traces already in hand.

    Separated from `run_calibration` so the extraction is testable offline and so
    `replay` can reproduce a published calibration from its saved traces without
    spending anything. Same code path either way -- an extraction that only ran
    once, inside the expensive command, is an extraction nobody checked.
    """
    fanouts = [fanout_traces] if isinstance(fanout_traces, Trace) else list(fanout_traces)
    usable = [t for t in serial_traces if t.calls]
    records = [c.as_record() for t in usable + fanouts for c in t.calls]

    # THE TIMING MODEL CAN LEGITIMATELY FAIL TO FIT, and losing the whole
    # calibration when it does would be wrong. It needs output length to VARY
    # across calls to identify throughput -- a run whose every turn emitted about
    # the same number of tokens leaves the slope unidentifiable, and `calibrate`
    # correctly refuses to invent one. But the DOLLAR curve and every overhead
    # constant on the cost axis need no timing model at all. So a failed fit
    # costs the minutes half and nothing else, and the reason is recorded rather
    # than raised over the top of runs that already happened.
    timing: TimingModel | None
    timing_why = ""
    try:
        timing = fit_timing_model(records)
    except ValueError as exc:
        timing, timing_why = None, str(exc)

    fit = timing or TimingModel(a_minutes=0.0, b_minutes_per_input_token=0.0,
                                output_tokens_per_minute=1.0)
    curves = block_curves(usable, scenario, price, fit)
    overhead = spawn_overhead(fanouts, price, fit)
    explore = explore_overhead(fanouts[0], price, fit) if fanouts else {"why": "no fanout run"}
    from .tools import MAX_CONCURRENCY

    throughput = throughput_by_concurrency(usable + fanouts, MAX_CONCURRENCY)

    constants, skipped = {}, {}
    minute_constants = {
        "block_minutes_curve",
        "brief_minutes",
        "brief_minutes_per_node",
        "absorb_minutes",
        "absorb_minutes_per_node",
        "explore_minutes",
    }
    candidates = {
        "block_dollars_curve": curves["block_dollars_curve"],
        "block_minutes_curve": curves["block_minutes_curve"],
        "spawn_fixed_dollars": overhead.get("spawn_fixed_dollars"),
        "brief_dollars_per_node": overhead.get("brief_dollars_per_node"),
        "brief_minutes": overhead.get("brief_minutes"),
        "brief_minutes_per_node": overhead.get("brief_minutes_per_node"),
        "absorb_dollars_per_node": overhead.get("absorb_dollars_per_node"),
        "absorb_minutes": overhead.get("absorb_minutes"),
        "absorb_minutes_per_node": overhead.get("absorb_minutes_per_node"),
        "explore_dollars": explore.get("explore_dollars"),
        "explore_minutes": explore.get("explore_minutes"),
        "throughput_curve": throughput.get("throughput_curve"),
    }
    for name, value in candidates.items():
        if timing is None and name in minute_constants:
            skipped[name] = f"no timing model: {timing_why}"
            continue
        if value is None:
            axis = "minutes" if "minutes" in name else "dollars"
            skipped[name] = (
                throughput.get("why") if name == "throughput_curve" else None
            ) or (
                overhead.get(f"{name.split('_')[0]}_{axis}_why")
                or overhead.get(f"{name.split('_')[0]}_why")
                or overhead.get("absorb_why")
                or explore.get("why")
                or "extraction produced no value"
            )
        else:
            constants[name] = value

    result = CalibrationResult(
        source=source,
        price_sheet=asdict(price),
        timing_model=asdict(timing) if timing else {"unfitted": timing_why},
        constants=constants,
        diagnostics={
            "dollar_disagreement": curves["dollar_disagreement"],
            "minute_disagreement": curves["minute_disagreement"],
            "dollar_holdout": curves["dollar_holdout"],
            "minute_holdout": curves["minute_holdout"],
            "n_serial_runs": curves["n_curves"],
            "n_subagents": overhead.get("n_subagents"),
            "n_brief_turns": overhead.get("n_brief_turns"),
            "explore_calls": explore.get("n_calls"),
            "throughput_samples": throughput.get("samples_per_level"),
            "timing_residual_rms_minutes": timing.residual_rms_minutes if timing else None,
        },
        skipped=skipped,
    )
    # Derived last, because it reads the diagnostics the rest of the extraction
    # produced. Reported on the result rather than mutated into the module-level
    # constant: a calibration should not reach in and change the meaning of
    # "equal" for every other run in the process.
    result.outcome_tol = materiality_floor(result)
    return result


def replay(out_dir: str | Path, scenario: Scenario, price: PriceSheet, *, source: str) -> CalibrationResult:
    """Re-derive a calibration from its saved traces. Zero spend.

    This is what makes the calibration reproducible rather than merely recorded.
    If `replay` disagrees with the published `calibration.json`, the extraction
    changed and every number downstream of it is suspect.
    """
    out_dir = Path(out_dir)
    serial = [Trace.load(p) for p in sorted(out_dir.glob("trace-serial-*.json"))]
    fanouts = [
        Trace.load(p)
        for name in ("trace-fanout.json", "trace-bundled.json")
        for p in [out_dir / name]
        if p.exists()
    ]
    return extract(serial, fanouts, scenario, price, source=source)
