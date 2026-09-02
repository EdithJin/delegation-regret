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
                     totals -- NET OF THE ORIENTATION PREFIX -- are the curve.
                     Corrected Aug 25: the prefix (everything before the lead's
                     first write) used to stay inside the curve, which put a
                     $0.05-0.27 startup hump in the first point, double-charged
                     exploration in the all-inline plan (`evaluate` adds
                     `explore_dollars` on top), and re-charged the lead's
                     startup once per subagent block. Every curve point now has
                     its own run's prefix subtracted, and the prefix itself is
                     what `explore_*` measures. Each ordering is FORCED via the
                     plan directive -- also corrected Aug 25: the driver used to
                     pass the same unordered plan for every "ordering", so all
                     six runs of the first real calibration completed n0..n5
                     identically and the ordering-disagreement diagnostic was
                     measuring repeat variance while claiming to test
                     composition-dependence.
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
                     block gives the regression a second x value. Corrected
                     Aug 25: the regression rows are PER SPAWN now, with x the
                     nodes that spawn actually covered (from attribution) and y
                     the spawn's instruction-length share of the turn's billed
                     output tokens. The old rows regressed whole-turn dollars on
                     the COUNT of spawn invocations in the turn -- so a bundled
                     spawn registered as x=1 no matter how many nodes it
                     covered, defeating this run's entire purpose, and the y was
                     dominated by context prefill that grows turn over turn,
                     which is what produced the negative slope the first real
                     calibration recorded as "confounded".

Both are ordinary `run_plan` calls against a real plan, which matters: the
constants are then measured on the same code path that scoring measures, rather
than on a bespoke harness that might differ.

A REQUIREMENT ON THE RUNS THEMSELVES, learned the hard way twice. The timing
model `minutes = a + b_fresh*fresh + b_cached*cache_reads + output/throughput`
needs its regressors to VARY across calls, or the fit is unidentifiable and
`fit_timing_model` correctly refuses to invent one. Two ways in: a scenario
whose every turn emits about the same number of tokens (pick a spread of node
sizes), and -- corrected Aug 25 -- a regressor that is secretly constant, which
is what regressing on billed `input_tokens` did under the pinned cache
breakpoints: the whole context billed as cache reads and writes, and billed
input was 2 tokens on every call of the first real calibration. The regressors
are context columns now, and `_assert_timing_regressors_vary` aborts a doomed
run on its first trace rather than after the full spend. A failed fit still
costs half a calibration rather than all of it -- the dollar side needs no
timing model. Check `diagnostics.timing_model` before trusting anything on the
latency axis.

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
    context_tokens,
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
    "completion_order",
    "orientation_prefix",
    "holdout_check",
    "holdout_check_runs",
    "materiality_floor",
    "dollar_floor",
    "minute_floor",
    "composition_check",
    "block_curves",
    "sub_block_curves",
    "spawn_overhead",
    "explore_overhead",
    "serial_orderings",
    "CalibrationResult",
    "run_calibration",
    "replay",
]


# ------------------------------------------------------- segmenting a run


def completion_order(trace: Trace, scenario: Scenario) -> list[tuple[str, float]]:
    """(node, t) in the order the lead actually finished them.

    A node is finished when its module was LAST written -- a later rewrite means
    the earlier one was wrong. Only the lead's writes count: this serves the
    forced-serial segmentation, and a subagent write means the plan was not
    followed. Also what `run_calibration` compares against the ordering it
    REQUESTED, so an ignored directive becomes a note instead of a silently
    mislabeled curve.
    """
    last: dict[str, float] = {}
    for event in trace.write_events:
        if event.actor != LEAD:
            continue
        for node_id in scenario.dag.ids:
            if event.path == module_path(node_id):
                last[node_id] = event.t
    return sorted(last.items(), key=lambda kv: kv[1])


def boundaries_from_trace(trace: Trace, scenario: Scenario) -> list[Boundary]:
    """When each node finished, and how much work was done by then.

    `units` is the running total of `size` in completion order, because the
    curve's x-axis is block size, not node index.
    """
    out: list[Boundary] = []
    units = 0
    for node_id, t in completion_order(trace, scenario):
        units += scenario.dag.by_id[node_id].size
        out.append(Boundary(units=units, t_done=t))
    return out


def orientation_prefix(trace: Trace) -> tuple[list, bool]:
    """(the lead's calls BEFORE its first action, whether it ever acted).

    The first action is the first turn that writes a file or spawns a subagent;
    everything before it is orientation -- reading the tasks, exploring the
    tree -- which every plan pays identically. That prefix must NOT stay inside
    the block curves: the model charges it once as `explore_*`, and a curve that
    keeps it charges it again for the all-inline plan and once more per
    subagent block (the curve's small-block end is read for every spawn).

    Synthetic logs may lack `tools_invoked`; the fallback anchors on the first
    lead write event and treats the call in flight at that moment as the acting
    turn, which coincides with the tool-marker definition on real traces.
    """
    lead = sorted((c for c in trace.calls if c.actor == LEAD), key=lambda c: c.index)
    for i, call in enumerate(lead):
        if {"write_file", "spawn_subagent"} & set(call.tools_invoked):
            return lead[:i], True
    writes = [w.t for w in trace.write_events if w.actor == LEAD]
    if writes:
        t_first = min(writes)
        started = [i for i, c in enumerate(lead) if c.t_request <= t_first]
        if started:
            return lead[: max(started)], True
    return lead, False


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

    The SINGLE-CURVE fallback. A serial run's curve is nested measurements --
    the point at 6 units contains the point at 4 -- so the largest point can be
    withheld from the fit at no extra spend. But on an averaged near-linear
    curve this check is close to vacuous: the first real calibration's version
    of it reported 0.13% while the runs disagreed by 30% at the same point,
    because holding one point out of an average only tests the average's
    smoothness. Whenever there is more than one run, `holdout_check_runs`
    replaces it.
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


def holdout_check_runs(curves: list[tuple], price_label: str = "dollars") -> dict:
    """Leave one RUN out: predict each run's top point from the other runs.

    This is the number that says whether the curve generalises, because it
    tests the thing the oracle actually does: price a run it never saw off the
    curve the other runs produced. A large gap falsifies the units-not-identity
    assumption, and the honest output is the gap rather than the curve.
    """
    from generator.oracle import _interpolate

    usable = [tuple(sorted(c)) for c in curves if c]
    if len(usable) < 2:
        return (
            holdout_check(average_curves(usable), price_label)
            if usable
            else {"axis": price_label, "checked": False, "why": "no curves"}
        )
    per_run = []
    for i, held in enumerate(usable):
        others = average_curves([c for j, c in enumerate(usable) if j != i])
        units, measured = held[-1]
        predicted = _interpolate(others, units)
        per_run.append(abs(measured - predicted) / measured if measured else float("inf"))
    return {
        "axis": price_label,
        "checked": True,
        "kind": "leave-one-run-out",
        "units": float(max(c[-1][0] for c in usable)),
        "mean_rel_error": statistics.fmean(per_run),
        "max_rel_error": max(per_run),
        "n_runs": len(usable),
    }


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

    Each run's ORIENTATION PREFIX is subtracted from its cumulative points
    before anything is averaged (see `orientation_prefix`), so the curve is
    block work and nothing else. The prefix varied five-fold across the first
    real calibration's runs, so leaving it in both mis-shaped the curve's first
    point and flooded this diagnostic with startup variance that has nothing to
    do with composition.
    """
    from .calibrate import call_minutes

    dollar_curves, minute_curves = [], []
    for trace in traces:
        boundaries = boundaries_from_trace(trace, scenario)
        if not boundaries:
            continue
        prefix, _ = orientation_prefix(trace)
        prefix_dollars = sum(call_dollars(c.as_record(), price) for c in prefix)
        prefix_minutes = sum(call_minutes(c.as_record(), timing) for c in prefix)
        raw_d = cumulative_points(trace.calls_as_records(), boundaries, price)
        raw_m = _minutes_points(trace, boundaries, timing)
        dollar_curves.append(tuple((u, v - prefix_dollars) for u, v in raw_d))
        minute_curves.append(tuple((u, v - prefix_minutes) for u, v in raw_m))
    if not dollar_curves:
        raise ValueError("no trace completed a node; nothing to segment")
    dollars = average_curves(dollar_curves)
    minutes = average_curves(minute_curves)
    return {
        "block_dollars_curve": dollars,
        "block_minutes_curve": minutes,
        "dollar_disagreement": curve_disagreement(dollar_curves),
        "minute_disagreement": curve_disagreement(minute_curves),
        "dollar_holdout": holdout_check_runs(dollar_curves, "dollars"),
        "minute_holdout": holdout_check_runs(minute_curves, "minutes"),
        "n_curves": len(dollar_curves),
    }


# --------------------------------------------------- the overhead constants


def explore_overhead(trace: Trace, price: PriceSheet, timing: TimingModel) -> dict:
    """Everything the lead spent STRICTLY BEFORE its first write or spawn.

    Orientation and planning together, charged once. The acting turn itself is
    excluded (corrected Aug 25): its cost belongs to the work it starts -- the
    block curve's first segment, or the briefing -- and including it here
    charged the same turn twice. This boundary is the same one the block curves
    subtract, so the two extractions partition the run instead of overlapping.

    Measured on EVERY run and averaged by the caller, because the prefix varied
    five-fold across the first real calibration's runs and a constant read from
    one draw of that distribution is mostly noise.
    """
    from .calibrate import call_minutes

    before, acted = orientation_prefix(trace)
    if not acted:
        return {"explore_dollars": None, "explore_minutes": None, "why": "lead never acted"}
    return {
        "explore_dollars": sum(call_dollars(c.as_record(), price) for c in before),
        "explore_minutes": sum(call_minutes(c.as_record(), timing) for c in before),
        "n_calls": len(before),
    }


def spawn_overhead(
    traces: list[Trace], price: PriceSheet, timing: TimingModel, scenario: Scenario | None = None
) -> dict:
    """Separate the three delegation charges from a fan-out run.

    `spawn_fixed_dollars` -- a subagent's FIRST call is what it costs before doing
    any work: system prompt, tool definitions, one round trip. Averaged over
    every subagent in every trace.

    `brief_*` -- one regression row PER SPAWN (corrected Aug 25; the old rows
    were per turn, regressed whole-turn dollars on the COUNT of spawn
    invocations, and were dominated by context prefill that grows turn over
    turn -- which is what produced the first real calibration's negative slope,
    and which made a bundled spawn register as x=1 no matter how many nodes it
    covered). Now:

      x = nodes the spawn actually covered, read from `node_attribution` (who
          wrote which module -- harness ground truth), falling back to the
          module files handed over, then to 1.
      y = the spawn's share of the turn's BILLED output tokens, apportioned by
          instruction length. The instruction IS lead output; apportioning
          preserves the provider's total for the turn, and any prose in the
          turn smears into the rows proportionally -- a small upward bias on
          the intercept, stated here rather than hidden.

    The dollar intercept is folded into `spawn_fixed_dollars` (the instruction
    boilerplate is also input to the subagent's first call) and the minutes
    intercept rides on top of the timing model's per-call floor, so
    `brief_minutes = a_minutes + intercept-of-generation-minutes`.

    `absorb_*` -- the lead's CONTEXT steps up when a result lands. Comparing the
    lead's context size on the turn after a batch returns against the turn that
    issued it isolates the step, divided by the nodes in the batch. Context, not
    billed `input_tokens` (corrected Aug 25): under the cache breakpoints the
    step lands almost entirely in the cache columns, and reading billed input
    made it invisibly zero on every real trace while this extractor blamed the
    trace shape.
    """
    modpaths = {module_path(n) for n in scenario.dag.ids} if scenario else None

    first_calls, brief_rows_d, brief_rows_m, absorb_rows = [], [], [], []
    n_brief_turns = 0
    followed = 0  # lead turns that came after a spawn -- for an honest absorb_why
    for trace in traces:
        by_actor: dict[str, list] = {}
        for call in trace.calls:
            by_actor.setdefault(call.actor, []).append(call)
        for actor, calls in by_actor.items():
            if actor == LEAD or not calls:
                continue
            first = min(calls, key=lambda c: c.index)
            first_calls.append(call_dollars(first.as_record(), price))

        nodes_by_actor: dict[str, int] = {}
        for _node, who in trace.node_attribution.items():
            nodes_by_actor[who] = nodes_by_actor.get(who, 0) + 1

        def spawn_nodes(s) -> int:
            n = nodes_by_actor.get(s.actor, 0)
            if not n and modpaths is not None:
                n = len(modpaths & set(s.files))
            return n or 1

        lead = sorted((c for c in trace.calls if c.actor == LEAD), key=lambda c: c.index)
        spawn_turns = [c for c in lead if "spawn_subagent" in c.tools_invoked]
        by_batch: dict[int, list] = {}
        for s in trace.spawns:
            by_batch.setdefault(s.batch, []).append(s)

        # The k-th spawn turn issued the k-th batch: batches are numbered in
        # issue order by the runner, and a turn's spawns share one batch id.
        for call, batch_id in zip(spawn_turns, sorted(by_batch)):
            batch = by_batch[batch_id]
            n_brief_turns += 1
            chars = [max(len(s.instruction), 1) for s in batch]
            total_chars = sum(chars)
            out_tokens = float(call.output_tokens or 0)
            for s, ch in zip(batch, chars):
                share = out_tokens * ch / total_chars
                x = spawn_nodes(s)
                brief_rows_d.append((x, share * price.output_per_mtok / 1_000_000.0))
                brief_rows_m.append((x, share / timing.output_tokens_per_minute))

        for i, call in enumerate(lead[:-1]):
            if "spawn_subagent" not in call.tools_invoked:
                continue
            followed += 1
            step = context_tokens(lead[i + 1].as_record()) - context_tokens(call.as_record())
            turn_pos = spawn_turns.index(call)
            batch_ids = sorted(by_batch)
            nodes = (
                sum(spawn_nodes(s) for s in by_batch[batch_ids[turn_pos]])
                if turn_pos < len(batch_ids)
                else sum(1 for n in call.tools_invoked if n == "spawn_subagent")
            )
            if step > 0 and nodes:
                absorb_rows.append((nodes, step))

    out: dict = {
        "n_subagents": len(first_calls),
        "n_brief_turns": n_brief_turns,
        "n_brief_rows": len(brief_rows_d),
    }
    out["spawn_fixed_dollars"] = statistics.fmean(first_calls) if first_calls else None

    out.update(_affine("brief", brief_rows_d, brief_rows_m))
    if out.get("brief_minutes") is not None:
        # The regression intercept is generation time every instruction carries
        # regardless of node count; the per-call latency floor sits underneath.
        fixed = timing.a_minutes + out["brief_minutes"]
        if fixed >= 0:
            out["brief_minutes"] = fixed
        else:
            out["brief_minutes"] = None
            out["brief_minutes_per_node"] = None
            out["brief_minutes_why"] = (
                f"fitted a negative fixed briefing time ({fixed:.3g} min); "
                "the rows cannot support the affine form"
            )

    if absorb_rows:
        per_node = statistics.fmean(step / n for n, step in absorb_rows)
        out["absorb_dollars_per_node"] = per_node * price.input_per_mtok / 1_000_000.0
        out["absorb_minutes"] = timing.a_minutes
        # Absorbed tokens sit in the lead's context from then on, and on every
        # subsequent turn they are cache READS -- the one-time write happened on
        # the turn that received the result. The cached coefficient prices them.
        out["absorb_minutes_per_node"] = per_node * timing.cached_minutes_per_token
    else:
        out["absorb_dollars_per_node"] = None
        out["absorb_minutes"] = None
        out["absorb_minutes_per_node"] = None
        # Two different failures, previously conflated -- the second one spent a
        # night being misdiagnosed as the first while the real cause was billed
        # input masquerading as context size.
        out["absorb_why"] = (
            "no lead turn followed a spawn; cannot see the context step"
            if followed == 0
            else f"lead context never grew across {followed} post-spawn turn(s); "
            "absorption is invisible in these traces"
        )
    return out


def sub_block_curves(
    fanout_traces: list[Trace], scenario: Scenario, price: PriceSheet, timing: TimingModel
) -> dict:
    """What a subagent costs to work a block, measured per subagent. Zero spend.

    The estimation gate's finding, turned into a measurement: spawned blocks
    were priced off the LEAD's serial curve, and that composition under-priced
    real delegation runs by ~2x, because a subagent pays fresh-context costs a
    warm serial lead never sees. The fan-out traces already contain every
    subagent's own calls, so the honest curve needs no new runs: each subagent
    is one direct (units worked, total billed) measurement -- no regression,
    no decomposition -- minus its FIRST call, which `spawn_fixed_dollars`
    already carries.

    Units come from attribution (who wrote which module), the same ground
    truth the briefing rows use. A subagent that wrote nothing attributable is
    not a measurement of a block and is skipped. Points from different
    subagents are merged by `average_curves`; its monotone clamp asserts here
    that a bigger block never bills less, which is the same physical claim the
    lead curve makes.
    """
    from .calibrate import call_minutes

    d_curves, m_curves = [], []
    for trace in fanout_traces:
        units_of: dict[str, int] = {}
        for node, who in trace.node_attribution.items():
            if who.startswith("subagent:"):
                units_of[who] = units_of.get(who, 0) + scenario.dag.by_id[node].size
        by_actor: dict[str, list] = {}
        for c in trace.calls:
            if c.actor.startswith("subagent:"):
                by_actor.setdefault(c.actor, []).append(c)
        for actor, calls in by_actor.items():
            units = units_of.get(actor)
            if not units:
                continue
            rest = sorted(calls, key=lambda c: c.index)[1:]
            d_curves.append(((units, sum(call_dollars(c.as_record(), price) for c in rest)),))
            m_curves.append(((units, sum(call_minutes(c.as_record(), timing) for c in rest)),))
    if not d_curves:
        return {
            "sub_block_dollars_curve": None,
            "sub_block_minutes_curve": None,
            "why": "no subagent with attributed nodes in the fan-out traces",
        }
    return {
        "sub_block_dollars_curve": average_curves(d_curves),
        "sub_block_minutes_curve": average_curves(m_curves),
        "n_subagent_blocks": len(d_curves),
        "sub_units_covered": sorted({u for c in d_curves for u, _ in c}),
    }


def _affine(prefix: str, dollar_rows: list, minute_rows: list) -> dict:
    """Least squares for `y = intercept + slope * nodes`, on both axes.

    Rows are one per spawn (see `spawn_overhead`). Returns `None`s rather than
    a fit when the rows carry only one distinct node count: the slope is then
    unidentifiable, and reporting a number would be fitting a line to a single
    point. That is the failure mode a fan-out calibration run falls into when
    every spawn covers exactly one node, so the calibration plan deliberately
    includes a bundled block.
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


def _axis_floor(spread: dict, curve) -> float | None:
    """The largest ABSOLUTE disagreement across the curve's buckets.

    Per bucket, not worst-relative-times-largest-value (corrected Aug 25): the
    old form multiplied the 140% relative spread at 3 units -- a percentage of
    a small mean -- by the value at 24 units, producing a floor larger than the
    entire all-inline objective. Relative spread only means something against
    the mean it was computed from.
    """
    if not (spread and curve):
        return None
    mean = {int(u): v for u, v in curve}
    vals = [s * mean[int(u)] for u, s in spread.items() if int(u) in mean]
    positive = [v for v in vals if v > 0]
    return max(positive) if positive else (0.0 if vals else None)


def dollar_floor(result: "CalibrationResult") -> float | None:
    """The smallest COST difference this calibration can resolve, in dollars."""
    diag = result.diagnostics or {}
    return _axis_floor(diag.get("dollar_disagreement") or {},
                       result.constants.get("block_dollars_curve"))


def minute_floor(result: "CalibrationResult") -> float | None:
    """The smallest LATENCY difference this calibration can resolve, in minutes.

    The larger of the minutes curve's cross-run disagreement and the timing
    model's residual RMS -- both bound how finely the minutes axis is measured.
    """
    diag = result.diagnostics or {}
    curve_part = _axis_floor(diag.get("minute_disagreement") or {},
                             result.constants.get("block_minutes_curve"))
    resid = diag.get("timing_residual_rms_minutes")
    candidates = [f for f in (curve_part, resid) if f is not None and f > 0]
    return max(candidates) if candidates else None


def materiality_floor(result: "CalibrationResult", beta: float = 0.0) -> float | None:
    """The smallest objective difference this calibration can resolve, AT A BETA.

    `OUTCOME_TOL` decides when two plans "have the same outcome", which gates the
    Tier-A test and the tie-break in `optimal_k_intervals`. Its default is 1e-9 --
    exact float equality, which tests nothing, because these are floats derived
    from measured constants that carry confidence intervals.

    The objective is `dollars + beta * minutes`, so its noise floor is
    `dollar_floor + beta * minute_floor` -- one number per beta, never a max
    over unlike units (corrected Aug 25: the old form took
    max(dollars, minutes) as if a dollar and a minute were comparable, which
    made the floor depend on which unit happened to be numerically larger).

    Returns None when the needed piece is unmeasured -- better a missing floor
    than one derived from a calibration that did not measure the spread. The
    per-axis floors are published on `CalibrationResult.floors` so scoring can
    gate each axis separately (`is_tier_a` compares cost and latency on their
    own tolerances).
    """
    d = dollar_floor(result)
    if beta == 0.0:
        return d
    m = minute_floor(result)
    if d is None or m is None:
        return None
    return d + beta * m


# --------------------------------------------------- the estimation gate


_COST_CRITICAL = (
    "block_dollars_curve",
    "sub_block_dollars_curve",
    "spawn_fixed_dollars",
    "brief_dollars_per_node",
    "absorb_dollars_per_node",
    "explore_dollars",
)


def composition_check(
    serial_traces: list[Trace],
    fanout_traces: list[Trace],
    scenario: Scenario,
    price: PriceSheet,
    timing: "TimingModel | None",
    cm: CostModel,
) -> dict:
    """Predict each executed calibration plan from the extracted constants and
    compare against what that run measurably cost. THE ESTIMATION GATE.

    Every other diagnostic tests an ingredient -- the curve interpolates, the
    slope has the right sign, the timing fit converges. None of them test the
    DISH: the oracle composing those constants into a plan price. The
    calibration already executed three different plans for real (all-inline,
    max-fanout, bundled), so the composed prediction can be checked against a
    measured total at zero extra spend -- and the first time this ran (Haiku
    preflight, Aug 25) it caught the composition under-pricing both delegation
    plans by 38-53% while the serial arm scattered evenly: the serial-measured
    block curve does not carry a subagent's fresh-context costs, and the miss
    is one-sided in the pro-spawn direction.

    Two verdicts matter more than the raw gaps:

      cost_ranking_preserved  -- does the oracle ORDER the executed plans the
                                 way their measured costs order? Ranking is the
                                 oracle's actual job; a uniform mispricing can
                                 still rank correctly, an inverted ranking
                                 means plan selection is answering from the
                                 model's error, not the world.
      rel gaps, SIGNED        -- (predicted - measured) / measured, so a
                                 negative gap on a delegation plan reads
                                 directly as "the oracle flatters spawning".

    The cost side runs whenever the cost-critical constants are measured; the
    minutes side additionally needs the timing model. Incomplete runs are
    excluded from the measured mean when any complete run exists -- a run that
    died mid-plan did not execute the plan being priced -- and flagged when
    nothing else is available.

    IN-SAMPLE CAVEAT, stated rather than hidden: once `sub_block_curves` is
    extracted from these same fan-out traces, the fan-out arm's prediction
    partly contains its own measurement. The serial arm stays out-of-sample
    for the subagent curve, the bundled-vs-fanout comparison still crosses
    arms, and the check's real force is on a FRESH calibration -- which is
    exactly when it runs.
    """
    from generator.oracle import all_inline, evaluate, max_fanout, unmeasured

    missing = unmeasured(cm)
    missing_cost = [m for m in missing if m in _COST_CRITICAL]
    if missing_cost:
        return {
            "checked": False,
            "why": "cost-critical constant(s) still placeholder: "
            + ", ".join(missing_cost),
        }
    minutes_ok = timing is not None and not any("minutes" in m for m in missing)

    dag = scenario.dag
    complete = [t for t in serial_traces if t.finished and t.succeeded and t.calls]
    serial_pool = complete or [t for t in serial_traces if t.calls]
    arms: list[tuple[str, object, list[Trace], bool]] = []
    if serial_pool:
        arms.append(("all-inline", all_inline(dag), serial_pool, not complete))
    for trace in fanout_traces:
        if trace.condition == "calibrate-fanout":
            arms.append(("max-fanout", max_fanout(dag), [trace],
                         not (trace.finished and trace.succeeded)))
        elif trace.condition == "calibrate-bundled":
            plan = bundled_plan(scenario)
            if plan is not None:
                arms.append(("bundled", plan, [trace],
                             not (trace.finished and trace.succeeded)))
    if len(arms) < 2:
        return {"checked": False,
                "why": f"only {len(arms)} executed plan(s) to compare; ranking needs two"}

    rows = []
    for name, plan, traces, incomplete in arms:
        predicted = evaluate(dag, plan, cm)
        measured_d = statistics.fmean(t.dollars(price) for t in traces)
        row = {
            "plan": name,
            "n_runs": len(traces),
            "all_runs_incomplete": incomplete,
            "predicted_dollars": predicted.cost,
            "measured_dollars": measured_d,
            "rel_gap_dollars": (predicted.cost - measured_d) / measured_d,
        }
        if minutes_ok:
            measured_m = statistics.fmean(t.analytic_minutes(timing) for t in traces)
            row["predicted_minutes"] = predicted.latency
            row["measured_minutes"] = measured_m
            row["rel_gap_minutes"] = (
                (predicted.latency - measured_m) / measured_m if measured_m else None
            )
        rows.append(row)

    def ranking(key_p, key_m):
        pred = [r["plan"] for r in sorted(rows, key=lambda r: r[key_p])]
        meas = [r["plan"] for r in sorted(rows, key=lambda r: r[key_m])]
        return pred == meas

    return {
        "checked": True,
        "arms": rows,
        "cost_ranking_preserved": ranking("predicted_dollars", "measured_dollars"),
        "latency_ranking_preserved": (
            ranking("predicted_minutes", "measured_minutes") if minutes_ok else None
        ),
    }


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
    # Per-axis noise floors: {"dollars": ..., "minutes": ...}. The objective
    # floor at any beta is dollars + beta * minutes; scoring passes the axes
    # separately (see `materiality_floor`). `outcome_tol` is the beta=0 floor,
    # kept for older files.
    floors: dict = field(default_factory=dict)
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
        d = (self.floors or {}).get("dollars")
        m = (self.floors or {}).get("minutes")
        if d is not None or m is not None:
            lines += [
                "",
                "  materiality floor: "
                + (f"${d:.3g}" if d is not None else "? dollars")
                + " + beta * "
                + (f"{m:.3g} min" if m is not None else "? min")
                + " -- differences smaller than this are inside the measurement's "
                "own noise. Pass the axes to is_tier_a / optimal_k_intervals as "
                "tolerances instead of OUTCOME_TOL's exact-equality default.",
            ]
        elif self.outcome_tol is not None:  # older files carry only the scalar
            lines += [
                "",
                f"  materiality floor: {self.outcome_tol:.3g} -- differences smaller than "
                "this are inside the measurement's own noise. Pass as OUTCOME_TOL.",
            ]
        if self.skipped:
            lines += ["", "left as placeholders:"]
            lines += [f"  {k:<28} {v}" for k, v in sorted(self.skipped.items())]
        holdout = self.diagnostics.get("dollar_holdout") or {}
        if holdout.get("kind") == "leave-one-run-out":
            lines += [
                "",
                f"  leave-one-run-out at {holdout['units']:.0f} units: each run's top "
                f"point predicted from the other {holdout['n_runs'] - 1} run(s) -- "
                f"mean {holdout['mean_rel_error']:.1%} off, worst "
                f"{holdout['max_rel_error']:.1%}",
            ]
            if holdout["max_rel_error"] > 0.20:
                lines.append(
                    "  ^ the curve does not predict a run it has not seen. The oracle "
                    "prices unseen runs off this curve, so treat its verdicts as noise-"
                    "bounded by this gap."
                )
        elif holdout.get("checked"):
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
        comp = self.diagnostics.get("composition") or {}
        if comp.get("checked"):
            lines += ["", "  composed estimate vs executed plans (the estimation gate):"]
            for a in comp["arms"]:
                line = (
                    f"    {a['plan']:<11} ${a['predicted_dollars']:.4f} predicted vs "
                    f"${a['measured_dollars']:.4f} measured ({a['rel_gap_dollars']:+.0%}"
                )
                if a.get("rel_gap_minutes") is not None:
                    line += f" $, {a['rel_gap_minutes']:+.0%} min"
                line += f"; {a['n_runs']} run(s)"
                if a.get("all_runs_incomplete"):
                    line += ", ALL INCOMPLETE"
                lines.append(line + ")")
            if comp.get("cost_ranking_preserved") is False:
                lines.append(
                    "  ^ COST RANKING INVERTED: the oracle orders these plans differently "
                    "than their measured costs do. Plan selection is answering from the "
                    "model's error, not the world -- do not trust oracle_k or implied "
                    "beta from this calibration."
                )
            else:
                worst_gap = max(abs(a["rel_gap_dollars"]) for a in comp["arms"])
                if worst_gap > 0.25:
                    lines.append(
                        f"  ^ gaps up to {worst_gap:.0%} but the ranking holds; regret's "
                        "measured numerator absorbs the level error, the plan ORDER is "
                        "what survives it."
                    )
            if comp.get("latency_ranking_preserved") is False:
                lines.append("  ^ LATENCY RANKING INVERTED on the minutes axis.")
        elif comp:
            lines += ["", f"  composed-estimate check skipped: {comp.get('why')}"]
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


def _assert_timing_regressors_vary(trace: Trace, tag: str) -> None:
    """Abort a calibration minutes in, not dollars later, when the timing fit
    is already doomed.

    The first real calibration ran its full eight runs overnight, and only the
    extraction afterwards revealed a constant regressor column ("not
    identifiable"). This gate reads the same columns `fit_timing_model` will
    regress on -- context size and output length -- off the FIRST serial trace,
    and refuses to launch the remaining runs when either is constant.

    Necessary, not sufficient: columns can each vary and still be collinear
    (the graceful-partial path pins that case), so `fit_timing_model` keeps the
    final word. The completed trace is on disk either way -- the dollar side of
    what was already paid for survives, per the module's costs-half-not-all
    rule.
    """
    rows = [r for r in (c.as_record() for c in trace.calls) if r.billable and r.total_s > 0]
    if len(rows) < 3:
        return  # too few calls to judge here; the extraction will say so itself
    problems = []
    if len({context_tokens(r) for r in rows}) < 2:
        problems.append("context size is constant across its calls")
    if len({r.output_tokens or 0 for r in rows}) < 2:
        problems.append("output length is constant across its calls")
    if problems:
        raise ValueError(
            f"calibration run {tag!r} cannot support a timing model: "
            + " and ".join(problems)
            + ". Aborting before the remaining runs spend anything -- the trace is "
            "on disk, and the scenario (or the usage accounting) has to change "
            "before a relaunch."
        )


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
    model's one approximation. Each ordering is a DISTINCT topological order
    stated in the plan directive -- an ordering the model chose for itself is
    the same ordering every run, which is how the first real calibration
    completed all six runs in identical order and tested nothing.

    Every trace is written to `out_dir` before anything is fitted, so a failed
    extraction never costs the runs. A fan-out or bundled run that dies
    MID-FLIGHT costs only itself: the failure becomes a note and the extraction
    proceeds on what exists, per the module's costs-half-not-all rule -- the
    first real calibration lost its bundled trace to exactly such a crash and
    the spend with it.

    EVERY RUN GOES THROUGH THE LOGGING PROXY, and `require_proxy` defaults to
    True for a reason specific to calibration. A client-side error in parsing the
    provider's usage block would be invisible in a single run, and here it would
    be baked into every constant -- and then into every reported dollar
    figure. The proxy parses the same blocks with different code, so a
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
        health = _health_notes(trace, tag)
        if health and "PROXY MISMATCH" in health[0] and require_proxy:
            raise ValueError(
                f"calibration run {tag!r} failed the proxy cross-check: {health[0]} "
                "Refusing to derive constants from token counts two parsers disagree "
                "about. The traces are on disk; fix the parser and re-run `extract`."
            )
        notes.extend(health)

    orders = serial_orderings(scenario.dag, orderings)
    if len(orders) < orderings:
        notes.append(
            f"only {len(orders)} distinct topological order(s) exist on shape "
            f"{scenario.dag.shape!r}; ordering disagreement partly measures repeat "
            "variance on this scenario"
        )
    for ordering in range(orderings):
        node_order = orders[ordering % len(orders)]
        for repeat in range(repeats):
            tag = f"serial-o{ordering}-r{repeat}"
            trace = run_plan(
                scenario,
                all_inline(scenario.dag),
                client,
                workspaces / tag,
                node_order=node_order,
                condition="calibrate-serial",
                repeat=repeat,
                max_turns=max_turns,
                budget=budget,
                retry=retry,
                proxy_log=out_dir / f"proxy-{tag}.jsonl",
            )
            files.append(str(trace.write(out_dir / f"trace-{tag}.json")))
            guard(trace, tag)
            realized = tuple(n for n, _ in completion_order(trace, scenario))
            if realized and realized != tuple(node_order):
                # Not a disqualification -- the curve is segmented on what
                # actually happened -- but an ignored directive means the
                # orderings did not vary the way the diagnostic assumes.
                notes.append(
                    f"{tag}: requested order {','.join(node_order)} but ran "
                    f"{','.join(realized)}"
                )
            serial_traces.append(trace)
            if len(serial_traces) == 1:
                _assert_timing_regressors_vary(trace, tag)

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
        try:
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
        except Exception as exc:  # noqa: BLE001 -- a lost run must not lose the rest
            notes.append(
                f"{tag} run died mid-flight and left no trace "
                f"({type(exc).__name__}: {exc}); constants that need it stay "
                f"placeholders, and proxy-{tag}.jsonl records what it spent"
            )
            continue
        files.append(str(trace.write(out_dir / f"trace-{tag}.json")))
        guard(trace, tag)
        fanout_traces.append(trace)

    result = extract(serial_traces, fanout_traces, scenario, price, source=source)
    result.trace_files = files
    result.notes = notes
    result.write(out_dir)
    return result


def _health_notes(trace: Trace, tag: str) -> list[str]:
    """Token-trust notes for one trace, shared by the driver and `replay` so a
    re-derived calibration carries the same provenance the original did."""
    bad = [n for n in trace.notes if n.startswith("PROXY MISMATCH")]
    if bad:
        return [f"{tag}: {bad[0]}"]
    if not trace.proxy_verified:
        return [f"{tag}: no independent proxy log; token counts are unverified"]
    return []


def serial_orderings(dag, count: int) -> list[tuple[str, ...]]:
    """Up to `count` DISTINCT topological orders, deterministically.

    Seed 0 is the canonical order (always pop the smallest ready node), so one
    run is comparable across calibrations; later seeds shuffle the ready set.
    On a constrained shape fewer than `count` distinct orders may exist -- a
    chain has exactly one -- and the caller notes that rather than pretending
    the repeats were orderings.
    """
    import random

    succs: dict[str, list[str]] = {}
    for u, v in dag.edges:
        succs.setdefault(u, []).append(v)
    orders: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for seed in range(max(count, 1) * 8):
        rng = random.Random(seed)
        indeg = {v: len(dag.preds[v]) for v in dag.ids}
        ready = sorted(v for v in dag.ids if indeg[v] == 0)
        out: list[str] = []
        while ready:
            v = ready.pop(rng.randrange(len(ready)) if seed else 0)
            out.append(v)
            for w in succs.get(v, ()):
                indeg[w] -= 1
                if indeg[w] == 0:
                    ready.append(w)
            ready.sort()
        key = tuple(out)
        if key not in seen:
            seen.add(key)
            orders.append(key)
        if len(orders) == count:
            break
    return orders


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
    overhead = spawn_overhead(fanouts, price, fit, scenario=scenario)
    sub = sub_block_curves(fanouts, scenario, price, fit)

    # Exploration is averaged over EVERY run, serial and fan-out alike: the
    # prefix is the same physical quantity in each, and it varied five-fold
    # across the first real calibration's runs -- one draw is mostly noise.
    per_run = [explore_overhead(t, price, fit) for t in usable + fanouts]
    measured = [e for e in per_run if e.get("explore_dollars") is not None]
    if measured:
        explore = {
            "explore_dollars": statistics.fmean(e["explore_dollars"] for e in measured),
            "explore_minutes": statistics.fmean(e["explore_minutes"] for e in measured),
            "n_calls": sum(e["n_calls"] for e in measured),
            "n_runs": len(measured),
        }
    else:
        explore = {"why": (per_run[0].get("why") if per_run else "no runs to read")}
    from .tools import MAX_CONCURRENCY

    throughput = throughput_by_concurrency(usable + fanouts, MAX_CONCURRENCY)

    constants, skipped = {}, {}
    minute_constants = {
        "block_minutes_curve",
        "sub_block_minutes_curve",
        "brief_minutes",
        "brief_minutes_per_node",
        "absorb_minutes",
        "absorb_minutes_per_node",
        "explore_minutes",
    }
    candidates = {
        "block_dollars_curve": curves["block_dollars_curve"],
        "block_minutes_curve": curves["block_minutes_curve"],
        "sub_block_dollars_curve": sub.get("sub_block_dollars_curve"),
        "sub_block_minutes_curve": sub.get("sub_block_minutes_curve"),
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
                sub.get("why") if name.startswith("sub_block") else None
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
            "n_subagent_blocks": sub.get("n_subagent_blocks"),
            "sub_units_covered": sub.get("sub_units_covered"),
            "n_brief_turns": overhead.get("n_brief_turns"),
            "n_brief_rows": overhead.get("n_brief_rows"),
            "explore_calls": explore.get("n_calls"),
            "explore_runs": explore.get("n_runs"),
            "throughput_samples": throughput.get("samples_per_level"),
            "timing_residual_rms_minutes": timing.residual_rms_minutes if timing else None,
        },
        skipped=skipped,
    )
    # THE ESTIMATION GATE: predict the calibration's own executed plans from
    # the constants just extracted and compare against their measured totals.
    # Every diagnostic above tests an ingredient; this one tests the dish, and
    # it is the only number that says whether the oracle's composed estimate
    # means anything. Zero spend -- the runs it needs are the ones that
    # produced the constants.
    result.diagnostics["composition"] = composition_check(
        usable, fanouts, scenario, price, timing, result.to_cost_model()
    )

    # Derived last, because they read the diagnostics the rest of the extraction
    # produced. Reported on the result rather than mutated into the module-level
    # constant: a calibration should not reach in and change the meaning of
    # "equal" for every other run in the process. The minutes floor is gated on
    # the timing fit -- a floor from the dummy model would be an invented number.
    result.floors = {
        "dollars": dollar_floor(result),
        "minutes": minute_floor(result) if timing else None,
    }
    result.outcome_tol = result.floors["dollars"]
    return result


def replay(out_dir: str | Path, scenario: Scenario, price: PriceSheet, *, source: str) -> CalibrationResult:
    """Re-derive a calibration from its saved traces. Zero spend.

    This is what makes the calibration reproducible rather than merely recorded.
    If `replay` disagrees with the published `calibration.json`, the extraction
    changed and every number downstream of it is suspect.

    Trace files and per-trace health notes are rebuilt from what is on disk, so
    a replayed calibration carries its provenance instead of publishing empty
    `notes` and `trace_files` -- which is how the first real calibration.json
    lost the record of which traces produced it.
    """
    out_dir = Path(out_dir)
    serial_paths = sorted(out_dir.glob("trace-serial-*.json"))
    fanout_paths = [
        p
        for name in ("trace-fanout.json", "trace-bundled.json")
        for p in [out_dir / name]
        if p.exists()
    ]
    serial = [Trace.load(p) for p in serial_paths]
    fanouts = [Trace.load(p) for p in fanout_paths]
    result = extract(serial, fanouts, scenario, price, source=source)
    result.trace_files = [str(p) for p in serial_paths + fanout_paths]
    for path, trace in zip(serial_paths + fanout_paths, serial + fanouts):
        tag = path.stem.replace("trace-", "")
        result.notes.extend(_health_notes(trace, tag))
    return result
