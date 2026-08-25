"""Turning a run's call log into the cost model's constants.

Design doc: Phase1-DelegationBench-Design.md section 4 Stage 3.
Low-level design: low-level-design.md, "Calibration".

The oracle prices thousands of plans it will never run. That is only legitimate
if the primitives it composes are measured rather than assumed, and the load-
bearing primitive is the BLOCK CURVE: what one agent is billed, and how long it
takes, to work a block of `u` size units in a single context.

Two opposing forces live in that curve. Context REUSE makes a block's later
subtasks cheaper -- the files are read, the approach is settled. Context DRAG
makes them dearer -- the whole conversation is re-sent and re-billed every turn,
so turn 20 pays for turns 1 through 19. Which one wins is an empirical fact
about a model and a workload. Encoding either as a parameter asserts the answer;
measuring the curve reports it.

WHAT THIS MODULE DOES NOT DO: drive the runs. It consumes `CallRecord`s that a
serial calibration run already produced, so every function here is testable
offline against a synthetic log, with no API key and no spend.

THE ASSUMPTION THIS BUYS, STATED ONCE: block cost depends on how MUCH work is in
the block, not on WHICH subtasks compose it. One testable approximation in place
of two unmeasurable parameters -- and `curve_disagreement` and `holdout_error`
below exist to test it rather than trust it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .proxy import CallRecord

__all__ = [
    "PriceSheet",
    "call_dollars",
    "Boundary",
    "cumulative_points",
    "average_curves",
    "curve_disagreement",
    "TimingModel",
    "fit_timing_model",
    "call_minutes",
    "holdout_error",
]


# ------------------------------------------------------------- price sheets


@dataclass(frozen=True)
class PriceSheet:
    """Dollars per million tokens, by category, for one model on one date.

    `as_of` is not decoration. Section 7 makes billed dollars the currency of
    every number in the report, and an undated dollar is not a unit -- provider
    prices move, and a curve measured under one price sheet does not compose
    with a matrix run under another. Record it, publish it, and refuse to mix.
    """

    model: str
    as_of: str  # ISO date, e.g. "2026-08-23"
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_per_mtok: float


def call_dollars(call: CallRecord, price: PriceSheet) -> float:
    """Bill one call. Cache reads are counted at their billed price, never as
    fresh input -- "tokens observed" and "tokens billed" are different numbers,
    and the cost model has to pick one and say which."""
    if not call.billable:
        raise ValueError(f"call {call.seq} has no usage block; it is not a measurement")
    m = 1_000_000.0
    return (
        (call.input_tokens or 0) * price.input_per_mtok
        + (call.output_tokens or 0) * price.output_per_mtok
        + (call.cache_read_tokens or 0) * price.cache_read_per_mtok
        + (call.cache_write_tokens or 0) * price.cache_write_per_mtok
    ) / m


# ------------------------------------------------------ segmenting a run log


@dataclass(frozen=True)
class Boundary:
    """A subtask completing, and how much work was done by that moment.

    `units` is CUMULATIVE size units finished, not this subtask's size. The
    curve is keyed on block size, so the x-axis is the running total.
    """

    units: int
    t_done: float  # wall clock, same clock as CallRecord.t_request


def cumulative_points(
    calls: list[CallRecord], boundaries: list[Boundary], price: PriceSheet
) -> tuple[tuple[int, float], ...]:
    """One serial run -> one curve.

    Calls are attributed to the boundary they precede. Agents interleave -- a
    file gets read for subtask 3 while subtask 1 is still open -- but that does
    not matter here, because the quantity is CUMULATIVE. Only the running total
    at each boundary is claimed, never a per-subtask cost.
    """
    if not boundaries:
        raise ValueError("no boundaries: a run with no completed subtask measures nothing")
    ordered = sorted(boundaries, key=lambda b: b.units)
    out = []
    for b in ordered:
        spend = sum(call_dollars(c, price) for c in calls if c.t_request <= b.t_done)
        out.append((b.units, spend))
    for (ua, va), (ub, vb) in zip(out, out[1:]):
        if vb < va - 1e-12:
            raise ValueError(
                f"cumulative spend fell from ${va:.4f} at {ua} units to ${vb:.4f} at {ub}; "
                "boundaries are probably out of order against the call clock"
            )
    return tuple(out)


def average_curves(curves: list[tuple[tuple[int, float], ...]]) -> tuple[tuple[int, float], ...]:
    """Mean across repeats and orderings, at each measured unit count.

    Repeats are not optional. A single run is one draw from a stochastic policy,
    and a curve fitted to one draw reports that draw's turn count as if it were
    a property of the model.
    """
    if not curves:
        raise ValueError("no curves to average")
    buckets: dict[int, list[float]] = {}
    for curve in curves:
        for units, value in curve:
            buckets.setdefault(units, []).append(value)
    return tuple((u, sum(vs) / len(vs)) for u, vs in sorted(buckets.items()))


def curve_disagreement(curves: list[tuple[tuple[int, float], ...]]) -> dict[int, float]:
    """Spread across curves at each unit count, as a fraction of the mean.

    THIS IS THE TEST OF THE MODULE'S ONE ASSUMPTION. Run the same node set in
    several orderings. If block cost really depends on total units and not on
    which subtasks compose the block, the orderings agree. Wide disagreement
    means the assumption is false, and it is far cheaper to learn that here than
    from a skeptical reader -- report the spread alongside the curve either way.
    """
    buckets: dict[int, list[float]] = {}
    for curve in curves:
        for units, value in curve:
            buckets.setdefault(units, []).append(value)
    out = {}
    for units, vs in sorted(buckets.items()):
        if len(vs) < 2:
            continue
        mean = sum(vs) / len(vs)
        out[units] = (max(vs) - min(vs)) / mean if mean else 0.0
    return out


# --------------------------------------------------------- the timing model


@dataclass(frozen=True)
class TimingModel:
    """`minutes = a + b * input_tokens + output_tokens / throughput`.

    Latency for the metric is reconstructed from this, never taken from the
    clock. Wall clock is confounded by rate limits, queueing, and provider load,
    so it is not comparable across vendors or even across two runs of the same
    scenario -- section 6 keeps it as a sanity check only.
    """

    a_minutes: float
    b_minutes_per_input_token: float
    output_tokens_per_minute: float
    n_calls: int = 0
    residual_rms_minutes: float = 0.0


def _solve3(m: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting. Three unknowns, no numpy."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(m)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-15:
            raise ValueError("timing model is not identifiable from these calls")
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(n):
            if r == col:
                continue
            f = aug[r][col] / aug[col][col]
            for c in range(col, n + 1):
                aug[r][c] -= f * aug[col][c]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def fit_timing_model(calls: list[CallRecord]) -> TimingModel:
    """Least squares over the call log for the three timing constants.

    Fitted against `total_s`, with `input_tokens` and `output_tokens` as the
    regressors. Time-to-first-byte is what makes the input term real: prefill
    scales with context length, so a large lead context is materially slower to
    START than a cold subagent -- which is why context drag lands in the latency
    column and not only in the cost column.
    """
    rows = [c for c in calls if c.billable and c.total_s > 0]
    if len(rows) < 3:
        raise ValueError(f"need at least 3 timed calls to fit three constants, got {len(rows)}")
    xs = [(1.0, float(c.input_tokens or 0), float(c.output_tokens or 0)) for c in rows]
    ys = [c.total_s / 60.0 for c in rows]
    m = [[sum(x[i] * x[j] for x in xs) for j in range(3)] for i in range(3)]
    rhs = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(3)]
    a, b, c_out = _solve3(m, rhs)
    if c_out <= 0:
        raise ValueError(
            "fitted a non-positive cost per output token; the log is too narrow to "
            "identify throughput -- vary output length across the calibration calls"
        )
    if b < 0:
        # A NEGATIVE PREFILL COEFFICIENT SAYS A BIGGER CONTEXT IS FASTER TO START,
        # which is not a thing. Left alone it does real damage: `b` is what puts
        # context drag in the latency column, so a negative one makes a long
        # serial context look FASTER as it grows and flips the sign of one of the
        # mechanisms the benchmark exists to measure. It also propagates -- the
        # absorption minutes are derived from it, and they would come out
        # negative too, making every absorbed result shorten the run.
        #
        # It happens when input length does not vary independently of output
        # length across the log, so the regression cannot separate the two. The
        # answer is more varied calls, not a clamp: clamping to zero would assert
        # prefill is free, which is the same unmeasured claim in the other
        # direction.
        raise ValueError(
            f"fitted a negative cost per input token ({b:.3g} min/token), which would make "
            "a larger context faster to start. Input and output length are too correlated "
            "in this log to separate prefill from generation -- vary them independently "
            "across the calibration calls"
        )
    resid = [(a + b * x[1] + c_out * x[2]) - y for x, y in zip(xs, ys)]
    rms = (sum(r * r for r in resid) / len(resid)) ** 0.5
    return TimingModel(
        a_minutes=a,
        b_minutes_per_input_token=b,
        output_tokens_per_minute=1.0 / c_out,
        n_calls=len(rows),
        residual_rms_minutes=rms,
    )


def call_minutes(call: CallRecord, tm: TimingModel) -> float:
    """Analytic duration for one call. Not `call.total_s`."""
    return (
        tm.a_minutes
        + tm.b_minutes_per_input_token * (call.input_tokens or 0)
        + (call.output_tokens or 0) / tm.output_tokens_per_minute
    )


# ------------------------------------------------------------- the held-out check


def holdout_error(
    curve: tuple[tuple[int, float], ...], units: int, measured: float
) -> dict[str, float]:
    """Predict a block the calibration never saw, then compare against running it.

    This is the number that decides whether the curve means anything. Everything
    above produces a curve from runs it was fitted on, which is not evidence that
    it generalizes. Take a block size or a node composition held out of the fit,
    predict, run it, and report the gap. A large gap falsifies the units-not-
    identity assumption, and the honest output is the gap rather than the curve.
    """
    from generator.oracle import _interpolate

    predicted = _interpolate(curve, units)
    err = measured - predicted
    return {
        "units": float(units),
        "predicted": predicted,
        "measured": measured,
        "abs_error": abs(err),
        "rel_error": abs(err) / measured if measured else float("inf"),
    }
