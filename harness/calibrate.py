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
    "fresh_tokens",
    "context_tokens",
    "holdout_error",
]


# ------------------------------------------------------------- price sheets


@dataclass(frozen=True)
class PriceSheet:
    """Dollars per million tokens, by category, for one model on one date.

    `as_of` is not decoration. Section 7 makes billed dollars the currency of
    every reported number, and an undated dollar is not a unit -- provider
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

    FORCED ORDERINGS MISALIGN THE BUCKETS (found by the first preflight that
    actually varied them): a run that starts with a size-5 node has no point at
    3 units, so bucketing by raw unit count averages each bucket over whichever
    runs happen to land on it -- and the mean curve came out NON-MONOTONE,
    pricing a 5-unit block below a 3-unit one purely as a coverage artifact.
    So each curve is read on the union grid via interpolation, a grid point
    averages only the curves whose MEASURED range covers it (no extrapolating a
    run beyond what it did, no pro-rata modelling below its first point), and
    the result is clamped monotone with a running max -- every run's own
    cumulative curve is monotone, so an inversion in the average is never a
    measurement.
    """
    from generator.oracle import _interpolate

    usable = [tuple(sorted(c)) for c in curves if c]
    if not usable:
        raise ValueError("no curves to average")
    grid = sorted({u for c in usable for u, _ in c})
    out: list[tuple[int, float]] = []
    floor = 0.0
    for u in grid:
        vals = [_interpolate(c, u) for c in usable if c[0][0] <= u <= c[-1][0]]
        if not vals:
            continue
        floor = max(floor, sum(vals) / len(vals))
        out.append((u, floor))
    return tuple(out)


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


def fresh_tokens(call: CallRecord) -> int:
    """Context the provider prefills at full price: uncached input plus cache
    writes. Cache reads are prefilled too, but on their own terms -- see
    `context_tokens` and the correction note on `TimingModel`."""
    return (call.input_tokens or 0) + (call.cache_write_tokens or 0)


def context_tokens(call: CallRecord) -> int:
    """The full prompt the model read: billed input + cache writes + cache reads.

    Billed `input_tokens` alone is NOT context size. Under the pinned cache
    breakpoints nearly the whole prompt bills as cache reads and writes, and
    billed input collapses to the residue after the last breakpoint -- measured
    at a constant 2 tokens per call across the first real calibration. Any code
    that means "how much did this call read" must use this sum, never the raw
    field.
    """
    return fresh_tokens(call) + (call.cache_read_tokens or 0)


@dataclass(frozen=True)
class TimingModel:
    """`minutes = a + b_fresh * fresh + b_cached * cache_reads + output / throughput`.

    `fresh` is uncached input plus cache writes -- tokens prefilled at full
    price. Cache reads are prefilled too, but faster; they carry their own
    coefficient when the log can identify one, and `b_cached = None` records a
    blended fit where a single coefficient covered every context token.

    THE REGRESSORS ARE CONTEXT COLUMNS, NOT BILLED `input_tokens` (corrected
    Aug 25). Under the pinned cache breakpoints billed input is a near-constant
    residue -- 2 tokens on every call of the first real calibration -- so a fit
    against it was singular while the context actually varied by tens of
    thousands of tokens. Where such a fit DID succeed (the Haiku preflight,
    whose higher cache floor left tail segments uncached), it priced prefill on
    that sliver and read 470K cache-read tokens as free: numerically plausible,
    physically wrong.

    Latency for the metric is reconstructed from this, never taken from the
    clock. Wall clock is confounded by rate limits, queueing, and provider load,
    so it is not comparable across vendors or even across two runs of the same
    scenario -- section 6 keeps it as a sanity check only.
    """

    a_minutes: float
    b_minutes_per_input_token: float  # minutes per FRESH context token
    output_tokens_per_minute: float
    b_cached_minutes_per_token: float | None = None  # per cache-read token; None = blended
    n_calls: int = 0
    residual_rms_minutes: float = 0.0
    # "total-s" = prefill separated by the least-squares fit against total_s.
    # "ttfb-anchor" = total_s could not separate prefill from generation (the
    # unconstrained coefficient came out negative), so prefill was pinned to the
    # slope of TTFB against context -- the direct prefill observation -- and only
    # the intercept and throughput were fitted against total_s.
    prefill_source: str = "total-s"

    @property
    def cached_minutes_per_token(self) -> float:
        """The cache-read coefficient, falling back to the fresh one for a
        blended fit -- so downstream consumers never branch."""
        b = self.b_cached_minutes_per_token
        return self.b_minutes_per_input_token if b is None else b


def _solve(m: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting. Any number of unknowns, no numpy."""
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


def _fit(xs: list[tuple], ys: list[float]) -> tuple[list[float], float]:
    """Normal-equations least squares plus the residual RMS of the fit."""
    k = len(xs[0])
    m = [[sum(x[i] * x[j] for x in xs) for j in range(k)] for i in range(k)]
    rhs = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(k)]
    coef = _solve(m, rhs)
    resid = [sum(c * xi for c, xi in zip(coef, x)) - y for x, y in zip(xs, ys)]
    return coef, (sum(r * r for r in resid) / len(resid)) ** 0.5


def fit_timing_model(calls: list[CallRecord]) -> TimingModel:
    """Least squares over the call log for the timing constants.

    Fitted against `total_s`, with CONTEXT columns and `output_tokens` as the
    regressors -- context, never billed `input_tokens`, per the correction on
    `TimingModel`. Time-to-first-byte is what makes the context term real:
    prefill scales with what the model reads, so a large lead context is
    materially slower to START than a cold subagent -- which is why context
    drag lands in the latency column and not only in the cost column.

    The fit is tried SPLIT first (separate fresh and cache-read coefficients),
    because the serial and fan-out arms differ systematically in cache mix and
    one blended coefficient would misprice exactly the serial-vs-parallel
    comparison. It falls back to blended when the log cannot tell the two
    apart: a singular system, a wrong-signed coefficient, or cached prefill
    fitting SLOWER than fresh -- physically backwards, so the split is noise
    there. The blended fit reports its own refusals.
    """
    rows = [c for c in calls if c.billable and c.total_s > 0]
    if len(rows) < 3:
        raise ValueError(f"need at least 3 timed calls to fit three constants, got {len(rows)}")
    ys = [c.total_s / 60.0 for c in rows]
    fresh = [float(fresh_tokens(c)) for c in rows]
    cached = [float(c.cache_read_tokens or 0) for c in rows]
    outs = [float(c.output_tokens or 0) for c in rows]

    if len(rows) >= 4 and len(set(cached)) >= 2:
        try:
            coef, rms = _fit([(1.0, f, cr, o) for f, cr, o in zip(fresh, cached, outs)], ys)
            a, b_f, b_c, c_split = coef
            if b_f >= 0.0 and 0.0 <= b_c <= b_f and c_split > 0.0:
                return TimingModel(
                    a_minutes=a,
                    b_minutes_per_input_token=b_f,
                    output_tokens_per_minute=1.0 / c_split,
                    b_cached_minutes_per_token=b_c,
                    n_calls=len(rows),
                    residual_rms_minutes=rms,
                )
        except ValueError:
            pass  # singular split; the blended fit below speaks for itself

    (a, b, c_out), rms = _fit([(1.0, f + cr, o) for f, cr, o in zip(fresh, cached, outs)], ys)
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
        # It happens when total_s cannot separate prefill from generation --
        # under a reasoning model, hidden thinking time swells `output_tokens`
        # in step with context depth, so the two regressors move together and
        # the split between them is arbitrary. A bare clamp to zero would
        # assert prefill is free without measuring it. But the log usually
        # carries a direct prefill observation total_s does not need: TTFB,
        # which ends when generation begins. When it does, pin `b` to the TTFB
        # slope against context (floored at physical zero) and fit only the
        # intercept and throughput against total_s. Refusal remains for logs
        # whose TTFB cannot identify a slope either.
        return _ttfb_anchored_fit(rows, ys, fresh, cached, outs, unconstrained_b=b)
    return TimingModel(
        a_minutes=a,
        b_minutes_per_input_token=b,
        output_tokens_per_minute=1.0 / c_out,
        n_calls=len(rows),
        residual_rms_minutes=rms,
    )


def _ttfb_anchored_fit(
    rows: list[CallRecord],
    ys: list[float],
    fresh: list[float],
    cached: list[float],
    outs: list[float],
    *,
    unconstrained_b: float,
) -> TimingModel:
    """The rescue for a log whose total_s fit put prefill below zero.

    Model-agnostic by construction: `fit_timing_model` reaches here only when
    the unconstrained fit is unphysical, whichever model produced the log. Legs
    whose primary fit succeeds (every Claude calibration to date) never enter.

    TTFB ends when generation begins, so its slope against context is the
    prefill price observed directly, free of the thinking-time confound that
    sank the total_s separation. Pin `b` to that slope (floored at physical
    zero), then fit intercept and throughput against total_s as usual. The
    original refusal stands when TTFB cannot identify a slope either -- fewer
    than 3 timed first bytes, or no context variation among them.
    """
    tt = [
        (r.ttfb_s / 60.0, f + cr)
        for r, f, cr in zip(rows, fresh, cached)
        if r.ttfb_s and r.ttfb_s > 0
    ]
    if len(tt) < 3 or len({ctx for _, ctx in tt}) < 2:
        raise ValueError(
            f"fitted a negative cost per input token ({unconstrained_b:.3g} min/token), "
            "which would make a larger context faster to start, and the log carries no "
            "usable TTFB to anchor prefill directly. Context and output length are too "
            "correlated in this log to separate prefill from generation -- vary them "
            "independently across the calibration calls"
        )
    (_, slope), _ = _fit([(1.0, ctx) for _, ctx in tt], [t for t, _ in tt])
    b_anchor = max(slope, 0.0)
    ys_less_prefill = [y - b_anchor * (f + cr) for y, f, cr in zip(ys, fresh, cached)]
    (a, c_out), rms = _fit([(1.0, o) for o in outs], ys_less_prefill)
    if c_out <= 0:
        raise ValueError(
            "fitted a non-positive cost per output token; the log is too narrow to "
            "identify throughput -- vary output length across the calibration calls"
        )
    return TimingModel(
        a_minutes=a,
        b_minutes_per_input_token=b_anchor,
        output_tokens_per_minute=1.0 / c_out,
        n_calls=len(rows),
        residual_rms_minutes=rms,
        prefill_source="ttfb-anchor",
    )


def call_minutes(call: CallRecord, tm: TimingModel) -> float:
    """Analytic duration for one call. Not `call.total_s`.

    Prices the context the model READ: fresh tokens at the fresh coefficient,
    cache reads at the cached one (identical under a blended fit). Reading
    billed `input_tokens` here was the Aug 25 correction -- it silently priced
    a 100K-token warm context as two tokens of prefill.
    """
    return (
        tm.a_minutes
        + tm.b_minutes_per_input_token * fresh_tokens(call)
        + tm.cached_minutes_per_token * (call.cache_read_tokens or 0)
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
