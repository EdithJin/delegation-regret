"""Stage 6: turning traces into delegation regret.

Low-level design: low-level-design.md, "Stage 6".

Three quantities and one division:

    regret = (agent objective − oracle-plan-as-EXECUTED objective)
             ────────────────────────────────────────────────────
             (worst degenerate objective − oracle objective)

0 is perfect. 1 is as bad as the dumbest defensible strategy. Above 1 is
possible and is not a bug: an agent that explores expensively and *then* chooses
badly can exceed the worst clairvoyant policy, because the oracle never paid for
discovery.

WHY THE NUMERATOR IS MEASURED AND THE DENOMINATOR IS NOT. They fail differently.

An error in the numerator's baseline **biases** regret -- it shifts every number
the same way -- so that one is obtained by executing the oracle plan for real and
measuring it, never by trusting the cost model's prediction. If a constant is
slightly off, the error would otherwise surface as regret the agent did not earn.

An error in the denominator merely **scales** regret, and the primary comparison
is model A against model B on the SAME scenario, where both divide by the
identical number and any error cancels exactly. So the denominator is read off
the plan table for free. That is not only cheaper -- roughly two dozen runs -- it
removes a known bias: a measured denominator is a maximum over two noisy
quantities, and a max over noise is inflated, which deflates every regret figure
reported.

WHAT IS REFUSED RATHER THAN APPROXIMATED. A scenario is EXCLUDED, explicitly and
with a reason, when the run that would anchor it is not sound: the agent failed a
node, the baseline failed a node, the baseline did not execute the plan it was
asked to, or either run edited the generated suites. Section 6.6 is direct about
this -- if executing the oracle plan fails, that scenario has no valid baseline
and must be excluded, not scored against a broken reference. Excluded scenarios
are reported, never silently dropped, because a model whose runs keep failing is
telling you something and a quietly shrinking denominator is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from generator.dag import DAG
from generator.oracle import (
    CostModel,
    all_inline,
    enumerate_plans,
    evaluate,
    is_tier_a,
    max_fanout,
    optimal_k_intervals,
)
from harness.calibrate import PriceSheet, TimingModel
from harness.trace import LEAD, Trace

__all__ = [
    "Stakes",
    "ScoreCard",
    "stakes",
    "score",
    "implied_beta",
    "intersect_beta",
    "quality_by_executor",
    "oracle_plan_health",
]


# ------------------------------------------------------ (c) how much is at stake


@dataclass(frozen=True)
class Stakes:
    """The denominator, and the plan table facts it comes from. No runs."""

    oracle_objective: float
    all_inline_objective: float
    max_fanout_objective: float
    oracle_k: int
    # Does ONE plan dominate on both cost and latency? Then every beta agrees and
    # the scenario is scoreable without arguing about what a minute is worth --
    # the strongest form of the result, and worth separating out rather than
    # averaging into scenarios whose answer depends on the persona.
    tier_a: bool = False

    @property
    def worst_degenerate(self) -> float:
        return max(self.all_inline_objective, self.max_fanout_objective)

    @property
    def spread(self) -> float:
        return self.worst_degenerate - self.oracle_objective


def stakes(dag: DAG, cm: CostModel, beta: float, results=None) -> Stakes:
    """Price the plan table once and read the three reference points off it.

    `results` lets a caller reuse an enumeration across betas -- feasibility does
    not depend on price, and on 8 nodes the evaluation is the expensive half.
    """
    if results is None:
        results = [evaluate(dag, p, cm) for p in enumerate_plans(dag)]
    by_plan = {r.plan: r for r in results}
    best = min(results, key=lambda r: (r.objective(beta), r.plan.k))
    return Stakes(
        tier_a=is_tier_a(results),
        oracle_objective=best.objective(beta),
        all_inline_objective=by_plan[all_inline(dag)].objective(beta),
        max_fanout_objective=by_plan[max_fanout(dag)].objective(beta),
        oracle_k=best.plan.k,
    )


# ----------------------------------------------------------------- the score


@dataclass(frozen=True)
class ScoreCard:
    """One scenario, one beta, one model. The unit the report aggregates."""

    scenario_id: str
    beta: float
    agent_objective: float | None = None
    baseline_objective: float | None = None
    stakes: Stakes | None = None
    regret: float | None = None
    agent_k: int = 0
    oracle_k: int = 0
    implied_beta: tuple[float, float] | None = None
    tier_a: bool = False
    agent_passed: bool = False
    baseline_passed: bool = False
    excluded: bool = False
    reason: str = ""
    warnings: tuple[str, ...] = ()

    @property
    def comparable(self) -> bool:
        """Scored AND in one currency. See `_scale_warning`."""
        return not self.excluded and not self.warnings

    @property
    def beat_all_inline(self) -> bool | None:
        """"Does this model beat always-serial?" -- a first-class pass/fail.

        A metric no trivial policy can top is worth more than a leaderboard, so
        this is reported next to regret rather than derived from it later.
        """
        if self.excluded or self.agent_objective is None or self.stakes is None:
            return None
        return self.agent_objective < self.stakes.all_inline_objective


def _scale_warning(cm: CostModel) -> tuple[str, ...]:
    """Are the numerator and the denominator in the same units?

    THE ONE FAILURE THAT PRODUCES A CLEAN TABLE OF MEANINGLESS NUMBERS, so it is
    checked here rather than left to a reader.

    The numerator is MEASURED: real billed dollars from the trace, plus beta
    times minutes reconstructed through the fitted timing model. The denominator
    is COMPUTED: objectives read off the oracle's plan table, in whatever units
    `CostModel`'s constants are denominated in. Section 6.4 argues the
    denominator need not be measured because it merely SCALES regret and cancels
    when comparing two models on one scenario -- and that argument is sound, but
    it silently assumes both sides share a currency.

    Under the default placeholder constants they do not, and not by a little. A
    size-unit costs a notional $0.10 and one notional minute, while a real run of
    the same node costs cents and seconds. The ratio then divides measured
    dollars by notional dollars and returns a number that is not a fraction of
    anything -- large, stable across repeats, and completely wrong.

    So regret is still computed, because refusing would make this module untestable
    before the calibration run exists, but the card is marked NOT COMPARABLE and
    `comparable` is False. Aggregating cards without checking that flag is the
    mistake this exists to prevent.
    """
    from generator.oracle import unmeasured

    missing = unmeasured(cm)
    if not missing:
        return ()
    return (
        "NOT COMPARABLE: the denominator comes from an uncalibrated cost model "
        f"({len(missing)} placeholder constant(s): {', '.join(missing[:4])}"
        f"{', ...' if len(missing) > 4 else ''}), so it is not denominated in the "
        "billed dollars and fitted minutes the numerator is measured in. Regret is "
        "reported for structure only until calibration lands.",
    )


def _disqualify(agent: Trace, baseline: Trace | None) -> str:
    """Why this scenario cannot be scored, or "" if it can."""
    if agent.tampered_tests:
        return f"agent edited generated suites ({', '.join(agent.tampered_tests)})"
    if not agent.succeeded:
        failed = sorted(n for n, ok in agent.verdicts.items() if not ok)
        return f"agent run did not verify ({', '.join(failed) or 'no verdicts'})"
    for trace, label in ((agent, "agent"), (baseline, "baseline")):
        if trace is None:
            continue
        bad = [n for n in trace.notes if n.startswith("PROXY MISMATCH")]
        if bad:
            # The client and an independent log of the same traffic disagreed
            # about what was billed. The run happened, but its cost is not a
            # measurement, and cost is half the objective.
            return f"{label} run failed the proxy cross-check: {bad[0][:120]}"
    if baseline is None:
        return "no baseline run supplied"
    if baseline.tampered_tests:
        return f"baseline edited generated suites ({', '.join(baseline.tampered_tests)})"
    if not baseline.succeeded:
        failed = sorted(n for n, ok in baseline.verdicts.items() if not ok)
        # Section 6.6: this is the case that would quietly break the metric.
        return f"oracle plan failed to verify ({', '.join(failed) or 'no verdicts'}); no valid baseline"
    if any(n.startswith("PLAN NOT FOLLOWED") for n in baseline.notes):
        return "baseline run did not execute the requested plan"
    return ""


def score(
    agent: Trace,
    baseline: Trace | None,
    dag: DAG,
    cm: CostModel,
    beta: float,
    price: PriceSheet,
    timing: TimingModel,
    results=None,
) -> ScoreCard:
    """Regret for one (scenario, model, beta), or an explicit exclusion.

    Both objectives are computed the same way from the same trace fields, which
    is the point of measuring the baseline rather than predicting it: any error
    in the cost model appears on both sides and cancels.
    """
    st = stakes(dag, cm, beta, results)
    warn = _scale_warning(cm)
    agent_k = agent.k
    interval = implied_beta(dag, cm, agent_k, results)

    reason = _disqualify(agent, baseline)
    if reason:
        return ScoreCard(
            scenario_id=agent.scenario_id,
            beta=beta,
            stakes=st,
            agent_k=agent_k,
            oracle_k=st.oracle_k,
            implied_beta=interval,
            tier_a=st.tier_a,
            agent_passed=agent.succeeded,
            baseline_passed=bool(baseline and baseline.succeeded),
            excluded=True,
            reason=reason,
            warnings=warn,
        )

    a = agent.objective(price, timing, beta)
    b = baseline.objective(price, timing, beta)
    if st.spread <= 0:
        # Every plan reaches the same outcome, so there is no mistake available
        # to make and no meaningful fraction of it to report. Tier A scenarios
        # can land here; dividing anyway would produce an infinity that means
        # "nothing was at stake", which is not a score.
        return ScoreCard(
            scenario_id=agent.scenario_id,
            beta=beta,
            agent_objective=a,
            baseline_objective=b,
            stakes=st,
            agent_k=agent_k,
            oracle_k=st.oracle_k,
            implied_beta=interval,
            tier_a=st.tier_a,
            agent_passed=True,
            baseline_passed=True,
            excluded=True,
            reason="no spread: every plan has the same objective at this beta",
            warnings=warn,
        )

    return ScoreCard(
        scenario_id=agent.scenario_id,
        beta=beta,
        agent_objective=a,
        baseline_objective=b,
        stakes=st,
        regret=(a - b) / st.spread,  # negative is a RESULT, never clipped
        agent_k=agent_k,
        oracle_k=st.oracle_k,
        implied_beta=interval,
        tier_a=st.tier_a,
        agent_passed=True,
        baseline_passed=True,
        warnings=warn,
    )


# --------------------------------------------------------------- implied beta


def implied_beta(dag: DAG, cm: CostModel, k: int, results=None) -> tuple[float, float] | None:
    """The beta range that makes spawning exactly `k` optimal, or None.

    Run the oracle backwards. `optimal_k_intervals` already returns, exactly and
    without a grid, the intervals on which each spawn count is globally optimal.
    Observing a model spawn `k` therefore identifies a range of latency prices
    its behaviour is consistent with.

    None means no latency price rationalizes that spawn count on this scenario,
    which is itself a finding rather than a missing measurement.

    Returns the convex hull when a k is optimal on several disjoint intervals.
    That is deliberately conservative for this report's purpose: a wider reported
    range makes a model look MORE rationalizable, so it cannot manufacture the
    incoherence result.
    """
    if results is None:
        results = [evaluate(dag, p, cm) for p in enumerate_plans(dag)]
    spans = [(lo, hi) for lo, hi, kk in optimal_k_intervals(results) if kk == k]
    if not spans:
        return None
    return (min(lo for lo, _ in spans), max(hi for _, hi in spans))


def intersect_beta(intervals) -> tuple[float, float] | None:
    """Intersect per-scenario ranges into one number for the model.

    *"This model behaves as if a developer-minute is worth $X."* None means the
    intersection is empty: no single latency price explains its choices across
    scenarios, so it is not trading cost against time coherently at all. That is
    a real result and is reported as one -- the estimator returns intervals and
    never a bare point estimate, because beta-rationalizability is precisely the
    thing under test.

    A scenario that rationalizes no k at all (a `None` interval) is treated as
    evidence against coherence, not skipped.
    """
    lo, hi = 0.0, math.inf
    seen = False
    for span in intervals:
        if span is None:
            return None
        seen = True
        lo, hi = max(lo, span[0]), min(hi, span[1])
        if lo >= hi:
            return None
    return (lo, hi) if seen else None


# ------------------------------------------------------- the quality checks


@dataclass
class ExecutorQuality:
    """Pass rate split by who did the work. Section 6.5."""

    lead_passed: int = 0
    lead_total: int = 0
    subagent_passed: int = 0
    subagent_total: int = 0

    def rate(self, who: str) -> float | None:
        p, t = (
            (self.lead_passed, self.lead_total)
            if who == LEAD
            else (self.subagent_passed, self.subagent_total)
        )
        return p / t if t else None

    def __str__(self) -> str:
        rows = []
        for label, who in (("nodes the lead did itself", LEAD), ("nodes handed to a subagent", "subagent")):
            r = self.rate(who)
            n = self.lead_total if who == LEAD else self.subagent_total
            rows.append(f"{label:<32} {'--' if r is None else f'{r:.0%} passed'}  (n={n})")
        return "\n".join(rows)


def quality_by_executor(traces) -> ExecutorQuality:
    """Cross-tabulate attribution against verdicts across runs.

    Answers the most predictable objection to the project: *"you measured only
    money and time; maybe delegation produces worse work."* The data is already
    collected, so this costs nothing beyond the arithmetic.

    Nodes nobody wrote are not counted on either side -- an unattributed failure
    is a failure to attempt, not evidence about an executor.
    """
    q = ExecutorQuality()
    for trace in traces:
        for node, who in trace.node_attribution.items():
            if node not in trace.verdicts:
                continue
            passed = trace.verdicts[node]
            if who == LEAD:
                q.lead_total += 1
                q.lead_passed += int(passed)
            else:
                q.subagent_total += 1
                q.subagent_passed += int(passed)
    return q


@dataclass
class OraclePlanHealth:
    """Section 6.6: does the oracle's own plan actually work?

    The oracle picks on cost and latency alone, assuming every subtask succeeds.
    The baseline runs already call `verify()`, so checking costs nothing -- and
    the answer decides whether regret means what it claims.
    """

    agent_pass_rate: float | None
    oracle_pass_rate: float | None
    n_agent: int
    n_oracle: int

    @property
    def verdict(self) -> str:
        if self.agent_pass_rate is None or self.oracle_pass_rate is None:
            return "insufficient runs"
        gap = self.oracle_pass_rate - self.agent_pass_rate
        if gap < -0.05:
            # The outcome that would quietly break the metric: the cost model is
            # recommending plans that are cheaper AND worse, typically by fanning
            # work out to subagents lacking the context to do it correctly.
            # Regret would then charge the agent for declining to do something
            # that does not work. Not a refinement -- it invalidates the metric
            # on those scenarios.
            return "INVALID: oracle plans pass less often than the agent's own choices"
        if gap > 0.05:
            return "oracle plans pass MORE often; the agent's choices cost quality too"
        return "oracle plans pass as often as the agent's own choices; regret is pure economics"


def oracle_plan_health(agent_traces, oracle_traces) -> OraclePlanHealth:
    def rate(traces):
        traces = [t for t in traces if t.verdicts]
        if not traces:
            return None, 0
        return sum(t.succeeded for t in traces) / len(traces), len(traces)

    a, na = rate(agent_traces)
    o, no = rate(oracle_traces)
    return OraclePlanHealth(agent_pass_rate=a, oracle_pass_rate=o, n_agent=na, n_oracle=no)
