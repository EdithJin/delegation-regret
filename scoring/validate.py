"""Checks on the method itself, not on any model.

Low-level design: low-level-design.md, "The check that keeps it honest" and 6.5.

Everything in `regret.py` assumes two things the rest of the repo cannot prove:
that computing a plan's cost is a substitute for running it, and that regret is
measuring allocation rather than discovery. Each has one cheap experiment
attached, and both are here.

**The anti-oracle check.** The whole derived-ground-truth argument is that you can
price 21,147 plans from a handful of measured primitives instead of running them.
That is only valid if the computation is right, so on a small subset execute the
oracle plan AND the WORST plan in the table, and verify reality agrees about
which is better. If the model says A beats B and reality disagrees, the cost
model is wrong -- and it is worth a few dollars on three scenarios to learn that
now rather than in review. This is the only experiment in the repo that can
falsify the method rather than measure a model.

**The discovery-cost split.** The oracle knows the dependency structure the agent
must infer, so regret bundles "allocated badly" with "paid to find out". The
disclosed condition renders the same scenario with the dependency list appended;
the gap between the two arms is what discovery cost. Without it the limitation is
merely admitted, which is weaker than measured.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from generator.dag import DAG
from generator.oracle import CostModel, PlanResult, Plan, enumerate_plans, evaluate

from .regret import ScoreCard

__all__ = [
    "anti_oracle",
    "OrderingCheck",
    "check_ordering",
    "DiscoverySplit",
    "discovery_cost",
    "OutOfOrder",
    "out_of_order_edits",
    "out_of_order_rate",
    "PlanCompliance",
    "plan_compliance",
]


# ------------------------------------------------------- the anti-oracle check


def anti_oracle(dag: DAG, cm: CostModel, beta: float, results=None) -> PlanResult:
    """The WORST plan in the table at this beta, and a legal one.

    Not the same as `max_fanout`, which is merely a thoughtless strategy and is
    often nowhere near worst. This is the genuine maximum of the objective over
    the feasible set -- the thing the cost model is most confident is bad, and
    therefore the sharpest test of whether it is confident about the right
    things. A cost model that gets the extremes wrong is not to be trusted in the
    middle.
    """
    if results is None:
        results = [evaluate(dag, p, cm) for p in enumerate_plans(dag)]
    return max(results, key=lambda r: (r.objective(beta), -r.plan.k))


@dataclass(frozen=True)
class OrderingCheck:
    """Did reality agree with the cost model about which plan is better?"""

    scenario_id: str
    beta: float
    predicted_oracle: float
    predicted_anti: float
    measured_oracle: float | None
    measured_anti: float | None
    agreed: bool | None
    reason: str = ""

    @property
    def predicted_gap(self) -> float:
        return self.predicted_anti - self.predicted_oracle

    @property
    def measured_gap(self) -> float | None:
        if self.measured_oracle is None or self.measured_anti is None:
            return None
        return self.measured_anti - self.measured_oracle

    def __str__(self) -> str:
        if self.agreed is None:
            return f"{self.scenario_id}: inconclusive -- {self.reason}"
        verdict = "agreed" if self.agreed else "DISAGREED"
        return (
            f"{self.scenario_id}: predicted gap {self.predicted_gap:+.4f}, "
            f"measured {self.measured_gap:+.4f} -> {verdict}"
        )


def check_ordering(
    scenario_id: str,
    beta: float,
    oracle_result: PlanResult,
    anti_result: PlanResult,
    measured_oracle: float | None,
    measured_anti: float | None,
    *,
    oracle_ok: bool = True,
    anti_ok: bool = True,
    oracle_complied: bool = True,
    anti_complied: bool = True,
    tol: float = 1e-9,
) -> OrderingCheck:
    """Compare the predicted ranking against the measured one.

    Only the SIGN is checked, not the magnitude. The cost model's job here is to
    rank plans, and it can be wrong about how much worse the anti-oracle is while
    still being right that it is worse -- which is all the plan selection needs.
    Testing magnitude would fail the check for being imprecise rather than for
    being wrong.

    A run that did not verify makes the comparison inconclusive rather than
    failed: an anti-oracle plan that produced broken artifacts tells you nothing
    about the ordering of their costs.

    TWO MORE WAYS TO BE INCONCLUSIVE, both of which would otherwise be read as
    the cost model failing.

    A run that did not EXECUTE THE PLAN IT WAS HANDED is not a measurement of
    that plan. If a model ignores the directive and does the same thing either
    way, the two runs cost the same because they *were* the same run -- which
    says nothing about whether the plans differ in cost. Caught here rather than
    reported as a ranking error, because a ranking error invalidates the whole
    method and this does not.

    A TIE is no signal either. Two measured objectives within `tol` mean the
    experiment could not distinguish the plans, not that the model ranked them
    backwards.
    """
    if not oracle_complied or not anti_complied:
        which = "oracle" if not oracle_complied else "anti-oracle"
        return OrderingCheck(
            scenario_id, beta,
            oracle_result.objective(beta), anti_result.objective(beta),
            measured_oracle, measured_anti, None,
            f"the {which} run did not execute the plan it was handed",
        )
    if not oracle_ok or not anti_ok:
        which = "oracle" if not oracle_ok else "anti-oracle"
        return OrderingCheck(
            scenario_id, beta,
            oracle_result.objective(beta), anti_result.objective(beta),
            measured_oracle, measured_anti, None,
            f"the {which} run did not verify",
        )
    if measured_oracle is None or measured_anti is None:
        return OrderingCheck(
            scenario_id, beta,
            oracle_result.objective(beta), anti_result.objective(beta),
            measured_oracle, measured_anti, None, "a measured objective is missing",
        )
    if abs(measured_anti - measured_oracle) <= tol:
        return OrderingCheck(
            scenario_id, beta,
            oracle_result.objective(beta), anti_result.objective(beta),
            measured_oracle, measured_anti, None,
            f"measured objectives are within {tol:g}; the runs did not distinguish the plans",
        )
    return OrderingCheck(
        scenario_id,
        beta,
        oracle_result.objective(beta),
        anti_result.objective(beta),
        measured_oracle,
        measured_anti,
        agreed=measured_anti > measured_oracle,
    )


@dataclass
class OrderingReport:
    """The subset result, and what it licenses.

    Deliberately blunt about the conclusion, because the temptation is to report
    "mostly agreed" and move on. If the ordering does not hold, every computed
    plan cost in the results is suspect, and that has to be said rather than
    softened.
    """

    checks: list[OrderingCheck] = field(default_factory=list)

    @property
    def conclusive(self) -> list[OrderingCheck]:
        return [c for c in self.checks if c.agreed is not None]

    @property
    def verdict(self) -> str:
        done = self.conclusive
        if not done:
            return "INCONCLUSIVE: no scenario produced two verified runs"
        wrong = [c for c in done if not c.agreed]
        if not wrong:
            return (
                f"ordering held on {len(done)}/{len(done)} scenarios; computing plan costs "
                "instead of running them is supported on this evidence"
            )
        return (
            f"ORDERING FAILED on {len(wrong)}/{len(done)} scenarios "
            f"({', '.join(c.scenario_id for c in wrong)}). The cost model ranks plans "
            "wrongly, so every computed plan cost downstream of it is suspect -- this "
            "invalidates the derived-ground-truth argument until the model is fixed."
        )

    def __str__(self) -> str:
        return "\n".join([*(str(c) for c in self.checks), "", self.verdict])


# ------------------------------------------------- can the model follow a plan?


@dataclass(frozen=True)
class PlanCompliance:
    """How often a forced-plan run actually executed the plan it was handed.

    THIS SILENTLY GATES THE WHOLE METRIC, which is why it is measured rather than
    assumed. Two things depend on a model being able to follow a plan directive:

      * the regret BASELINE. `run_plan` states the plan in the prompt and checks
        afterwards; a non-compliant baseline is excluded, because a run that did
        something else is not a measurement of the plan it claims to be. So a
        model that cannot follow a directive produces no regret at all -- not a
        worse number, no number.
      * the ANTI-ORACLE CHECK. If the oracle and anti-oracle runs both ignore
        their plans they are the same run, their costs tie, and the check is
        inconclusive. The one experiment that can falsify the method never
        actually runs.

    Both failures are quiet: the first shows up as a rising exclusion count, the
    second as a permanently inconclusive verdict. Neither says "this model cannot
    follow instructions", which is the actual finding and a reportable one.

    A low rate is not a reason to relax the compliance check. It is a reason to
    say so in the findings: regret over execution plans is only measurable on models
    that can be told which plan to execute.
    """

    complied: int
    total: int
    by_condition: dict = field(default_factory=dict)

    @property
    def rate(self) -> float | None:
        return self.complied / self.total if self.total else None

    @property
    def verdict(self) -> str:
        if self.rate is None:
            return "no forced-plan runs to judge"
        if self.rate >= 0.95:
            return f"plan compliance {self.rate:.0%} ({self.complied}/{self.total})"
        if self.rate >= 0.5:
            return (
                f"plan compliance {self.rate:.0%} ({self.complied}/{self.total}) -- "
                "excluded baselines will thin the sample; report the rate"
            )
        return (
            f"PLAN COMPLIANCE {self.rate:.0%} ({self.complied}/{self.total}). This model "
            "largely cannot execute a stated plan, so most scenarios have no valid "
            "baseline and the anti-oracle check cannot run. Regret is not measurable "
            "here, and that is the finding rather than a gap in the data."
        )

    def __str__(self) -> str:
        extra = "  ".join(
            f"{k} {v[0]}/{v[1]}" for k, v in sorted(self.by_condition.items())
        )
        return self.verdict + (f"   [{extra}]" if extra else "")


def plan_compliance(traces) -> PlanCompliance:
    """Count forced-plan runs that followed their directive.

    Only runs that were GIVEN a plan are counted: an ordinary agent run has no
    plan to comply with, and including it would dilute the rate toward 100% with
    runs that were never tested.
    """
    complied = total = 0
    by_condition: dict[str, list] = {}
    for trace in traces:
        if not any(
            n.startswith("PLAN NOT FOLLOWED") for n in trace.notes
        ) and trace.condition not in _FORCED_CONDITIONS:
            continue
        if trace.condition not in _FORCED_CONDITIONS:
            continue
        total += 1
        ok = not any(n.startswith("PLAN NOT FOLLOWED") for n in trace.notes)
        complied += int(ok)
        bucket = by_condition.setdefault(trace.condition, [0, 0])
        bucket[0] += int(ok)
        bucket[1] += 1
    return PlanCompliance(
        complied=complied,
        total=total,
        by_condition={k: tuple(v) for k, v in by_condition.items()},
    )


# Conditions produced by `run_plan`. An ordinary agent run is not among them.
_FORCED_CONDITIONS = frozenset(
    {"oracle-plan", "anti-oracle", "all-inline", "calibrate-serial",
     "calibrate-fanout", "calibrate-bundled"}
)


# --------------------------------------------- the soft-dependency measurement


@dataclass(frozen=True)
class OutOfOrder:
    """How often a run edited a successor before its predecessor was correct.

    The oracle models edges as strictly BLOCKING. Under bug injection they are
    not: an agent can open a successor module, find its defect, and fix it
    correctly while the predecessor is still broken -- it simply cannot *confirm*
    the fix. The honest formulation is "no verified result for the successor until
    the predecessor is correct", not "no work on the successor".

    So a real agent can beat the "optimal" plan, and the design says to check the
    traces for out-of-order edits and REPORT THE RATE rather than assume it is
    zero. `negative_regret_scenarios` in the aggregate is the downstream
    signature; this is the direct measurement.
    """

    edits: int
    out_of_order: int
    pairs_at_risk: int

    @property
    def rate(self) -> float | None:
        return self.out_of_order / self.edits if self.edits else None

    def __str__(self) -> str:
        if self.rate is None:
            return "no module writes observed"
        return (
            f"{self.out_of_order}/{self.edits} module writes were out of order "
            f"({self.rate:.1%}) across {self.pairs_at_risk} dependency pair(s)"
        )


def out_of_order_edits(trace, scenario) -> OutOfOrder:
    """Count writes to a node made before every predecessor's last write.

    "Before its predecessor was last written" is the operative test, not "before
    it was first written": a predecessor rewritten later was wrong until then, so
    a successor edited in between was still working against broken upstream code.

    A write with no recorded predecessor write counts as out of order too -- the
    successor was edited while the predecessor had not been touched at all, which
    is the strongest version of the same thing.

    Wide scenarios have no dependency pairs and therefore no risk; the count is
    reported alongside so a 0% rate on `wide` is not mistaken for evidence.
    """
    from generator.templates import module_path

    by_node: dict[str, list[float]] = {}
    for event in trace.write_events:
        for node_id in scenario.dag.ids:
            if event.path == module_path(node_id):
                by_node.setdefault(node_id, []).append(event.t)

    last = {n: max(ts) for n, ts in by_node.items()}
    pairs = sum(len(scenario.dag.preds[n]) for n in scenario.dag.ids)
    edits = out = 0
    for node_id, times in by_node.items():
        preds = scenario.dag.preds[node_id]
        if not preds:
            continue
        for t in times:
            edits += 1
            if any(p not in last or t < last[p] for p in preds):
                out += 1
    return OutOfOrder(edits=edits, out_of_order=out, pairs_at_risk=pairs)


def out_of_order_rate(traces, scenarios) -> OutOfOrder:
    """Pooled across runs. `scenarios` maps scenario_id -> Scenario."""
    edits = out = pairs = 0
    for trace in traces:
        scenario = scenarios.get(trace.scenario_id)
        if scenario is None:
            continue
        one = out_of_order_edits(trace, scenario)
        edits += one.edits
        out += one.out_of_order
        pairs += one.pairs_at_risk
    return OutOfOrder(edits=edits, out_of_order=out, pairs_at_risk=pairs)


# ----------------------------------------------------- the discovery-cost split


@dataclass(frozen=True)
class DiscoverySplit:
    """Regret with the structure hidden versus disclosed.

    The oracle is clairvoyant: it knows the dependency graph the agent has to
    infer from imports or from where the suite fails. So hidden-arm regret is
    allocation error PLUS whatever discovery cost. The disclosed arm removes the
    inference, and the difference is the part of regret that was never about
    delegation at all.
    """

    beta: float
    hidden_regret: float | None
    disclosed_regret: float | None
    n_hidden: int
    n_disclosed: int

    @property
    def discovery_share(self) -> float | None:
        """How much of hidden-arm regret disclosure removes, as a fraction.

        None when either arm is empty or hidden regret is not positive -- a
        negative or zero denominator makes the share meaningless rather than
        large, and reporting a ratio there would be the same class of error as
        dividing measured dollars by notional ones.
        """
        if self.hidden_regret is None or self.disclosed_regret is None:
            return None
        if self.hidden_regret <= 0:
            return None
        return (self.hidden_regret - self.disclosed_regret) / self.hidden_regret

    def __str__(self) -> str:
        if self.hidden_regret is None or self.disclosed_regret is None:
            return f"beta={self.beta:g}: not enough scored runs in both arms"
        share = self.discovery_share
        tail = (
            f"; disclosure removes {share:.0%} of it"
            if share is not None
            else "; share undefined (hidden regret is not positive)"
        )
        return (
            f"beta={self.beta:g}: hidden {self.hidden_regret:+.3f} "
            f"(n={self.n_hidden})  disclosed {self.disclosed_regret:+.3f} "
            f"(n={self.n_disclosed}){tail}"
        )


def discovery_cost(
    hidden: list[ScoreCard], disclosed: list[ScoreCard], beta: float | None = None
) -> DiscoverySplit:
    """Compare the two arms on the scenarios BOTH arms scored.

    Restricted to the intersection on purpose. If one arm excluded a scenario the
    other kept, comparing arm means would compare different scenario sets, and on
    shapes whose spread differs by a factor of four that difference alone could
    produce the whole effect.
    """
    def scored(cards):
        return {c.scenario_id: c for c in cards if not c.excluded and c.regret is not None}

    a, b = scored(hidden), scored(disclosed)
    shared = sorted(set(a) & set(b))
    if beta is None:
        betas = {c.beta for c in list(hidden) + list(disclosed)}
        beta = next(iter(betas), 0.0) if len(betas) <= 1 else float("nan")
    if not shared:
        return DiscoverySplit(beta, None, None, len(a), len(b))
    return DiscoverySplit(
        beta=beta,
        hidden_regret=statistics.fmean(a[s].regret for s in shared),
        disclosed_regret=statistics.fmean(b[s].regret for s in shared),
        n_hidden=len(shared),
        n_disclosed=len(shared),
    )
