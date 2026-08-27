"""A manifest in, a results file out. The command that produces the table.

Low-level design: low-level-design.md, sections 1.4, 5 and 6.

Every piece this composes already existed -- manifest, runner, oracle, scoring,
aggregation -- and composing them by hand is the same reproducibility problem
calibration had before it got a driver. A table assembled in a notebook cannot be
re-derived, cannot say which manifest or which calibration produced it, and
quietly becomes a different table every time someone reruns a cell.

WHAT A RESULTS FILE MUST CARRY TO BE INTERPRETABLE AT ALL

Three stamps, and a results file missing any of them is uninterpretable rather
than merely undocumented:

  manifest fingerprint  which scenarios, and proof the list was not edited after
  calibration source    which run measured the constants, and at which price sheet
  price sheet date      an undated dollar is not a unit

`Results.write` refuses to omit them.

PER SCENARIO, WHAT GETS RUN

  agent arm      the model decides. This is the measurement.
  baseline arm   the oracle plan, executed. This is regret's denominator-side
                 anchor, and it is measured rather than predicted so that an
                 error in the constants cancels instead of becoming fake regret.
  disclosed arm  optional. The same scenario with the dependency graph stated,
                 which separates "failed to discover the structure" from
                 "discovered it and chose wrong".
  anti-oracle    optional, on a SUBSET. The worst plan in the table. Not a
                 measurement of the model at all -- it is the only experiment
                 that can falsify the derived-ground-truth argument, so it runs
                 on a few scenarios rather than all of them.

Note the baseline arm depends on beta: the oracle plan at beta=0 is usually not
the oracle plan at beta=1. So one baseline run is executed per (scenario, beta)
whose oracle plan differs, and identical plans reuse one run rather than paying
twice for the same execution.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from generator.manifest import Manifest, ScenarioSpec
from generator.oracle import CostModel, all_inline, enumerate_plans, evaluate, unmeasured
from scoring.aggregate import Aggregate, aggregate
from scoring.regret import ScoreCard, model_best, score
from scoring.validate import (
    OrderingReport,
    anti_oracle,
    check_ordering,
    discovery_cost,
    out_of_order_rate,
    plan_compliance,
)

from .calibrate import PriceSheet, TimingModel
from .client import Client
from .runner import Budget, RetryPolicy, run_agent, run_plan
from .trace import Trace

__all__ = ["Results", "run_experiment"]


def _plan_key(plan) -> tuple:
    return (tuple(sorted(plan.inline)), tuple(sorted(tuple(sorted(b)) for b in plan.blocks)))


@dataclass
class Results:
    """One experiment: every card, every aggregate, and what produced them."""

    manifest_fingerprint: str
    manifest_name: str
    calibration_source: str
    price_sheet: dict
    model: str
    betas: tuple[float, ...]
    cards: list[ScoreCard] = field(default_factory=list)
    aggregates: dict = field(default_factory=dict)  # str(beta) -> Aggregate
    ordering: OrderingReport | None = None
    discovery: dict = field(default_factory=dict)  # str(beta) -> DiscoverySplit
    out_of_order: dict = field(default_factory=dict)
    compliance: object = None
    trace_files: list[str] = field(default_factory=list)
    placeholder_constants: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    @property
    def comparable(self) -> bool:
        """Are these numbers in one currency? See `scoring.regret._scale_warning`."""
        return not self.placeholder_constants

    def write(self, path: str | Path) -> Path:
        if not self.manifest_fingerprint or not self.calibration_source:
            raise ValueError(
                "a results file without a manifest fingerprint and a calibration source "
                "cannot be interpreted later; refusing to write one"
            )
        if not self.price_sheet.get("as_of"):
            raise ValueError("the price sheet has no date; an undated dollar is not a unit")
        path = Path(path)
        payload = {
            "manifest": {"name": self.manifest_name, "fingerprint": self.manifest_fingerprint},
            "calibration_source": self.calibration_source,
            "price_sheet": self.price_sheet,
            "model": self.model,
            "betas": list(self.betas),
            "comparable": self.comparable,
            "placeholder_constants": list(self.placeholder_constants),
            "cards": [asdict(c) for c in self.cards],
            "aggregates": {k: str(v) for k, v in self.aggregates.items()},
            "ordering": str(self.ordering) if self.ordering else None,
            "discovery": {k: str(v) for k, v in self.discovery.items()},
            "out_of_order": {k: str(v) for k, v in self.out_of_order.items()},
            "plan_compliance": str(self.compliance) if self.compliance else None,
            "trace_files": self.trace_files,
            "notes": self.notes,
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def report(self) -> str:
        lines = [
            f"model {self.model}   manifest {self.manifest_name} "
            f"({self.manifest_fingerprint})   calibration {self.calibration_source}",
            f"price sheet as of {self.price_sheet.get('as_of', '??')}",
        ]
        if not self.comparable:
            lines.append(
                f"! NOT COMPARABLE: {len(self.placeholder_constants)} constant(s) still "
                f"placeholder ({', '.join(self.placeholder_constants[:4])}"
                f"{', ...' if len(self.placeholder_constants) > 4 else ''}). Regret is a "
                "ratio between two currencies until calibration lands."
            )
        for note in self.notes:
            lines.append(f"! {note}")
        for beta in self.betas:
            lines += ["", f"=== beta = {beta:g} ===", str(self.aggregates[str(beta)])]
            split = self.discovery.get(str(beta))
            if split is not None:
                lines.append(f"discovery  {split}")
        if self.out_of_order:
            lines += ["", "out-of-order edits (the soft-dependency approximation):"]
            lines += [f"  {k}: {v}" for k, v in sorted(self.out_of_order.items())]
        if self.compliance is not None:
            lines += ["", str(self.compliance)]
        if self.ordering is not None:
            lines += ["", "anti-oracle ordering check:", str(self.ordering)]
        return "\n".join(lines)


def run_experiment(
    manifest: Manifest,
    client: Client,
    price: PriceSheet,
    timing: TimingModel,
    cost_model: CostModel,
    out_dir: str | Path,
    *,
    calibration_source: str,
    betas=(0.0, 1.0),
    repeats: int = 1,
    disclosed: bool = True,
    ordering_subset: int = 3,
    max_turns: int = 60,
    budget: Budget | None = None,
    retry: RetryPolicy | None = None,
    use_proxy: bool = True,
    floors: tuple | None = None,
) -> Results:
    """Run a manifest end to end and write a stamped results file.

    `calibration_source` is required, not defaulted. It is the string
    `CostModel.calibrate` stamped onto every measured constant, and without it a
    results file cannot say which run its dollars came from.

    `floors` is the calibration's (dollar, minute) materiality pair
    (`CalibrationResult.floors`); it sets the tolerance at which two plans
    count as the same outcome in the Tier-A test and the implied-beta
    tie-break. Omitted, both run at exact float equality -- which the oracle's
    own provenance notes is not a meaningful test on measured constants.

    Traces are written as they are produced, so a crash halfway through costs the
    remaining scenarios and not the completed ones.
    """
    out_dir = Path(out_dir)
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    betas = tuple(betas)
    results = Results(
        manifest_fingerprint=manifest.fingerprint,
        manifest_name=manifest.name,
        calibration_source=calibration_source,
        price_sheet=asdict(price),
        model=client.model,
        betas=betas,
        placeholder_constants=unmeasured(cost_model),
    )
    shape_by_id: dict[str, str] = {}
    scenarios: dict[str, object] = {}
    agent_traces: list[Trace] = []
    baseline_traces: list[Trace] = []
    forced_traces: list[Trace] = []
    disclosed_cards: dict[float, list[ScoreCard]] = {b: [] for b in betas}
    hidden_cards: dict[float, list[ScoreCard]] = {b: [] for b in betas}
    ordering = OrderingReport()

    def run(spec, plan, tag, *, condition, order=None, disclose=False):
        root = out_dir / "traces" / f"{spec.id}-{tag}"
        kwargs = dict(
            max_turns=max_turns, budget=budget, retry=retry, disclose_dag=disclose,
            proxy_log=(root.parent / f"{spec.id}-{tag}-proxy.jsonl") if use_proxy else None,
        )
        scenario = scenarios[spec.id]
        if plan is None:
            trace = run_agent(scenario, client, root, condition=condition, **kwargs)
        else:
            trace = run_plan(scenario, plan, client, root, order=order,
                             condition=condition, **kwargs)
        results.trace_files.append(str(trace.write(out_dir / "traces" / f"{spec.id}-{tag}.json")))
        if plan is not None:
            forced_traces.append(trace)
        return trace

    for index, spec in enumerate(manifest.specs):
        scenario = spec.build()
        scenarios[spec.id] = scenario
        shape_by_id[spec.id] = spec.shape
        priced = [evaluate(scenario.dag, p, cost_model) for p in enumerate_plans(scenario.dag)]

        for repeat in range(repeats):
            agent = run(spec, None, f"agent-r{repeat}", condition="agent")
            agent_traces.append(agent)

            # One baseline per DISTINCT oracle plan across the beta list. The
            # oracle plan at beta=0 is usually not the plan at beta=1, and paying
            # twice for the same execution would be waste -- but scoring beta=1
            # against beta=0's plan would be measuring the wrong baseline.
            # `model_best` is floor-aware: the estimation gate showed the model
            # fumbles only near-ties, so among plans inside the noise floor the
            # simplest one is executed rather than letting the model call a
            # coin flip.
            best_by_beta = {b: model_best(priced, b, floors) for b in betas}
            baselines: dict[tuple, Trace] = {}
            for b in betas:
                key = _plan_key(best_by_beta[b].plan)
                if key not in baselines:
                    baselines[key] = run(
                        spec, best_by_beta[b].plan, f"oracle-b{b:g}-r{repeat}",
                        condition="oracle-plan", order=best_by_beta[b].order,
                    )
                    baseline_traces.append(baselines[key])

            # ALWAYS-SERIAL IS EXECUTED, NOT PRICED (corrected Aug 25). The
            # beats-all-inline flag used to compare the agent's measured
            # objective against the plan table's all-inline price -- which the
            # estimation gate showed runs ~13% low, making serial artificially
            # hard to beat, a bias toward the headline. One extra run per
            # scenario buys a measurement-vs-measurement flag AND a fresh
            # out-of-sample estimation-gate sample on every scenario (the
            # model's inline prediction is on the card next to this run's
            # measured objective). Reused when a baseline already IS all-inline.
            inline_plan = all_inline(scenario.dag)
            inline_run = baselines.get(_plan_key(inline_plan))
            if inline_run is None:
                inline_run = run(spec, inline_plan, f"inline-r{repeat}",
                                 condition="all-inline")
            inline_ok = inline_run.succeeded and not any(
                n.startswith("PLAN NOT FOLLOWED") for n in inline_run.notes
            )

            arm = run(spec, None, f"disclosed-r{repeat}", condition="agent-disclosed",
                      disclose=True) if disclosed else None

            for b in betas:
                base = baselines[_plan_key(best_by_beta[b].plan)]
                inline_obj = inline_run.objective(price, timing, b) if inline_ok else None
                card = score(agent, base, scenario.dag, cost_model, b, price, timing,
                             priced, floors=floors, measured_all_inline=inline_obj)
                results.cards.append(card)
                hidden_cards[b].append(card)
                if arm is not None:
                    disclosed_cards[b].append(
                        score(arm, base, scenario.dag, cost_model, b, price, timing,
                              priced, floors=floors, measured_all_inline=inline_obj)
                    )

        # The anti-oracle check runs on a subset: it is not a measurement of the
        # model, it is the only experiment that can falsify the method, and a few
        # scenarios settle it.
        if index < ordering_subset:
            beta = betas[-1]
            best = min(priced, key=lambda r: (r.objective(beta), r.plan.k))
            worst = anti_oracle(scenario.dag, cost_model, beta, priced)
            anti = run(spec, worst.plan, "anti", condition="anti-oracle", order=worst.order)
            base = run(spec, best.plan, "anti-baseline", condition="oracle-plan",
                       order=best.order)
            followed = lambda t: not any(
                n.startswith("PLAN NOT FOLLOWED") for n in t.notes
            )
            ordering.checks.append(
                check_ordering(
                    spec.id, beta, best, worst,
                    base.objective(price, timing, beta) if base.succeeded else None,
                    anti.objective(price, timing, beta) if anti.succeeded else None,
                    oracle_ok=base.succeeded, anti_ok=anti.succeeded,
                    oracle_complied=followed(base), anti_complied=followed(anti),
                )
            )

    for b in betas:
        results.aggregates[str(b)] = aggregate(
            hidden_cards[b], shape_by_id.get, beta=b,
            manifest_fingerprint=manifest.fingerprint,
        )
        if disclosed:
            results.discovery[str(b)] = discovery_cost(hidden_cards[b], disclosed_cards[b], b)

    by_shape: dict[str, list[Trace]] = {}
    for trace in agent_traces:
        by_shape.setdefault(shape_by_id.get(trace.scenario_id, "?"), []).append(trace)
    for shape, group in sorted(by_shape.items()):
        results.out_of_order[shape] = out_of_order_rate(group, scenarios)

    # Measured over every forced-plan run, because non-compliance is what makes
    # a baseline excluded and the ordering check inconclusive -- both quiet
    # failures that otherwise look like missing data rather than a finding.
    results.compliance = plan_compliance(forced_traces)
    if results.compliance.rate is not None and results.compliance.rate < 0.5:
        results.notes.append(results.compliance.verdict)
    results.ordering = ordering if ordering.checks else None
    results.write(out_dir / "results.json")
    return results
