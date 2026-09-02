"""The committee audit: run every plan that could win, and grade the calculation.

Design record: TASKS-AND-OPEN-ISSUES.md section 4 item 13. The oracle's
measured role is a COMMITTEE -- an exhaustive enumerator
plus a bounded-error pruner whose one unforgivable failure is excluding a plan
that could actually win. This module is the instrument that grades it, and the
grade is executed, not argued:

  enumerate every legal plan          (the space is finite; no unknown paths)
  -> collapse to predicted-outcome   (at noise-floor resolution wide8's 21,147
     representatives at floor           plans are 36 predicted outcomes)
     resolution
  -> cross-check structural shapes   (on equal-size wide DAGs, record whether
                                        those representatives cover every
                                        node-relabeling allocation class)
  -> EXECUTE each representative      (compliance-gated, proxy-witnessed --
     via the ordinary forced-plan       every row is a certified run of its
     runner                             assigned plan or an explicit exclusion)
  -> scorecard                        (the headline numbers)

THE SCORECARD, prospectively specified here before any audit data exists, so the rule
cannot be tuned to the result:

  champion retention   Is the MEASURED-best plan inside the model's 2-epsilon
                       band? The committee's only unforgivable error, checked
                       directly.
  safe filter rate     Sort the table by PREDICTED objective; r = the measured
                       champion's predicted rank (1 = predicted best), N = table
                       size. x = 1 - r/N is the largest bottom fraction of the
                       calculated table discardable on this scenario without
                       discarding the champion. THE REPORTED X% IS min(x) ACROSS
                       AUDITED SCENARIOS, reported with every (r, N) pair, worst
                       case first, never the mean alone.
  rank agreement       Kendall tau between predicted and measured orderings --
                       the one-number form of "sharp at extremes, fuzzy at ties".
  error by width       |predicted - measured| / measured per plan, grouped by
                       spawn count k. Feeds the model fixes, not the verdict.

Epsilon is not chosen; it is READ OFF the calibration's own estimation gate
(`eps_from_calibration`): the worst relative gap the composed model showed on
its executed plans. An audit that picked its own epsilon would be grading the
committee on a curve the committee drew.

WHAT AN EXCLUDED ROW MEANS. A representative whose run failed verification,
tampered, or did not execute its assigned plan is recorded with its reason and
drops out of the measured ranking -- the scorecard is then marked incomplete
rather than silently computed over survivors. A committee audit that quietly
drops the plans that were hard to execute is auditing nothing.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from generator.oracle import CostModel, enumerate_plans, evaluate

from .calibrate import PriceSheet, TimingModel
from .client import Client
from .runner import Budget, RetryPolicy, run_plan

__all__ = [
    "outcome_key",
    "distinct_outcomes",
    "allocation_shape_key",
    "allocation_shape_classes",
    "committee_band",
    "eps_from_calibration",
    "kendall_tau",
    "AuditRow",
    "AuditResult",
    "run_audit",
    "audit_summary",
]


# ------------------------------------------------------------- pure machinery


def outcome_key(result, floors) -> tuple[int, int]:
    """A plan's predicted (cost, latency) bucket at floor resolution.

    Sharing a key means only that the compositional model cannot resolve the
    plans at these floors.  It does not establish empirical exchangeability;
    the independent structural check below records when a stronger symmetry
    argument is available.
    """
    d, m = floors
    return (round(result.cost / (d or 1e-9)), round(result.latency / (m or 1e-9)))


def distinct_outcomes(results, floors) -> list:
    """One representative per predicted-outcome bucket, deterministically chosen.

    Among plans sharing an outcome, the representative is the simplest (fewest
    subagents, then lexicographic blocks) -- the same conservative tie-break
    `model_best` uses, so the audit executes the plan scoring would have.
    Ordered by predicted cost so audit tags are stable across re-runs.
    """
    def plan_sig(r):
        return (r.plan.k, tuple(sorted(tuple(sorted(b)) for b in r.plan.blocks)),
                tuple(sorted(r.plan.inline)))

    by_key: dict[tuple, object] = {}
    for r in sorted(results, key=plan_sig):
        by_key.setdefault(outcome_key(r, floors), r)
    return sorted(by_key.values(), key=lambda r: (r.cost, r.latency))


def allocation_shape_key(dag, plan) -> tuple[int, tuple[int, ...]] | None:
    """Structural allocation class for an equal-size, independent-work DAG.

    For this deliberately narrow case, node identities do not reach the oracle:
    a plan is characterized by how many tasks remain inline and the multiset of
    delegated block sizes.  The key is undefined for dependency edges or
    heterogeneous node sizes; treating those plans as relabeling-equivalent
    would silently assume away structure or workload differences.

    This key is independent of the calibrated cost model and its materiality
    floors.  It therefore cross-checks -- rather than defines -- the audit's
    historical predicted-outcome collapse.
    """
    if dag.edges or len({node.size for node in dag.nodes}) != 1:
        return None
    return (
        len(plan.inline),
        tuple(sorted(len(block) for block in plan.blocks)),
    )


def allocation_shape_classes(dag, plans) -> dict[tuple[int, tuple[int, ...]], list]:
    """Group every legal plan by :func:`allocation_shape_key` when defined."""
    groups: dict[tuple[int, tuple[int, ...]], list] = {}
    for plan in plans:
        key = allocation_shape_key(dag, plan)
        if key is None:
            return {}
        groups.setdefault(key, []).append(plan)
    return groups


def committee_band(reps, beta: float, eps: float) -> list:
    """The plans that could win: predicted objective within (1 + 2*eps) of the
    predicted minimum. With per-plan relative error <= eps, the true optimum
    cannot sit outside this band -- arithmetic on a measured bound, not trust."""
    lo = min(r.objective(beta) for r in reps)
    return [r for r in reps if r.objective(beta) <= lo * (1.0 + 2.0 * eps)]


def eps_from_calibration(calibration) -> float | None:
    """The audit's epsilon is the calibration's own worst composed-model gap.

    Read from the estimation gate's per-arm relative errors, both axes, largest
    magnitude. None when the gate never ran -- an audit without a measured
    epsilon has no band and must run the full table.
    """
    comp = (calibration.diagnostics or {}).get("composition") or {}
    if not comp.get("checked"):
        return None
    gaps = []
    for arm in comp.get("arms", ()):
        for k in ("rel_gap_dollars", "rel_gap_minutes"):
            if arm.get(k) is not None:
                gaps.append(abs(arm[k]))
    return max(gaps) if gaps else None


def kendall_tau(xs, ys) -> float | None:
    """Kendall tau-a over paired values. O(n^2), no dependencies; ties count as
    disagreement-neutral. None below three pairs -- a tau of two points is a
    coin report."""
    n = len(xs)
    if n < 3:
        return None
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (xs[i] - xs[j]) * (ys[i] - ys[j])
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    total = n * (n - 1) // 2
    return (concordant - discordant) / total if total else None


# ------------------------------------------------------------------ the audit


@dataclass
class AuditRow:
    """One predicted-outcome representative: model versus execution."""

    tag: str
    k: int
    inline: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    shape_key: dict = field(default_factory=dict)
    shape_class_size: int | None = None
    predicted_cost: float = 0.0
    predicted_latency: float = 0.0
    predicted_objective: float = 0.0
    in_band: bool = False
    measured_cost: float | None = None
    measured_latency: float | None = None
    measured_objective: float | None = None
    rel_gap_objective: float | None = None  # (predicted - measured) / measured
    excluded: bool = False
    reason: str = ""


@dataclass
class AuditResult:
    """One scenario's committee grade, and every number behind it."""

    scenario_id: str
    beta: float
    eps: float | None
    floors: tuple
    n_plans: int  # raw enumeration size
    n_outcomes: int  # predicted-outcome representatives at floor resolution
    n_band: int
    rows: list = field(default_factory=list)
    # Independent structural cross-check for equal-size wide DAGs.  The audit
    # still selects rows by predicted outcome; these fields say whether that
    # selection happens to cover every relabeling-defined allocation shape.
    n_shape_classes: int | None = None
    n_shape_represented: int | None = None
    shape_complete: bool | None = None
    shape_definition: str = ""
    shape_classes: list = field(default_factory=list)
    # the scorecard
    champion_tag: str | None = None
    champion_predicted_rank: int | None = None  # r: 1 = predicted best
    safe_filter_x: float | None = None  # 1 - r/N
    champion_in_band: bool | None = None
    tau: float | None = None
    complete: bool = False  # every representative measured
    dry_run: bool = False
    spend_dollars: float | None = None
    notes: list = field(default_factory=list)

    def write(self, out_dir) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "audit.json"
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True, default=str),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "AuditResult":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["rows"] = [AuditRow(**r) for r in raw["rows"]]
        raw["floors"] = tuple(raw["floors"])
        return cls(**raw)

    def report(self) -> str:
        lines = [
            f"committee audit: {self.scenario_id}  beta={self.beta:g}  "
            f"eps={'-' if self.eps is None else f'{self.eps:.0%}'}",
            f"  {self.n_plans} plans -> {self.n_outcomes} predicted-outcome reps -> "
            f"{self.n_band} in the 2-eps band",
        ]
        if self.n_shape_classes is not None:
            lines.append(
                "  structural shape cross-check: "
                f"{self.n_shape_represented}/{self.n_shape_classes} represented "
                f"(complete: {self.shape_complete})"
            )
        for n in self.notes:
            lines.append(f"  ! {n}")
        for row in self.rows:
            mark = "*" if row.tag == self.champion_tag else " "
            band = "band" if row.in_band else "    "
            if row.excluded:
                right = f"EXCLUDED: {row.reason}"
            elif row.measured_objective is None:
                right = "(not run)"
            else:
                right = (f"measured {row.measured_objective:.4f} "
                         f"({row.rel_gap_objective:+.0%})")
            lines.append(f"  {mark} {row.tag:<12} k={row.k}  {band}  "
                         f"predicted {row.predicted_objective:.4f}  {right}")
        if self.dry_run:
            lines.append(
                f"  DRY RUN -- predicted spend to execute: "
                f"${sum(r.predicted_cost for r in self.rows):.2f}"
            )
            return "\n".join(lines)
        if self.champion_tag is not None:
            lines += [
                "",
                f"  measured champion: {self.champion_tag}  "
                f"predicted rank {self.champion_predicted_rank}/{self.n_outcomes}  "
                f"-> safe filter x = {self.safe_filter_x:.0%}",
                f"  champion in band: {self.champion_in_band}   "
                f"rank agreement tau = "
                f"{'-' if self.tau is None else f'{self.tau:+.2f}'}   "
                f"complete: {self.complete}   spend ${self.spend_dollars:.2f}",
            ]
        return "\n".join(lines)


def _plan_of(row_or_rep):
    return row_or_rep.plan


def run_audit(
    scenario,
    client: "Client | None",
    price: PriceSheet,
    timing: TimingModel,
    cm: CostModel,
    out_dir,
    *,
    beta: float = 1.0,
    eps: float | None,
    floors: tuple,
    band_only: bool = False,
    dry_run: bool = False,
    max_turns: int = 60,
    budget: Budget | None = None,
    retry: RetryPolicy | None = None,
    runner=run_plan,
) -> AuditResult:
    """Execute one scenario's committee audit and write audit.json.

    `dry_run` prices the table and stops before spending -- the reproducible
    form of "what would this audit cost", and the path tests exercise offline.
    `runner` is injectable so the gating logic is testable without a socket.
    `band_only` restricts execution to the 2-eps band (cheaper follow-up
    audits); the FIRST audit of a scenario should run the full table, because
    the safe-filter rate is only meaningful if the discarded region was
    actually checked.
    """
    out_dir = Path(out_dir)
    dag = scenario.dag
    plans = enumerate_plans(dag)
    priced = [evaluate(dag, p, cm) for p in plans]
    reps = distinct_outcomes(priced, floors)
    shape_groups = allocation_shape_classes(dag, plans)
    represented_shape_keys = {
        allocation_shape_key(dag, rep.plan) for rep in reps
    } if shape_groups else set()
    band = committee_band(reps, beta, eps) if eps is not None else list(reps)
    band_keys = {outcome_key(r, floors) for r in band}
    to_run = band if band_only else reps

    result = AuditResult(
        scenario_id=scenario.id,
        beta=beta,
        eps=eps,
        floors=tuple(floors),
        n_plans=len(plans),
        n_outcomes=len(reps),
        n_band=len(band),
        n_shape_classes=len(shape_groups) if shape_groups else None,
        n_shape_represented=len(represented_shape_keys) if shape_groups else None,
        shape_complete=(
            represented_shape_keys == set(shape_groups)
            and len(reps) == len(shape_groups)
            if shape_groups else None
        ),
        shape_definition=(
            "(inline task count, sorted delegated block task counts)"
            if shape_groups else ""
        ),
        dry_run=dry_run,
    )
    if eps is None:
        result.notes.append(
            "no epsilon from the calibration's estimation gate; band = full table"
        )
    if band_only:
        result.notes.append(
            "band-only audit: the safe-filter rate is NOT computable (the "
            "discarded region was not checked); champion retention only"
        )

    # rows in predicted-objective order: rank 1 first, tags stable
    ordered = sorted(to_run, key=lambda r: (r.objective(beta), r.cost))
    rank_of = {id(r): i + 1 for i, r in enumerate(
        sorted(reps, key=lambda r: (r.objective(beta), r.cost)))}
    for i, rep in enumerate(ordered):
        shape_key = allocation_shape_key(dag, rep.plan)
        result.rows.append(AuditRow(
            tag=f"plan{i:03d}-k{rep.plan.k}",
            k=rep.plan.k,
            inline=sorted(rep.plan.inline),
            blocks=[sorted(b) for b in rep.plan.blocks],
            shape_key=(
                {
                    "inline_tasks": shape_key[0],
                    "delegated_block_sizes": list(shape_key[1]),
                }
                if shape_key is not None else {}
            ),
            shape_class_size=(
                len(shape_groups[shape_key]) if shape_key is not None else None
            ),
            predicted_cost=rep.cost,
            predicted_latency=rep.latency,
            predicted_objective=rep.objective(beta),
            in_band=outcome_key(rep, floors) in band_keys,
        ))

    if shape_groups:
        row_tag_by_key = {
            allocation_shape_key(dag, rep.plan): row.tag
            for row, rep in zip(result.rows, ordered)
        }
        result.shape_classes = [
            {
                "inline_tasks": key[0],
                "delegated_block_sizes": list(key[1]),
                "labeled_plan_count": len(members),
                "predicted_outcome_represented": key in represented_shape_keys,
                "execution_row": row_tag_by_key.get(key),
            }
            for key, members in sorted(shape_groups.items())
        ]

    if dry_run:
        result.write(out_dir)
        return result

    spend = 0.0
    measured: list[tuple[AuditRow, float]] = []
    for row, rep in zip(result.rows, ordered):
        trace = runner(
            scenario, rep.plan, client, out_dir / "workspaces" / row.tag,
            order=rep.order, condition="audit",
            max_turns=max_turns, budget=budget, retry=retry,
            proxy_log=out_dir / f"proxy-{row.tag}.jsonl",
        )
        trace.write(out_dir / f"trace-{row.tag}.json")
        spend += trace.dollars(price)
        bad = [n for n in trace.notes if n.startswith("PROXY MISMATCH")]
        if trace.tampered_tests:
            row.excluded, row.reason = True, "edited generated suites"
        elif bad:
            row.excluded, row.reason = True, f"proxy cross-check failed: {bad[0][:80]}"
        elif not trace.succeeded:
            row.excluded, row.reason = True, "run did not verify"
        elif any(n.startswith("PLAN NOT FOLLOWED") for n in trace.notes):
            row.excluded, row.reason = True, "did not execute its assigned plan"
        else:
            row.measured_cost = trace.dollars(price)
            row.measured_latency = trace.analytic_minutes(timing)
            row.measured_objective = row.measured_cost + beta * row.measured_latency
            row.rel_gap_objective = (
                (row.predicted_objective - row.measured_objective)
                / row.measured_objective
            )
            measured.append((row, rank_of[id(rep)]))

    result.spend_dollars = spend
    result.complete = all(not r.excluded and r.measured_objective is not None
                          for r in result.rows)
    if not result.complete:
        excluded = [r.tag for r in result.rows if r.excluded]
        result.notes.append(
            f"INCOMPLETE: {len(excluded)} representative(s) excluded "
            f"({', '.join(excluded)}); the scorecard covers the measured rows only"
        )
    if measured:
        champ, champ_rank = min(measured, key=lambda rw: rw[0].measured_objective)
        result.champion_tag = champ.tag
        result.champion_predicted_rank = champ_rank
        result.champion_in_band = champ.in_band
        if not band_only:
            result.safe_filter_x = 1.0 - champ_rank / result.n_outcomes
        pairs = [(r.predicted_objective, r.measured_objective) for r, _ in measured]
        result.tau = kendall_tau([p for p, _ in pairs], [m for _, m in pairs])
    result.write(out_dir)
    return result


def audit_summary(audit_paths) -> dict:
    """Combine per-scenario audits into the headline numbers.

    THE X% RULE, applied: X = min over audited scenarios of each scenario's
    safe-filter x -- worst case, never the mean. Reported next to every (r, N)
    pair and the retention count, and refusing a value when any contributing
    audit is band-only or incomplete on the champion side.
    """
    audits = [AuditResult.load(p) for p in audit_paths]
    rows = []
    xs = []
    retained = 0
    caveats = []
    for a in audits:
        rows.append({
            "scenario": a.scenario_id,
            "r": a.champion_predicted_rank,
            "N": a.n_outcomes,
            "x": a.safe_filter_x,
            "in_band": a.champion_in_band,
            "tau": a.tau,
            "complete": a.complete,
        })
        if a.champion_in_band:
            retained += 1
        if a.safe_filter_x is not None and a.complete:
            xs.append(a.safe_filter_x)
        else:
            caveats.append(
                f"{a.scenario_id}: no certified x "
                f"({'band-only' if a.safe_filter_x is None else 'incomplete'})"
            )
    return {
        "n_audits": len(audits),
        "champion_retention": f"{retained}/{len(audits)}",
        "safe_filter_X": min(xs) if xs and len(xs) == len(audits) else None,
        "per_scenario": sorted(rows, key=lambda r: (r["x"] is None, r["x"])),
        "mean_tau": statistics.fmean(t for a in audits if (t := a.tau) is not None)
        if any(a.tau is not None for a in audits) else None,
        "caveats": caveats,
    }
