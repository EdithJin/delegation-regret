"""From score cards to a result: per class first, pooled second, with intervals.

Low-level design: low-level-design.md, sections 1.2, 1.4 and 6.

Regret on one scenario is not a result. It is one draw from a stochastic policy
on one instance of a randomly generated structure, and reporting it alone would
present sampling noise as a finding.

THREE RULES, AND EACH REPLACES A TEMPTING SHORTCUT.

**Per class is primary; pooled is secondary.** The four shape classes have
opposite correct answers -- fan-out is free on `wide` and pure loss on `chain` --
so a pooled mean is a weighted average whose weights are the class mix. Left
unpinned that mix is a free parameter that silently sets what always-serial
scores; pinned in `generator.manifest`, it is at least visible. Reporting per
class as primary means a reader who distrusts the mix can reweight it, which they
cannot do from a pooled number.

**The bootstrap resamples SCENARIOS, not runs.** Repeats of one scenario are
correlated -- same structure, same defects, same seed -- so treating each run as
an independent observation understates the interval, and it understates it worse
the more repeats you add. That is the wrong direction: more data would look like
more certainty while telling you nothing new about the population of scenarios.
So the resampling unit is the scenario cluster, and repeats within a cluster are
averaged first.

**Exclusions are reported, never dropped.** A model whose runs keep failing is
telling you something, and a quietly shrinking denominator is how that signal
gets lost. Every summary carries the excluded count and the reasons.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass, field

from .regret import ScoreCard

__all__ = [
    "ClassSummary",
    "Aggregate",
    "aggregate",
    "cluster_bootstrap",
    "beats_all_inline_rate",
]


def _mean(xs) -> float | None:
    xs = list(xs)
    return statistics.fmean(xs) if xs else None


def cluster_bootstrap(
    clusters: list[list[float]],
    *,
    resamples: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float] | None:
    """Percentile bootstrap over CLUSTERS, each cluster averaged first.

    `clusters` is one list per scenario, holding that scenario's repeat values.
    Resampling clusters with replacement and averaging within them is what keeps
    the interval honest when repeats are correlated: the sample size that matters
    is the number of scenarios, not the number of runs.

    Seeded, because an interval that moves between two runs of the analysis is
    not a number anyone can check.
    """
    usable = [c for c in clusters if c]
    if len(usable) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        draw = [usable[rng.randrange(len(usable))] for _ in usable]
        means.append(statistics.fmean(statistics.fmean(c) for c in draw))
    means.sort()
    lo_i = int((1 - level) / 2 * resamples)
    hi_i = min(resamples - 1, int((1 + level) / 2 * resamples))
    return (means[lo_i], means[hi_i])


def beats_all_inline_rate(cards: list[ScoreCard]) -> tuple[int, int]:
    """(beat, of) -- "does this model beat always-serial?" as a pass/fail count.

    A first-class number rather than something derived later. A metric no trivial
    policy can top is worth more than a leaderboard, and if a frontier model does
    not clear always-serial, that is the finding.
    """
    scored = [c for c in cards if not c.excluded and c.beat_all_inline is not None]
    return sum(1 for c in scored if c.beat_all_inline), len(scored)


@dataclass(frozen=True)
class ClassSummary:
    """One shape class at one beta."""

    shape: str
    beta: float
    n_scenarios: int
    n_runs: int
    n_excluded: int
    mean_regret: float | None
    ci95: tuple[float, float] | None
    negative_regret_scenarios: int
    beat_all_inline: tuple[int, int]
    mean_agent_k: float | None
    mean_oracle_k: float | None
    tier_a_scenarios: int = 0
    tier_a_mean_regret: float | None = None
    exclusion_reasons: dict = field(default_factory=dict)
    comparable: bool = True

    def __str__(self) -> str:
        if self.mean_regret is None:
            return (
                f"{self.shape:<9} beta={self.beta:<5g} "
                f"-- no scorable scenarios ({self.n_excluded} excluded)"
            )
        ci = f"[{self.ci95[0]:+.3f}, {self.ci95[1]:+.3f}]" if self.ci95 else "(too few clusters)"
        beat, of = self.beat_all_inline
        tier = (
            f"  tierA {self.tier_a_scenarios}/{self.n_scenarios}"
            + (f" @{self.tier_a_mean_regret:+.3f}" if self.tier_a_mean_regret is not None else "")
        )
        return (
            f"{self.shape:<9} beta={self.beta:<5g} regret {self.mean_regret:+.3f} {ci:<20}"
            f" k={self.mean_agent_k:.1f} vs oracle {self.mean_oracle_k:.1f}"
            f"  beats-serial {beat}/{of}{tier}"
            f"  excluded {self.n_excluded}"
            + ("" if self.comparable else "   [NOT COMPARABLE]")
        )


@dataclass(frozen=True)
class Aggregate:
    """Per class first, pooled second, with everything that qualifies them."""

    beta: float
    per_class: tuple[ClassSummary, ...]
    pooled_regret: float | None
    pooled_ci95: tuple[float, float] | None
    n_scenarios: int
    n_excluded: int
    manifest_fingerprint: str = ""
    comparable: bool = True
    warnings: tuple[str, ...] = ()

    def __str__(self) -> str:
        lines = []
        if self.manifest_fingerprint:
            lines.append(f"manifest {self.manifest_fingerprint}")
        for warning in self.warnings:
            lines.append(f"! {warning}")
        lines += [str(c) for c in self.per_class]
        if self.pooled_regret is None:
            lines.append("pooled    -- nothing scorable")
        else:
            ci = (
                f"[{self.pooled_ci95[0]:+.3f}, {self.pooled_ci95[1]:+.3f}]"
                if self.pooled_ci95
                else "(too few clusters)"
            )
            # Said explicitly, every time, because a pooled number invites being
            # quoted alone.
            lines.append(
                f"pooled    {self.pooled_regret:+.3f} {ci}"
                f"   <- unweighted mean of the classes above; the mix is the manifest's, "
                f"not a property of the metric"
            )
        return "\n".join(lines)


def aggregate(
    cards: list[ScoreCard],
    shape_of,
    *,
    beta: float | None = None,
    resamples: int = 2000,
    seed: int = 0,
    manifest_fingerprint: str = "",
) -> Aggregate:
    """Group cards by shape, summarize each, then pool the class means.

    `shape_of` maps a scenario id to its shape class. Passed in rather than
    parsed out of the id, so the aggregation does not depend on an id format that
    could change.

    Pooling averages the CLASS MEANS rather than all scenarios, so a class with
    more scenarios does not get more weight than the manifest gave it. With an
    equal mix the two agree; the distinction matters the moment the mix is not
    equal, which is exactly when someone would be tempted not to notice.
    """
    if beta is None:
        betas = {c.beta for c in cards}
        if len(betas) > 1:
            raise ValueError(f"cards span several betas {sorted(betas)}; aggregate one at a time")
        beta = next(iter(betas), 0.0)

    by_shape: dict[str, list[ScoreCard]] = {}
    for card in cards:
        by_shape.setdefault(shape_of(card.scenario_id), []).append(card)

    summaries = []
    for shape in sorted(by_shape):
        group = by_shape[shape]
        scored = [c for c in group if not c.excluded and c.regret is not None]
        excluded = [c for c in group if c.excluded]
        clusters: dict[str, list[float]] = {}
        tier_a_clusters: dict[str, list[float]] = {}
        for card in scored:
            clusters.setdefault(card.scenario_id, []).append(card.regret)
            if card.tier_a:
                # Reported as its own subset rather than folded in. On a Tier-A
                # scenario every beta agrees, so its regret is a claim about the
                # agent alone; pooling it with persona-dependent scenarios mixes
                # two different kinds of statement.
                tier_a_clusters.setdefault(card.scenario_id, []).append(card.regret)
        summaries.append(
            ClassSummary(
                shape=shape,
                beta=beta,
                n_scenarios=len(clusters),
                n_runs=len(scored),
                n_excluded=len(excluded),
                mean_regret=_mean(statistics.fmean(v) for v in clusters.values()),
                ci95=cluster_bootstrap(
                    list(clusters.values()), resamples=resamples, seed=seed
                ),
                negative_regret_scenarios=sum(
                    1 for v in clusters.values() if statistics.fmean(v) < 0
                ),
                beat_all_inline=beats_all_inline_rate(group),
                mean_agent_k=_mean(c.agent_k for c in scored),
                mean_oracle_k=_mean(c.oracle_k for c in scored),
                tier_a_scenarios=len(tier_a_clusters),
                tier_a_mean_regret=_mean(
                    statistics.fmean(v) for v in tier_a_clusters.values()
                ),
                exclusion_reasons=dict(Counter(c.reason for c in excluded)),
                comparable=all(c.comparable for c in scored) if scored else True,
            )
        )

    class_means = [s.mean_regret for s in summaries if s.mean_regret is not None]
    class_clusters = [[m] for m in class_means]
    warnings = sorted({w for c in cards for w in c.warnings})

    return Aggregate(
        beta=beta,
        per_class=tuple(summaries),
        pooled_regret=_mean(class_means),
        pooled_ci95=cluster_bootstrap(class_clusters, resamples=resamples, seed=seed),
        n_scenarios=sum(s.n_scenarios for s in summaries),
        n_excluded=sum(s.n_excluded for s in summaries),
        manifest_fingerprint=manifest_fingerprint,
        comparable=all(s.comparable for s in summaries) and not warnings,
        warnings=tuple(warnings),
    )
