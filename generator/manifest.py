"""The pre-registered scenario set: which scenarios, fixed before any model runs.

Low-level design: low-level-design.md, sections 1.2 and 1.4.

WHY A MANIFEST EXISTS AT ALL, GIVEN SCENARIOS ARE FREE TO GENERATE

Nothing here stores task content. `sample_dag(shape, n, sizes, seed)` regenerates
a byte-identical scenario from four integers, so a scenario "database" would be a
cache of something cheaper to recompute. What has to be pinned is not the content
but the CHOICE: which shapes, how many of each, at what sizes.

That choice is not a detail. The headline aggregate is a direct function of how
many wide versus chain scenarios are included, because the two shapes have
opposite correct answers -- fan-out is free on `wide` and pure loss on `chain`.
Left unpinned, the class mix is a free parameter that silently sets what
always-serial scores, and it is a parameter whose value can be chosen after
seeing results. Every mix is defensible in isolation; that is exactly the
problem.

So: the mix is committed here, in code, before any model is run, and the aggregate
is reported PER CLASS as primary with the pooled number secondary. A reader who
distrusts the mix can then reweight it themselves, which they cannot do from a
pooled figure alone.

THREE SETS, AND THEY ARE NOT INTERCHANGEABLE

`CORE`      the pre-registered graded set. Fixed seeds. This is what the
            headline numbers come from, and it does not change.

`HELDOUT`   generated from a seed range disjoint from CORE, on demand. Because
            generation is deterministic and free, a fresh held-out set can be
            drawn at any time -- which is what makes memorization pointless
            rather than merely discouraged. Nothing about a model's training data
            can cover a scenario nobody has drawn yet.

`ANCHOR`    a small set kept deliberately tiny and hand-checkable. Its purpose is
            to be worked through by hand when a number looks wrong, so the
            three-node worked example in the design doc stays runnable.

WHAT THE MANIFEST DOES NOT DO. It does not pick beta, and it does not pick the
cost model. Those are reported across their whole range, so pinning them here
would smuggle an answer into the input.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .dag import SHAPES, sample_dag
from .scenario import Scenario, build_scenario

__all__ = [
    "ScenarioSpec",
    "Manifest",
    "CORE",
    "ANCHOR",
    "heldout",
    "CORE_SEED_MAX",
    "HELDOUT_SEED_MIN",
]

# The seed ranges are disjoint BY CONSTRUCTION and the boundary is a published
# constant, not a convention. An overlapping held-out set is not held out, and
# the overlap would be invisible in results.
CORE_SEED_MAX = 999
HELDOUT_SEED_MIN = 10_000


@dataclass(frozen=True)
class ScenarioSpec:
    """Everything needed to regenerate one scenario, and nothing else."""

    shape: str
    n: int
    size: int
    seed: int

    def __post_init__(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"unknown shape {self.shape!r}; expected one of {sorted(SHAPES)}")
        if self.shape == "diamond" and self.n < 3:
            raise ValueError("a diamond needs at least 3 nodes")
        if self.n < 1 or self.size < 1:
            raise ValueError("n and size must be positive")

    @property
    def id(self) -> str:
        return f"{self.shape}{self.n}-s{self.size}-{self.seed}"

    def build(self) -> Scenario:
        dag = sample_dag(self.shape, self.n, sizes=(self.size,), seed=self.seed)
        return build_scenario(dag, self.id, seed=self.seed)


@dataclass(frozen=True)
class Manifest:
    """A named, ordered, hashable set of scenario specs."""

    name: str
    specs: tuple[ScenarioSpec, ...]
    note: str = ""

    def __len__(self) -> int:
        return len(self.specs)

    def by_shape(self) -> dict[str, tuple[ScenarioSpec, ...]]:
        """The class mix, which is the thing reporting is grouped by."""
        out: dict[str, list[ScenarioSpec]] = {}
        for spec in self.specs:
            out.setdefault(spec.shape, []).append(spec)
        return {k: tuple(v) for k, v in sorted(out.items())}

    @property
    def fingerprint(self) -> str:
        """A short hash of the exact spec list.

        Published alongside results. If the manifest is edited, the fingerprint
        changes, and a results table quoting the old one is visibly stale -- which
        is the whole point of pre-registering. A mix silently widened after seeing
        an unflattering aggregate would otherwise leave no trace.
        """
        blob = json.dumps([asdict(s) for s in self.specs], sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def build(self):
        """Generate every scenario. Deterministic, free, and never cached."""
        for spec in self.specs:
            yield spec, spec.build()

    def summary(self) -> str:
        lines = [f"{self.name}  ({len(self)} scenarios, fingerprint {self.fingerprint})"]
        if self.note:
            lines.append(f"  {self.note}")
        for shape, specs in self.by_shape().items():
            sizes = sorted({s.size for s in specs})
            ns = sorted({s.n for s in specs})
            lines.append(
                f"  {shape:<9} {len(specs):>3} scenarios   n={ns}   size={sizes}"
            )
        return "\n".join(lines)


def _grid(shape: str, ns, sizes, seeds) -> list[ScenarioSpec]:
    return [
        ScenarioSpec(shape=shape, n=n, size=size, seed=seed)
        for n in ns
        for size in sizes
        for seed in seeds
    ]


# --------------------------------------------------------------- the core set
#
# EQUAL COUNTS PER SHAPE, deliberately. An unequal mix would need a defence, and
# any defence would be a claim about which shapes matter more -- which is a
# finding, not an input. Equal weights make the pooled figure a plain average of
# the four classes, so a reader who wants a different weighting can compute it
# from the per-class numbers.
#
# n is capped at 8 because the plan space is B(n+1): 21,147 plans at n=8 and
# 27.6M at n=12, and evaluation (not enumeration) is the cost.

_CORE_NS = (4, 6, 8)
_CORE_SIZES = (3,)  # one size until the block curve says which sizes are distinct
_CORE_SEEDS = (11, 23, 37)

CORE = Manifest(
    name="core",
    note=(
        "Pre-registered graded set. Equal counts per shape so the pooled figure is "
        "an unweighted mean of the four classes. Fixed seeds; do not edit without "
        "changing the fingerprint and saying so."
    ),
    specs=tuple(
        spec
        for shape in ("wide", "chain", "diamond", "mixed")
        for spec in _grid(shape, _CORE_NS, _CORE_SIZES, _CORE_SEEDS)
    ),
)

# --------------------------------------------------------------- anchor set
#
# Three nodes, one size, both interesting shapes: this is the worked example the
# design doc walks through by hand. Kept because a benchmark whose smallest case
# cannot be checked on report is a benchmark nobody can audit.

ANCHOR = Manifest(
    name="anchor",
    note="The design doc's worked example. Small enough to check by hand.",
    specs=(
        ScenarioSpec("wide", 3, 5, 1),
        ScenarioSpec("chain", 3, 5, 1),
    ),
)


def heldout(count: int = 12, *, offset: int = 0, shapes=("wide", "chain", "diamond", "mixed"),
            n: int = 6, size: int = 3) -> Manifest:
    """A fresh set drawn from seeds disjoint from CORE.

    Generated on demand rather than stored, which is what makes the refresh
    mechanism work: no fixed set can be memorized if the set does not exist until
    it is asked for. `offset` walks the seed range so successive refreshes do not
    collide.

    Deliberately parameterised rather than pinned. A held-out set that never
    changes is just a second core set.
    """
    if count < 1:
        raise ValueError("a held-out set needs at least one scenario")
    specs = []
    for i in range(count):
        shape = shapes[i % len(shapes)]
        specs.append(
            ScenarioSpec(shape=shape, n=n, size=size, seed=HELDOUT_SEED_MIN + offset + i)
        )
    return Manifest(
        name=f"heldout+{offset}",
        note="Drawn from a seed range disjoint from core; regenerate freely.",
        specs=tuple(specs),
    )


def write(manifest: Manifest, path: str | Path) -> Path:
    """Persist a manifest next to the results it produced.

    The results are only interpretable against the exact set that produced them,
    and "the exact set" is four integers per scenario plus a fingerprint.
    """
    path = Path(path)
    path.write_text(
        json.dumps(
            {
                "name": manifest.name,
                "note": manifest.note,
                "fingerprint": manifest.fingerprint,
                "specs": [asdict(s) for s in manifest.specs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load(path: str | Path) -> Manifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = Manifest(
        name=raw["name"],
        note=raw.get("note", ""),
        specs=tuple(ScenarioSpec(**s) for s in raw["specs"]),
    )
    if raw.get("fingerprint") and raw["fingerprint"] != manifest.fingerprint:
        raise ValueError(
            f"manifest fingerprint mismatch: file says {raw['fingerprint']}, "
            f"specs hash to {manifest.fingerprint}. The spec list was edited after "
            "the fingerprint was written, so results quoting it are stale."
        )
    return manifest
