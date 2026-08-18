"""A scenario: a DAG made executable, rendered as a workspace, and verifiable.

Design doc: Phase1-DelegationBench-Design.md section 4, Stages 2 and 4.

This is the join between the two halves of the benchmark. The oracle side reads
a scenario as a DAG with node sizes and prices plans over it. The measurement
side reads the same scenario as a directory an agent is pointed at, plus a
verifier that says whether each node's artifact is right. Both views are built
from one object, so the plan the oracle scores and the work the agent does
cannot drift apart.

What the agent sees is a natural task description and a file tree. What it does
not see is the graph: dependencies are discoverable from the instructions' file
paths, which is work the oracle is not charged for. That asymmetry is priced in
section 6 and is the whole reason for `disclose_dag=True`, which renders the
same scenario with the structure handed over -- section 7's DAG-disclosed
condition, and the one ablation the sprint's drop order never drops, because it
separates discovery failure from decision failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dag import DAG
from .templates import (
    CheckResult,
    FAMILIES,
    SEED_MODULE,
    Subtask,
    apply_family,
    build_seed,
    merge_modules,
    verify_output,
)

__all__ = ["Scenario", "build_scenario", "SEED_PATH"]

SEED_PATH = SEED_MODULE


@dataclass(frozen=True)
class Scenario:
    """One executable instance of a dependency graph.

    `reference` is the ground-truth artifact text per node, computed by applying
    each family's transformation along the true topological order. It is the
    answer key, and section 5.2's realized oracle execution runs against it.
    """

    id: str
    dag: DAG
    subtasks: dict[str, Subtask]
    seed: str
    reference: dict[str, str]

    # -- the agent-facing surface (Stage 4) -------------------------------

    def surface(self, disclose_dag: bool = False) -> str:
        """The task description. The graph appears only if disclosure is on."""
        lines = [
            f"# Refactor pass `{self.id}`",
            "",
            f"`{SEED_PATH}` is a generated module. Below are {len(self.subtasks)} "
            "transformations to apply to it. Each one reads the files it names and "
            "writes exactly one new file; write no other files, and modify nothing "
            "that a task does not tell you to modify.",
            "",
            "## Tasks",
            "",
        ]
        for node_id in self.dag.topo_order:
            task = self.subtasks[node_id]
            lines.append(f"- **{node_id}** — {task.instruction}")
        if disclose_dag:
            lines += ["", "## Dependencies", ""]
            edges = sorted(self.dag.edges)
            if edges:
                lines += [f"- `{u}` must finish before `{v}` starts" for u, v in edges]
            else:
                lines.append("- none")
            independent = [
                f"`{a}`/`{b}`"
                for i, a in enumerate(self.dag.ids)
                for b in self.dag.ids[i + 1 :]
                if self.dag.is_independent(a, b)
            ]
            lines += [
                "",
                "Every other pair is independent and may run in any order or at the "
                "same time: " + (", ".join(independent) if independent else "none") + ".",
            ]
        return "\n".join(lines) + "\n"

    def materialize(self, root: str | Path, disclose_dag: bool = False) -> Path:
        """Write the workspace an agent is pointed at. Returns its path."""
        root = Path(root)
        (root / "seed").mkdir(parents=True, exist_ok=True)
        (root / "work").mkdir(parents=True, exist_ok=True)
        (root / SEED_PATH).write_text(self.seed, encoding="utf-8")
        (root / "TASKS.md").write_text(self.surface(disclose_dag), encoding="utf-8")
        return root

    def write_reference(self, root: str | Path) -> Path:
        """Fill `work/` with the answer key.

        Used two ways: to test that the verifier accepts a correct run, and to
        execute the oracle plan for section 5.2's realized-versus-realized
        regret baseline.
        """
        root = Path(root)
        (root / "work").mkdir(parents=True, exist_ok=True)
        for node_id, text in self.reference.items():
            (root / self.subtasks[node_id].output).write_text(text, encoding="utf-8")
        return root

    # -- verification -----------------------------------------------------

    def verify(self, root: str | Path) -> dict[str, CheckResult]:
        """Check every node's artifact against the answer key, behaviourally."""
        root = Path(root)
        results: dict[str, CheckResult] = {}
        for node_id in self.dag.topo_order:
            path = root / self.subtasks[node_id].output
            actual = path.read_text(encoding="utf-8") if path.is_file() else None
            results[node_id] = verify_output(actual, self.reference[node_id], node_id)
        return results

    def succeeded(self, root: str | Path) -> bool:
        """Section 6: regret is only comparable among runs that produced artifacts."""
        return all(self.verify(root).values())


def build_scenario(dag: DAG, scenario_id: str = "s0") -> Scenario:
    """Instantiate a DAG as executable work.

    A node's index is its position in topological order, which is what keeps a
    node's target distinct from every target on any path through it: topo
    position strictly increases along an edge, so no node's transformation can
    have been consumed by an ancestor or be pre-empted by a descendant.
    """
    order = dag.topo_order
    index_of = {node_id: i for i, node_id in enumerate(order)}
    size = max(node.size for node in dag.nodes)
    seed = build_seed(len(order), size)

    subtasks: dict[str, Subtask] = {}
    for node_id in order:
        node = dag.by_id[node_id]
        if node.family not in FAMILIES:
            raise ValueError(
                f"node {node_id} has family {node.family!r}; "
                f"expected one of {sorted(FAMILIES)}"
            )
        preds = sorted(dag.preds[node_id], key=lambda p: index_of[p])
        inputs = tuple(f"work/{p}.py" for p in preds) or (SEED_PATH,)
        subtasks[node_id] = Subtask(
            node_id=node_id,
            family=node.family,
            size=node.size,
            index=index_of[node_id],
            inputs=inputs,
            output=f"work/{node_id}.py",
        )

    reference: dict[str, str] = {}
    for node_id in order:
        task = subtasks[node_id]
        preds = sorted(dag.preds[node_id], key=lambda p: index_of[p])
        sources = [reference[p] for p in preds] or [seed]
        reference[node_id] = apply_family(
            task.family, merge_modules(sources, base=seed), task.index, task.size
        )

    return Scenario(
        id=scenario_id, dag=dag, subtasks=subtasks, seed=seed, reference=reference
    )
