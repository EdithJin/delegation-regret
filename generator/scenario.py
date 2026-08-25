"""A scenario: a DAG made executable, rendered as a workspace, and verifiable.

Design doc: Phase1-DelegationBench-Design.md section 4, Stages 2 and 4.
Low-level design: low-level-design.md, "Stage 3".

This is the join between the two halves of the benchmark. The oracle reads a
scenario as a DAG with node sizes and prices plans over it. The measurement side
reads the same scenario as a directory an agent is pointed at, plus a verifier
that says whether each node's tests pass. Both views are built from one object,
so the plan the oracle scores and the work the agent does cannot drift apart.

WHAT THE AGENT SEES AND DOES NOT SEE. It gets a task list naming each module and
its suite, and a package whose tests fail. It does NOT get the dependency graph:
which module imports which is discoverable by reading the imports, or by running
the suite and seeing where the failures originate. That inference is part of the
measured skill, which is what `disclose_dag=True` exists to switch off -- the one
ablation that separates "failed to see the structure" from "saw it and chose
wrong."

The DECOMPOSITION is disclosed on purpose, and the distinction matters. The task
list names the nodes because that is the granularity contract making the oracle
comparable at all: the oracle's claim is exact over execution plans OF A GIVEN
decomposition, not over every strategy an agent might invent. If the agent
invented its own decomposition, regret would mix "chose a different
decomposition" with "allocated badly" and neither could be read off. It is also
what makes per-node attribution and the disjoint-footprint guarantee work.

THE ANSWER KEY IS DERIVED, NOT AUTHORED. Expected values come from EXECUTING the
correct modules, so no expectation is hand-written and no model appears anywhere
in the ground-truth path. That is required by the claim that ground truth here is
derived rather than labelled.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .dag import DAG
from .templates import (
    CheckResult,
    PACKAGE,
    PROBE_ARGS,
    Subtask,
    TEST_DIR,
    build_module,
    build_tests,
    inject,
    module_path,
    test_module_dotted,
    test_path,
)

__all__ = ["Scenario", "build_scenario"]


_PROBE = r"""
import json, sys, importlib
sys.path.insert(0, sys.argv[1])
out = {}
for node_id in sys.argv[3].split(","):
    mod = importlib.import_module("%s.mod_" + node_id if False else "%s.mod_%%s" %% node_id)
    vals = {}
    for name in sorted(vars(mod)):
        if not name.startswith("step_" + node_id):
            continue
        fn = getattr(mod, name)
        if not callable(fn):
            continue
        vals[name] = [[a, fn(a)] for a in json.loads(sys.argv[2])]
    out[node_id] = vals
print(json.dumps(out, sort_keys=True))
""" % (PACKAGE, PACKAGE)


def _capture_expected(modules: dict[str, str], node_ids: list[str]) -> dict[str, dict]:
    """Run the CORRECT package and record what every function returns.

    In a subprocess, so a generated module can never pollute the generator's own
    interpreter, and so an import error surfaces as a generator failure rather
    than as a mysterious state change.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / PACKAGE).mkdir()
        (root / PACKAGE / "__init__.py").write_text("", encoding="utf-8")
        for node_id, src in modules.items():
            (root / module_path(node_id)).write_text(src, encoding="utf-8")
        probe = root / "_probe.py"
        probe.write_text(_PROBE, encoding="utf-8")
        done = subprocess.run(
            [sys.executable, str(probe), str(root), json.dumps(list(PROBE_ARGS)), ",".join(node_ids)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    if done.returncode != 0:
        raise RuntimeError(f"reference package does not execute: {(done.stderr or '').strip()[-400:]}")
    return json.loads(done.stdout)


@dataclass(frozen=True)
class Scenario:
    """One executable instance of a dependency graph."""

    id: str
    dag: DAG
    subtasks: dict[str, Subtask]
    modules: dict[str, str]  # node_id -> DEFECTIVE source; what the agent is given
    tests: dict[str, str]  # node_id -> generated suite
    reference: dict[str, str]  # node_id -> CORRECT source; the answer key

    # -- the agent-facing surface ----------------------------------------

    def surface(self, disclose_dag: bool = False) -> str:
        lines = [
            f"# Test repair pass `{self.id}`",
            "",
            f"`{PACKAGE}/` is a Python package whose test suite is failing. Below are "
            f"{len(self.subtasks)} repair tasks. Each names one module and the suite "
            "that judges it.",
            "",
            "Fix only the module a task names. Do not edit any test file.",
            "",
            "## Tasks",
            "",
        ]
        for node_id in self.dag.topo_order:
            lines.append(f"- **{node_id}** — {self.subtasks[node_id].instruction}")
        if disclose_dag:
            lines += ["", "## Dependencies", ""]
            edges = sorted(self.dag.edges)
            if edges:
                lines += [f"- `{u}` must be correct before `{v}` can pass" for u, v in edges]
            else:
                lines.append("- None. Every task is independent of every other.")
        return "\n".join(lines) + "\n"

    def materialize(self, root: str | Path, disclose_dag: bool = False) -> Path:
        """Write the workspace an agent is pointed at. Returns its path."""
        root = Path(root)
        (root / PACKAGE).mkdir(parents=True, exist_ok=True)
        (root / TEST_DIR).mkdir(parents=True, exist_ok=True)
        (root / PACKAGE / "__init__.py").write_text("", encoding="utf-8")
        (root / TEST_DIR / "__init__.py").write_text("", encoding="utf-8")
        for node_id, src in self.modules.items():
            (root / module_path(node_id)).write_text(src, encoding="utf-8")
        for node_id, src in self.tests.items():
            (root / test_path(node_id)).write_text(src, encoding="utf-8")
        (root / "TASKS.md").write_text(self.surface(disclose_dag), encoding="utf-8")
        return root

    def restore_tests(self, root: str | Path) -> Path:
        """Put the generated suites back exactly as issued.

        THE SUITE IS THE MEASUREMENT INSTRUMENT, NOT PART OF THE WORKSPACE. The
        task list tells the agent not to edit a test, but an instruction is not
        an enforcement, and `write_file` cannot refuse the write without also
        refusing the legitimate reads the agent needs. So the suites are
        restored from the generator's own copies immediately before grading.

        Without this, the cheapest passing strategy is to replace every suite
        with `pass` -- three writes, no repairs, no delegation, and a clean
        `succeeded()`. That run lands as the cheapest plan in the table WITH the
        success gate satisfied, which is not a small measurement error: it is a
        maximally negative regret produced by doing none of the work.
        """
        root = Path(root)
        (root / TEST_DIR).mkdir(parents=True, exist_ok=True)
        (root / TEST_DIR / "__init__.py").write_text("", encoding="utf-8")
        for node_id, src in self.tests.items():
            (root / test_path(node_id)).write_text(src, encoding="utf-8")
        return root

    def tampered_tests(self, root: str | Path) -> tuple[str, ...]:
        """Node ids whose suite the run did not leave as issued.

        Restoring silently would hide the attempt. A run that edited a test has
        told you something about the model worth reporting, and section 6 should
        be able to flag it rather than merely be protected from it.
        """
        root = Path(root)
        out = []
        for node_id, src in self.tests.items():
            path = root / test_path(node_id)
            try:
                current = path.read_text(encoding="utf-8")
            except OSError:
                out.append(node_id)
                continue
            if current != src:
                out.append(node_id)
        return tuple(out)

    def write_reference(self, root: str | Path) -> Path:
        """Overwrite the modules with the correct versions.

        Two jobs. It proves the verifier ACCEPTS a correct run -- without which a
        verifier that rejected everything would look like a hard benchmark. And
        it is how section 6 executes the oracle plan to obtain the regret
        baseline.
        """
        root = Path(root)
        (root / PACKAGE).mkdir(parents=True, exist_ok=True)
        for node_id, src in self.reference.items():
            (root / module_path(node_id)).write_text(src, encoding="utf-8")
        return root

    # -- verification -----------------------------------------------------

    def verify(self, root: str | Path, timeout: float = 60.0) -> dict[str, CheckResult]:
        """Run each node's suite in a clean subprocess.

        The agent may run these itself, and should -- that is its feedback loop.
        The VERDICT comes from here, after it has finished. A self-report is
        never a measurement.

        BYTECODE CACHES ARE PURGED FIRST, and this is a correctness requirement
        rather than hygiene. CPython decides a `.pyc` is current by comparing the
        source's mtime AND SIZE, so a repair that changes neither -- swapping the
        two branches of a conditional reorders tokens without changing length --
        can leave a stale cache in place and be graded against the code it
        replaced. Found the hard way: a correctly repaired module reported as
        failing because the defective bytecode was still cached. Every subprocess
        also runs with bytecode writing disabled, so a verification never leaves
        a cache behind for the next one to trip over.

        THE SUITES ARE RESTORED FIRST, for the reason `restore_tests` gives: the
        verdict has to come from the tests as ISSUED, not from whatever is on
        disk when the agent stops. Grading the workspace's copy makes rewriting
        the suite the cheapest way to pass.
        """
        root = Path(root)
        self.restore_tests(root)
        for cache in root.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        results: dict[str, CheckResult] = {}
        for node_id in self.dag.topo_order:
            try:
                done = subprocess.run(
                    [sys.executable, "-B", "-m", "unittest", "-q", test_module_dotted(node_id)],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                results[node_id] = CheckResult(node_id, False, f"suite timed out after {timeout:g}s")
                continue
            passed = done.returncode == 0
            detail = "" if passed else (done.stderr or done.stdout or "").strip().splitlines()[-1:]
            results[node_id] = CheckResult(node_id, passed, detail[0] if detail else "")
        return results

    def succeeded(self, root: str | Path) -> bool:
        """Regret is only comparable among runs that produced working artifacts."""
        return all(self.verify(root).values())


def build_scenario(dag: DAG, scenario_id: str = "s0", seed: int = 0) -> Scenario:
    """DAG -> executable package. Deterministic: one seed, one scenario.

    Order matters. The correct modules are built first and executed to capture
    the answer key, because a suite generated from defective code would assert
    the bug. Defects go in last, after the expectations are already fixed.
    """
    import random

    rng = random.Random(seed)
    order = list(dag.topo_order)

    reference: dict[str, str] = {}
    for node_id in order:
        node = dag.by_id[node_id]
        preds = tuple(sorted(dag.preds[node_id]))
        reference[node_id] = build_module(node_id, node.size, preds, rng)

    captured = _capture_expected(reference, order)

    tests: dict[str, str] = {}
    modules: dict[str, str] = {}
    subtasks: dict[str, Subtask] = {}
    for node_id in order:
        node = dag.by_id[node_id]
        expected = {name: [tuple(p) for p in pairs] for name, pairs in captured[node_id].items()}
        tests[node_id] = build_tests(node_id, expected)
        broken, manifest = inject(reference[node_id], node.size, rng)
        modules[node_id] = broken
        subtasks[node_id] = Subtask(
            node_id=node_id,
            module=module_path(node_id),
            test_module=test_path(node_id),
            defects=manifest,
            imports=tuple(sorted(dag.preds[node_id])),
            size=node.size,
        )
    return Scenario(
        id=scenario_id, dag=dag, subtasks=subtasks, modules=modules, tests=tests, reference=reference
    )
