"""What the payload actually guarantees.

Every scenario is generated, so the properties the benchmark rests on have to be
asserted rather than inspected: that a correct repair passes, that a wrong one
does not, that dependencies bite, and that independent nodes really are
independent. A generator whose scenarios quietly did none of these would still
produce numbers.
"""

from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator.dag import DAG, Node, sample_dag, SHAPES
from generator.scenario import build_scenario
from generator.templates import (
    DEFECT_KINDS,
    build_module,
    inject,
    module_path,
    test_path,
)

SIZES = (2,)


def scenario(shape: str, n: int = 4, size: int = 2, seed: int = 5):
    return build_scenario(sample_dag(shape, n, sizes=(size,), seed=seed), shape, seed=seed)


class TestReferenceVerifies(unittest.TestCase):
    """If the answer key does not pass, every failure downstream is meaningless."""

    def test_answer_key_passes_on_every_shape(self) -> None:
        for shape in SHAPES:
            with tempfile.TemporaryDirectory() as tmp:
                s = scenario(shape)
                s.materialize(tmp)
                s.write_reference(tmp)
                self.assertTrue(s.succeeded(tmp), f"{shape}: reference package fails its own tests")

    def test_as_generated_everything_fails(self) -> None:
        # The other half. A suite that passed before any repair would measure
        # nothing at all.
        for shape in SHAPES:
            with tempfile.TemporaryDirectory() as tmp:
                s = scenario(shape)
                s.materialize(tmp)
                self.assertFalse(any(r.passed for r in s.verify(tmp).values()), shape)


class TestStaleBytecodeCannotMisgrade(unittest.TestCase):
    """Regression. CPython invalidates a .pyc on source mtime AND SIZE, so a
    repair that changes neither -- swapping a conditional's branches reorders
    tokens without changing length -- was graded against the code it replaced."""

    def test_repair_is_graded_after_an_earlier_failing_verify(self) -> None:
        for shape in SHAPES:
            with tempfile.TemporaryDirectory() as tmp:
                s = scenario(shape)
                s.materialize(tmp)
                s.verify(tmp)  # compiles the DEFECTIVE modules into __pycache__
                s.write_reference(tmp)
                self.assertTrue(
                    s.succeeded(tmp), f"{shape}: correct repair graded against stale bytecode"
                )


class TestTestsCannotBeTamperedWith(unittest.TestCase):
    """Regression. The suite is the measurement instrument, not part of the
    workspace, and `verify` used to grade whatever was on disk when the agent
    stopped. So the cheapest passing strategy was to overwrite every suite with
    `pass`: no repairs, no delegation, a clean `succeeded()`, and the lowest
    cost in the plan table -- a maximally negative regret earned by doing none
    of the work. The instruction not to edit tests is not an enforcement."""

    def test_rewriting_the_suite_does_not_produce_a_pass(self) -> None:
        for shape in SHAPES:
            with tempfile.TemporaryDirectory() as tmp:
                s = scenario(shape)
                s.materialize(tmp)
                for node_id in s.dag.ids:
                    (Path(tmp) / test_path(node_id)).write_text(
                        "import unittest\n\n\nclass T(unittest.TestCase):\n"
                        "    def test_ok(self):\n        pass\n",
                        encoding="utf-8",
                    )
                self.assertFalse(s.succeeded(tmp), f"{shape}: a rewritten suite graded as a pass")

    def test_deleting_the_suite_does_not_produce_a_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            s = scenario("wide")
            s.materialize(tmp)
            for node_id in s.dag.ids:
                (Path(tmp) / test_path(node_id)).unlink()
            self.assertFalse(s.succeeded(tmp))

    def test_tampering_is_reported_and_not_merely_prevented(self) -> None:
        # Restoring silently would hide the attempt. A model that rewrites its
        # own tests has told you something worth reporting.
        with tempfile.TemporaryDirectory() as tmp:
            s = scenario("wide")
            s.materialize(tmp)
            self.assertEqual(s.tampered_tests(tmp), ())
            victim = s.dag.ids[0]
            (Path(tmp) / test_path(victim)).write_text("# gone\n", encoding="utf-8")
            self.assertEqual(s.tampered_tests(tmp), (victim,))

    def test_restoring_does_not_disturb_a_genuine_repair(self) -> None:
        # The guard must not cost an honest run its pass.
        with tempfile.TemporaryDirectory() as tmp:
            s = scenario("chain")
            s.materialize(tmp)
            s.write_reference(tmp)
            self.assertTrue(s.succeeded(tmp))


class TestDependenciesAreReal(unittest.TestCase):
    """A dependency that bit only sometimes would be worse than none."""

    def test_successor_cannot_pass_while_predecessor_is_broken(self) -> None:
        s = scenario("chain", n=3)
        order = list(s.dag.topo_order)
        with tempfile.TemporaryDirectory() as tmp:
            s.materialize(tmp)
            for node_id in order[1:]:
                Path(tmp, module_path(node_id)).write_text(s.reference[node_id], encoding="utf-8")
            results = s.verify(tmp)
            for node_id in order[1:]:
                self.assertFalse(
                    results[node_id].passed,
                    f"{node_id} passed with its predecessor still broken -- the edge is decorative",
                )

    def test_independent_nodes_do_not_block_each_other(self) -> None:
        # The symmetric half: on `wide`, repairing one node must be enough for
        # that node. Otherwise "dependencies are real" would just mean
        # "everything is coupled", and fan-out could never verify.
        s = scenario("wide", n=4)
        with tempfile.TemporaryDirectory() as tmp:
            s.materialize(tmp)
            target = s.dag.topo_order[0]
            Path(tmp, module_path(target)).write_text(s.reference[target], encoding="utf-8")
            results = s.verify(tmp)
            self.assertTrue(results[target].passed, "an independent node needs no one else")
            for other in s.dag.topo_order[1:]:
                self.assertFalse(results[other].passed)


class TestFootprintsAreDisjoint(unittest.TestCase):
    """Two concurrent subagents must not be able to touch the same file. A merge
    conflict between them would be a confound, not a finding."""

    def test_one_node_one_module_one_suite(self) -> None:
        for shape in SHAPES:
            s = scenario(shape)
            mods = [t.module for t in s.subtasks.values()]
            suites = [t.test_module for t in s.subtasks.values()]
            self.assertEqual(len(set(mods)), len(mods), shape)
            self.assertEqual(len(set(suites)), len(suites), shape)
            for node_id, t in s.subtasks.items():
                self.assertEqual(t.module, module_path(node_id))
                self.assertEqual(t.test_module, test_path(node_id))


class TestSizeIsDefectCount(unittest.TestCase):
    """`size` is the cost model's x-axis. If it meant two things in one run, the
    block curve would be keyed on a quantity that does not exist."""

    def test_every_node_carries_exactly_size_defects(self) -> None:
        for size in (1, 2, 4):
            s = scenario("wide", n=3, size=size)
            for node_id, t in s.subtasks.items():
                self.assertEqual(len(t.defects), size, node_id)
                self.assertEqual(t.size, size)

    def test_defects_land_on_distinct_functions(self) -> None:
        s = scenario("wide", n=3, size=4)
        for t in s.subtasks.values():
            fns = [d.function for d in t.defects]
            self.assertEqual(len(set(fns)), len(fns), "two defects in one function is one decision")

    def test_kinds_come_from_the_published_catalogue(self) -> None:
        s = scenario("mixed", n=5, size=3)
        for t in s.subtasks.values():
            for d in t.defects:
                self.assertIn(d.kind, DEFECT_KINDS)


class TestDefectMixIsBalanced(unittest.TestCase):
    """`size` is the cost model's only handle on how much work a node is, and
    `block_dollars` prices a block on total units while explicitly not looking at
    which nodes compose it. That holds only if a unit of size means the same
    thing everywhere -- so the defect mix has to be balanced, not merely random.

    Measured before the deal was balanced, over 120 size-5 nodes: the two
    structural kinds appeared 0.36-0.39 times per node against 1.32-1.48 for the
    arithmetic ones, and single nodes carried one kind three times."""

    def test_a_node_gets_each_kind_the_same_number_of_times(self) -> None:
        import collections

        for size in (5, 10):
            for seed in range(6):
                s = scenario("wide", n=3, size=size, seed=seed)
                for node_id, task in s.subtasks.items():
                    mix = collections.Counter(d.kind for d in task.defects)
                    self.assertEqual(len(mix), len(DEFECT_KINDS), f"{node_id}: {dict(mix)}")
                    self.assertEqual(
                        max(mix.values()) - min(mix.values()), 0, f"{node_id}: {dict(mix)}"
                    )

    def test_an_uneven_size_is_off_by_at_most_one(self) -> None:
        # size 7 cannot be even across 5 kinds; 2+2+1+1+1 is the best possible.
        import collections

        for seed in range(6):
            s = scenario("wide", n=3, size=7, seed=seed)
            for node_id, task in s.subtasks.items():
                mix = collections.Counter(d.kind for d in task.defects)
                counts = [mix.get(k, 0) for k in DEFECT_KINDS]
                self.assertLessEqual(max(counts) - min(counts), 1, f"{node_id}: {dict(mix)}")

    def test_small_nodes_are_not_systematically_arithmetic_only(self) -> None:
        # A fixed tie-break made every size-3 node draw the same three kinds, so
        # small nodes never carried a structural defect. The block curve would
        # then measure a changing MIX as size rose rather than more work.
        import collections

        seen: collections.Counter = collections.Counter()
        for seed in range(40):
            s = scenario("wide", n=3, size=3, seed=seed)
            for task in s.subtasks.values():
                seen.update(d.kind for d in task.defects)
        self.assertEqual(set(seen), set(DEFECT_KINDS), dict(seen))
        share = [seen[k] / sum(seen.values()) for k in DEFECT_KINDS]
        self.assertLess(max(share) - min(share), 0.10, dict(seen))

    def test_enough_conditional_bodies_exist_for_the_structural_kinds(self) -> None:
        # The balance is only reachable because `_body_plan` reserves them. This
        # asserts the reservation rather than the consequence.
        import ast

        from generator.templates import STRUCTURAL_KINDS, _body_plan

        rng = random.Random(0)
        for size in (5, 10, 20):
            plan = _body_plan(size, rng)
            conditional = sum(1 for body in plan if " if " in body)
            self.assertGreaterEqual(conditional, len(STRUCTURAL_KINDS) * (size // 5), size)
        self.assertEqual(len(_body_plan(0, rng)), 0)


class TestVerifierDiscriminates(unittest.TestCase):
    """A verifier that accepts anything is worse than no verifier."""

    def test_a_plausible_but_wrong_repair_fails(self) -> None:
        s = scenario("wide", n=2)
        target = s.dag.topo_order[0]
        with tempfile.TemporaryDirectory() as tmp:
            s.materialize(tmp)
            # Syntactically fine, imports cleanly, wrong by one.
            wrong = s.reference[target].replace("return ", "return 1 + ", 1)
            Path(tmp, module_path(target)).write_text(wrong, encoding="utf-8")
            self.assertFalse(s.verify(tmp)[target].passed)

    def test_a_module_that_does_not_import_fails_rather_than_crashes(self) -> None:
        s = scenario("wide", n=2)
        target = s.dag.topo_order[0]
        with tempfile.TemporaryDirectory() as tmp:
            s.materialize(tmp)
            Path(tmp, module_path(target)).write_text("def (\n", encoding="utf-8")
            result = s.verify(tmp)[target]
            self.assertFalse(result.passed)


class TestSurface(unittest.TestCase):
    def test_the_dag_is_hidden_unless_disclosed(self) -> None:
        s = scenario("chain", n=3)
        plain = s.surface()
        for u, v in s.dag.edges:
            self.assertNotIn(f"`{u}` must be correct before", plain)
        disclosed = s.surface(disclose_dag=True)
        for u, v in sorted(s.dag.edges):
            self.assertIn(f"`{u}` must be correct before `{v}`", disclosed)

    def test_every_node_is_named_in_the_task_list(self) -> None:
        # The decomposition IS disclosed, deliberately: it is the granularity
        # contract that makes the oracle's plan space comparable.
        s = scenario("mixed", n=5)
        surface = s.surface()
        for node_id in s.dag.ids:
            self.assertIn(f"**{node_id}**", surface)


class TestDeterminism(unittest.TestCase):
    """One seed, one scenario. This is what makes held-out refresh possible."""

    def test_same_seed_same_package(self) -> None:
        a = scenario("diamond", n=4, seed=11)
        b = scenario("diamond", n=4, seed=11)
        self.assertEqual(a.modules, b.modules)
        self.assertEqual(a.tests, b.tests)
        self.assertEqual(a.reference, b.reference)

    def test_different_seed_different_package(self) -> None:
        a = scenario("diamond", n=4, seed=11)
        b = scenario("diamond", n=4, seed=12)
        self.assertNotEqual(a.modules, b.modules)


class TestInjection(unittest.TestCase):
    def test_injection_changes_the_source(self) -> None:
        rng = random.Random(2)
        src = build_module("n0", 3, (), rng)
        broken, manifest = inject(src, 3, rng)
        self.assertNotEqual(src, broken)
        self.assertEqual(len(manifest), 3)

    def test_cannot_inject_more_defects_than_functions(self) -> None:
        rng = random.Random(2)
        src = build_module("n0", 2, (), rng)
        with self.assertRaises(ValueError):
            inject(src, 3, rng)


if __name__ == "__main__":
    unittest.main()
