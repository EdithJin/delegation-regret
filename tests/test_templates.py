"""What the executable payload guarantees.

The scenario library's whole job is to make three claims true at once: the
reference answer verifies, a wrong answer does not, and running a chain in
parallel produces a wrong answer *by itself*, without the scorer's help. If the
third fails, the benchmark measures nothing -- an agent could fan out a strict
chain and still be graded as successful, and every regret number computed over
that run would be arithmetic on a fiction.

These tests are slower than the rest of the suite because verification imports
generated modules in a subprocess. That is the point: section 6 forbids an LLM
judge anywhere in this loop, so correctness is decided by execution.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from generator.dag import DAG, Node, sample_dag
from generator.scenario import SEED_PATH, build_scenario
from generator.templates import (
    FAMILIES,
    apply_family,
    build_seed,
    merge_modules,
    probe_source,
    verify_output,
)

TEMPLATE_FAMILIES = tuple(sorted(FAMILIES))


def _scenario(shape: str, n: int, seed: int = 3, size: int = 1):
    dag = sample_dag(shape, n, families=TEMPLATE_FAMILIES, sizes=(size,), seed=seed)
    return build_scenario(dag, f"{shape}-{n}-{seed}")


class TestReferenceVerifies(unittest.TestCase):
    """The answer key must pass on every shape and every family mix."""

    def test_reference_passes_on_every_shape(self) -> None:
        for shape, n in (("wide", 4), ("chain", 4), ("diamond", 5), ("mixed", 5)):
            with self.subTest(shape=shape):
                sc = _scenario(shape, n)
                with tempfile.TemporaryDirectory() as td:
                    sc.materialize(td)
                    sc.write_reference(td)
                    self.assertTrue(sc.succeeded(td))

    def test_every_family_is_exercised_and_passes_alone(self) -> None:
        # A family that is never sampled is a family that is never tested, so
        # pin each one on its own rather than trusting the random mix above.
        for family in FAMILIES:
            with self.subTest(family=family):
                dag = DAG(tuple(Node(f"n{i}", family, 1) for i in range(3)), frozenset())
                sc = build_scenario(dag, f"solo-{family}")
                with tempfile.TemporaryDirectory() as td:
                    sc.materialize(td)
                    sc.write_reference(td)
                    self.assertTrue(sc.succeeded(td), sc.verify(td))


class TestDependenciesAreReal(unittest.TestCase):
    """The environment punishes a wrong parallelization decision on its own.

    This is the property that separates this benchmark from one where the
    dependency graph is a label the scorer consults. Here the graph is enforced
    by the artifacts: a node that ignores its predecessor cannot produce the
    right file, because the definitions it needs do not exist yet.
    """

    def test_fanning_out_a_chain_breaks_every_downstream_node(self) -> None:
        sc = _scenario("chain", 4)
        with tempfile.TemporaryDirectory() as td:
            root = Path(sc.materialize(td))
            # An agent that spawns four subagents at once: each transforms the
            # seed, because nothing upstream has been written yet.
            for task in sc.subtasks.values():
                (root / task.output).write_text(
                    apply_family(task.family, sc.seed, task.index, task.size),
                    encoding="utf-8",
                )
            results = sc.verify(td)
        order = sc.dag.topo_order
        self.assertTrue(results[order[0]].passed, "the root has no unmet dependency")
        for node_id in order[1:]:
            self.assertFalse(results[node_id].passed, f"{node_id} should have failed")

    def test_fanning_out_a_wide_scenario_is_correct(self) -> None:
        # The symmetric half: on independent work the same fully-parallel
        # execution is right. Without this, the test above would be satisfied
        # by a payload that simply punishes all delegation.
        sc = _scenario("wide", 4)
        with tempfile.TemporaryDirectory() as td:
            root = Path(sc.materialize(td))
            for task in sc.subtasks.values():
                (root / task.output).write_text(
                    apply_family(task.family, sc.seed, task.index, task.size),
                    encoding="utf-8",
                )
            self.assertTrue(sc.succeeded(td), sc.verify(td))

    def test_a_missing_artifact_fails_rather_than_crashes(self) -> None:
        sc = _scenario("chain", 3)
        with tempfile.TemporaryDirectory() as td:
            sc.materialize(td)
            results = sc.verify(td)
        self.assertFalse(any(results.values()))
        self.assertIn("missing", results[sc.dag.topo_order[0]].reason)


class TestFootprintsAreDisjoint(unittest.TestCase):
    """Concurrent subagents must not be able to collide (section 7).

    A merge conflict between two subagents would be a confound: the run would
    fail for a reason that has nothing to do with the delegation decision under
    test. Disjointness is structural here -- one node, one output file -- and
    this test is what keeps it that way.
    """

    def test_independent_nodes_never_share_an_output(self) -> None:
        for shape, n in (("wide", 6), ("diamond", 6), ("mixed", 7)):
            with self.subTest(shape=shape):
                sc = _scenario(shape, n)
                for a in sc.dag.ids:
                    for b in sc.dag.ids:
                        if a != b and sc.dag.is_independent(a, b):
                            self.assertNotEqual(
                                sc.subtasks[a].output, sc.subtasks[b].output
                            )

    def test_a_nodes_inputs_are_exactly_its_predecessors(self) -> None:
        sc = _scenario("mixed", 7)
        for node_id, task in sc.subtasks.items():
            preds = sc.dag.preds[node_id]
            expected = (
                tuple(f"work/{p}.py" for p in sorted(preds, key=sc.dag.topo_order.index))
                if preds
                else (SEED_PATH,)
            )
            self.assertEqual(task.inputs, expected, node_id)


class TestVerifierDiscriminates(unittest.TestCase):
    """It has to reject wrong answers and accept differently-shaped right ones."""

    def test_wrong_rename_target_is_caught(self) -> None:
        dag = DAG((Node("n0", "rename", 1),), frozenset())
        sc = build_scenario(dag, "rename-1")
        wrong = sc.reference["n0"].replace("core_0", "core_wrong")
        self.assertFalse(verify_output(wrong, sc.reference["n0"], "n0").passed)

    def test_port_with_the_arguments_not_swapped_is_caught(self) -> None:
        # The failure mode a regex-minded agent actually produces: the call is
        # migrated, the name is right, and the operands are backwards. Only a
        # behavioural check catches this -- a textual diff on names would not.
        dag = DAG((Node("n0", "port", 1),), frozenset())
        sc = build_scenario(dag, "port-1")
        swapped = sc.reference["n0"].replace(
            "return b - a + 0 if mode == 'strict' else a - b + 0",
            "return a - b + 0 if mode == 'strict' else b - a + 0",
        )
        self.assertNotEqual(swapped, sc.reference["n0"], "the sabotage must apply")
        result = verify_output(swapped, sc.reference["n0"], "n0")
        self.assertFalse(result.passed)
        self.assertIn("wrong value", result.reason)

    def test_missing_annotations_are_caught(self) -> None:
        # Annotations do not change what a function computes, so a values-only
        # verifier would pass this. The annotate family exists precisely to
        # exercise a check that behaviour alone cannot make.
        dag = DAG((Node("n0", "annotate", 1),), frozenset())
        sc = build_scenario(dag, "annotate-1")
        stripped = sc.reference["n0"].replace("(x: int) -> int", "(x)")
        self.assertNotEqual(stripped, sc.reference["n0"], "the sabotage must apply")
        result = verify_output(stripped, sc.reference["n0"], "n0")
        self.assertFalse(result.passed)
        self.assertIn("annotations differ", result.reason)

    def test_reformatting_a_correct_answer_still_passes(self) -> None:
        # An agent's whitespace, comments, and definition order are not the
        # thing being measured, and a verifier that scored them would report
        # style variance as delegation error.
        sc = _scenario("wide", 4)
        for node_id, text in sc.reference.items():
            noisy = "# agent scratch notes\n\n" + text.replace("\n\n", "\n\n\n") + "\n"
            self.assertTrue(verify_output(noisy, text, node_id).passed, node_id)

    def test_an_unimportable_artifact_fails_without_raising(self) -> None:
        sc = _scenario("wide", 3)
        node_id = sc.dag.topo_order[0]
        result = verify_output("def broken(:\n", sc.reference[node_id], node_id)
        self.assertFalse(result.passed)
        self.assertIn("does not import", result.reason)


class TestMerge(unittest.TestCase):
    """Multi-predecessor nodes combine their inputs by a rule stated in one line."""

    def test_a_stale_sibling_does_not_overwrite_an_edit(self) -> None:
        # Regression test for the merge correction. Under the obvious last-wins
        # rule this returns a == 1: source B carries an untouched copy of `a`
        # and clobbers source A's edit, so in a diamond only the last middle's
        # work reaches the sink and the join stops meaning anything.
        base = "def a(x):\n    return 1\n\n\ndef b(x):\n    return 2\n"
        edited_a = "def a(x):\n    return 10\n\n\ndef b(x):\n    return 2\n"
        edited_b = "def a(x):\n    return 1\n\n\ndef b(x):\n    return 20\n"
        values = probe_source(merge_modules([edited_a, edited_b], base=base))["values"]
        self.assertEqual((values["a"], values["b"]), ("10", "20"))

    def test_reformatting_is_not_mistaken_for_an_edit(self) -> None:
        # The correction judges change structurally, because the port family
        # unparses the whole module on its way past: every definition comes out
        # textually different from the seed while only one is really changed.
        base = "def a(x):\n    return 1\n\n\ndef b(x):\n    return 2\n"
        reformatted = "def a(x):\n\n    return  1\n\n\ndef b(x):\n    return 2\n"
        edited = "def a(x):\n    return 99\n\n\ndef b(x):\n    return 2\n"
        values = probe_source(merge_modules([edited, reformatted], base=base))["values"]
        self.assertEqual(values["a"], "99")

    def test_later_source_wins_a_name_collision(self) -> None:
        merged = merge_modules(["def f(x):\n    return 1\n", "def f(x):\n    return 2\n"])
        self.assertEqual(probe_source(merged)["values"]["f"], "2")

    def test_non_binding_statements_survive(self) -> None:
        # An agent is free to add an import; the merge must neither drop it nor
        # treat it as a collision with another file's import.
        merged = merge_modules(
            ["import math\n\ndef f(x):\n    return 1\n", "import json\n\ndef g(x):\n    return 2\n"]
        )
        self.assertIn("import math", merged)
        self.assertIn("import json", merged)

    def test_a_diamond_sink_needs_both_middles(self) -> None:
        # Dropping either middle's contribution must fail, or the diamond --
        # the shape the whole fan-out-then-join argument rests on -- would not
        # actually require the join.
        sc = _scenario("diamond", 5)
        sink = sc.dag.topo_order[-1]
        middles = sorted(sc.dag.preds[sink])
        self.assertGreaterEqual(len(middles), 2)
        for dropped in middles:
            kept = [m for m in middles if m != dropped]
            partial = merge_modules([sc.reference[m] for m in kept], base=sc.seed)
            task = sc.subtasks[sink]
            partial = apply_family(task.family, partial, task.index, task.size)
            with self.subTest(dropped=dropped):
                self.assertFalse(verify_output(partial, sc.reference[sink], sink).passed)


class TestSurface(unittest.TestCase):
    """What the agent is shown, and what it is not."""

    def test_the_graph_is_hidden_by_default(self) -> None:
        sc = _scenario("mixed", 6)
        surface = sc.surface()
        self.assertNotIn("Dependencies", surface)
        for node_id in sc.dag.ids:
            self.assertIn(node_id, surface)

    def test_disclosure_states_every_edge(self) -> None:
        # Section 7's DAG-disclosed condition: the ablation that separates
        # discovery failure from decision failure only means something if the
        # disclosed surface really carries the whole graph.
        sc = _scenario("mixed", 6)
        disclosed = sc.surface(disclose_dag=True)
        for u, v in sc.dag.edges:
            self.assertIn(f"`{u}` must finish before `{v}` starts", disclosed)

    def test_materialize_writes_the_seed_and_an_empty_work_dir(self) -> None:
        sc = _scenario("wide", 4)
        with tempfile.TemporaryDirectory() as td:
            root = Path(sc.materialize(td))
            self.assertTrue((root / SEED_PATH).is_file())
            self.assertTrue((root / "work").is_dir())
            self.assertEqual(list((root / "work").iterdir()), [])


class TestSeedScaling(unittest.TestCase):
    """Node size is the token-volume dial the run budget is priced in."""

    def test_size_scales_the_seed_monotonically(self) -> None:
        lengths = [len(build_seed(4, size)) for size in (1, 2, 4)]
        self.assertEqual(lengths, sorted(lengths))
        self.assertLess(lengths[0], lengths[-1])

    def test_each_index_gets_its_own_group(self) -> None:
        seed = build_seed(5, 1)
        for i in range(5):
            self.assertIn(f"def h{i}(x)", seed)
            self.assertIn(f"def legacy_op_{i}(a, b)", seed)


if __name__ == "__main__":
    unittest.main()
