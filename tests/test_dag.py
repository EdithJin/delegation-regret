"""Structural invariants of the generator.

The DAG is the ground truth every downstream number is derived from, so a
malformed graph is not a caught error later -- it is a wrong benchmark result
that still looks plausible. These tests are cheap and they run first.
"""

from __future__ import annotations

import unittest

from generator.dag import DAG, Node, sample_dag, SHAPES


class TestValidation(unittest.TestCase):
    def test_duplicate_ids_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DAG((Node("a"), Node("a")), frozenset())

    def test_self_edge_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DAG((Node("a"),), frozenset({("a", "a")}))

    def test_unknown_endpoint_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DAG((Node("a"),), frozenset({("a", "b")}))

    def test_cycle_rejected_at_construction(self) -> None:
        # Not deferred to first use: __post_init__ touches topo_order so a cyclic
        # graph can never be handed to the oracle in the first place.
        with self.assertRaises(ValueError):
            DAG((Node("a"), Node("b")), frozenset({("a", "b"), ("b", "a")}))


class TestShapes(unittest.TestCase):
    def test_every_shape_yields_requested_node_count(self) -> None:
        # diamond takes width but sample_dag adapts it, so "an n-node scenario"
        # means the same thing across shapes. Cost is priced in nodes.
        for shape in SHAPES:
            for n in range(3, 9):
                self.assertEqual(sample_dag(shape, n).n, n, f"{shape} n={n}")

    def test_wide_is_fully_independent(self) -> None:
        dag = sample_dag("wide", 6)
        self.assertEqual(dag.edges, frozenset())
        for a in dag.ids:
            for b in dag.ids:
                if a != b:
                    self.assertTrue(dag.is_independent(a, b))

    def test_chain_is_totally_ordered(self) -> None:
        dag = sample_dag("chain", 6)
        self.assertEqual(len(dag.edges), 5)
        for a in dag.ids:
            for b in dag.ids:
                if a != b:
                    self.assertFalse(dag.is_independent(a, b))

    def test_diamond_middles_are_parallel(self) -> None:
        dag = sample_dag("diamond", 6)  # n0 -> n1..n4 -> n5
        mids = ["n1", "n2", "n3", "n4"]
        self.assertEqual(len(dag.edges), 2 * len(mids))
        for a in mids:
            for b in mids:
                if a != b:
                    self.assertTrue(dag.is_independent(a, b))
        self.assertEqual(dag.preds["n5"], frozenset(mids))
        self.assertEqual(dag.succs["n0"], frozenset(mids))

    def test_topo_order_respects_every_edge(self) -> None:
        for shape in SHAPES:
            dag = sample_dag(shape, 8, sizes=(1, 2, 3))
            pos = {v: i for i, v in enumerate(dag.topo_order)}
            self.assertEqual(len(pos), dag.n)
            for u, v in dag.edges:
                self.assertLess(pos[u], pos[v], f"{shape}: {u}->{v}")

    def test_reachability_agrees_with_edge_closure(self) -> None:
        for shape in SHAPES:
            dag = sample_dag(shape, 7)
            for u in dag.ids:
                self.assertNotIn(u, dag.reachable[u])
                for v in dag.succs[u]:
                    self.assertIn(v, dag.reachable[u])
                    self.assertTrue(dag.reachable[v] <= dag.reachable[u])


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_graph(self) -> None:
        # Scenarios are reproducible from their id; nothing is stored.
        for shape in SHAPES:
            a = sample_dag(shape, 7, sizes=(1, 2, 3), seed=11)
            b = sample_dag(shape, 7, sizes=(1, 2, 3), seed=11)
            self.assertEqual(a.edges, b.edges)
            self.assertEqual(a.nodes, b.nodes)

    def test_different_seed_changes_a_random_shape(self) -> None:
        a = sample_dag("mixed", 8, seed=1)
        b = sample_dag("mixed", 8, seed=2)
        self.assertNotEqual(a.edges, b.edges)


if __name__ == "__main__":
    unittest.main()
