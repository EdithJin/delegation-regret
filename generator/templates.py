"""The executable payload: working modules, generated tests, and injected defects.

Design doc: Phase1-DelegationBench-Design.md section 4, Stage 2.
Low-level design: low-level-design.md, "Stage 2".

A node's task is "make this module's tests pass." The generator writes a package
of small modules with definite behaviour, generates a test suite by EXECUTING the
correct version, then breaks each module in `size` distinct ways.

WHY NOT TRANSFORMATIONS. The previous payload asked for bulk edits -- rename this
helper at every call site, port these calls to a new signature -- and sized a node
by how many call sites it had. That makes `size` count REPETITIONS OF ONE
DECISION, and repetition is precisely what a script eliminates: an agent with a
code-execution tool collapses the whole scenario into one find-and-replace,
finishes in about three turns, and spawns nothing on any shape. Two things break.
The benchmark observes no delegation decisions to score, and the cost model's
`cost proportional to size` assumption fails, because a script over two hundred
sites costs what a script over twenty does.

The fix has to live in the TASK, not the harness. Restricting the tool surface
would work here and would not carry to Claude Code or Codex, which bring their own
tools -- and comparing against shipped products is the point. So `size` counts
DISTINCT decisions instead. Five unlike defects require five separate read-locate-
fix cycles, and there is no regular expression that finds them.

HOW DEPENDENCIES WORK, AND HOW STRONG THEY ARE. A successor module imports its
predecessors and calls into them, so while a predecessor is broken the successor's
tests fail for reasons that have nothing to do with the successor. Every function
in a successor calls a predecessor function, so a predecessor defect always
propagates -- a dependency that bit only sometimes would be worse than none.

Be precise about what that buys, because it is weaker than what it replaces. The
old file dependency was HARD: a successor read a file that did not exist until its
predecessor ran, so it could not be completed early. This one is SOFT. An agent
can open a successor module, find its defect, and fix it correctly while the
predecessor is still broken -- it simply cannot CONFIRM the fix. The honest
formulation is "no verified result for the successor until the predecessor is
correct," not "no work on the successor." That is the dependency shape real
software has, and the penalty for parallelising a chain is real but finite.

The risk this creates is worth writing down where the code lives: the oracle
models edges as strictly blocking, so if agents do make genuine out-of-order
progress, a real agent can beat the "optimal" plan. The signature is systematic
NEGATIVE REGRET on chain and diamond scenarios. Check traces for out-of-order
edits and report the rate rather than assuming it is zero.
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass

__all__ = [
    "Defect",
    "Subtask",
    "CheckResult",
    "DEFECT_KINDS",
    "module_path",
    "test_path",
    "test_module_dotted",
    "build_module",
    "inject",
    "build_tests",
    "PACKAGE",
    "TEST_DIR",
]

PACKAGE = "pkg"
TEST_DIR = "tests"

# Probe arguments the generated tests call each function with. Fixed, small, and
# including a negative and a zero so a defect that only shows on one sign is not
# invisible to the suite that is supposed to catch it.
PROBE_ARGS = (3, 0, -2, 11)


# ------------------------------------------------------------------ the pieces


@dataclass(frozen=True)
class Defect:
    """One injected fault, and where it went.

    `kind` is drawn from DEFECT_KINDS. Each kind produces a different failure
    signature, which is what stops a node from being one decision applied N
    times: five unlike defects cannot be found by one edit.
    """

    kind: str
    function: str


@dataclass(frozen=True)
class Subtask:
    """One node's work: a module to repair and the suite that judges it."""

    node_id: str
    module: str  # "pkg/mod_n0.py" -- this node's SOLE footprint
    test_module: str  # "tests/test_n0.py"
    defects: tuple[Defect, ...]
    imports: tuple[str, ...]  # predecessor node ids
    size: int  # == len(defects)

    @property
    def instruction(self) -> str:
        return (
            f"Make the tests in `{self.test_module}` pass by repairing "
            f"`{self.module}`. Modify no other file, and do not edit any test."
        )


@dataclass(frozen=True)
class CheckResult:
    node_id: str
    passed: bool
    detail: str = ""

    def __bool__(self) -> bool:
        return self.passed


def module_path(node_id: str) -> str:
    return f"{PACKAGE}/mod_{node_id}.py"


def test_path(node_id: str) -> str:
    return f"{TEST_DIR}/test_{node_id}.py"


def test_module_dotted(node_id: str) -> str:
    return f"{TEST_DIR}.test_{node_id}"


def _fn(node_id: str, i: int) -> str:
    return f"step_{node_id}_{i}"


# --------------------------------------------------------- generating modules


# Body templates. Each is a single return over one integer parameter, so every
# function has definite behaviour, runs instantly, and cannot loop forever in a
# verifier subprocess. Variety exists so that injected defects land on genuinely
# different structures rather than on N copies of one expression.
#
# SPLIT BY WHAT THEY ADMIT, because the split is load-bearing. Two of the five
# defect kinds are STRUCTURAL: `swapped_branches` needs a conditional expression
# and `inverted_condition` needs a comparison. Only the conditional bodies carry
# either. So the body mix is not free -- it decides which defect kinds can be
# placed at all, and a module of pure arithmetic can only ever receive arithmetic
# defects.
_COND_BODIES = (
    "return ({u} + {a}) // {b} if {u} >= 0 else {u} - {b}",
    "return {a} * {u} - {b} if {u} > {a} else {b} - {u}",
)
_PLAIN_BODIES = (
    "return {u} * {a} + {b}",
    "return ({u} - {a}) * {b}",
    "return {u} * {u} - {a}",
    "return abs({u} - {a}) + {b}",
)
_BODIES = _PLAIN_BODIES + _COND_BODIES

# Kinds that need a conditional body. Everything else lands on any arithmetic.
STRUCTURAL_KINDS = ("inverted_condition", "swapped_branches")


def _body_plan(size: int, rng: random.Random) -> list[str]:
    """Which body shape each of a module's `size` functions gets.

    Enough conditional bodies are reserved to let the defect deal come out
    BALANCED -- see `inject`. A balanced deal wants each of the five kinds
    ceil(size/5) times, and the two structural kinds need a conditional function
    each, so the quota is 2 * ceil(size/5), capped at `size`.

    Without this the mix was whatever `rng.choice` produced, and the structural
    kinds appeared about a quarter as often as the arithmetic ones -- not because
    of the kind selection but because there was nowhere to put them. Two nodes of
    equal `size` then carried materially different work, which is exactly the
    drift the cost model cannot see.
    """
    if size <= 0:
        return []
    want_cond = min(size, 2 * -(-size // len(DEFECT_KINDS)))
    plan = [_COND_BODIES[i % len(_COND_BODIES)] for i in range(want_cond)]
    plan += [_PLAIN_BODIES[i % len(_PLAIN_BODIES)] for i in range(size - want_cond)]
    rng.shuffle(plan)
    return plan


def build_module(
    node_id: str,
    size: int,
    imports: tuple[str, ...],
    rng: random.Random,
) -> str:
    """The CORRECT module for one node. Defects are injected separately.

    Every function takes one integer and returns one integer. Functions in a
    module with predecessors call into them, so the import edge is load-bearing
    rather than decorative: a broken predecessor makes this module's tests fail.
    """
    lines = [f'"""Module owned by node {node_id}."""', ""]
    upstream: list[str] = []
    for dep in imports:
        # Import a fixed, known name from each predecessor. Which one it is does
        # not matter; that it is called by every function here does.
        lines.append(f"from {PACKAGE}.mod_{dep} import {_fn(dep, 0)}")
        upstream.append(_fn(dep, 0))
    if imports:
        lines.append("")

    plan = _body_plan(size, rng)
    for i in range(size):
        a = rng.randint(2, 9)
        b = rng.randint(2, 9)
        body = plan[i]
        if upstream:
            # Route the argument through a predecessor. Every function does this,
            # so any predecessor defect propagates into every test here.
            call = upstream[i % len(upstream)]
            expr = body.format(u=f"{call}(x)", a=a, b=b)
        else:
            expr = body.format(u="x", a=a, b=b)
        lines += [f"def {_fn(node_id, i)}(x):", f"    {expr}", ""]
    return "\n".join(lines).rstrip() + "\n"


# ----------------------------------------------------------- injecting defects


DEFECT_KINDS = (
    "off_by_one",
    "wrong_constant",
    "flipped_operator",
    "inverted_condition",
    "swapped_branches",
)


class _Mutate(ast.NodeTransformer):
    """Apply exactly one defect of a given kind inside one function."""

    def __init__(self, kind: str, rng: random.Random) -> None:
        self.kind = kind
        self.rng = rng
        self.done = False

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if self.done or self.kind not in ("off_by_one", "wrong_constant"):
            return node
        if not isinstance(node.value, int) or isinstance(node.value, bool):
            return node
        self.done = True
        options = (1, -1) if self.kind == "off_by_one" else (3, 5, -4)
        # Never land on zero: a mutated divisor of 0 turns a findable wrong
        # answer into a crash, which is a different and worse task.
        deltas = [d for d in options if node.value + d != 0] or [max(options)]
        return ast.Constant(value=node.value + self.rng.choice(deltas))

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if self.done or self.kind != "flipped_operator":
            return node
        swap = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult}
        repl = swap.get(type(node.op))
        if repl is None:
            return node
        self.done = True
        return ast.BinOp(left=node.left, op=repl(), right=node.right)

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if self.done or self.kind != "inverted_condition" or len(node.ops) != 1:
            return node
        swap = {ast.Gt: ast.LtE, ast.GtE: ast.Lt, ast.Lt: ast.GtE, ast.LtE: ast.Gt}
        repl = swap.get(type(node.ops[0]))
        if repl is None:
            return node
        self.done = True
        return ast.Compare(left=node.left, ops=[repl()], comparators=node.comparators)

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.generic_visit(node)
        if self.done or self.kind != "swapped_branches":
            return node
        self.done = True
        return ast.IfExp(test=node.test, body=node.orelse, orelse=node.body)


def _applicable(fn: ast.FunctionDef, rng: random.Random) -> tuple[str, ...]:
    """Which defect kinds can actually be placed in this function."""
    out = []
    for kind in DEFECT_KINDS:
        probe = _Mutate(kind, random.Random(0))
        probe.visit(ast.parse(ast.unparse(fn)))
        if probe.done:
            out.append(kind)
    return tuple(out)


def inject(source: str, size: int, rng: random.Random) -> tuple[str, tuple[Defect, ...]]:
    """Break `size` distinct functions, one defect each, in a BALANCED mix.

    Returns the defective source and the manifest.

    WHY BALANCE IS NOT COSMETIC. `size` is the cost model's only handle on how
    much work a node is, and `block_dollars` prices a block on total size units
    while explicitly not looking at which nodes compose it. That approximation
    only holds if a unit of size means the same thing everywhere. Defect kinds
    are not equally hard to find -- a swapped conditional reads differently from
    an off-by-one -- so a node that happened to draw three off-by-ones is not the
    same work as one that drew five different kinds, even though both report
    `size == 5`. The variance lands inside the measured block curve as noise and
    the oracle mis-prices every lopsided node.

    Measured before the fix, over 120 size-5 nodes: the two structural kinds
    appeared 0.36-0.39 times per node against 1.32-1.48 for the arithmetic ones,
    and single nodes carried the same kind three times.

    THE DEAL. Kinds are dealt so that each appears floor(size/5) or ceil(size/5)
    times. Functions are visited MOST-CONSTRAINED FIRST -- a conditional body
    admits all five kinds, an arithmetic one admits only three -- because
    assigning the flexible functions first can strand a structural kind with
    nowhere to go. Among the kinds a function admits, the least-used wins.
    `_body_plan` has already reserved enough conditional bodies for the deal to
    come out even.

    A defect that cannot be placed anywhere raises rather than being skipped: a
    node claiming `size` defects while carrying fewer would make `size` mean two
    things in one run.
    """
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if size > len(funcs):
        raise ValueError(f"cannot inject {size} defects into {len(funcs)} functions")
    targets = rng.sample(funcs, size)

    options = {id(fn): _applicable(fn, rng) for fn in targets}
    for fn in targets:
        if not options[id(fn)]:
            raise ValueError(f"no defect kind applies to {fn.name}")

    used = {kind: 0 for kind in DEFECT_KINDS}
    # Tie-break order is shuffled PER MODULE, and that matters whenever `size` is
    # not a multiple of the kind count. With a fixed tie-break, a size-3 node
    # always drew the same first three kinds, so every small node in the
    # benchmark was arithmetic-only while size-5 nodes carried structural defects
    # too. The block curve would then measure a CHANGING MIX as size rose,
    # conflating "more work" with "different work" -- which is precisely the
    # confound the balanced deal exists to remove. Shuffling spreads the
    # remainder uniformly across the population while leaving the within-node
    # balance untouched, since `used[k]` dominates the key.
    priority = list(DEFECT_KINDS)
    rng.shuffle(priority)
    # Fewest options first; assigning the flexible functions first can strand a
    # structural kind with nowhere to go.
    order = sorted(targets, key=lambda fn: (len(options[id(fn)]), fn.name))
    chosen: dict[int, str] = {}
    for fn in order:
        pick = min(options[id(fn)], key=lambda k: (used[k], priority.index(k)))
        chosen[id(fn)] = pick
        used[pick] += 1

    manifest: list[Defect] = []
    for fn in targets:  # manifest follows source order, not deal order
        kind = chosen[id(fn)]
        mut = _Mutate(kind, rng)
        candidate = mut.visit(ast.parse(ast.unparse(fn)))
        if not mut.done:  # pragma: no cover -- _applicable already proved it fits
            raise ValueError(f"{kind} failed to apply to {fn.name}")
        tree.body[tree.body.index(fn)] = candidate.body[0]
        manifest.append(Defect(kind=kind, function=fn.name))

    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n", tuple(manifest)


# ----------------------------------------------------------- generating tests


def build_tests(node_id: str, expected: dict[str, list[tuple[int, int]]]) -> str:
    """The suite for one node, asserting values captured from the CORRECT modules.

    `expected` comes from executing the pristine package, so the answer key is
    derived rather than authored -- there is no hand-written expectation to be
    wrong, and no model anywhere in the ground-truth path.
    """
    lines = [
        f'"""Generated suite for node {node_id}. Do not edit."""',
        "",
        "import unittest",
        "",
        f"from {PACKAGE}.mod_{node_id} import (",
    ]
    lines += [f"    {name}," for name in sorted(expected)]
    lines += [")", "", "", f"class TestNode{node_id.upper()}(unittest.TestCase):"]
    for name in sorted(expected):
        lines.append(f"    def test_{name}(self):")
        for arg, want in expected[name]:
            lines.append(f"        self.assertEqual({name}({arg}), {want})")
        lines.append("")
    lines += ['if __name__ == "__main__":', "    unittest.main()"]
    return "\n".join(lines) + "\n"
