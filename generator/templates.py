"""Executable subtasks: the payload that makes the dependency structure real.

Design doc: Phase1-DelegationBench-Design.md section 4, Stage 2.

The oracle reasons about an abstract DAG. This module is what turns a node of
that DAG into work a model can actually do and a verifier can actually check.
Three properties are load-bearing, and each one is a design constraint rather
than an implementation detail:

TOKEN-HEAVY BUT MECHANICAL. Section 5.3's triangle needs subtasks with enough
work per node that latency savings can beat spawn overhead, success probability
near 1 so the decision stays purely economic, and a low dollar cost per run.
Bulk renames, API ports, annotation passes, and derived tables satisfy all
three: many tokens, no reasoning cliff, deterministic answer.

DEPENDENCIES ARE REAL, NOT DECORATIVE. A node's inputs are its predecessors'
output files. Run a chain out of order and the downstream node reads a file that
does not exist yet, so it cannot produce the right artifact. The environment
punishes a wrong parallelization decision on its own, before the scorer sees it.

DISJOINT FOOTPRINTS BY CONSTRUCTION. Every node writes exactly one file, named
for the node. Independent nodes therefore never touch the same path, so
concurrent subagents cannot produce a merge conflict -- section 7 needs that to
be structurally impossible rather than merely unlikely, since a conflict would
be a confound and not a finding.

All four families operate on the same generated module structure, indexed by the
node's position in topological order. Because topo position strictly increases
along every edge, the target a node transforms is distinct from every target its
ancestors and descendants transform -- so no node's work is accidentally a no-op
after an upstream node has run, on any shape.

Verification is execution-based and compares BEHAVIOUR, not text. The reference
answer and the agent's answer are both imported in a subprocess and probed for
what they compute, which names they define, and how they are annotated. An agent
that reformats, reorders, or re-comments passes; an agent that renames the wrong
symbol does not.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Subtask",
    "CheckResult",
    "FAMILIES",
    "build_seed",
    "merge_modules",
    "apply_family",
    "instruction_for",
    "probe_source",
    "verify_output",
    "PROBE_ARG",
    "TABULATE_ARG",
    "SEED_MODULE",
]

# Fixed inputs the probe and the tabulate family evaluate at. Constants, not
# parameters: a verifier whose sampling point varies is not deterministic.
PROBE_ARG = 7
TABULATE_ARG = 11

# Where the fixture lands in a materialized workspace. It appears in the
# instructions, so it lives beside them rather than in the renderer.
SEED_MODULE = "seed/base.py"


@dataclass(frozen=True)
class Subtask:
    """One node's executable work, with its footprint stated explicitly.

    `inputs` is what the node reads -- its predecessors' outputs, or the seed
    file if it is a root. `output` is the single file it writes, and it is the
    whole footprint, which is what makes independent nodes conflict-free.
    """

    node_id: str
    family: str
    size: int
    index: int
    inputs: tuple[str, ...]
    output: str

    @property
    def instruction(self) -> str:
        return instruction_for(self)


@dataclass(frozen=True)
class CheckResult:
    node_id: str
    passed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.passed


# ------------------------------------------------------------------ the seed
#
# One seed module serves every family. For each index i it defines a helper, a
# deprecated two-argument operation, and `size * 3` call sites that use both.
# A family's transformation at index i touches only the i-th group, so what a
# node has to do is unambiguous and what it must leave alone is everything else.


def build_seed(count: int, size: int) -> str:
    """Source for a scenario's seed module: `count` groups of `size * 3` sites."""
    if count < 1 or size < 1:
        raise ValueError("count and size must both be at least 1")
    blocks = ['"""Generated fixture. Every group is independent of every other."""']
    for i in range(count):
        blocks.append(f"def h{i}(x):\n    return x * {2 + i} + {i}")
        blocks.append(f"def legacy_op_{i}(a, b):\n    return a - b + {i}")
        for j in range(size * 3):
            blocks.append(
                f"def g{i}_use_{j}(x):\n"
                f"    return h{i}(x) + legacy_op_{i}(x, {j}) + {j}"
            )
    return "\n\n\n".join(blocks) + "\n"


# ----------------------------------------------------------------- the merge
#
# A node with several predecessors must combine their modules before doing its
# own work. Combining by top-level binding, later input winning a collision, is
# deterministic and is a rule that can be stated in one sentence to an agent.


def _binding_name(stmt: ast.stmt) -> str | None:
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return stmt.name
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id
    return None


def _normalized(segment: str) -> str:
    """Structure of a statement, free of formatting. Used to spot real changes."""
    try:
        return ast.dump(ast.parse(segment))
    except SyntaxError:
        return segment


def merge_modules(sources: list[str], base: str | None = None) -> str:
    """Union of top-level definitions; on a collision the *changed* version wins.

    The obvious rule -- last source wins -- is wrong here, and wrong on the
    shape the benchmark most depends on. Every node carries its inputs forward
    whole, so in a diamond each middle's output still contains an untouched copy
    of every other middle's group. Under last-wins, the middle listed last
    overwrites its siblings' work with stale definitions, only its own changes
    survive into the sink, and dropping any other middle changes nothing. The
    join would be decorative: the sink would not actually need what it waited
    for, which is exactly the property the fan-out-then-join argument rests on.

    Because a node's index is its topological position, each group is owned by
    exactly one node, so at most one input can have modified any given
    definition and "keep the changed one" is unambiguous rather than a tiebreak.
    Change is judged structurally, against `base`, so a node that reformatted
    its whole module on the way past is not mistaken for one that edited it.

    Statements binding no single name (imports, bare expressions) are kept under
    synthetic keys so they survive and never collide -- an agent is free to add
    an import, and the merge must neither lose it nor crash on it.
    """
    if len(sources) == 1:
        return sources[0]
    base_sigs: dict[str, str] = {}
    if base is not None:
        for stmt in ast.parse(base).body:
            name = _binding_name(stmt)
            if name is not None:
                base_sigs[name] = _normalized(ast.get_source_segment(base, stmt) or "")

    kept: dict[str, str] = {}
    changed: set[str] = set()
    for src in sources:
        tree = ast.parse(src)
        for pos, stmt in enumerate(tree.body):
            segment = ast.get_source_segment(src, stmt) or ast.unparse(stmt)
            name = _binding_name(stmt)
            if name is None:
                kept[f"__stmt_{len(kept)}_{pos}"] = segment
                continue
            differs = name not in base_sigs or _normalized(segment) != base_sigs[name]
            if name in changed and not differs:
                continue  # a stale copy must not overwrite a sibling's edit
            kept[name] = segment
            if differs:
                changed.add(name)
    return "\n\n\n".join(kept.values()) + "\n"


# -------------------------------------------------------------- the families


def _rename(src: str, index: int, size: int) -> str:
    """Bulk symbol rename: `h{i}` becomes `core_{i}` at every occurrence."""
    return re.sub(rf"\bh{index}\b", f"core_{index}", src)


class _PortCalls(ast.NodeTransformer):
    """`legacy_op_i(a, b)` -> `op_v2_i(b, a, mode="strict")`.

    The argument swap is the point: a rename is a substitution, a port is a
    rewrite, and an agent that reaches for a regex gets the arguments backwards.
    """

    def __init__(self, index: int) -> None:
        self.old = f"legacy_op_{index}"
        self.new = f"op_v2_{index}"

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == self.old and len(node.args) == 2:
            return ast.Call(
                func=ast.Name(id=self.new, ctx=ast.Load()),
                args=[node.args[1], node.args[0]],
                keywords=[ast.keyword(arg="mode", value=ast.Constant(value="strict"))],
            )
        return node


def _port(src: str, index: int, size: int) -> str:
    tree = ast.parse(src)
    tree = _PortCalls(index).visit(tree)
    # Replace the deprecated definition with the new signature. Under mode
    # "strict" the operands are taken in the swapped order, so a correctly
    # migrated call site computes exactly what it computed before.
    replacement = ast.parse(
        f"def op_v2_{index}(a, b, mode='strict'):\n"
        f"    return (b - a + {index}) if mode == 'strict' else (a - b + {index})"
    ).body[0]
    body = []
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == f"legacy_op_{index}":
            body.append(replacement)
        else:
            body.append(stmt)
    tree.body = body
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def _annotate(src: str, index: int, size: int) -> str:
    """Add `x: int` / `-> int` to the index's call sites. Behaviour unchanged."""
    tree = ast.parse(src)
    prefix = f"g{index}_use_"
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name.startswith(prefix):
            for arg in stmt.args.args:
                arg.annotation = ast.Name(id="int", ctx=ast.Load())
            stmt.returns = ast.Name(id="int", ctx=ast.Load())
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def _tabulate(src: str, index: int, size: int) -> str:
    """Append `TABLE_{i}`: every call site in the group, evaluated at a fixed x.

    The only family that requires running the code rather than editing it, and
    the reason the reference implementation executes the module: the answer is
    a computed table, not a transformation of the text.
    """
    namespace: dict[str, object] = {}
    exec(compile(src, "<reference>", "exec"), namespace)  # noqa: S102 - generator-owned source
    prefix = f"g{index}_use_"
    table = {
        name: namespace[name](TABULATE_ARG)  # type: ignore[operator]
        for name in sorted(namespace)
        if name.startswith(prefix) and callable(namespace[name])
    }
    literal = ", ".join(f"{name!r}: {value!r}" for name, value in table.items())
    return src.rstrip("\n") + f"\n\n\nTABLE_{index} = {{{literal}}}\n"


@dataclass(frozen=True)
class Family:
    name: str
    apply: object  # (src, index, size) -> src
    describe: object  # (subtask) -> str


def _instr_rename(t: Subtask) -> str:
    return (
        f"Rename the helper `h{t.index}` to `core_{t.index}` everywhere it appears, "
        f"including every call site. Change nothing else."
    )


def _instr_port(t: Subtask) -> str:
    return (
        f"Migrate every call to the deprecated `legacy_op_{t.index}(a, b)` onto its "
        f"replacement `op_v2_{t.index}(a, b, mode='strict')`, which takes its operands "
        f"in the opposite order: a call `legacy_op_{t.index}(X, Y)` becomes "
        f"`op_v2_{t.index}(Y, X, mode='strict')`. Replace the old definition with the "
        f"new one so that every migrated call computes exactly what it computed before. "
        f"Leave every other group's operations alone."
    )


def _instr_annotate(t: Subtask) -> str:
    return (
        f"Add type annotations to every function whose name starts with "
        f"`g{t.index}_use_`: each parameter is `int` and each return is `int`. "
        f"Do not change what any function computes."
    )


def _instr_tabulate(t: Subtask) -> str:
    return (
        f"Append a module-level dict `TABLE_{t.index}` mapping the name of every "
        f"function whose name starts with `g{t.index}_use_` to that function's value "
        f"at x = {TABULATE_ARG}. Keys sorted. Leave the rest of the module unchanged."
    )


FAMILIES: dict[str, Family] = {
    "rename": Family("rename", _rename, _instr_rename),
    "port": Family("port", _port, _instr_port),
    "annotate": Family("annotate", _annotate, _instr_annotate),
    "tabulate": Family("tabulate", _tabulate, _instr_tabulate),
}


def apply_family(family: str, src: str, index: int, size: int) -> str:
    """The reference transformation for one node. Deterministic, no I/O."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {sorted(FAMILIES)}")
    return FAMILIES[family].apply(src, index, size)  # type: ignore[operator]


def instruction_for(subtask: Subtask) -> str:
    """What the agent is told to do at this node, inputs and output included."""
    family = FAMILIES[subtask.family]
    if len(subtask.inputs) == 1:
        source = f"Start from `{subtask.inputs[0]}`."
    else:
        listed = ", ".join(f"`{p}`" for p in subtask.inputs)
        source = (
            f"Combine the top-level definitions of {listed} into one module first. "
            f"Where the same definition appears in more than one of them, keep the "
            f"one that has been changed from `{SEED_MODULE}` -- at most one input "
            f"will have changed any given definition -- and keep the unchanged form "
            f"only if none of them changed it."
        )
    return f"{source} {family.describe(subtask)} Write the result to `{subtask.output}`."  # type: ignore[operator]


# ------------------------------------------------------------ the verifier
#
# Execution-based and behavioural. Both the reference answer and the agent's
# answer are imported in a subprocess and reduced to what they compute, which
# public names they bind, and how they are annotated. Comparing that instead of
# text is what keeps an agent's formatting choices out of the score -- and
# section 6 forbids an LLM judge anywhere in this loop, so the comparison has to
# be exact on something, and behaviour is the something worth being exact about.

_PROBE = r'''
import importlib.util, json, sys

def canon(value):
    if isinstance(value, dict):
        return "dict{" + ", ".join(f"{k!r}: {canon(v)}" for k, v in sorted(value.items(), key=repr)) + "}"
    if isinstance(value, (set, frozenset)):
        return "set{" + ", ".join(sorted(map(repr, value))) + "}"
    return repr(value)

spec = importlib.util.spec_from_file_location("subject", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

out = {"values": {}, "annotations": {}}
for name in sorted(vars(module)):
    if name.startswith("_"):
        continue
    value = getattr(module, name)
    if callable(value):
        for args in ((ARG,), (ARG, 3), ()):
            try:
                out["values"][name] = canon(value(*args))
                break
            except TypeError:
                continue
        else:
            out["values"][name] = "<uncallable>"
        annotations = getattr(value, "__annotations__", None) or {}
        out["annotations"][name] = {
            k: getattr(v, "__name__", str(v)) for k, v in sorted(annotations.items())
        }
    else:
        out["values"][name] = canon(value)
print(json.dumps(out, sort_keys=True))
'''


def probe_source(source: str, timeout: float = 30.0) -> dict:
    """Import `source` in a subprocess and report what it computes.

    Raises RuntimeError if the module does not import or the probe times out --
    both are verifier failures for the node, not crashes of the run.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subject = root / "subject_module.py"
        subject.write_text(source, encoding="utf-8")
        runner = root / "probe.py"
        runner.write_text(_PROBE.replace("ARG", str(PROBE_ARG)), encoding="utf-8")
        try:
            done = subprocess.run(
                [sys.executable, str(runner), str(subject)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"probe timed out after {timeout:g}s") from None
    if done.returncode != 0:
        tail = (done.stderr or "").strip().splitlines()
        raise RuntimeError(tail[-1] if tail else f"probe exited {done.returncode}")
    return json.loads(done.stdout)


def verify_output(actual: str | None, expected: str, node_id: str) -> CheckResult:
    """Compare an agent's artifact against the reference answer, behaviourally.

    `actual` is None when the file was never written -- the normal outcome for a
    node whose inputs did not exist when it ran, which is exactly how an
    incorrect parallelization decision surfaces as a failure.
    """
    if actual is None:
        return CheckResult(node_id, False, "output file missing")
    try:
        got = probe_source(actual)
    except RuntimeError as exc:
        return CheckResult(node_id, False, f"artifact does not import: {exc}")
    want = probe_source(expected)
    if got == want:
        return CheckResult(node_id, True)

    missing = sorted(set(want["values"]) - set(got["values"]))
    extra = sorted(set(got["values"]) - set(want["values"]))
    if missing:
        return CheckResult(node_id, False, f"missing definitions: {', '.join(missing[:4])}")
    if extra:
        return CheckResult(node_id, False, f"unexpected definitions: {', '.join(extra[:4])}")
    wrong = [n for n in want["values"] if got["values"][n] != want["values"][n]]
    if wrong:
        first = wrong[0]
        return CheckResult(
            node_id,
            False,
            f"{len(wrong)} definition(s) compute the wrong value, e.g. {first}: "
            f"got {got['values'][first]}, expected {want['values'][first]}",
        )
    bad = [n for n in want["annotations"] if got["annotations"].get(n) != want["annotations"][n]]
    return CheckResult(node_id, False, f"annotations differ on {len(bad)} function(s), e.g. {bad[0]}")
