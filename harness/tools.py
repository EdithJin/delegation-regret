"""The subagent tool surface, and a workspace that can execute it safely.

Design doc: Phase1-DelegationBench-Design.md section 7.

This is the harness's action space, and it is also the oracle's. Section 5.1 is
explicit that an oracle with powers the agent does not have systematically
penalizes the agent for a constraint it never faced, so the two must be pinned
to each other: delegation is one level deep, `MAX_CONCURRENCY` is published to
the model in the tool description rather than silently enforced, and the oracle
is capped identically.

The tool set is deliberately small and boring. Every tool an agent has to reason
about is a chance for a weaker model to fumble the protocol, and a protocol
fumble in the trace looks exactly like a bad delegation decision. Section 10's
open-weights leg is the binding constraint on that, which is what the smoke test
in `smoke.py` exists to check before the leg is committed to.

`Workspace` refuses to read or write outside its root. That is not a security
posture -- the code being run is ours -- it is a measurement guarantee: a run
that escaped into the repo would contaminate every subsequent scenario in the
same session, and the failure would surface days later as unexplained variance.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["MAX_CONCURRENCY", "Workspace", "ToolCall", "anthropic_tools", "openai_tools"]

# Published to the model in the spawn tool's description and matched exactly by
# the oracle's action space. A real harness and a real rate limit both impose
# one; an uncapped oracle would price a parallelism the agent cannot buy.
MAX_CONCURRENCY = 4

_SPAWN_DESCRIPTION = (
    "Delegate a self-contained piece of work to a fresh subagent that shares no "
    "context with you. The subagent sees only the instruction and file list you "
    "give it, runs on its own, and returns a single summary; you cannot talk to "
    f"it while it runs. At most {MAX_CONCURRENCY} subagents run at once, and "
    "subagents cannot themselves delegate. Spawning costs tokens and time, so "
    "delegate when the parallelism is worth more than the overhead."
)

TOOL_SPECS: list[tuple[str, str, dict]] = [
    (
        "list_files",
        "List files under a directory in the workspace.",
        {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory, default '.'"}},
        },
    ),
    (
        "read_file",
        "Read a UTF-8 text file from the workspace.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    ),
    (
        "write_file",
        "Write a UTF-8 text file to the workspace, creating or replacing it.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    ),
    (
        "run_python",
        "Run a short Python snippet with the workspace as the working directory "
        "and return its stdout and stderr.",
        {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    ),
    (
        "spawn_subagent",
        _SPAWN_DESCRIPTION,
        {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "The complete brief. The subagent sees nothing else.",
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Workspace paths the subagent may read.",
                },
            },
            "required": ["instruction"],
        },
    ),
    (
        "finish",
        "Declare the whole task complete. Call this exactly once, last.",
        {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    ),
]


def anthropic_tools() -> list[dict]:
    return [{"name": n, "description": d, "input_schema": s} for n, d, s in TOOL_SPECS]


def openai_tools() -> list[dict]:
    return [
        {"type": "function", "function": {"name": n, "description": d, "parameters": s}}
        for n, d, s in TOOL_SPECS
    ]


@dataclass
class ToolCall:
    """One tool invocation and what came back. The trace's atomic unit."""

    name: str
    arguments: dict
    result: str = ""
    ok: bool = True
    duration_s: float = 0.0


@dataclass
class Workspace:
    """A scenario directory an agent may act inside, and nowhere else."""

    root: Path
    calls: list[ToolCall] = field(default_factory=list)
    python_timeout: float = 30.0

    def _resolve(self, path: str) -> Path:
        target = (self.root / path).resolve()
        root = self.root.resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"path {path!r} escapes the workspace")
        return target

    # -- the tools -------------------------------------------------------

    def list_files(self, path: str = ".") -> str:
        target = self._resolve(path)
        if not target.is_dir():
            return f"not a directory: {path}"
        entries = sorted(
            (p.relative_to(self.root).as_posix() + ("/" if p.is_dir() else ""))
            for p in target.rglob("*")
            if "__pycache__" not in p.parts
        )
        return "\n".join(entries) or "(empty)"

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.is_file():
            return f"no such file: {path}"
        return target.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> str:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} characters to {path}"

    def run_python(self, code: str) -> str:
        try:
            done = subprocess.run(
                [sys.executable, "-c", code],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=self.python_timeout,
            )
        except subprocess.TimeoutExpired:
            return f"timed out after {self.python_timeout:g}s"
        parts = []
        if done.stdout:
            parts.append(done.stdout.rstrip())
        if done.stderr:
            parts.append("stderr:\n" + done.stderr.rstrip())
        return "\n".join(parts) or "(no output)"

    # -- dispatch --------------------------------------------------------

    def invoke(self, name: str, arguments: dict) -> ToolCall:
        """Run one tool call, recording it whether it succeeds or not.

        `spawn_subagent` is not executed here. A real spawn is the harness's
        job -- it owns the concurrency slots, the child's own trace, and the
        absorption accounting -- so this returns a placeholder and leaves the
        call in the record, which is exactly what the smoke test needs: proof
        the model can *emit* a well-formed spawn, without paying for one.
        """
        import time

        started = time.perf_counter()
        try:
            if name == "spawn_subagent":
                result, ok = "(spawn recorded; not executed in this mode)", True
            elif name == "finish":
                result, ok = "acknowledged", True
            elif name in {"list_files", "read_file", "write_file", "run_python"}:
                result, ok = str(getattr(self, name)(**arguments)), True
            else:
                result, ok = f"unknown tool: {name}", False
        except TypeError as exc:  # wrong or missing arguments: a protocol fumble
            result, ok = f"bad arguments for {name}: {exc}", False
        except Exception as exc:
            result, ok = f"{type(exc).__name__}: {exc}", False
        call = ToolCall(name, arguments, result, ok, time.perf_counter() - started)
        self.calls.append(call)
        return call

    def summary(self) -> dict:
        """Protocol reliability, which is what the open-weights leg is judged on."""
        return {
            "calls": len(self.calls),
            "failed": sum(1 for c in self.calls if not c.ok),
            "by_tool": {
                name: sum(1 for c in self.calls if c.name == name)
                for name in sorted({c.name for c in self.calls})
            },
            "spawns": sum(1 for c in self.calls if c.name == "spawn_subagent"),
        }


def parse_arguments(raw: object) -> dict:
    """Tool arguments as the model actually sends them.

    OpenAI-shaped endpoints deliver arguments as a JSON *string*, and weaker
    models routinely emit one that does not parse. That is precisely the failure
    the open-weights smoke test is looking for, so it is surfaced as a bad call
    rather than repaired.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw or "{}")
        except ValueError:
            raise ValueError(f"arguments are not valid JSON: {raw[:120]!r}") from None
        if not isinstance(parsed, dict):
            raise ValueError(f"arguments are not an object: {raw[:120]!r}")
        return parsed
    raise ValueError(f"unsupported argument payload: {type(raw).__name__}")
