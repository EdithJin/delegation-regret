"""Pins the stated-beta instrument (matrix.py + DATASET-BOUNDARY-DEFECT.md
section 5, step-2 amendment).

Two guarantees, each of which failed silently would corrupt the matrix:

1. The objective directive reaches the lead's FIRST user message verbatim, and
   the blind condition leaves no trace of any objective. If the directive were
   dropped, every stated-beta cell would silently run blind and the flip test
   would grade an agent on a price it never heard.
2. The frozen wordings do not drift. The directives are instrument constants;
   an edit between cells would make runs incomparable.
"""

import tempfile
import unittest

from generator.dag import sample_dag
from generator.scenario import build_scenario, module_path
from harness.client import Reply, ScriptedClient, ToolRequest
from harness.runner import run_agent

from matrix import DIRECTIVE_B0, DIRECTIVE_B1


def _scenario():
    return build_scenario(sample_dag("wide", 3, sizes=(1,), seed=5), "wide3", seed=5)


def _finish() -> Reply:
    return Reply(
        text="", tool_calls=(ToolRequest(id="f", name="finish", arguments={}),),
        input_tokens=10, output_tokens=5, total_s=1.0, ttfb_s=0.5,
    )


class _RecordingClient(ScriptedClient):
    """ScriptedClient that keeps the initial (system, user) it was started with."""

    def start(self, system: str, user: str, actor: str = "lead") -> list:
        if actor == "lead":
            self.lead_first_user = user
        return super().start(system, user, actor)


def _run_with(directive: str):
    scn = _scenario()
    ids = list(scn.dag.ids)
    fixes = [
        ToolRequest(id=f"w-{n}", name="write_file",
                    arguments={"path": module_path(n), "content": scn.reference[n]})
        for n in ids
    ]
    client = _RecordingClient(script={"lead": [
        Reply(text="", tool_calls=tuple(fixes), input_tokens=10, output_tokens=5,
              total_s=1.0, ttfb_s=0.5),
        _finish(),
    ]})
    with tempfile.TemporaryDirectory() as tmp:
        run_agent(scn, client, tmp, directive=directive)
    return client.lead_first_user


class TestStatedBetaDirective(unittest.TestCase):
    def test_stated_b1_directive_lands_verbatim_in_first_user_message(self) -> None:
        prompt = _run_with(DIRECTIVE_B1)
        self.assertIn(DIRECTIVE_B1.strip(), prompt)

    def test_stated_b0_directive_lands_verbatim_in_first_user_message(self) -> None:
        prompt = _run_with(DIRECTIVE_B0)
        self.assertIn(DIRECTIVE_B0.strip(), prompt)

    def test_blind_condition_carries_no_objective_text(self) -> None:
        prompt = _run_with("")
        self.assertNotIn("Objective:", prompt)
        self.assertNotIn("$1.00", prompt)

    def test_frozen_wordings_have_not_drifted(self) -> None:
        # The instrument constants, pinned. Changing them between cells makes
        # runs incomparable; changing them requires a new prospective specification.
        self.assertIn("$1.00 per", DIRECTIVE_B1)
        self.assertIn("minimizing total cost = dollars spent + $1.00 x elapsed minutes",
                      DIRECTIVE_B1)
        self.assertIn("Elapsed time costs nothing", DIRECTIVE_B0)
        for directive in (DIRECTIVE_B0, DIRECTIVE_B1):
            self.assertNotIn("spawn", directive.lower())  # neither may mention spawning


if __name__ == "__main__":
    unittest.main()
