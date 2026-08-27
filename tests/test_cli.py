"""The entry point's wiring, tested without a key and without a run.

What a CLI test can and cannot cover here: the drivers themselves
(`run_calibration`, `run_experiment`) are exercised end to end against
`ProtocolUpstream` in tests/test_calibration.py and tests/test_experiment.py.
What is new in harness/cli.py is the part a driver test never sees -- the
pinned price sheets, the refusals, and the argument-to-object wiring -- and a
mistake there is exactly the kind that surfaces at 11pm against a paid
endpoint. So this file tests the wiring and the refusals, and leaves the
driving to the driver tests.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from datetime import date
from pathlib import Path

from generator.manifest import CORE, load as load_manifest
from harness.calibrate import PriceSheet
from harness.calibration import CalibrationResult
from harness.cli import PRICE_SHEETS, calibration_scenario, main, price_sheet


def run_cli(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = main(list(argv))
    return code, out.getvalue()


class TestPriceSheets(unittest.TestCase):
    def test_every_sheet_is_complete_and_dated(self) -> None:
        for key, sheet in PRICE_SHEETS.items():
            self.assertIsInstance(sheet, PriceSheet)
            date.fromisoformat(sheet.as_of)  # raises if undated or malformed
            for rate in (sheet.input_per_mtok, sheet.output_per_mtok,
                         sheet.cache_read_per_mtok, sheet.cache_write_per_mtok):
                self.assertGreater(rate, 0.0, key)

    def test_registry_key_names_the_model_it_prices(self) -> None:
        # "claude-sonnet-5@list" prices claude-sonnet-5; every key is the model
        # id plus an optional @variant. A key pricing a different model would
        # let --model select a sheet that bills the wrong rates silently.
        for key, sheet in PRICE_SHEETS.items():
            self.assertEqual(key.split("@")[0], sheet.model, key)

    def test_cache_pricing_carries_the_pinned_ttl_premium(self) -> None:
        # Reads at 0.1x input, writes at the 2x one-hour-TTL premium. If the
        # pinned TTL ever changes to 5 minutes, the write premium is 1.25x and
        # this test is the reminder that the sheets must change with it.
        for key, sheet in PRICE_SHEETS.items():
            self.assertAlmostEqual(sheet.cache_read_per_mtok, 0.1 * sheet.input_per_mtok,
                                   places=6, msg=key)
            self.assertAlmostEqual(sheet.cache_write_per_mtok, 2.0 * sheet.input_per_mtok,
                                   places=6, msg=key)

    def test_unknown_model_is_refused_with_the_roster(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            price_sheet("claude-nonexistent")
        self.assertIn("claude-opus-5", str(ctx.exception))


class TestModelsCommand(unittest.TestCase):
    def test_prints_every_pinned_sheet(self) -> None:
        code, out = run_cli("models")
        self.assertEqual(code, 0)
        for key in PRICE_SHEETS:
            self.assertIn(key, out)


class TestManifestCommand(unittest.TestCase):
    def test_core_roundtrips_with_its_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            code, out = run_cli("manifest", "--out", str(path))
            self.assertEqual(code, 0)
            self.assertIn(CORE.fingerprint, out)
            self.assertEqual(load_manifest(path).fingerprint, CORE.fingerprint)

    def test_seed_trim_shrinks_the_set_and_changes_the_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            code, _ = run_cli("manifest", "--seeds", "11,23", "--out", str(path))
            self.assertEqual(code, 0)
            trimmed = load_manifest(path)
            self.assertEqual(len(trimmed), 24)  # 36 core specs, one of three seeds dropped
            self.assertNotEqual(trimmed.fingerprint, CORE.fingerprint)
            self.assertNotEqual(trimmed.name, CORE.name)

    def test_a_trim_keeping_nothing_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                run_cli("manifest", "--seeds", "999", "--out", str(Path(tmp) / "m.json"))

    def test_heldout_seeds_never_collide_with_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "heldout.json"
            code, _ = run_cli("manifest", "--set", "heldout", "--out", str(path))
            self.assertEqual(code, 0)
            core_seeds = {s.seed for s in CORE.specs}
            for spec in load_manifest(path).specs:
                self.assertNotIn(spec.seed, core_seeds)


class _Args:
    shape = "wide"
    n = 6
    sizes = "2,3,5"
    seed = 11


class TestCalibrationScenario(unittest.TestCase):
    def test_default_flags_yield_varied_node_sizes(self) -> None:
        # The whole reason the defaults exist: uniform outputs make the timing
        # model unfittable and calibration.py refuses the fit. If a defaults
        # change trips this, pick a seed whose draw is mixed.
        scenario = calibration_scenario(_Args())
        sizes = {node.size for node in scenario.dag.nodes}
        self.assertGreaterEqual(len(sizes), 2)
        self.assertTrue(sizes <= {2, 3, 5})

    def test_same_flags_rebuild_the_identical_scenario(self) -> None:
        # --replay re-derives constants from traces against a rebuilt scenario,
        # which only means anything if rebuilding is exact.
        a, b = calibration_scenario(_Args()), calibration_scenario(_Args())
        self.assertEqual(a.id, b.id)
        self.assertEqual([n.size for n in a.dag.nodes], [n.size for n in b.dag.nodes])
        self.assertEqual(a.dag.edges, b.dag.edges)


class TestReplayWritesTheFileTheNextCommandNeeds(unittest.TestCase):
    """The rescue case: a killed run whose traces survived but whose extraction
    never ran. `replay` itself is a checker and never writes -- the CLI must,
    or its own printed `next: --calibration ...` hint points at a file that
    does not exist. That is exactly how the first real calibration ended."""

    def test_replay_writes_calibration_json(self) -> None:
        from generator.templates import module_path
        from harness.trace import LEAD, ModelCall, Trace, WriteEvent

        scn = calibration_scenario(_Args())
        with tempfile.TemporaryDirectory() as out:
            trace = Trace(scenario_id=scn.id, model="claude-opus-5")
            for i, node in enumerate(scn.dag.ids):
                trace.calls.append(
                    ModelCall(actor=LEAD, index=i, input_tokens=700 + 300 * i,
                              output_tokens=60 + 40 * i, total_s=4.0 + i,
                              t_request=float(i))
                )
                trace.write_events.append(
                    WriteEvent(path=module_path(node), actor=LEAD, t=float(i) + 0.5)
                )
            trace.write(Path(out) / "trace-serial-o0-r0.json")
            code, _ = run_cli("calibrate", "--model", "claude-opus-5", "--replay",
                              "--out", out)
            self.assertEqual(code, 0)
            self.assertTrue((Path(out) / "calibration.json").exists())
            # And a second replay against its own published file is silent
            # agreement, not a spurious overwrite warning.
            code, text = run_cli("calibrate", "--model", "claude-opus-5", "--replay",
                                 "--out", out)
            self.assertEqual(code, 0)
            self.assertNotIn("DISAGREES", text)


class TestExperimentRefusals(unittest.TestCase):
    def _calibration(self, tmp: str, *, model: str, timing: dict) -> Path:
        result = CalibrationResult(
            source=f"{model} test", price_sheet={"model": model}, timing_model=timing,
        )
        return result.write(tmp)

    def _manifest(self, tmp: str) -> Path:
        path = Path(tmp) / "manifest.json"
        run_cli("manifest", "--seeds", "11", "--out", str(path))
        return path

    def test_a_calibration_from_another_model_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calib = self._calibration(tmp, model="claude-haiku-4-5",
                                      timing={"a_minutes": 0.01,
                                              "b_minutes_per_input_token": 1e-7,
                                              "output_tokens_per_minute": 3000.0})
            with self.assertRaises(SystemExit) as ctx:
                run_cli("experiment", "--model", "claude-opus-5",
                        "--manifest", str(self._manifest(tmp)),
                        "--calibration", str(calib), "--out", tmp)
            self.assertIn("claude-haiku-4-5", str(ctx.exception))

    def test_a_calibration_without_a_timing_model_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calib = self._calibration(tmp, model="claude-opus-5", timing={})
            with self.assertRaises(SystemExit) as ctx:
                run_cli("experiment", "--model", "claude-opus-5",
                        "--manifest", str(self._manifest(tmp)),
                        "--calibration", str(calib), "--out", tmp)
            self.assertIn("timing model", str(ctx.exception))

    def test_a_missing_api_key_is_a_message_not_a_traceback(self) -> None:
        # Refusals above fire before the client is built; a run that passes them
        # must then fail on the absent key, naming the variable to export.
        with tempfile.TemporaryDirectory() as tmp:
            calib = self._calibration(tmp, model="claude-opus-5",
                                      timing={"a_minutes": 0.01,
                                              "b_minutes_per_input_token": 1e-7,
                                              "output_tokens_per_minute": 3000.0})
            with self.assertRaises(SystemExit) as ctx:
                run_cli("experiment", "--model", "claude-opus-5",
                        "--manifest", str(self._manifest(tmp)),
                        "--calibration", str(calib), "--out", tmp,
                        "--api-key-env", "DELEGATION_TEST_KEY_THAT_IS_NOT_SET")
            self.assertIn("DELEGATION_TEST_KEY_THAT_IS_NOT_SET", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
