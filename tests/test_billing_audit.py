from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from billing_audit import audit


class TestSavedBillingAudit(unittest.TestCase):
    def _pair(self, root: Path, *, trace_writes: int, proxy_writes: int) -> None:
        (root / "run-0-trace.json").write_text(
            json.dumps(
                {
                    "model": "claude-opus-5",
                    "proxy_calls": 1,
                    "calls": [
                        {
                            "input_tokens": 2,
                            "output_tokens": 3,
                            "cache_read_tokens": 4,
                            "cache_write_tokens": trace_writes,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (root / "run-0-proxy.jsonl").write_text(
            json.dumps(
                {
                    "model": "claude-opus-5",
                    "input_tokens": 2,
                    "output_tokens": 3,
                    "cache_read_tokens": 4,
                    "cache_write_tokens": proxy_writes,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def test_matching_pair_passes(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            self._pair(root, trace_writes=5, proxy_writes=5)
            report = audit(root)
        self.assertEqual(report["paired_traces"], 1)
        self.assertEqual(report["mismatches"], [])
        self.assertEqual(report["archived_proxy_ledgers"], 1)
        self.assertEqual(report["archived_proxy_calls"], 1)
        self.assertEqual(report["archived_billable_calls"], 1)
        self.assertEqual(report["unpriced_billable_records"], [])
        self.assertGreater(report["total_spend_dollars"], 0.0)

    def test_cache_write_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            self._pair(root, trace_writes=5, proxy_writes=6)
            report = audit(root)
        self.assertIn("cache_write_tokens", report["mismatches"][0]["problems"])

    def test_unpaired_billable_ledger_requires_a_price_sheet(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            (root / "orphan-proxy.jsonl").write_text(
                json.dumps(
                    {
                        "model": "unknown-model",
                        "input_tokens": 2,
                        "output_tokens": 3,
                        "cache_read_tokens": 0,
                        "cache_write_tokens": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = audit(root)
        self.assertEqual(report["mismatches"], [])
        self.assertEqual(report["unpriced_billable_records"][0]["records"], 1)


if __name__ == "__main__":
    unittest.main()
