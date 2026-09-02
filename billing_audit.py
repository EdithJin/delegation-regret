#!/usr/bin/env python3
"""Recheck saved trace/proxy pairs and total every archived proxy ledger.

The online runner stamps ``proxy_verified`` when its client-side usage parse
matches the independent logging proxy.  This offline audit repeats that check
from the saved artifacts, including cache-write tokens, so an old stamp is not
treated as proof merely because it is present in a trace.

Run from the benchmark root::

    python3 billing_audit.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
)
PRICE_SHEETS = {
    "claude-haiku-4-5": ROOT / "results" / "preflight-v2" / "calibration.json",
    "claude-opus-5": ROOT / "results" / "cal-opus-v2" / "calibration.json",
    "gpt-5.6-sol": ROOT / "results" / "cal-gpt-responses" / "calibration.json",
    "kimi-k3": ROOT / "results" / "cal-kimi" / "calibration.json",
}


def proxy_for_trace(trace_path: Path) -> Path | None:
    """Return the adjacent proxy path for every naming convention in results/."""
    name = trace_path.name
    candidates: list[Path] = []
    if name.startswith("trace-") and name.endswith(".json"):
        candidates.append(trace_path.with_name("proxy-" + name[6:-5] + ".jsonl"))
    if name.endswith("-trace.json"):
        candidates.append(trace_path.with_name(name[:-11] + "-proxy.jsonl"))
    if name.endswith(".json"):
        candidates.append(trace_path.with_name(name[:-5] + "-proxy.jsonl"))
    return next((path for path in candidates if path.exists()), None)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def field_total(rows: list[dict], field: str) -> int:
    return sum(int(row.get(field) or 0) for row in rows)


def load_prices() -> dict[str, dict]:
    prices: dict[str, dict] = {}
    for model, path in PRICE_SHEETS.items():
        raw = json.loads(path.read_text(encoding="utf-8"))
        sheet = raw["price_sheet"]
        if sheet["model"] != model:
            raise ValueError(f"{path}: price-sheet model {sheet['model']!r} != {model!r}")
        prices[model] = sheet
    return prices


def dollars(rows: list[dict], sheet: dict) -> float:
    rates = {
        "input_tokens": sheet["input_per_mtok"],
        "output_tokens": sheet["output_per_mtok"],
        "cache_read_tokens": sheet["cache_read_per_mtok"],
        "cache_write_tokens": sheet["cache_write_per_mtok"],
    }
    return sum(field_total(rows, field) * rate / 1_000_000 for field, rate in rates.items())


def audit(results_dir: Path) -> dict:
    prices = load_prices()
    pairs: list[tuple[Path, Path, dict]] = []
    for trace_path in sorted(results_dir.rglob("*.json")):
        proxy_path = proxy_for_trace(trace_path)
        if proxy_path is None:
            continue
        raw = json.loads(trace_path.read_text(encoding="utf-8"))
        if not isinstance(raw.get("calls"), list):
            continue
        pairs.append((trace_path, proxy_path, raw))

    mismatches: list[dict] = []
    model_pairs: Counter[str] = Counter()
    paired_spend_by_model: Counter[str] = Counter()
    billable_calls = proxy_calls = 0
    for trace_path, proxy_path, trace in pairs:
        records = load_jsonl(proxy_path)
        billable = [
            row
            for row in records
            if row.get("input_tokens") is not None and row.get("output_tokens") is not None
        ]
        calls = trace["calls"]
        model = str(trace.get("model") or "")
        model_pairs[model] += 1
        proxy_calls += len(records)
        billable_calls += len(billable)
        if model in prices:
            paired_spend_by_model[model] += dollars(billable, prices[model])

        problems: dict[str, object] = {}
        if len(billable) != len(calls):
            problems["billable_call_count"] = {
                "trace": len(calls),
                "proxy": len(billable),
            }
        recorded_proxy_calls = trace.get("proxy_calls")
        if recorded_proxy_calls is not None and recorded_proxy_calls != len(records):
            problems["proxy_call_count"] = {
                "trace": recorded_proxy_calls,
                "proxy": len(records),
            }
        for field in FIELDS:
            trace_total = field_total(calls, field)
            proxy_total = field_total(billable, field)
            if trace_total != proxy_total:
                problems[field] = {"trace": trace_total, "proxy": proxy_total}

        proxy_models = {
            str(row.get("model"))
            for row in billable
            if row.get("model") is not None
        }
        if proxy_models and proxy_models != {model}:
            problems["model"] = {"trace": model, "proxy": sorted(proxy_models)}
        if model not in prices:
            problems["price_sheet"] = {"missing_for_model": model}

        if problems:
            mismatches.append(
                {
                    "trace": str(trace_path.relative_to(ROOT)),
                    "proxy": str(proxy_path.relative_to(ROOT)),
                    "problems": problems,
                }
            )

    # Spend is a property of the archived ledgers, including an aborted run for
    # which no complete trace exists.  Sum every proxy file exactly once rather
    # than silently conditioning the compute disclosure on trace survival.
    ledger_spend_by_model: Counter[str] = Counter()
    all_proxy_paths = sorted(results_dir.rglob("*proxy*.jsonl"))
    paired_proxy_paths = {proxy.resolve() for _, proxy, _ in pairs}
    unpaired_proxy_ledgers: list[str] = []
    unpriced_billable_records: list[dict] = []
    archived_proxy_calls = archived_billable_calls = 0
    for proxy_path in all_proxy_paths:
        records = load_jsonl(proxy_path)
        archived_proxy_calls += len(records)
        billable = [
            row
            for row in records
            if row.get("input_tokens") is not None and row.get("output_tokens") is not None
        ]
        archived_billable_calls += len(billable)
        if proxy_path.resolve() not in paired_proxy_paths:
            unpaired_proxy_ledgers.append(str(proxy_path.relative_to(ROOT)))
        for model in {str(row.get("model") or "") for row in billable}:
            model_rows = [row for row in billable if str(row.get("model") or "") == model]
            if not model or model not in prices:
                unpriced_billable_records.append(
                    {
                        "proxy": str(proxy_path.relative_to(ROOT)),
                        "model": model or None,
                        "records": len(model_rows),
                    }
                )
                continue
            ledger_spend_by_model[model] += dollars(model_rows, prices[model])

    return {
        "results_root": str(results_dir.relative_to(ROOT)),
        "paired_traces": len(pairs),
        "archived_proxy_ledgers": len(all_proxy_paths),
        "archived_proxy_calls": archived_proxy_calls,
        "archived_billable_calls": archived_billable_calls,
        "paired_proxy_calls": proxy_calls,
        "paired_billable_calls": billable_calls,
        "pairs_by_model": dict(sorted(model_pairs.items())),
        "paired_spend_by_model": dict(sorted(paired_spend_by_model.items())),
        "archived_ledger_spend_by_model": dict(sorted(ledger_spend_by_model.items())),
        "total_spend_dollars": sum(ledger_spend_by_model.values()),
        "unpaired_proxy_ledgers": unpaired_proxy_ledgers,
        "unpriced_billable_records": unpriced_billable_records,
        "mismatches": mismatches,
    }


def main() -> int:
    report = audit(ROOT / "results")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["mismatches"] or report["unpriced_billable_records"] else 0


if __name__ == "__main__":
    sys.exit(main())
