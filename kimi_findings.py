"""Recompute the report's Kimi K3-only findings from saved artifacts.

The matrix, ladder, mini-calibration, and full calibration all live in this
canonical checkout. This script checks every cell against its trace and proxy
log and fails loudly if a report-facing denominator changes.

Run from the benchmark root:

    python3 kimi_findings.py
"""

from __future__ import annotations

import json
from pathlib import Path

from harness.calibrate import PriceSheet, TimingModel
from harness.calibration import CalibrationResult
from harness.trace import LEAD, Trace


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
KIMI = "kimi-k3"
SIZES = (3, 8, 15, 25)


def ratio(rows: list[dict], predicate=lambda row: row["k"] > 0) -> dict[str, int]:
    return {"numerator": sum(bool(predicate(row)) for row in rows), "runs": len(rows)}


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT.parent))


def load_timekeeper() -> tuple[CalibrationResult, PriceSheet, TimingModel]:
    path = RESULTS / "cal-kimi" / "calibration.json"
    calibration = CalibrationResult.load(path)
    price = PriceSheet(**calibration.price_sheet)
    timing = TimingModel(**calibration.timing_model)
    assert price.model == KIMI, path
    return calibration, price, timing


def proxy_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def matrix_rows(price: PriceSheet, timing: TimingModel) -> list[dict]:
    rows: list[dict] = []
    for cell_path in sorted((RESULTS / "matrix-kimi").glob("*/cell.json")):
        raw = json.loads(cell_path.read_text(encoding="utf-8"))
        assert raw["model"] == KIMI, cell_path
        meta = raw["cell"]
        for run, recorded in enumerate(raw["runs"]):
            trace_path = cell_path.parent / f"run-{run}-trace.json"
            proxy_path = cell_path.parent / f"run-{run}-proxy.jsonl"
            trace = Trace.load(trace_path)
            proxies = proxy_rows(proxy_path)
            assert trace.model == KIMI, trace_path
            assert trace.k == recorded["k"], trace_path
            assert trace.succeeded == recorded["succeeded"], trace_path
            assert trace.proxy_verified == recorded["proxy_verified"], trace_path
            assert len(proxies) == trace.proxy_calls == len(trace.calls), trace_path
            assert all(200 <= row["status"] < 300 for row in proxies), proxy_path

            batches: dict[int, int] = {}
            for spawn in trace.spawns:
                batches[spawn.batch] = batches.get(spawn.batch, 0) + 1
            lead_calls = [call for call in trace.calls if call.actor == LEAD]
            lead_writes = [event for event in trace.write_events if event.actor == LEAD]
            module_writes = {
                event.path for event in lead_writes if event.path.startswith("pkg/mod_")
            }
            last_lead = lead_calls[-1]
            rows.append(
                {
                    "source": relative(trace_path),
                    "cell": meta["id"],
                    "shape": meta["shape"],
                    "size": meta["size"],
                    "seed": meta["seed"],
                    "condition": meta["cond"],
                    "reference": meta["cond"] == "ref",
                    "undisclosed": "undisc" in meta["id"],
                    "run": run,
                    "k": trace.k,
                    "batch_sizes": sorted(batches.values(), reverse=True),
                    "succeeded": trace.succeeded,
                    "finished": trace.finished,
                    "proxy_verified": trace.proxy_verified,
                    "calls": len(trace.calls),
                    "dollars": trace.dollars(price),
                    "analytic_minutes": trace.analytic_minutes(timing),
                    "objective_beta1": trace.objective(price, timing, 1.0),
                    "wall_minutes": trace.wall_seconds / 60.0,
                    "node_passes": sum(trace.verdicts.values()),
                    "nodes": len(trace.verdicts),
                    "last_lead_stop": last_lead.stop_reason,
                    "last_lead_output_tokens": last_lead.output_tokens,
                    "last_lead_tools": list(last_lead.tools_invoked),
                    "lead_module_writes": len(module_writes),
                    "lead_used_python": any(
                        "run_python" in call.tools_invoked for call in lead_calls
                    ),
                    "lead_python_calls": sum(
                        "run_python" in call.tools_invoked for call in lead_calls
                    ),
                    "contested_writes": len(trace.contested_writes),
                }
            )
    return rows


def matrix_summary(rows: list[dict], calibration: CalibrationResult) -> dict:
    agents = [row for row in rows if not row["reference"]]
    references = [row for row in rows if row["reference"]]

    def select(*, condition: str, shape: str, undisclosed: bool | None = None) -> list[dict]:
        selected = [
            row
            for row in agents
            if row["condition"] == condition and row["shape"] == shape
        ]
        if undisclosed is not None:
            selected = [row for row in selected if row["undisclosed"] == undisclosed]
        return selected

    failed = [row for row in rows if not row["succeeded"]]
    successful = [row for row in rows if row["succeeded"]]
    successful_wide_agents = [
        row for row in agents if row["shape"] == "wide" and row["succeeded"]
    ]
    w25_stated = [row for row in agents if row["cell"] == "w25-stated-b1"]
    w25_success = [row for row in w25_stated if row["succeeded"]]
    assert len(w25_success) == 1

    return {
        "runs": {
            "cells": len({row["cell"] for row in rows}),
            "all": len(rows),
            "agent": len(agents),
            "reference": len(references),
            "succeeded": len(successful),
            "proxy_verified": sum(row["proxy_verified"] for row in rows),
            "calls": sum(row["calls"] for row in rows),
            "dollars": sum(row["dollars"] for row in rows),
            "contested_writes": sum(row["contested_writes"] for row in rows),
        },
        "decision_tallies": {
            "all_free_choice": ratio(agents),
            "blind_wide": ratio(select(condition="blind", shape="wide")),
            "blind_chain": ratio(select(condition="blind", shape="chain")),
            "stated_beta1_wide_disclosed": ratio(
                select(condition="stated-b1", shape="wide", undisclosed=False)
            ),
            "stated_beta1_wide_undisclosed": ratio(
                select(condition="stated-b1", shape="wide", undisclosed=True)
            ),
            "stated_beta1_chain": ratio(
                select(condition="stated-b1", shape="chain")
            ),
            "stated_beta0_wide": ratio(
                select(condition="stated-b0", shape="wide")
            ),
        },
        "quality": {
            "succeeded": ratio(rows, lambda row: row["succeeded"]),
            "node_pass_patterns": sorted(
                {
                    f'{row["node_passes"]}/{row["nodes"]}' for row in rows
                }
            ),
            "failures": len(failed),
            "failed_cells": [row["cell"] for row in failed],
            "all_failures_free_choice": all(not row["reference"] for row in failed),
            "all_failures_serial": all(row["k"] == 0 for row in failed),
            "all_failures_unfinished": all(not row["finished"] for row in failed),
            "all_failures_zero_nodes": all(row["node_passes"] == 0 for row in failed),
            "all_failures_end_at_length": all(
                row["last_lead_stop"] == "length" for row in failed
            ),
            "all_failures_end_at_16000_tokens": all(
                row["last_lead_output_tokens"] == 16000 for row in failed
            ),
            "all_failures_end_without_tool_call": all(
                not row["last_lead_tools"] for row in failed
            ),
            "failure_sizes": sorted({row["size"] for row in failed}),
        },
        "successful_free_choice_strategy": {
            "wide_runs": len(successful_wide_agents),
            "wide_used_python": sum(row["lead_used_python"] for row in successful_wide_agents),
            "wide_zero_recorded_module_writes": sum(
                row["lead_module_writes"] == 0 for row in successful_wide_agents
            ),
            "w25_stated_beta1": {
                "successful_runs": len(w25_success),
                "failed_runs": len(w25_stated) - len(w25_success),
                "successful_objective": w25_success[0]["objective_beta1"],
                "successful_module_writes": w25_success[0]["lead_module_writes"],
                "successful_python_calls": w25_success[0]["lead_python_calls"],
                "beta1_materiality_floor": (
                    calibration.floors["dollars"] + calibration.floors["minutes"]
                ),
            },
        },
    }


def ladder_row(size: int, calibration: CalibrationResult) -> dict:
    path = RESULTS / f"probe-kimi-wide4-s{size}" / "probe.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    serial = raw["measured"]["serial"]
    fanout = raw["measured"]["fanout"]
    serial_trace = Trace.load(path.parent / "serial-trace.json")
    fanout_trace = Trace.load(path.parent / "fanout-trace.json")
    assert raw["model"] == serial_trace.model == fanout_trace.model == KIMI, path
    assert raw["valid"] and serial_trace.succeeded and fanout_trace.succeeded, path
    batches: dict[int, int] = {}
    for spawn in fanout_trace.spawns:
        batches[spawn.batch] = batches.get(spawn.batch, 0) + 1
    serial_output = sum(call.output_tokens for call in serial_trace.calls)
    fanout_output = sum(call.output_tokens for call in fanout_trace.calls)
    subagent_output = sum(
        call.output_tokens for call in fanout_trace.calls if call.actor != LEAD
    )
    dollar_premium = fanout["dollars"] - serial["dollars"]
    minutes_saved = serial["minutes"] - fanout["minutes"]
    nominal_beta = dollar_premium / minutes_saved if minutes_saved > 0 else None
    assert nominal_beta == raw["measured_beta_star"]
    return {
        "source": relative(path),
        "scenario": raw["scenario"],
        "size": size,
        "serial": {"dollars": serial["dollars"], "minutes": serial["minutes"]},
        "fanout": {"dollars": fanout["dollars"], "minutes": fanout["minutes"]},
        "dollar_premium": dollar_premium,
        "minutes_saved": minutes_saved,
        "dollar_difference_clears_floor": (
            abs(dollar_premium) > calibration.floors["dollars"]
        ),
        "latency_difference_clears_floor": (
            abs(minutes_saved) > calibration.floors["minutes"]
        ),
        "observed_dominated": dollar_premium > 0 and minutes_saved < 0,
        "nominal_beta_star": nominal_beta,
        "floor_cleared_beta_star": (
            nominal_beta
            if dollar_premium > calibration.floors["dollars"]
            and minutes_saved > calibration.floors["minutes"]
            else None
        ),
        "fanout_batch_sizes": sorted(batches.values(), reverse=True),
        "serial_output_tokens": serial_output,
        "fanout_output_tokens": fanout_output,
        "fanout_to_serial_output_ratio": fanout_output / serial_output,
        "subagent_to_serial_output_ratio": subagent_output / serial_output,
    }


def ladder_summary(calibration: CalibrationResult) -> dict:
    rows = [ladder_row(size, calibration) for size in SIZES]
    return {
        "rungs": rows,
        "forced_fanout_one_batch": ratio(
            rows, lambda row: row["fanout_batch_sizes"] == [4]
        ),
        "aggregate_output_exceeds_serial": ratio(
            rows, lambda row: row["fanout_to_serial_output_ratio"] > 1
        ),
        "floor_cleared_finite_boundaries": [
            row["size"] for row in rows if row["floor_cleared_beta_star"] is not None
        ],
        "nonmonotonic_observed_latency": [row["minutes_saved"] for row in rows],
    }


def forced_execution_patterns() -> list[dict]:
    paths = [
        *[
            (f"ladder-size-{size}", RESULTS / f"probe-kimi-wide4-s{size}" / "fanout-trace.json")
            for size in SIZES
        ],
        ("matrix-max-fanout", RESULTS / "matrix-kimi/w15b-ref-fanout/run-0-trace.json"),
        ("matrix-split-2+2", RESULTS / "matrix-kimi/int25-split22/run-0-trace.json"),
        ("mini-calibration-fanout", RESULTS / "preflight-kimi/trace-fanout.json"),
        ("full-calibration-fanout", RESULTS / "cal-kimi/trace-fanout.json"),
    ]
    rows = []
    for label, path in paths:
        trace = Trace.load(path)
        batches: dict[int, int] = {}
        for spawn in trace.spawns:
            batches[spawn.batch] = batches.get(spawn.batch, 0) + 1
        rows.append(
            {
                "label": label,
                "source": relative(path),
                "k": trace.k,
                "batch_sizes": sorted(batches.values(), reverse=True),
            }
        )
    return rows


def saved_call_census(price: PriceSheet) -> dict:
    groups = {
        "mini_calibration": sorted((RESULTS / "preflight-kimi").glob("trace-*.json")),
        "full_calibration": sorted((RESULTS / "cal-kimi").glob("trace-*.json")),
        "ladder": sorted(RESULTS.glob("probe-kimi-wide4-s*/*-trace.json")),
        "matrix": sorted((RESULTS / "matrix-kimi").glob("*/*-trace.json")),
    }
    traces = [(group, path, Trace.load(path)) for group, paths in groups.items() for path in paths]
    calls = [(group, path, call) for group, path, trace in traces for call in trace.calls]
    subagent_calls = [row for row in calls if row[2].actor != LEAD]
    proxy_records: list[dict] = []
    for _, path, trace in traces:
        if path.name.startswith("trace-"):
            proxy_name = "proxy-" + path.name.removeprefix("trace-").removesuffix(".json") + ".jsonl"
        else:
            proxy_name = path.name.removesuffix("-trace.json") + "-proxy.jsonl"
        records = proxy_rows(path.with_name(proxy_name))
        assert len(records) == trace.proxy_calls == len(trace.calls), path
        proxy_records.extend(records)
    return {
        "scope": {
            "mini_calibration": "bench/results/preflight-kimi/",
            "full_calibration": "bench/results/cal-kimi/",
            "ladder": "bench/results/probe-kimi-wide4-s{3,8,15,25}/",
            "matrix": "bench/results/matrix-kimi/",
            "excludes": "unarchived ad-hoc wire probes",
        },
        "traces": len(traces),
        "proxy_verified_traces": sum(trace.proxy_verified for _, _, trace in traces),
        "calls": len(calls),
        "all_proxy_calls_2xx": all(200 <= row["status"] < 300 for row in proxy_records),
        "calls_with_cache_reads": sum(call.cache_read_tokens > 0 for _, _, call in calls),
        "cache_read_tokens": sum(call.cache_read_tokens for _, _, call in calls),
        "subagent_calls": len(subagent_calls),
        "lead_length_stops": sum(call.stop_reason == "length" for _, _, call in calls if call.actor == LEAD),
        "subagent_length_stops": sum(call.stop_reason == "length" for _, _, call in subagent_calls),
        "all_models_kimi": all(call.actor and trace.model == KIMI for _, _, trace in traces for call in trace.calls),
        "groups": {
            group: {
                "traces": len(paths),
                "calls": sum(len(trace.calls) for name, _, trace in traces if name == group),
                "dollars": round(
                    sum(trace.dollars(price) for name, _, trace in traces if name == group),
                    10,
                ),
            }
            for group, paths in groups.items()
        },
        "saved_artifact_dollars": round(
            sum(trace.dollars(price) for _, _, trace in traces), 10
        ),
    }


def calibration_summary(calibration: CalibrationResult) -> dict:
    dollars = calibration.diagnostics["dollar_holdout"]
    minutes = calibration.diagnostics["minute_holdout"]
    composition = calibration.diagnostics["composition"]
    return {
        "source": "bench/results/cal-kimi/calibration.json",
        "timing_calls": calibration.timing_model["n_calls"],
        "prefill_source": calibration.timing_model["prefill_source"],
        "floors": calibration.floors,
        "beta1_objective_floor": calibration.floors["dollars"] + calibration.floors["minutes"],
        "holdout": {
            "dollars_mean_relative_error": dollars["mean_rel_error"],
            "dollars_max_relative_error": dollars["max_rel_error"],
            "minutes_mean_relative_error": minutes["mean_rel_error"],
            "minutes_max_relative_error": minutes["max_rel_error"],
        },
        "composition": composition,
    }


def main() -> None:
    calibration, price, timing = load_timekeeper()
    rows = matrix_rows(price, timing)
    matrix = matrix_summary(rows, calibration)
    ladder = ladder_summary(calibration)
    saved_calls = saved_call_census(price)
    patterns = forced_execution_patterns()

    w25_free = matrix["successful_free_choice_strategy"]["w25_stated_beta1"]
    w25_forced = next(row for row in ladder["rungs"] if row["size"] == 25)
    w25_free["forced_fanout_objective"] = (
        w25_forced["fanout"]["dollars"] + w25_forced["fanout"]["minutes"]
    )
    w25_free["free_minus_forced_fanout"] = (
        w25_free["successful_objective"] - w25_free["forced_fanout_objective"]
    )
    w25_free["difference_inside_materiality_floor"] = (
        abs(w25_free["free_minus_forced_fanout"])
        < w25_free["beta1_materiality_floor"]
    )

    report = {
        "model": KIMI,
        "calibration": calibration_summary(calibration),
        "ladder": ladder,
        "matrix": matrix,
        "forced_execution_patterns": patterns,
        "saved_call_census": saved_calls,
    }

    # Report-facing invariants.
    assert ladder["forced_fanout_one_batch"] == {"numerator": 4, "runs": 4}
    assert ladder["aggregate_output_exceeds_serial"] == {"numerator": 4, "runs": 4}
    assert ladder["floor_cleared_finite_boundaries"] == [25]
    assert round(w25_forced["floor_cleared_beta_star"], 3) == 0.047
    assert matrix["runs"] == {
        "cells": 21,
        "all": 35,
        "agent": 30,
        "reference": 5,
        "succeeded": 30,
        "proxy_verified": 35,
        "calls": 328,
        "dollars": 10.4159388,
        "contested_writes": 0,
    }
    assert matrix["decision_tallies"] == {
        "all_free_choice": {"numerator": 0, "runs": 30},
        "blind_wide": {"numerator": 0, "runs": 8},
        "blind_chain": {"numerator": 0, "runs": 3},
        "stated_beta1_wide_disclosed": {"numerator": 0, "runs": 11},
        "stated_beta1_wide_undisclosed": {"numerator": 0, "runs": 1},
        "stated_beta1_chain": {"numerator": 0, "runs": 2},
        "stated_beta0_wide": {"numerator": 0, "runs": 5},
    }
    assert matrix["quality"]["failures"] == 5
    assert matrix["quality"]["node_pass_patterns"] == ["0/4", "4/4"]
    assert all(
        matrix["quality"][key]
        for key in (
            "all_failures_free_choice",
            "all_failures_serial",
            "all_failures_unfinished",
            "all_failures_zero_nodes",
            "all_failures_end_at_length",
            "all_failures_end_at_16000_tokens",
            "all_failures_end_without_tool_call",
        )
    )
    assert saved_calls["traces"] == 54
    assert saved_calls["proxy_verified_traces"] == 54
    assert saved_calls["calls"] == 691
    assert saved_calls["all_proxy_calls_2xx"]
    assert saved_calls["calls_with_cache_reads"] == 600
    assert saved_calls["cache_read_tokens"] == 2_868_224
    assert saved_calls["subagent_calls"] == 164
    assert saved_calls["lead_length_stops"] == 5
    assert saved_calls["subagent_length_stops"] == 0
    assert round(saved_calls["saved_artifact_dollars"], 6) == 15.736174
    assert [row["batch_sizes"] for row in patterns] == [
        [4], [4], [4], [4], [4], [1, 1], [1, 1, 1], [4, 2]
    ]
    assert w25_free["difference_inside_materiality_floor"]

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
