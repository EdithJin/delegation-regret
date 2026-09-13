"""Recompute the reported GPT-only findings from saved artifacts.

All canonical and supplemental GPT artifacts are stored in this checkout.
This script prints the exact reported denominators. In particular,
it does not silently pool the reasoning-none retry runs with the canonical
matrix.

Run from the benchmark root:

    python3 gpt_findings.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.calibrate import PriceSheet, TimingModel
from harness.calibration import CalibrationResult
from harness.trace import LEAD, Trace


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
GPT = "gpt-5.6-sol"
PACKED_CONFIRM_SIZES = (3, 8, 15, 25)
PACKED_CONFIRM_REPEATS = (1, 2)


def ratio(rows: list[dict], predicate=lambda row: row["k"] > 0) -> dict[str, int]:
    return {"numerator": sum(bool(predicate(row)) for row in rows), "runs": len(rows)}


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_calibration(path: Path) -> tuple[CalibrationResult, PriceSheet, TimingModel]:
    calibration = CalibrationResult.load(path)
    price = PriceSheet(**calibration.price_sheet)
    timing = TimingModel(**calibration.timing_model)
    assert price.model == GPT, path
    return calibration, price, timing


def load_proxy_statuses(path: Path) -> list[int]:
    return [
        json.loads(line)["status"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_matrix(matrix_dir: Path, price: PriceSheet, timing: TimingModel) -> list[dict]:
    rows: list[dict] = []
    for cell_path in sorted(matrix_dir.glob("*/cell.json")):
        raw = json.loads(cell_path.read_text(encoding="utf-8"))
        assert raw["model"] == GPT, cell_path
        meta = raw["cell"]
        for run, recorded in enumerate(raw["runs"]):
            trace_path = cell_path.parent / f"run-{run}-trace.json"
            proxy_path = cell_path.parent / f"run-{run}-proxy.jsonl"
            trace = Trace.load(trace_path)
            statuses = load_proxy_statuses(proxy_path)
            assert trace.model == GPT, trace_path
            assert trace.k == recorded["k"], trace_path
            assert trace.succeeded == recorded["succeeded"], trace_path
            assert trace.proxy_verified == recorded["proxy_verified"], trace_path
            assert len(statuses) == trace.proxy_calls, trace_path
            assert sum(200 <= status < 300 for status in statuses) == len(trace.calls), trace_path

            batches = {spawn.batch for spawn in trace.spawns}
            lead_writes = [event for event in trace.write_events if event.actor == LEAD]
            lead_module_writes = {
                event.path for event in lead_writes if event.path.startswith("pkg/mod_")
            }
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
                    "spawn_batches": len(batches),
                    "packed": trace.k > 1 and len(batches) == 1,
                    "serialized": trace.k > 1 and len(batches) == trace.k,
                    "lead_writes": len(lead_writes),
                    "lead_module_writes": len(lead_module_writes),
                    "lead_used_python": any(
                        call.actor == LEAD and "run_python" in call.tools_invoked
                        for call in trace.calls
                    ),
                    "contested_writes": len(trace.contested_writes),
                    "succeeded": trace.succeeded,
                    "proxy_verified": trace.proxy_verified,
                    "calls": len(trace.calls),
                    "all_2xx": all(200 <= status < 300 for status in statuses),
                    "dollars": trace.dollars(price),
                    "analytic_minutes": trace.analytic_minutes(timing),
                    "wall_minutes": trace.wall_seconds / 60.0,
                }
            )
    return rows


def matrix_summary(rows: list[dict]) -> dict:
    agents = [row for row in rows if not row["reference"]]
    references = [row for row in rows if row["reference"]]
    blind_wide = [
        row for row in agents if row["shape"] == "wide" and row["condition"] == "blind"
    ]
    stated_one_wide_disclosed = [
        row
        for row in agents
        if row["shape"] == "wide"
        and row["condition"] == "stated-b1"
        and not row["undisclosed"]
    ]
    undisclosed_wide = [row for row in agents if row["undisclosed"]]
    stated_zero_wide = [
        row
        for row in agents
        if row["shape"] == "wide" and row["condition"] == "stated-b0"
    ]
    blind_chain = [
        row for row in agents if row["shape"] == "chain" and row["condition"] == "blind"
    ]
    stated_one_chain = [
        row
        for row in agents
        if row["shape"] == "chain" and row["condition"] == "stated-b1"
    ]
    free_multi = [row for row in agents if row["k"] > 1]
    forced_multi = [row for row in references if row["k"] > 1]
    failed = [row for row in rows if not row["succeeded"]]
    serial_wide_agents = [
        row for row in agents if row["shape"] == "wide" and row["k"] == 0
    ]
    serial_chain_agents = [
        row for row in agents if row["shape"] == "chain" and row["k"] == 0
    ]
    stated_chain_cell = [row for row in agents if row["cell"] == "c15-stated-b1"]
    stated_chain_serial = [row for row in stated_chain_cell if row["k"] == 0]
    stated_chain_spawn = [row for row in stated_chain_cell if row["k"] > 0]

    return {
        "runs": {
            "cells": len({row["cell"] for row in rows}),
            "all": len(rows),
            "agent": len(agents),
            "reference": len(references),
            "succeeded": sum(row["succeeded"] for row in rows),
            "proxy_verified": sum(row["proxy_verified"] for row in rows),
            "calls": sum(row["calls"] for row in rows),
            "all_calls_2xx": all(row["all_2xx"] for row in rows),
            "dollars": sum(row["dollars"] for row in rows),
        },
        "decision_tallies": {
            "blind_wide_all_sampled_sizes": ratio(blind_wide),
            "blind_wide_size3": ratio([row for row in blind_wide if row["size"] == 3]),
            "stated_beta1_wide_disclosed": ratio(stated_one_wide_disclosed),
            "stated_beta1_wide_including_undisclosed": ratio(
                stated_one_wide_disclosed + undisclosed_wide
            ),
            "undisclosed_beta1_wide": ratio(undisclosed_wide),
            "stated_beta0_wide": ratio(stated_zero_wide),
            "blind_chain": ratio(blind_chain),
            "stated_beta1_chain": ratio(stated_one_chain),
        },
        "packing": {
            "free_choice_multi_spawn": ratio(free_multi, lambda row: row["packed"]),
            "forced_multi_spawn_packed": ratio(forced_multi, lambda row: row["packed"]),
            "forced_multi_spawn_serialized": ratio(
                forced_multi, lambda row: row["serialized"]
            ),
        },
        "serial_agent_strategy": {
            "wide_runs": len(serial_wide_agents),
            "wide_hand_repaired_every_module": sum(
                row["lead_module_writes"] == 4 for row in serial_wide_agents
            ),
            "wide_used_python": sum(row["lead_used_python"] for row in serial_wide_agents),
            "chain_runs": len(serial_chain_agents),
            "chain_hand_repaired_every_module": sum(
                row["lead_module_writes"] == 4 for row in serial_chain_agents
            ),
            "chain_used_python": sum(row["lead_used_python"] for row in serial_chain_agents),
        },
        "within_condition_outcomes": {
            "c15_stated_beta1": {
                "serial_objectives": [
                    row["dollars"] + row["analytic_minutes"]
                    for row in stated_chain_serial
                ],
                "spawn_objectives": [
                    row["dollars"] + row["analytic_minutes"]
                    for row in stated_chain_spawn
                ],
                "spawn_minus_serial": [
                    spawn["dollars"]
                    + spawn["analytic_minutes"]
                    - serial["dollars"]
                    - serial["analytic_minutes"]
                    for spawn in stated_chain_spawn
                    for serial in stated_chain_serial
                ],
            }
        },
        "failures": {
            "runs": len(failed),
            "all_serial": all(row["k"] == 0 for row in failed),
            "cells": sorted({row["cell"] for row in failed}),
        },
        "contested_writes": sum(row["contested_writes"] for row in rows),
    }


def load_probe(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    serial = raw["measured"]["serial"]
    fanout = raw["measured"]["fanout"]
    trace = Trace.load(path.parent / "fanout-trace.json")
    batches = {spawn.batch for spawn in trace.spawns}
    dominated = (
        fanout["dollars"] > serial["dollars"]
        and fanout["minutes"] > serial["minutes"]
    )
    assert raw["model"] == trace.model == GPT, path
    assert raw["valid"], path
    assert raw["measured_beta_star"] is None if dominated else True, path
    return {
        "source": relative(path),
        "scenario": raw["scenario"],
        "size": int(raw["scenario"].split("-s", 1)[1].split("-", 1)[0]),
        "serial": {"dollars": serial["dollars"], "minutes": serial["minutes"]},
        "fanout": {"dollars": fanout["dollars"], "minutes": fanout["minutes"]},
        "fanout_batches": len(batches),
        "fanout_packed": len(batches) == 1,
        "fanout_serialized": len(batches) == trace.k,
        "dominated": dominated,
        "beta_star": raw["measured_beta_star"],
    }


def ladder_summary(result_root: Path, stem: str) -> dict:
    rows = [load_probe(result_root / f"{stem}{size}" / "probe.json") for size in (3, 8, 15, 25)]
    size_ge_8 = [row for row in rows if row["size"] >= 8]
    return {
        "rungs": rows,
        "dominated": ratio(rows, lambda row: row["dominated"]),
        "serialized_at_sizes_8_15_25": ratio(
            size_ge_8, lambda row: row["fanout_serialized"]
        ),
        "size3_packed": rows[0]["fanout_packed"],
    }


def packed_probe(path: Path, serial: dict) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    fanout = raw["measured"]["fanout"]
    trace = Trace.load(path.parent / "fanout-trace.json")
    batches = {spawn.batch for spawn in trace.spawns}
    minutes_saved = serial["minutes"] - fanout["minutes"]
    dollar_premium = fanout["dollars"] - serial["dollars"]
    beta_star = dollar_premium / minutes_saved if minutes_saved > 0 else None
    assert raw["model"] == trace.model == GPT, path
    return {
        "source": relative(path),
        "fanout": {"dollars": fanout["dollars"], "minutes": fanout["minutes"]},
        "spawn_batches": len(batches),
        "packing_obeyed": len(batches) == 1,
        "plan_complied": fanout["complied"],
        "beta_star_against_frozen_serial": beta_star,
    }


def packed_recovery(high_ladder: dict) -> dict:
    serial_by_size = {row["size"]: row["serial"] for row in high_ladder["rungs"]}
    runs: dict[str, list[dict]] = {}
    for size in (15, 25):
        paths = [
            RESULTS / f"probe-gpt-hi-wide4-s{size}-packed" / "probe.json",
            RESULTS / f"probe-gpt-hi-wide4-s{size}-packed-r2" / "probe.json",
        ]
        runs[str(size)] = [packed_probe(path, serial_by_size[size]) for path in paths]
    return {
        "runs": runs,
        "packing_obedience": {
            size: ratio(rows, lambda row: row["packing_obeyed"])
            for size, rows in runs.items()
        },
        "finite_beta_star": {
            size: [
                row["beta_star_against_frozen_serial"]
                for row in rows
                if row["beta_star_against_frozen_serial"] is not None
            ]
            for size, rows in runs.items()
        },
    }


def paired_packed_probe(path: Path, price: PriceSheet, timing: TimingModel) -> dict:
    """Read one pre-registered serial/packed-fanout confirmatory pair."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["model"] == GPT, path
    assert raw["pack_spawns"] is True, path
    assert set(raw["measured"]) == {"serial", "fanout"}, path

    serial = raw["measured"]["serial"]
    fanout = raw["measured"]["fanout"]
    serial_trace = Trace.load(path.parent / "serial-trace.json")
    fanout_trace = Trace.load(path.parent / "fanout-trace.json")
    assert serial_trace.model == fanout_trace.model == GPT, path
    assert serial_trace.k == 0 and fanout_trace.k == 4, path

    batches = {spawn.batch for spawn in fanout_trace.spawns}
    minutes_saved = serial["minutes"] - fanout["minutes"]
    dollar_premium = fanout["dollars"] - serial["dollars"]
    beta_star = dollar_premium / minutes_saved if minutes_saved > 0 else None
    floors = raw["floors"]
    admitted = all(
        arm["succeeded"] and arm["complied"] and arm["proxy_verified"]
        for arm in (serial, fanout)
    )
    minute_floor_cleared = minutes_saved > floors["minutes"]
    dollar_floor_cleared = abs(dollar_premium) > floors["dollars"]
    floor_cleared = minute_floor_cleared and dollar_floor_cleared

    return {
        "source": relative(path),
        "serial": {"dollars": serial["dollars"], "minutes": serial["minutes"]},
        "fanout": {"dollars": fanout["dollars"], "minutes": fanout["minutes"]},
        "spawn_batches": len(batches),
        "packing_obeyed": len(batches) == 1 and len(fanout_trace.spawns) == 4,
        "admitted": admitted,
        "minutes_saved": minutes_saved,
        "dollar_premium": dollar_premium,
        "minute_floor_cleared": minute_floor_cleared,
        "dollar_floor_cleared": dollar_floor_cleared,
        "floor_cleared": floor_cleared,
        "beta_star": beta_star,
        "in_scored_band": bool(
            admitted and floor_cleared and beta_star is not None and 0 <= beta_star <= 1
        ),
        "dollars": serial_trace.dollars(price) + fanout_trace.dollars(price),
        "analytic_minutes": (
            serial_trace.analytic_minutes(timing) + fanout_trace.analytic_minutes(timing)
        ),
    }


def paired_packed_ladder(price: PriceSheet, timing: TimingModel) -> dict:
    """Read the exact eight-probe confirmatory set frozen before execution."""
    runs: dict[str, list[dict]] = {}
    for size in PACKED_CONFIRM_SIZES:
        rows = []
        for repeat in PACKED_CONFIRM_REPEATS:
            path = (
                RESULTS
                / f"probe-gpt-hi-wide4-s{size}-packed-confirm-r{repeat}"
                / "probe.json"
            )
            rows.append(paired_packed_probe(path, price, timing))
        runs[str(size)] = rows

    flat = [row for rows in runs.values() for row in rows]
    return {
        "design": {
            "sizes": list(PACKED_CONFIRM_SIZES),
            "repeats_per_size": len(PACKED_CONFIRM_REPEATS),
            "paired_probes": len(flat),
            "paid_arms": 2 * len(flat),
            "run_record": "docs/run-record-gpt-packed-ladder.md",
        },
        "runs": runs,
        "admission": ratio(flat, lambda row: row["admitted"]),
        "packing_obedience": ratio(flat, lambda row: row["packing_obeyed"]),
        "floor_cleared_finite": ratio(
            flat, lambda row: row["floor_cleared"] and row["beta_star"] is not None
        ),
        "in_scored_band": ratio(flat, lambda row: row["in_scored_band"]),
        "total_spend_dollars": sum(row["dollars"] for row in flat),
    }


def calibration_summary(calibration: CalibrationResult) -> dict:
    minutes = calibration.diagnostics["minute_holdout"]
    dollars = calibration.diagnostics["dollar_holdout"]
    return {
        "source": relative(RESULTS / "cal-gpt-responses" / "calibration.json"),
        "prefill_source": calibration.timing_model["prefill_source"],
        "floors": calibration.floors,
        "beta1_objective_floor": calibration.floors["dollars"] + calibration.floors["minutes"],
        "minute_holdout": {
            "mean_relative_error": minutes["mean_rel_error"],
            "max_relative_error": minutes["max_rel_error"],
        },
        "dollar_holdout": {
            "mean_relative_error": dollars["mean_rel_error"],
            "max_relative_error": dollars["max_rel_error"],
        },
    }


def figure_payload(
    high_cal: CalibrationResult,
    high_ladder: dict,
    packed_confirmatory: dict,
) -> dict:
    return {
        "what": (
            "GPT-5.6 Sol delegation curve at reasoning high: executed serial vs "
            "max-fanout arm pairs under the frozen forced-plan instruction. beta* "
            "is null at every rung because fan-out loses both dollars and analytic "
            "minutes. This curve does not price packed free-choice delegation."
        ),
        "date": "2026-08-29",
        "model": GPT,
        "reasoning_effort": "high",
        "calibration": "results/cal-gpt-responses/calibration.json",
        "floors": high_cal.floors,
        "total_spend_dollars": sum(
            row[arm]["dollars"]
            for row in high_ladder["rungs"]
            for arm in ("serial", "fanout")
        ),
        "single_draw_caveat": "one executed draw per arm at each rung",
        "execution_mode_caveat": (
            "all four forced fan-out arms issued one spawn per assistant turn; "
            "reasoning-high free-choice multi-spawn matrix runs packed 21/21"
        ),
        "packed_confirmatory": {
            "what": (
                "Pre-registered paired packed-mode ladder: two fresh serial/fanout "
                "pairs per size; all eight fanout arms packed in one batch."
            ),
            "run_record": "docs/run-record-gpt-packed-ladder.md",
            "runs": packed_confirmatory["runs"],
        },
        "rungs": [
            {
                "node_size": row["size"],
                "n_nodes": 4,
                "scenario": row["scenario"],
                "serial": row["serial"],
                "fanout": row["fanout"],
                "beta_star_per_min": row["beta_star"],
                "latency_ratio_serial_over_fanout": (
                    row["serial"]["minutes"] / row["fanout"]["minutes"]
                ),
                "valid": True,
                "fanout_spawn_batches": row["fanout_batches"],
                "provenance": {
                    "kind": "executed forced-plan arm pair, single draw each",
                    "path": str(Path(row["source"]).parent),
                },
            }
            for row in high_ladder["rungs"]
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write-figure",
        action="store_true",
        help="regenerate figures/beta-star-ladder-gpt.json from reasoning-high probes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    high_cal_path = RESULTS / "cal-gpt-responses" / "calibration.json"
    none_cal_path = RESULTS / "cal-gpt" / "calibration.json"
    high_cal, high_price, high_timing = load_calibration(high_cal_path)
    _, none_price, none_timing = load_calibration(none_cal_path)

    high_ladder = ladder_summary(RESULTS, "probe-gpt-hi-wide4-s")
    none_ladder = ladder_summary(RESULTS, "probe-gpt-wide4-s")
    high_matrix_rows = load_matrix(RESULTS / "matrix-gpt-hi", high_price, high_timing)
    none_matrix_rows = load_matrix(RESULTS / "matrix-gpt", none_price, none_timing)
    none_retry_rows = load_matrix(RESULTS / "matrix-gpt-r2", none_price, none_timing)

    high_matrix = matrix_summary(high_matrix_rows)
    none_matrix = matrix_summary(none_matrix_rows)
    none_retry = matrix_summary(none_retry_rows)
    packed_confirmatory = paired_packed_ladder(high_price, high_timing)
    none_main_free = [
        row for row in none_matrix_rows if not row["reference"] and row["k"] > 1
    ]
    none_retry_free = [
        row for row in none_retry_rows if not row["reference"] and row["k"] > 1
    ]

    report = {
        "model": GPT,
        "reasoning_high": {
            "calibration": calibration_summary(high_cal),
            "ladder": high_ladder,
            "packed_recovery": packed_recovery(high_ladder),
            "packed_confirmatory_ladder": packed_confirmatory,
            "matrix": high_matrix,
        },
        "reasoning_none": {
            "status": "superseded operating point",
            "ladder": none_ladder,
            "canonical_matrix": none_matrix,
            "retry_artifacts": none_retry,
            "packing_denominators": {
                "canonical_free_choice_multi_spawn": ratio(
                    none_main_free, lambda row: row["packed"]
                ),
                "retry_free_choice_multi_spawn": ratio(
                    none_retry_free, lambda row: row["packed"]
                ),
                "archive_free_choice_multi_spawn": ratio(
                    none_main_free + none_retry_free, lambda row: row["packed"]
                ),
            },
        },
    }

    # Reported invariants: fail loudly if an artifact changes underneath a claim.
    assert high_ladder["dominated"] == {"numerator": 4, "runs": 4}
    assert high_ladder["serialized_at_sizes_8_15_25"] == {"numerator": 3, "runs": 3}
    assert high_matrix["runs"]["succeeded"] == high_matrix["runs"]["all"] == 35
    assert high_matrix["runs"]["proxy_verified"] == 35
    assert high_matrix["runs"]["calls"] == 598
    assert high_matrix["decision_tallies"]["blind_wide_all_sampled_sizes"] == {
        "numerator": 8,
        "runs": 8,
    }
    assert high_matrix["decision_tallies"]["stated_beta1_wide_disclosed"] == {
        "numerator": 9,
        "runs": 11,
    }
    assert high_matrix["decision_tallies"]["stated_beta1_wide_including_undisclosed"] == {
        "numerator": 10,
        "runs": 12,
    }
    assert high_matrix["decision_tallies"]["stated_beta0_wide"] == {
        "numerator": 0,
        "runs": 5,
    }
    assert high_matrix["decision_tallies"]["blind_chain"] == {
        "numerator": 3,
        "runs": 3,
    }
    assert high_matrix["packing"]["free_choice_multi_spawn"] == {
        "numerator": 21,
        "runs": 21,
    }
    assert high_matrix["packing"]["forced_multi_spawn_serialized"] == {
        "numerator": 2,
        "runs": 2,
    }
    packed_confirm = report["reasoning_high"]["packed_confirmatory_ladder"]
    assert packed_confirm["design"] == {
        "sizes": [3, 8, 15, 25],
        "repeats_per_size": 2,
        "paired_probes": 8,
        "paid_arms": 16,
        "run_record": "docs/run-record-gpt-packed-ladder.md",
    }
    assert packed_confirm["admission"] == {"numerator": 8, "runs": 8}
    assert packed_confirm["packing_obedience"] == {"numerator": 8, "runs": 8}
    assert [row["beta_star"] for row in packed_confirm["runs"]["3"]] == [None, None]
    assert not any(row["minute_floor_cleared"] for row in packed_confirm["runs"]["8"])
    assert [round(row["beta_star"], 3) for row in packed_confirm["runs"]["15"]] == [
        0.513,
        0.569,
    ]
    assert [round(row["beta_star"], 3) for row in packed_confirm["runs"]["25"]] == [
        0.586,
        1.690,
    ]
    high_chain = high_matrix["within_condition_outcomes"]["c15_stated_beta1"]
    assert [round(value, 3) for value in high_chain["serial_objectives"]] == [1.164]
    assert [round(value, 3) for value in high_chain["spawn_objectives"]] == [1.869]
    assert [round(value, 3) for value in high_chain["spawn_minus_serial"]] == [0.705]
    assert none_matrix["runs"]["succeeded"] == 30
    assert none_matrix["runs"]["all"] == none_matrix["runs"]["proxy_verified"] == 35
    assert none_matrix["failures"]["runs"] == 5
    assert none_matrix["failures"]["all_serial"]
    assert report["reasoning_none"]["packing_denominators"] == {
        "canonical_free_choice_multi_spawn": {"numerator": 23, "runs": 23},
        "retry_free_choice_multi_spawn": {"numerator": 2, "runs": 2},
        "archive_free_choice_multi_spawn": {"numerator": 25, "runs": 25},
    }

    if args.write_figure:
        figure_path = ROOT / "figures" / "beta-star-ladder-gpt.json"
        figure_path.write_text(
            json.dumps(
                figure_payload(high_cal, high_ladder, packed_confirmatory), indent=2
            )
            + "\n",
            encoding="utf-8",
        )

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
