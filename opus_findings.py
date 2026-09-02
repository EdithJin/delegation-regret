"""Recompute the reported Opus-only findings from saved artifacts.

This script deliberately ignores GPT and Kimi artifacts, even when they live
under a ``results/matrix*`` directory.  Headline numbers should be copied
from this report, not re-counted by hand.

Run from the benchmark root:

    python3 opus_findings.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from generator.dag import sample_dag
from generator.oracle import Plan, enumerate_plans
from harness.audit import allocation_shape_classes, allocation_shape_key
from harness.calibrate import PriceSheet, TimingModel, call_dollars
from harness.calibration import CalibrationResult, orientation_prefix
from harness.trace import LEAD, Trace


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OPUS = "claude-opus-5"
MATRIX_DIRS = (
    RESULTS / "matrix",
    RESULTS / "matrix-reserve",
    RESULTS / "matrix-reserve2",
    RESULTS / "matrix-reserve3",
)
MODULE_PATH = re.compile(r"^pkg/mod_n\d+\.py$")


def spawn_ratio(numerator: int, denominator: int) -> dict[str, int]:
    return {"spawned": numerator, "runs": denominator}


def load_timekeeper() -> tuple[CalibrationResult, PriceSheet, TimingModel]:
    calibration = CalibrationResult.load(RESULTS / "cal-opus-v2" / "calibration.json")
    price = PriceSheet(**calibration.price_sheet)
    timing = TimingModel(**calibration.timing_model)
    assert price.model == OPUS
    return calibration, price, timing


def edit_strategy(trace: Trace, n: int) -> dict:
    """Classify persistent module edits across the complete agent team.

    ``write_events`` only covers writes made through the workspace ``write_file``
    tool.  A successful run with no recorded module write and one or more
    ``run_python`` calls is therefore classified as programmatic, rather than as
    a lead that performed no work.  ``run_python`` alone is not evidence of
    programmatic mutation because direct-edit runs also use it for verification.
    """
    module_events = [
        event for event in trace.write_events if MODULE_PATH.fullmatch(event.path)
    ]
    lead_module_paths = {
        event.path for event in module_events if event.actor == LEAD
    }
    subagent_module_paths = {
        event.path for event in module_events if event.actor != LEAD
    }
    team_module_paths = lead_module_paths | subagent_module_paths
    run_python_calls = sum(
        "run_python" in call.tools_invoked for call in trace.calls
    )
    if len(team_module_paths) == n:
        strategy = "direct-per-module"
    elif not team_module_paths and run_python_calls:
        strategy = "programmatic"
    else:
        strategy = "hybrid"
    return {
        "lead_module_writes": len(lead_module_paths),
        "subagent_module_writes": len(subagent_module_paths),
        "team_module_writes": len(team_module_paths),
        "direct_edit_complete": len(team_module_paths) == n,
        "zero_recorded_module_writes": not team_module_paths,
        "run_python_calls": run_python_calls,
        "execution_strategy": strategy,
    }


def scenario_width(scenario_id: str) -> int:
    match = re.match(r"^[a-z]+(\d+)-", scenario_id)
    assert match is not None, scenario_id
    return int(match.group(1))


def load_matrix_runs(price: PriceSheet, timing: TimingModel) -> list[dict]:
    rows: list[dict] = []
    for matrix_dir in MATRIX_DIRS:
        for cell_path in sorted(matrix_dir.glob("*/cell.json")):
            cell = json.loads(cell_path.read_text(encoding="utf-8"))
            if cell.get("model") != OPUS:
                continue
            meta = cell["cell"]
            for index, recorded in enumerate(cell["runs"]):
                trace_path = cell_path.parent / f"run-{index}-trace.json"
                trace = Trace.load(trace_path)
                assert trace.model == OPUS, trace_path
                assert trace.k == recorded["k"], trace_path
                assert trace.succeeded == recorded["succeeded"], trace_path
                assert trace.proxy_verified == recorded["proxy_verified"], trace_path

                edits = edit_strategy(trace, meta["n"])
                realized = recorded["realized"]
                max_fanout = (
                    trace.k == meta["n"]
                    and len(realized) == meta["n"]
                    and all(len(block) == 1 for block in realized)
                )
                batch_sizes: dict[int, int] = {}
                for spawn in trace.spawns:
                    batch_sizes[spawn.batch] = batch_sizes.get(spawn.batch, 0) + 1
                contract_briefings = sum(
                    "CONTRACT YOU MAY RELY ON" in spawn.instruction
                    or "authoritative values" in spawn.instruction
                    for spawn in trace.spawns
                )
                rows.append(
                    {
                        "source": str(trace_path.relative_to(ROOT)),
                        "cell": meta["id"],
                        "scenario": cell["scenario"],
                        "shape": meta["shape"],
                        "n": meta["n"],
                        "size": meta["size"],
                        "seed": meta["seed"],
                        "condition": meta["cond"],
                        "undisclosed": "undisc" in meta["id"],
                        "reference": meta["cond"] == "ref",
                        "run": index,
                        "k": trace.k,
                        "max_fanout": max_fanout,
                        "spawn_batch_sizes": sorted(batch_sizes.values(), reverse=True),
                        "all_spawns_one_batch": trace.k > 0 and len(batch_sizes) == 1,
                        "contract_briefings": contract_briefings,
                        **edits,
                        "succeeded": trace.succeeded,
                        "proxy_verified": trace.proxy_verified,
                        "contested_writes": len(trace.contested_writes),
                        "dollars": trace.dollars(price),
                        "minutes": trace.analytic_minutes(timing),
                        "objective_beta1": trace.objective(price, timing, 1.0),
                    }
                )
    return rows


def tally(rows: list[dict]) -> dict[str, int]:
    return spawn_ratio(sum(row["k"] > 0 for row in rows), len(rows))


def probe_row(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    serial = raw["measured"]["serial"]
    fanout = raw["measured"]["fanout"]
    if fanout["dollars"] > serial["dollars"] and fanout["minutes"] >= serial["minutes"]:
        measured_beta_star: float | None = None
        verdict = "fan-out dominated"
    else:
        measured_beta_star = (
            (fanout["dollars"] - serial["dollars"])
            / (serial["minutes"] - fanout["minutes"])
        )
        verdict = "finite crossover"
    assert measured_beta_star == raw["measured_beta_star"]
    return {
        "scenario": raw["scenario"],
        "valid": raw["valid"],
        "serial": serial,
        "fanout": fanout,
        "beta_star": measured_beta_star,
        "verdict": verdict,
        "serial_minutes_prediction_error": (
            serial["minutes"] - raw["predicted"]["serial"]["minutes"]
        )
        / raw["predicted"]["serial"]["minutes"],
        "fanout_dollars_prediction_error": (
            fanout["dollars"] - raw["predicted"]["fanout"]["dollars"]
        )
        / raw["predicted"]["fanout"]["dollars"],
    }


def matrix_summary(rows: list[dict], price: PriceSheet, timing: TimingModel) -> dict:
    agents = [row for row in rows if not row["reference"]]
    references = [row for row in rows if row["reference"]]
    wide_flip = [
        row for row in agents if row["shape"] == "wide" and row["size"] >= 15
    ]
    blind_wide = [row for row in wide_flip if row["condition"] == "blind"]
    stated_one_wide = [
        row
        for row in wide_flip
        if row["condition"] == "stated-b1" and not row["undisclosed"]
    ]
    stated_zero_wide = [row for row in wide_flip if row["condition"] == "stated-b0"]
    undisclosed_wide = [row for row in wide_flip if row["undisclosed"]]
    blind_chain = [
        row for row in agents if row["shape"] == "chain" and row["condition"] == "blind"
    ]
    stated_chain = [
        row
        for row in agents
        if row["shape"] == "chain" and row["condition"] == "stated-b1"
    ]
    wide = [row for row in agents if row["shape"] == "wide"]
    chain = [row for row in agents if row["shape"] == "chain"]
    wide_serial = [row for row in wide if row["k"] == 0]
    wide_spawned = [row for row in wide if row["k"] > 0]
    chain_serial = [row for row in chain if row["k"] == 0]
    spawned_chain = [row for row in chain if row["k"] > 0]
    spawned = [row for row in agents if row["k"] > 0]

    def strategy_counts(selected: list[dict]) -> dict[str, int]:
        return {
            "runs": len(selected),
            "direct_all_modules": sum(
                row["direct_edit_complete"] for row in selected
            ),
            "programmatic": sum(
                row["execution_strategy"] == "programmatic" for row in selected
            ),
            "hybrid": sum(
                row["execution_strategy"] == "hybrid" for row in selected
            ),
            "zero_recorded_module_writes": sum(
                row["zero_recorded_module_writes"] for row in selected
            ),
        }

    audit = json.loads((RESULTS / "audit-w15" / "audit.json").read_text(encoding="utf-8"))
    audit_dag = sample_dag("wide", 4, sizes=(15,), seed=11)
    labeled_plans = enumerate_plans(audit_dag)
    shape_classes = allocation_shape_classes(audit_dag, labeled_plans)
    selected_shape_keys = {
        allocation_shape_key(
            audit_dag,
            Plan(
                frozenset(row["inline"]),
                tuple(frozenset(block) for block in row["blocks"]),
            ),
        )
        for row in audit["rows"]
    }
    assert audit["n_plans"] == len(labeled_plans) == 52
    assert audit["n_outcomes"] == len(shape_classes) == 12
    assert selected_shape_keys == set(shape_classes)
    assert audit.get("n_shape_classes") == len(shape_classes)
    assert audit.get("n_shape_represented") == len(selected_shape_keys)
    assert audit.get("shape_complete") is True
    assert sum(
        row["labeled_plan_count"] for row in audit.get("shape_classes", ())
    ) == len(labeled_plans)
    audit_champion = min(
        (row for row in audit["rows"] if not row["excluded"]),
        key=lambda row: row["measured_objective"],
    )
    audit_serial = next(row for row in audit["rows"] if row["k"] == 0)

    def matrix_objective(cell: str) -> float:
        selected = [row for row in rows if row["cell"] == cell]
        assert len(selected) == 1, (cell, len(selected))
        return selected[0]["objective_beta1"]

    probes = {
        8: probe_row(RESULTS / "probe-wide4-s8" / "probe.json"),
        25: probe_row(RESULTS / "probe-wide4-s25" / "probe.json"),
        40: probe_row(RESULTS / "probe-wide4-s40" / "probe.json"),
    }
    comparators = {
        "wide4-s15-11": audit_champion["measured_objective"],
        "wide4-s15-23": matrix_objective("w15b-ref-fanout"),
        "wide4-s25-11": (
            probes[25]["fanout"]["dollars"] + probes[25]["fanout"]["minutes"]
        ),
        "wide4-s40-11": (
            probes[40]["fanout"]["dollars"] + probes[40]["fanout"]["minutes"]
        ),
        "wide4-s8-11": (
            probes[8]["serial"]["dollars"] + probes[8]["serial"]["minutes"]
        ),
        "chain4-s15-11": matrix_objective("c15-ref-serial"),
    }
    stated_beta_one = [row for row in agents if row["condition"] == "stated-b1"]
    beat_comparator = [
        row
        for row in stated_beta_one
        if row["objective_beta1"] < comparators[row["scenario"]]
    ]

    blind_spawn_objectives = [
        row["objective_beta1"] for row in blind_wide if row["k"] > 0
    ]
    blind_serial_objectives = [
        row["objective_beta1"] for row in blind_wide if row["k"] == 0
    ]

    original_beta_zero = [
        row
        for row in rows
        if row["source"].startswith("results/matrix/w15-stated-b0/")
    ]
    beta_zero_serial = next(row for row in original_beta_zero if row["k"] == 0)
    beta_zero_penalties = sorted(
        row["dollars"] - beta_zero_serial["dollars"]
        for row in original_beta_zero
        if row["k"] > 0
    )
    chain_same_cell_serial = next(
        row
        for row in rows
        if row["source"].startswith("results/matrix/c15-blind/")
        and row["k"] == 0
    )
    chain_seed11_spawn = next(
        row
        for row in rows
        if row["cell"] == "c15-blind"
        and row["k"] > 0
        and row["seed"] == 11
    )

    return {
        "runs": {
            "all": len(rows),
            "agent": len(agents),
            "reference": len(rows) - len(agents),
            "succeeded": sum(row["succeeded"] for row in rows),
            "proxy_verified": sum(row["proxy_verified"] for row in rows),
        },
        "decision_tallies": {
            "blind_wide_size_15_to_25": tally(blind_wide),
            "blind_wide_size_15_to_25_seed11": tally(
                [row for row in blind_wide if row["seed"] == 11]
            ),
            "blind_wide_size_15_to_25_seed23": tally(
                [row for row in blind_wide if row["seed"] == 23]
            ),
            "stated_beta1_wide_size_15_to_40": tally(stated_one_wide),
            "stated_beta1_wide_size_15_to_25_seed11": tally(
                [
                    row
                    for row in stated_one_wide
                    if row["seed"] == 11 and row["size"] in (15, 25)
                ]
            ),
            "stated_beta1_wide_size_15_to_25_seed23": tally(
                [
                    row
                    for row in stated_one_wide
                    if row["seed"] == 23 and row["size"] in (15, 25)
                ]
            ),
            "stated_beta0_wide_size_15_to_25": tally(stated_zero_wide),
            "undisclosed_beta1_wide_size_15": tally(undisclosed_wide),
            "stated_beta1_wide_size_8": tally(
                [row for row in agents if row["cell"] == "w8-stated-b1"]
            ),
            "blind_chain_size_15": tally(blind_chain),
            "all_chain_size_15": tally(blind_chain + stated_chain),
            "all_agent_runs": tally(agents),
        },
        "spawn_form": {
            "spawned_runs": len(spawned),
            "max_fanout_singletons": sum(row["max_fanout"] for row in spawned),
            "partial": sum(not row["max_fanout"] for row in spawned),
        },
        "packing": {
            # V10's Opus side: same-turn batch census over multi-spawn runs
            # (k >= 2; a single spawn has no packing question to answer).
            "free_multi_spawn": {
                "runs": sum(row["k"] >= 2 for row in agents),
                "single_batch": sum(
                    row["all_spawns_one_batch"] for row in agents if row["k"] >= 2
                ),
            },
            "forced_multi_spawn": {
                "runs": sum(row["k"] >= 2 for row in references),
                "single_batch": sum(
                    row["all_spawns_one_batch"]
                    for row in references
                    if row["k"] >= 2
                ),
            },
        },
        "strategy": {
            "wide": strategy_counts(wide),
            "wide_serial": strategy_counts(wide_serial),
            "wide_spawned": strategy_counts(wide_spawned),
            "chain": strategy_counts(chain),
            "chain_serial": strategy_counts(chain_serial),
            "chain_spawned": strategy_counts(spawned_chain),
            "references": strategy_counts(references),
        },
        "chain_contract_parallelism": {
            "spawned_runs": len(spawned_chain),
            "single_batch_runs": sum(row["all_spawns_one_batch"] for row in spawned_chain),
            "runs": [
                {
                    "source": row["source"],
                    "seed": row["seed"],
                    "k": row["k"],
                    "spawn_batch_sizes": row["spawn_batch_sizes"],
                    "lead_module_writes": row["lead_module_writes"],
                    "contract_briefings": row["contract_briefings"],
                }
                for row in spawned_chain
            ],
        },
        "outcomes_beta1": {
            "comparators": comparators,
            "stated_runs_beating_executed_comparator": {
                "beating": len(beat_comparator),
                "runs": len(stated_beta_one),
            },
            "serial_stated_runs_beating_executed_comparator": {
                "beating": sum(row["k"] == 0 for row in beat_comparator),
                "runs": sum(row["k"] == 0 for row in stated_beta_one),
            },
            "blind_spawn_range": [min(blind_spawn_objectives), max(blind_spawn_objectives)],
            "blind_serial_range": [min(blind_serial_objectives), max(blind_serial_objectives)],
            "best_stated_beta1_programmatic_serial": {
                str(size): min(
                    row["objective_beta1"]
                    for row in wide
                    if row["size"] == size
                    and row["k"] == 0
                    and row["execution_strategy"] == "programmatic"
                    and row["condition"] == "stated-b1"
                    and not row["undisclosed"]
                )
                for size in (15, 25, 40)
            },
        },
        "beta0": {
            "original_batch_serial_dollars": beta_zero_serial["dollars"],
            "spawn_penalties_dollars": beta_zero_penalties,
        },
        "chain_placebo_seed11": {
            "spawn_dollar_penalty": (
                chain_seed11_spawn["dollars"] - chain_same_cell_serial["dollars"]
            ),
            "spawn_minutes_saved": (
                chain_same_cell_serial["minutes"] - chain_seed11_spawn["minutes"]
            ),
        },
        "audit_w15": {
            "plans": audit["n_outcomes"],
            "labeled_plans": len(labeled_plans),
            "allocation_shapes": len(shape_classes),
            "shape_complete": selected_shape_keys == set(shape_classes),
            "shape_class_sizes": sorted(
                len(members) for members in shape_classes.values()
            ),
            "selection_basis": "floor-resolved predicted outcomes",
            "structural_cross_check": (
                "(inline task count, sorted delegated block task counts)"
            ),
            "complete": audit["complete"],
            "champion": audit_champion["tag"],
            "champion_objective": audit_champion["measured_objective"],
            "serial_objective": audit_serial["measured_objective"],
            "intermediate_range": [
                min(row["measured_objective"] for row in audit["rows"] if row["k"] > 0 and row["tag"] != audit_champion["tag"]),
                max(row["measured_objective"] for row in audit["rows"] if row["k"] > 0 and row["tag"] != audit_champion["tag"]),
            ],
            "champion_predicted_rank": audit["champion_predicted_rank"],
            "champion_in_band": audit["champion_in_band"],
            "safe_filter_x": audit["safe_filter_x"],
            "kendall_tau": audit["tau"],
        },
    }


def calibration_summary(calibration: CalibrationResult) -> dict:
    composition = calibration.diagnostics["composition"]
    dollar_disagreement = calibration.diagnostics["dollar_disagreement"]
    minute_disagreement = calibration.diagnostics["minute_disagreement"]
    return {
        "floors": calibration.floors,
        "composition": {
            row["plan"]: {
                "dollars_gap": row["rel_gap_dollars"],
                "minutes_gap": row["rel_gap_minutes"],
            }
            for row in composition["arms"]
        },
        "ranking_preserved": {
            "dollars": composition["cost_ranking_preserved"],
            "minutes": composition["latency_ranking_preserved"],
        },
        "dollar_holdout": calibration.diagnostics["dollar_holdout"],
        "minute_holdout": calibration.diagnostics["minute_holdout"],
        "max_cross_ordering_disagreement": {
            "dollars": max(dollar_disagreement.values()),
            "minutes": max(minute_disagreement.values()),
        },
        "block_curves": {
            "lead_dollars": calibration.constants["block_dollars_curve"],
            "subagent_dollars": calibration.constants["sub_block_dollars_curve"],
        },
    }


def calibration_v1_summary(price: PriceSheet) -> dict:
    prefixes = []
    totals = []
    for path in sorted((RESULTS / "cal-opus").glob("trace-serial-*.json")):
        trace = Trace.load(path)
        prefix, acted = orientation_prefix(trace)
        assert acted
        prefixes.append(sum(call_dollars(call.as_record(), price) for call in prefix))
        totals.append(trace.dollars(price))
    return {
        "runs": len(totals),
        "orientation_prefix_dollars_range": [min(prefixes), max(prefixes)],
        "total_dollars_range": [min(totals), max(totals)],
    }


def pass1_and_smoke_summary() -> dict:
    smoke = json.loads((RESULTS / "smoke" / "results.json").read_text(encoding="utf-8"))
    pass1 = json.loads((RESULTS / "run-opus-1" / "results.json").read_text(encoding="utf-8"))
    smoke_beta_one = [card for card in smoke["cards"] if card["beta"] == 1.0]
    pass1_valid = [card for card in pass1["cards"] if not card["excluded"]]
    pass1_wide_agents = [
        Trace.load(path)
        for path in sorted((RESULTS / "run-opus-1" / "traces").glob("wide*-agent-r0.json"))
    ]
    smoke_agents = [
        Trace.load(path)
        for path in sorted((RESULTS / "smoke" / "traces").glob("*-agent-r0.json"))
    ]
    pass1_strategies = [
        edit_strategy(trace, scenario_width(trace.scenario_id))
        for trace in pass1_wide_agents
    ]
    smoke_strategies = [
        edit_strategy(trace, scenario_width(trace.scenario_id))
        for trace in smoke_agents
    ]
    return {
        "smoke_beta1": [
            {
                "scenario": card["scenario_id"],
                "agent_k": card["agent_k"],
                "agent_objective": card["agent_objective"],
                "baseline_objective": card["baseline_objective"],
                "regret": card["regret"],
            }
            for card in smoke_beta_one
        ],
        "ordering": {
            "smoke": smoke["ordering"],
            "pass1": pass1["ordering"],
        },
        "pass1": {
            "valid_cards": len(pass1_valid),
            "valid_shapes": sorted({card["scenario_id"].split("4", 1)[0].split("6", 1)[0].split("8", 1)[0] for card in pass1_valid}),
            "wide_agent_runs": len(pass1_wide_agents),
            "wide_agent_spawns": sum(trace.k for trace in pass1_wide_agents),
            "direct_all_modules": sum(
                row["direct_edit_complete"] for row in pass1_strategies
            ),
            "programmatic": sum(
                row["execution_strategy"] == "programmatic"
                for row in pass1_strategies
            ),
            "aggregates": pass1["aggregates"],
        },
        "smoke_strategy": {
            "agent_runs": len(smoke_strategies),
            "direct_all_modules": sum(
                row["direct_edit_complete"] for row in smoke_strategies
            ),
            "programmatic": sum(
                row["execution_strategy"] == "programmatic"
                for row in smoke_strategies
            ),
        },
    }


def contested_write_summary(matrix_rows: list[dict]) -> dict:
    paths = []
    paths.extend(row["source"] for row in matrix_rows)
    paths.extend(
        str(path.relative_to(ROOT))
        for path in sorted((RESULTS / "audit-w15").glob("trace-*.json"))
    )
    for probe_dir in (
        "probe-wide4-s3",
        "probe-wide4-s8",
        "probe-wide4-s15",
        "probe-wide4-s25",
        "probe-wide4-s40",
        "probe-wide8-s15",
    ):
        paths.extend(
            str(path.relative_to(ROOT))
            for path in sorted((RESULTS / probe_dir).glob("*-trace.json"))
        )
    traces = [Trace.load(ROOT / path) for path in paths]
    return {
        "traces": len(traces),
        "with_recorded_contested_write_events": sum(
            bool(trace.contested_writes) for trace in traces
        ),
        "scope": "workspace write_file events; run_python mutations are unattributed",
    }


def validate_report_macros(report: dict) -> None:
    """Fail if a headline Opus macro drifts from the recomputed artifacts.

    Reads the macro file named by the REPORT_TEX environment variable; when
    unset, the validation is skipped so the recompute stays self-contained.
    """
    tex_path = os.environ.get("REPORT_TEX")
    if tex_path is None:
        print("macro validation skipped: REPORT_TEX not set")
        return
    tex = Path(tex_path).read_text(encoding="utf-8")

    def macro(name: str) -> str:
        match = re.search(rf"\\newcommand\{{\\{name}\}}\{{([^}}]*)\}}", tex)
        assert match is not None, f"missing report macro: {name}"
        return match.group(1)

    matrix = report["matrix"]
    tallies = matrix["decision_tallies"]
    outcomes = matrix["outcomes_beta1"]
    audit = matrix["audit_w15"]
    ladder = report["ladder"]
    beta0 = matrix["beta0"]

    expected = {
        "betastarEight": rf"\${ladder['wide4']['8']['beta_star']:.2f}",
        "betastarFifteen": rf"\${ladder['wide4']['15']['beta_star']:.2f}",
        "betastarTwentyfive": rf"\${ladder['wide4']['25']['beta_star']:.2f}",
        "betastarForty": rf"\${ladder['wide4']['40']['beta_star']:.2f}",
        "nAxisBetastar": rf"\${ladder['wide8_size15']['beta_star']:.2f}",
        "champObj": f"{audit['champion_objective']:.2f}",
        "serialObj": f"{audit['serial_objective']:.2f}",
        "valleyRange": (
            f"{audit['intermediate_range'][0]:.2f}--"
            f"{audit['intermediate_range'][1]:.2f}"
        ),
        "safeX": rf"{audit['safe_filter_x']:.0%}".replace("%", r"\%"),
        "rankTau": f"${audit['kendall_tau']:.2f}$",
        "blindRate": (
            f"{tallies['blind_wide_size_15_to_25']['spawned']}/"
            f"{tallies['blind_wide_size_15_to_25']['runs']}"
        ),
        "statedBoneRate": (
            f"{tallies['stated_beta1_wide_size_15_to_40']['spawned']}/"
            f"{tallies['stated_beta1_wide_size_15_to_40']['runs']}"
        ),
        "statedBzeroRate": (
            f"{tallies['stated_beta0_wide_size_15_to_25']['spawned']}/"
            f"{tallies['stated_beta0_wide_size_15_to_25']['runs']}"
        ),
        "zeroBelow": "0/6",
        "blindChainRate": (
            f"{tallies['blind_chain_size_15']['spawned']}/"
            f"{tallies['blind_chain_size_15']['runs']}"
        ),
        "chainSpawns": (
            f"{tallies['all_chain_size_15']['spawned']} of "
            f"{tallies['all_chain_size_15']['runs']}"
        ),
        "beatComparator": (
            f"{outcomes['stated_runs_beating_executed_comparator']['beating']} of "
            f"{outcomes['stated_runs_beating_executed_comparator']['runs']}"
        ),
        "serialBeatComparator": (
            f"{outcomes['serial_stated_runs_beating_executed_comparator']['beating']} of "
            f"{outcomes['serial_stated_runs_beating_executed_comparator']['runs']}"
        ),
        "wideDirectAll": (
            f"{matrix['strategy']['wide']['direct_all_modules']} of "
            f"{matrix['strategy']['wide']['runs']}"
        ),
        "chainDirectAll": (
            f"{matrix['strategy']['chain']['direct_all_modules']} of "
            f"{matrix['strategy']['chain']['runs']}"
        ),
        "wideSerialProgrammatic": (
            f"{matrix['strategy']['wide_serial']['programmatic']}/"
            f"{matrix['strategy']['wide_serial']['runs']}"
        ),
        "wideSpawnDirect": (
            f"{matrix['strategy']['wide_spawned']['direct_all_modules']}/"
            f"{matrix['strategy']['wide_spawned']['runs']}"
        ),
        "chainSerialDirect": (
            f"{matrix['strategy']['chain_serial']['direct_all_modules']}/"
            f"{matrix['strategy']['chain_serial']['runs']}"
        ),
        "chainSpawnDirect": (
            f"{matrix['strategy']['chain_spawned']['direct_all_modules']}/"
            f"{matrix['strategy']['chain_spawned']['runs']}"
        ),
        "allAgentSpawns": (
            f"{tallies['all_agent_runs']['spawned']}/"
            f"{tallies['all_agent_runs']['runs']}"
        ),
        "maximalSpawnForm": (
            f"{matrix['spawn_form']['max_fanout_singletons']}/"
            f"{matrix['spawn_form']['spawned_runs']}"
        ),
        "partialSpawnForm": (
            f"{matrix['spawn_form']['partial']}/"
            f"{matrix['spawn_form']['spawned_runs']}"
        ),
        "matchedWideSizes": "15/25",
        "seedElevenBlind": (
            f"{tallies['blind_wide_size_15_to_25_seed11']['spawned']}/"
            f"{tallies['blind_wide_size_15_to_25_seed11']['runs']}"
        ),
        "seedElevenStated": (
            f"{tallies['stated_beta1_wide_size_15_to_25_seed11']['spawned']}/"
            f"{tallies['stated_beta1_wide_size_15_to_25_seed11']['runs']}"
        ),
        "matrixVerified": (
            f"{matrix['runs']['succeeded']}/{matrix['runs']['all']}"
        ),
        "statedBoneWEight": (
            f"{tallies['stated_beta1_wide_size_8']['spawned']}/"
            f"{tallies['stated_beta1_wide_size_8']['runs']}"
        ),
        "statedBoneChain": "0/2",
        "undiscRate": (
            f"{tallies['undisclosed_beta1_wide_size_15']['spawned']}/"
            f"{tallies['undisclosed_beta1_wide_size_15']['runs']}"
        ),
        "bZeroPenaltyA": rf"+\${max(beta0['spawn_penalties_dollars']):.2f}",
        "bZeroPenaltyB": rf"+\${min(beta0['spawn_penalties_dollars']):.2f}",
        "bestProgrammaticFifteen": f"{outcomes['best_stated_beta1_programmatic_serial']['15']:.2f}",
        "bestProgrammaticTwentyfive": f"{outcomes['best_stated_beta1_programmatic_serial']['25']:.2f}",
        "bestProgrammaticForty": f"{outcomes['best_stated_beta1_programmatic_serial']['40']:.2f}",
        "blindSpawnObjectives": (
            f"{outcomes['blind_spawn_range'][0]:.2f}--"
            f"{outcomes['blind_spawn_range'][1]:.2f}"
        ),
        "blindSerialObjectives": (
            f"{outcomes['blind_serial_range'][0]:.2f}--"
            f"{outcomes['blind_serial_range'][1]:.2f}"
        ),
        "concurrentChainRuns": (
            f"{matrix['chain_contract_parallelism']['single_batch_runs']}/"
            f"{matrix['chain_contract_parallelism']['spawned_runs']}"
        ),
        "seedElevenChainBatch": str(
            next(
                row["k"]
                for row in matrix["chain_contract_parallelism"]["runs"]
                if row["seed"] == 11
            )
        ),
        "seedTwentythreeChainBatch": str(
            next(
                row["k"]
                for row in matrix["chain_contract_parallelism"]["runs"]
                if row["seed"] == 23
            )
        ),
        "auditPlanCount": str(audit["plans"]),
        "auditLabeledCount": str(audit["labeled_plans"]),
        "referenceTwentyfive": f"{outcomes['comparators']['wide4-s25-11']:.2f}",
        "referenceForty": f"{outcomes['comparators']['wide4-s40-11']:.2f}",
        "chainWaste": (
            rf"+\${matrix['chain_placebo_seed11']['spawn_dollar_penalty']:.2f}"
        ),
    }
    packing = matrix["packing"]
    ladder_packing = report["forced_fanout_packing"]
    assert ladder_packing["wide8_batch_sizes"] == [4, 4], (
        "wide-8 forced arm no longer batches at the concurrency cap: "
        f"{ladder_packing['wide8_batch_sizes']}"
    )
    expected["opusFreePacked"] = (
        f"{packing['free_multi_spawn']['single_batch']}/"
        f"{packing['free_multi_spawn']['runs']}"
    )
    expected["opusForcedPacked"] = (
        f"{packing['forced_multi_spawn']['single_batch'] + ladder_packing['wide4_single_batch']}/"
        f"{packing['forced_multi_spawn']['runs'] + ladder_packing['wide4_arms']}"
    )
    large = tallies["stated_beta1_wide_size_15_to_40"]
    eight = tallies["stated_beta1_wide_size_8"]
    expected["statedBoneAllWide"] = (
        f"{large['spawned'] + eight['spawned']}/{large['runs'] + eight['runs']}"
    )
    rank_words = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
                  6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth",
                  10: "tenth", 11: "eleventh", 12: "twelfth"}
    expected["champPredictedRank"] = rank_words[audit["champion_predicted_rank"]]
    serial_errors = [
        ladder["wide4"][str(size)]["serial_minutes_prediction_error"]
        for size in (8, 15, 25)
    ]
    fanout_errors = [
        ladder["wide4"][str(size)]["fanout_dollars_prediction_error"]
        for size in (8, 15, 25)
    ]
    expected["ladderSerialMiss"] = (
        "$" + "$/$".join(f"{value:.0%}".removesuffix("%") for value in serial_errors)
        + r"\%$"
    )
    expected["ladderFanoutMiss"] = (
        "$" + "$/$".join(f"{value:+.0%}".removesuffix("%") for value in fanout_errors)
        + r"\%$"
    )
    mismatches = {
        name: {"macro": macro(name), "data": value}
        for name, value in expected.items()
        if macro(name) != value
    }
    assert not mismatches, f"Opus report macro drift: {mismatches}"


def forced_fanout_packing() -> dict:
    """Spawn batching of the forced ladder fan-out arms (V10, Opus side).

    The four scored wide-4 rungs each request four spawns; the wide-8 axis
    probe requests eight, which cannot fit one batch under the concurrency
    cap of four and is therefore reported by its batch sizes, not pooled
    into the single-batch tally.
    """
    arms: dict[str, list[int]] = {}
    for name in ("probe-wide4-s3", "probe-wide4-s8", "probe-wide4-s15",
                 "probe-wide4-s25"):
        trace = Trace.load(RESULTS / name / "fanout-trace.json")
        assert trace.model == OPUS, name
        batches: dict[int, int] = {}
        for spawn in trace.spawns:
            batches[spawn.batch] = batches.get(spawn.batch, 0) + 1
        arms[name] = sorted(batches.values(), reverse=True)
    wide8 = Trace.load(RESULTS / "probe-wide8-s15" / "fanout-trace.json")
    assert wide8.model == OPUS
    batches8: dict[int, int] = {}
    for spawn in wide8.spawns:
        batches8[spawn.batch] = batches8.get(spawn.batch, 0) + 1
    return {
        "wide4_scored_rungs": arms,
        "wide4_arms": len(arms),
        "wide4_single_batch": sum(len(sizes) == 1 for sizes in arms.values()),
        "wide8_batch_sizes": sorted(batches8.values(), reverse=True),
    }


def main() -> None:
    calibration, price, timing = load_timekeeper()
    matrix_rows = load_matrix_runs(price, timing)
    ladder = {
        "wide4": {
            str(size): probe_row(RESULTS / f"probe-wide4-s{size}" / "probe.json")
            for size in (3, 8, 15, 25, 40)
        },
        "wide8_size15": probe_row(RESULTS / "probe-wide8-s15" / "probe.json"),
    }
    report = {
        "scope": {
            "model": OPUS,
            "excluded": ["haiku", "gpt", "kimi"],
        },
        "calibration_v2": calibration_summary(calibration),
        "calibration_v1": calibration_v1_summary(price),
        "ladder": ladder,
        "matrix": matrix_summary(matrix_rows, price, timing),
        "forced_fanout_packing": forced_fanout_packing(),
        "smoke_and_pass1": pass1_and_smoke_summary(),
        "contested_writes": contested_write_summary(matrix_rows),
    }
    validate_report_macros(report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
