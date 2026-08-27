"""The discriminating mini-matrix — the run design captured as code.

Pre-registered in DATASET-BOUNDARY-DEFECT.md section 5 (the stated-beta
amendment; wording of the objective directives below is an INSTRUMENT CONSTANT,
frozen at sign-off — do not edit between cells). One invocation executes one
cell in its own fresh process (the probe pattern, which is what makes the
matrix independent of the item-14 leak fix). Nothing here launches by itself:

    python3 matrix.py --list             # the full run plan + cost estimate, no key
    python3 matrix.py --cell w15-stated-b1 --dry-run
    python3 matrix.py --cell int8-bundle          # one cell, for real

Conditions:
- stated-b1 / stated-b0 : agent arm with the objective directive appended to
  the task prompt (runner.py's `directive` plumbing; the agent otherwise runs
  the standard protocol). Scored ONLY at the stated beta — scoring a stated-b1
  run at beta=0 would repeat the beta-blindness mistake inverted.
- blind    : agent arm, current protocol, no objective stated. The revealed-
  preference arm; scored at both betas, continuous with pass-1.
- ref-*    : run_plan of a named fixed plan. `serial`/`fanout` references are
  already banked by the ladder probes; the `bundle1`/`split13`/`split22` cells
  are the intermediate-plan check (committee logic scoped to the matrix cells).

Primary observable: k (spawn decision), read from SpawnRecords — immune to the
V1 run_python attribution hole. Realized-plan compliance is secondary and is
attribution-based; treat it as advisory until item 14's module-hashing fix.

Cell ordering matters and is encoded in CELLS: the three intermediate checks
run FIRST. Re-anchor rule, pre-registered: if `int8-bundle` beats the banked
size-8 serial arm on the beta=1 objective (beyond the floors), size 8 is no
longer serial-correct — skip `w8-stated-b1` and run `w3-stated-b1` instead
(size 3 is measured-dominated and immune to intermediate-plan risk).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from generator.manifest import ScenarioSpec
from generator.oracle import Plan, all_inline, max_fanout
from harness.calibrate import PriceSheet
from harness.cli import ANTHROPIC_API, client_for_model, endpoint_defaults, price_sheet
from harness.runner import Budget, run_agent, run_plan

# ----------------------------------------------------------------- constants

# INSTRUMENT CONSTANTS — the stated-beta objective directives, appended
# verbatim to the agent's first user message. Symmetric wording; neither
# mentions spawning. Frozen at pre-registration sign-off.
DIRECTIVE_B1 = (
    "\n\nObjective: treat elapsed wall-clock time as a real cost of $1.00 per "
    "minute, in addition to API spend. Complete every task while minimizing "
    "total cost = dollars spent + $1.00 x elapsed minutes."
)
DIRECTIVE_B0 = (
    "\n\nObjective: minimize API dollars spent. Elapsed time costs nothing."
)


def named_plan(name: str, dag) -> Plan:
    """Fixed plans by name. Node ids in wide-4 sampling order."""
    ids = list(dag.topo_order)
    if name == "serial":
        return all_inline(dag)
    if name == "fanout":
        return max_fanout(dag)
    if name == "bundle1":  # k=1: one subagent takes everything
        return Plan(frozenset(), (frozenset(ids),))
    if name == "split13":  # k=2: blocks of 1 and 3
        return Plan(frozenset(), (frozenset(ids[:1]), frozenset(ids[1:])))
    if name == "split22":  # k=2: blocks of 2 and 2
        return Plan(frozenset(), (frozenset(ids[:2]), frozenset(ids[2:])))
    if name == "splithalf":  # k=2 at any n: first half / second half
        h = len(ids) // 2
        return Plan(frozenset(), (frozenset(ids[:h]), frozenset(ids[h:])))
    raise SystemExit(f"unknown plan {name!r}")


# The design as data. Order = execution order. `est` is a per-run cost
# estimate in dollars, from the measured ladder arms at the same size.
CELLS = [
    # -- intermediate-plan check (committee logic, runs first) --------------
    dict(id="int8-bundle",  shape="wide", n=4, size=8,  seed=11, cond="ref", plan="bundle1", runs=1, est=0.6,
         note="does k=1 beat serial at size 8? if yes: re-anchor rule fires"),
    # wide-15 gets the FULL outcome-distinct table instead of a single
    # intermediate check — it is the flip cell, and full execution is cheap
    # (~11-12 runs after the symmetry/floor collapse). Run via the existing
    # audit machinery, NOT this script:
    #   python3 -m harness.cli audit --model claude-opus-5 \
    #     --calibration results/cal-opus-v2/calibration.json \
    #     --shape wide --n 4 --size 15 --seed 11 --beta 1.0 \
    #     --out results/audit-w15   (~$8-10; also yields the safe-filter X%)
    dict(id="int25-split22", shape="wide", n=4, size=25, seed=11, cond="ref", plan="split22", runs=1, est=1.2,
         note="predicted-best at 25; same one-sided stakes as int15"),
    # -- the flip pair (the report's core cells) -----------------------------
    dict(id="w15-stated-b1", shape="wide", n=4, size=15, seed=11, cond="stated-b1", runs=3, est=0.9,
         correct="delegate", note="flip test; measured margin +$0.25"),
    dict(id="w15-stated-b0", shape="wide", n=4, size=15, seed=11, cond="stated-b0", runs=3, est=0.7,
         correct="serial", note="flip's other half; over-delegation detector"),
    # -- second-seed flip cell: instance generality for the core claim ------
    # Everything else runs on seed 11; if the flip replicates on an
    # independently generated instance, the headline claim stops resting on
    # one draw of the generator. Needs its own references (nothing banked).
    dict(id="w15b-ref-serial", shape="wide", n=4, size=15, seed=23, cond="ref", plan="serial", runs=1, est=0.6,
         note="seed-23 serial reference"),
    dict(id="w15b-ref-fanout", shape="wide", n=4, size=15, seed=23, cond="ref", plan="fanout", runs=1, est=0.9,
         note="seed-23 fan-out reference"),
    dict(id="w15b-stated-b1", shape="wide", n=4, size=15, seed=23, cond="stated-b1", runs=2, est=0.9,
         correct="delegate", note="flip replication on a second generated instance"),
    dict(id="w15b-stated-b0", shape="wide", n=4, size=15, seed=23, cond="stated-b0", runs=1, est=0.7,
         correct="serial", note="flip replication, beta=0 side"),
    # -- size contrast ------------------------------------------------------
    dict(id="w25-stated-b1", shape="wide", n=4, size=25, seed=11, cond="stated-b1", runs=2, est=1.2,
         correct="delegate", note="fat-margin delegate cell (+$0.51)"),
    dict(id="w25-stated-b0", shape="wide", n=4, size=25, seed=11, cond="stated-b0", runs=1, est=0.9,
         correct="serial", note="flip replication"),
    dict(id="w8-stated-b1",  shape="wide", n=4, size=8,  seed=11, cond="stated-b1", runs=2, est=0.6,
         correct="serial", note="near-boundary serial cell; SKIP if re-anchor rule fires"),
    dict(id="w3-stated-b1",  shape="wide", n=4, size=3,  seed=11, cond="stated-b1", runs=2, est=0.4,
         correct="serial", note="RESERVE: runs only if re-anchor rule fires"),
    # DECISION-ONLY cell beyond the payload validity boundary (F13): k from
    # SpawnRecords is immune to the attribution hole; correct answer robust
    # (delegate by $2.68 at beta=1 — serial must improve 40% to flip). NO
    # compliance, quality, or cost-curve claims from this cell. Refs banked
    # (results/probe-wide4-s40/).
    dict(id="w40-stated-b1", shape="wide", n=4, size=40, seed=11, cond="stated-b1", runs=2, est=1.3,
         correct="delegate", note="DECISION-ONLY margin stress test: does it delegate at a $2.68 margin?"),
    # -- shape placebo ------------------------------------------------------
    dict(id="c15-ref-serial", shape="chain", n=4, size=15, seed=11, cond="ref", plan="serial", runs=1, est=0.7,
         note="chain reference (not banked by the ladder)"),
    dict(id="c15-stated-b1", shape="chain", n=4, size=15, seed=11, cond="stated-b1", runs=2, est=0.8,
         correct="serial", note="size-heuristic placebo: big, time-priced, no parallelism"),
    dict(id="c15-blind", shape="chain", n=4, size=15, seed=11, cond="blind", runs=1, est=0.8,
         correct="serial", note="dominated-region blind placebo: spawning here is wrong at EVERY beta"),
    # -- revealed preference (current protocol, continuity with pass-1) ----
    dict(id="w15-blind", shape="wide", n=4, size=15, seed=11, cond="blind", runs=2, est=0.8,
         note="revealed-beta bracket, lower rung"),
    dict(id="w25-blind", shape="wide", n=4, size=25, seed=11, cond="blind", runs=2, est=1.0,
         note="revealed-beta bracket, upper rung"),
    # -- optional discovery-vs-decision contrast ----------------------------
    dict(id="w15-undisc-b1", shape="wide", n=4, size=15, seed=11, cond="stated-b1", runs=1, est=0.9,
         correct="delegate", disclose=False, note="OPTIONAL: undisclosed DAG contrast"),
    # -- queued dominance probe (AFTER Phase D, budget allowing) ------------
    # F13's valid best-case configuration: few workers, big blocks, every
    # module inside the hand-repair regime (size 25 = 5 reps/kind). Chasing
    # dominance above size 40 is invalid (payload boundary) — this is the
    # in-range version: 200 units total, k=2, 4 modules per subagent.
    dict(id="dom8s25-serial", shape="wide", n=8, size=25, seed=11, cond="ref", plan="serial", runs=1, est=1.6,
         note="QUEUED: dominance config serial reference (200 units, validated regime)"),
    dict(id="dom8s25-k2", shape="wide", n=8, size=25, seed=11, cond="ref", plan="splithalf", runs=1, est=1.8,
         note="QUEUED: few-workers/big-blocks best case — k=2, 100 units per subagent"),
]


def cell_by_id(cid: str) -> dict:
    for c in CELLS:
        if c["id"] == cid:
            return c
    raise SystemExit(f"unknown cell {cid!r}; run --list")


def stage(out_root: str = "results/matrix-staging") -> int:
    """Materialize every matrix scenario + enumeration for MANUAL REVIEW.

    Zero spend, no key. Per scenario: workspace/ (what agents will see,
    disclosed variant), reference/ (the answer key), plans.json (the full
    enumeration with nomination-only predicted prices), checks.json. Live
    checks: broken workspace fails with one failure per defect; reference
    passes; on chain, a repaired successor STILL fails while its predecessor
    is broken (dependency propagation); on wide, one repaired node passes
    alone (independence).
    """
    import shutil
    import subprocess

    from generator.oracle import enumerate_plans, evaluate
    from generator.scenario import module_path, test_module_dotted
    from harness.calibration import CalibrationResult

    cm = CalibrationResult.load("results/cal-opus-v2/calibration.json").to_cost_model()

    def suite(cwd, dotted=None) -> bool:
        tail = ["-q", dotted] if dotted else ["discover", "-q"]
        cmd = [sys.executable, "-B", "-m", "unittest"] + tail
        return subprocess.run(cmd, capture_output=True, cwd=str(cwd), timeout=300).returncode == 0

    seen, failures = {}, 0
    for c in CELLS:
        key = (c["shape"], c["n"], c["size"], c["seed"])
        if key in seen:
            continue
        scenario = ScenarioSpec(*key).build()
        seen[key] = scenario
        root = Path(out_root) / scenario.id
        if root.exists():
            shutil.rmtree(root)
        work = scenario.materialize(root / "workspace", disclose_dag=True)
        ref = scenario.write_reference(root / "reference")

        nodes = list(scenario.dag.topo_order)
        checks = {"broken_fails": not suite(work), "reference_passes": suite(ref)}
        # repair ONE node in a copy; on chain pick the successor (its suite
        # must STILL fail: predecessor broken); on wide any node (must pass).
        probe_node = nodes[1] if scenario.dag.edges else nodes[0]
        copy = root / "check-one-repair"
        shutil.copytree(work, copy)
        shutil.copy(ref / module_path(probe_node), copy / module_path(probe_node))
        one = suite(copy, test_module_dotted(probe_node))
        if scenario.dag.edges:
            checks["chain_propagation (repaired successor still fails)"] = not one
            shutil.copy(ref / module_path(nodes[0]), copy / module_path(nodes[0]))
            checks["chain_repair_in_order_passes"] = suite(copy, test_module_dotted(probe_node))
        else:
            checks["wide_independence (one repaired node passes alone)"] = one

        plans = enumerate_plans(scenario.dag)
        priced = sorted(
            ({"k": len(p.blocks), "inline": sorted(p.inline), "blocks": [sorted(b) for b in p.blocks],
              "predicted_dollars_NOMINATION_ONLY": round(evaluate(scenario.dag, p, cm).cost, 4),
              "predicted_minutes_NOMINATION_ONLY": round(evaluate(scenario.dag, p, cm).latency, 4)}
             for p in plans), key=lambda r: r["predicted_dollars_NOMINATION_ONLY"] + r["predicted_minutes_NOMINATION_ONLY"])
        (root / "plans.json").write_text(json.dumps(
            {"n_plans": len(plans), "note": "prices are calibration nominations, out-of-range above size 5; "
             "verdicts come from execution only", "plans": priced[:40]}, indent=2))
        (root / "checks.json").write_text(json.dumps(checks, indent=2))
        ok = all(checks.values())
        failures += 0 if ok else 1
        print(f"  {scenario.id:<14} plans={len(plans):>5}  " +
              "  ".join(f"{k.split(' ')[0]}={'OK' if v else 'FAIL'}" for k, v in checks.items()) +
              ("" if ok else "   <-- INSPECT"))
    print(f"\n  staged {len(seen)} scenarios under {out_root}/ "
          f"({'ALL CHECKS PASS' if failures == 0 else f'{failures} SCENARIO(S) FAILED CHECKS'})")
    return 0 if failures == 0 else 1


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", action="store_true",
                   help="materialize all scenarios + enumeration for manual review; no key, no spend")
    p.add_argument("--list", action="store_true", help="print the run plan and estimated cost")
    p.add_argument("--cell", default=None, help="cell id to execute")
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--out-root", default="results/matrix")
    p.add_argument("--max-turns", type=int, default=120)
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--base-url", default=ANTHROPIC_API)
    p.add_argument("--api-key-env", default="ANTHROPIC_API_KEY")
    p.add_argument("--max-output-tokens", type=int, default=200_000)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.stage:
        return stage()

    if args.list or not args.cell:
        total_runs = sum(c["runs"] for c in CELLS)
        total_est = sum(c["runs"] * c["est"] for c in CELLS)
        print(f"  {len(CELLS)} cells, {total_runs} runs, ~${total_est:.0f} estimated "
              f"(reserve for adaptive repeats not included)")
        for c in CELLS:
            correct = c.get("correct", "-")
            print(f"  {c['id']:<16} {c['shape']}{c['n']}-s{c['size']:<3} {c['cond']:<10} "
                  f"runs={c['runs']}  ~${c['runs']*c['est']:.1f}  correct={correct:<8} {c['note']}")
        return 0

    cell = cell_by_id(args.cell)
    spec = ScenarioSpec(cell["shape"], cell["n"], cell["size"], cell["seed"])
    scenario = spec.build()
    disclose = cell.get("disclose", True)
    out = Path(args.out_root) / cell["id"]

    price = price_sheet(args.model)
    print(f"  cell {cell['id']}  scenario {scenario.id}  cond {cell['cond']}  "
          f"runs {cell['runs']}  disclose_dag={disclose}")
    if args.dry_run:
        print("  dry run: nothing executed.")
        return 0

    import os
    # The provider dispatch lives in harness.cli so this entry point cannot
    # drift from the others; untouched Anthropic defaults re-route for gpt-*.
    base_url, key_env = endpoint_defaults(price.model, args.base_url, args.api_key_env)
    key = os.environ.get(key_env, "")
    if not key:
        raise SystemExit(f"{key_env} is not set.")
    client = client_for_model(price.model, key, base_url, args.max_tokens, args.timeout)
    out.mkdir(parents=True, exist_ok=True)

    directive = {"stated-b1": DIRECTIVE_B1, "stated-b0": DIRECTIVE_B0}.get(cell["cond"], "")
    runs = []
    for i in range(cell["runs"]):
        print(f"\n  run {i + 1}/{cell['runs']} ...", flush=True)
        common = dict(
            max_turns=args.max_turns,
            budget=Budget(max_output_tokens=args.max_output_tokens),
            proxy_log=out / f"run-{i}-proxy.jsonl",
        )
        if cell["cond"] == "ref":
            plan = named_plan(cell["plan"], scenario.dag)
            trace = run_plan(scenario, plan, client, out / f"work-{i}",
                             condition=f"matrix-ref-{cell['plan']}", **common)
        else:
            trace = run_agent(scenario, client, out / f"work-{i}",
                              disclose_dag=disclose, condition=f"matrix-{cell['cond']}",
                              repeat=i, directive=directive, **common)
        trace.write(out / f"run-{i}-trace.json")
        runs.append(dict(
            k=trace.k, succeeded=trace.succeeded, turns=trace.turns,
            proxy_verified=trace.proxy_verified,
            realized=[sorted(b) for b in trace.realized_plan()[1]],
            notes=[n for n in trace.notes if n.startswith("PLAN NOT FOLLOWED")],
        ))
        print(f"  k={trace.k}  succeeded={trace.succeeded}  turns={trace.turns}")

    summary = dict(cell=cell, scenario=scenario.id, model=price.model,
                   directive=directive, runs=runs)
    (out / "cell.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"\n  wrote {out}/cell.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
