"""The boundary probe: one ladder rung, pulled forward as the go/no-go.

DATASET-BOUNDARY-DEFECT.md section 5 pre-registers a size ladder before any of
it runs. This executes ONE rung -- default wide-4 at node size 15, seed 11, the
same DAG family as the section-4 table -- with the two reference arms, and
reports the MEASURED break-even beta* next to the oracle's extrapolated one.

Why this rung, and why serial runs first: at size 15 the fan-out arm's subagent
blocks are 15 units each, inside the measured 0-24-unit calibration range, so
the prediction's uncertainty is concentrated in a single number -- the serial
arm's 60-unit block, priced today by extrapolation. The serial arm runs first
so the load-bearing measurement exists even if the process dies afterwards.

Leak note (TASKS section 4 item 14): the harness degrades around ~31 runs in
one process; this probe is 2 runs in a fresh process, below the threshold, so
it does not wait on the leak fix. Do NOT extend this script into a loop over
rungs in one process until item 14 lands -- run rungs as separate invocations.

    python3 probe.py --dry-run     # predicted table only; no key, no spend
    python3 probe.py               # both arms; ~$1.5-3 at cal-opus-v2 prices
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

from generator.manifest import ScenarioSpec
from generator.oracle import all_inline, evaluate, max_fanout
from harness.calibrate import PriceSheet, TimingModel
from harness.calibration import CalibrationResult
from harness.cli import ANTHROPIC_API, client_for_model, endpoint_defaults, price_sheet
from harness.runner import Budget, run_plan


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--calibration", default="results/cal-opus-v2/calibration.json")
    p.add_argument("--shape", default="wide")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--size", type=int, default=15)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", default=None, help="default: results/probe-{shape}{n}-s{size}")
    p.add_argument("--arms", default="serial,fanout",
                   help="comma subset of serial,fanout; serial is the load-bearing one")
    p.add_argument("--max-turns", type=int, default=120,
                   help="pass-1 ran 60 for size-3 nodes; a 60-defect serial arm needs headroom")
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--base-url", default=ANTHROPIC_API)
    p.add_argument("--api-key-env", default="ANTHROPIC_API_KEY")
    p.add_argument("--max-output-tokens", type=int, default=200_000,
                   help="per-arm Budget ceiling; ~$5 worst case at opus output rates")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--pack-spawns", action="store_true",
                   help="add the single-turn spawn-packing line to the plan directive "
                        "(the disambiguation condition; a separate labeled arm, never "
                        "comparable with baseline-wording runs)")
    return p.parse_args(argv)


def load_calibration(path: str, price: PriceSheet):
    """(timing, cost_model, floors), with cmd_experiment's model refusal."""
    calib = CalibrationResult.load(path)
    calibrated_model = (calib.price_sheet or {}).get("model", "")
    if calibrated_model != price.model:
        raise SystemExit(
            f"calibration {path} was measured on {calibrated_model!r}, but --model "
            f"resolves to {price.model!r}. Curves do not transfer; calibrate first."
        )
    timing_keys = {f.name for f in fields(TimingModel)}
    timing_dict = {k: v for k, v in (calib.timing_model or {}).items() if k in timing_keys}
    if not timing_dict:
        raise SystemExit("the calibration has no timing model; the latency axis would be a guess.")
    floors = ((calib.floors or {}).get("dollars"), (calib.floors or {}).get("minutes"))
    return TimingModel(**timing_dict), calib.to_cost_model(), floors


def beta_star(serial_d, serial_m, fan_d, fan_m):
    """$/min at which fan-out starts paying; None if it never does (no latency win)."""
    if serial_m <= fan_m:
        return None
    return (fan_d - serial_d) / (serial_m - fan_m)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out or f"results/probe-{args.shape}{args.n}-s{args.size}")
    price = price_sheet(args.model)
    timing, cm, floors = load_calibration(args.calibration, price)

    scenario = ScenarioSpec(args.shape, args.n, args.size, args.seed).build()
    dag = scenario.dag
    plans = {"serial": all_inline(dag), "fanout": max_fanout(dag)}
    predicted = {name: evaluate(dag, plan, cm) for name, plan in plans.items()}

    pb = beta_star(predicted["serial"].cost, predicted["serial"].latency,
                   predicted["fanout"].cost, predicted["fanout"].latency)
    print(f"  probe {scenario.id}  model {price.model}  calibration {args.calibration}")
    for name in ("serial", "fanout"):
        pr = predicted[name]
        units = args.size * args.n if name == "serial" else args.size
        blocks = "block" if name == "serial" else "blocks"
        regime = "EXTRAPOLATED" if units > 24 else "measured range"
        print(f"  predicted {name:<6}  ${pr.cost:.2f}  {pr.latency:.2f} min"
              f"   ({units}-unit {blocks}: {regime})")
    print(f"  predicted beta*  {'none (no latency win)' if pb is None else f'${pb:.3f}/min'}")

    if args.dry_run:
        print("  dry run: no key, no spend.")
        return 0

    # The provider dispatch lives in harness.cli so this entry point cannot
    # drift from the others; untouched Anthropic defaults re-route for gpt-*.
    base_url, key_env = endpoint_defaults(price.model, args.base_url, args.api_key_env)
    key = os.environ.get(key_env, "")
    if not key:
        raise SystemExit(f"{key_env} is not set.")
    client = client_for_model(price.model, key, base_url, args.max_tokens, args.timeout)

    out.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    measured = {}
    for name in arms:  # serial first: the load-bearing number survives a died process
        plan = plans[name]
        print(f"\n  running {name} arm ...", flush=True)
        trace = run_plan(
            scenario, plan, client, out / f"work-{name}",
            condition=f"probe-{name}" + ("-packed" if args.pack_spawns else ""),
            max_turns=args.max_turns,
            budget=Budget(max_output_tokens=args.max_output_tokens),
            proxy_log=out / f"{name}-proxy.jsonl",
            pack_spawns=args.pack_spawns,
        )
        trace.write(out / f"{name}-trace.json")
        d, m = trace.dollars(price), trace.analytic_minutes(timing)
        complied = not any(n.startswith("PLAN NOT FOLLOWED") for n in trace.notes)
        measured[name] = {
            "dollars": d, "minutes": m, "succeeded": trace.succeeded,
            "complied": complied, "k": trace.k, "turns": trace.turns,
            "proxy_verified": trace.proxy_verified,
        }
        print(f"  measured  {name:<6}  ${d:.2f}  {m:.2f} min  "
              f"succeeded={trace.succeeded}  complied={complied}  k={trace.k}  "
              f"turns={trace.turns}/{args.max_turns}  proxy_verified={trace.proxy_verified}")
        pr = predicted[name]
        print(f"  gap vs predicted   dollars {d - pr.cost:+.2f} ({(d - pr.cost) / pr.cost:+.0%})"
              f"   minutes {m - pr.latency:+.2f} ({(m - pr.latency) / pr.latency:+.0%})")

    summary = {
        "scenario": scenario.id, "model": price.model, "calibration": args.calibration,
        "pack_spawns": args.pack_spawns,
        "predicted": {k: {"dollars": v.cost, "minutes": v.latency} for k, v in predicted.items()},
        "predicted_beta_star": pb,
        "measured": measured,
        "floors": {"dollars": floors[0], "minutes": floors[1]},
    }

    verdict = []
    if len(measured) == 2:
        s, f = measured["serial"], measured["fanout"]
        valid = all(a["succeeded"] and a["complied"] for a in measured.values())
        mb = beta_star(s["dollars"], s["minutes"], f["dollars"], f["minutes"])
        summary["measured_beta_star"] = mb
        summary["valid"] = valid
        delta_m = s["minutes"] - f["minutes"]
        if not valid:
            verdict.append("NOT VALID: an arm failed verification or did not follow its plan; "
                           "numbers above are not a measurement of the boundary.")
        elif mb is None:
            verdict.append("no latency win at this size -> probe the next rung; if flat there "
                           "too, decision rule 3 (scoped negative) is the report.")
        else:
            verdict.append(f"MEASURED beta* = ${mb:.3f}/min at node size {args.size} "
                           f"(predicted {'none' if pb is None else f'${pb:.3f}'}).")
            if floors[1] is not None and delta_m < floors[1]:
                verdict.append(f"CAUTION: latency delta {delta_m:.2f} min is inside the noise "
                               f"floor ({floors[1]:.2f} min); treat beta* as unresolved, not small.")
            elif mb < 1.0:
                verdict.append("GREENLIGHT: delegation pays within the scored beta range at this "
                               "size. Run the full ladder + discriminating mini-matrix as "
                               "pre-registered (after the item-14 leak fix).")
            else:
                verdict.append("MARGINAL: crossover exists but above the scored betas; "
                               "probe size 25 (as its own fresh process) before committing.")
    summary["verdict"] = verdict

    (out / "probe.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print()
    for line in verdict:
        print(f"  {line}")
    print(f"\n  wrote {out}/probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
