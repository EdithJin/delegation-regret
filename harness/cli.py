"""The real-endpoint entry point: one command from a pinned model to a file.

    python -m harness.cli models
    python -m harness.cli manifest   --out scenarios/core/manifest.json
    python -m harness.cli calibrate  --model claude-opus-5 --out results/cal-opus
    python -m harness.cli experiment --model claude-opus-5 \\
        --manifest scenarios/core/manifest.json \\
        --calibration results/cal-opus/calibration.json --out results/run-opus

Until this module existed, nothing in the repo could touch a real endpoint
without a hand-written script: `run_calibration` and `run_experiment` take an
already-constructed client and price sheet, no `PriceSheet` had ever been
instantiated, and no model ID was pinned anywhere. This is where those choices
live -- in code, dated, and greppable -- rather than in whatever script someone
typed the night of the run.

Three decisions are deliberately hard-coded rather than accepted as flags:

* THE PRICE SHEETS ARE PINNED, NOT PASSED. An undated dollar is not a unit
  (see `calibrate.PriceSheet`), and a rate typed on the command line at 11pm is
  exactly how a units error enters a report. `--model` selects from `PRICE_SHEETS`;
  changing a rate means editing this file, which leaves a diff.
* EXPERIMENT REFUSES A CALIBRATION FROM A DIFFERENT MODEL. The block curves are
  read off one model's runs under one price sheet; scoring another model's
  matrix against them would divide two currencies again -- the precise mistake
  the NOT COMPARABLE gate exists to stop.
* THE CALIBRATION SCENARIO DEFAULTS TO WIDE WITH MIXED NODE SIZES. Two
  identifiability traps live in `calibration.py`: uniform output lengths make
  the timing model unfittable, and a chain has no legal bundled plan, so the
  briefing slope stays a placeholder. wide + sizes drawn from {2,3,5} avoids
  both. The flags can override the shape; the default should not invite the
  trap.

The Anthropic key is read from ANTHROPIC_API_KEY (overridable via
--api-key-env), which no other module reads: smoke.py's native leg defers to
the `claude` CLI's own auth, and everything else runs keyless against the fake
upstream. `--base-url` exists so tests can point the same code at
`ProtocolUpstream`; the default is the real endpoint, and the runner routes
every call through the logging proxy on its own.

The OpenAI wire format has no entry here on purpose: the open-weights leg was
dropped on Aug 24 (TASKS-AND-OPEN-ISSUES.md section 3) after its go/no-go
deadline passed unmet. When it returns, it needs an `OpenAIClient` branch and a
provider-pinned price sheet -- the price vector is provider-specific, not
model-specific.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields
from pathlib import Path

from generator.dag import SHAPES, sample_dag
from generator.manifest import ANCHOR, CORE, Manifest, heldout
from generator.manifest import load as load_manifest
from generator.manifest import write as write_manifest
from generator.oracle import unmeasured
from generator.scenario import Scenario, build_scenario

from .calibrate import PriceSheet, TimingModel
from .calibration import CalibrationResult, replay, run_calibration
from .client import AnthropicClient
from .experiment import run_experiment
from .runner import Budget

ANTHROPIC_API = "https://api.anthropic.com"

# Dollars per million tokens, from the vendor's published rates, checked
# 2026-08-24. Cache reads are 0.1x input; cache writes carry the 2x premium of
# the ONE-HOUR TTL, because that is the TTL the harness pins (design doc
# section 7 -- constants measured under an unpinned TTL do not reproduce).
# NOTE: the client does not emit `cache_control` yet, so until the pinning task
# lands these two columns bill zero tokens; they are recorded now so the sheet
# is complete when the first cached run happens.
#
# claude-sonnet-5 is on introductory pricing through 2026-08-31. The intro
# sheet is the default because every planned run lands inside the window; the
# "@list" entry exists so any absolute dollar figure can be restated at list
# rates, which the write-up is required to do (Cost-And-Funding-Notes item 5).
PRICE_SHEETS: dict[str, PriceSheet] = {
    "claude-opus-5": PriceSheet(
        model="claude-opus-5", as_of="2026-08-24",
        input_per_mtok=5.00, output_per_mtok=25.00,
        cache_read_per_mtok=0.50, cache_write_per_mtok=10.00,
    ),
    "claude-sonnet-5": PriceSheet(
        model="claude-sonnet-5", as_of="2026-08-24",
        input_per_mtok=2.00, output_per_mtok=10.00,
        cache_read_per_mtok=0.20, cache_write_per_mtok=4.00,
    ),
    "claude-sonnet-5@list": PriceSheet(
        model="claude-sonnet-5", as_of="2026-08-24",
        input_per_mtok=3.00, output_per_mtok=15.00,
        cache_read_per_mtok=0.30, cache_write_per_mtok=6.00,
    ),
    "claude-haiku-4-5": PriceSheet(
        model="claude-haiku-4-5", as_of="2026-08-24",
        input_per_mtok=1.00, output_per_mtok=5.00,
        cache_read_per_mtok=0.10, cache_write_per_mtok=2.00,
    ),
}


def price_sheet(key: str) -> PriceSheet:
    try:
        return PRICE_SHEETS[key]
    except KeyError:
        known = ", ".join(sorted(PRICE_SHEETS))
        raise SystemExit(
            f"no price sheet for {key!r}. Pinned models: {known}. A new model "
            "needs its rates entered in harness/cli.py with an as_of date -- "
            "an undated dollar is not a unit."
        )


def build_client(args, price: PriceSheet) -> AnthropicClient:
    key = os.environ.get(args.api_key_env, "")
    if not key:
        raise SystemExit(
            f"{args.api_key_env} is not set. Export it, or name another "
            "variable with --api-key-env. There is no keyless mode against a "
            "real endpoint; for a keyless dry run, point --base-url at a "
            "ProtocolUpstream the way tests/test_cli.py does."
        )
    return AnthropicClient(
        model=price.model,
        api_key=key,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )


def calibration_scenario(args) -> Scenario:
    """Deterministic from its flags, so `--replay` rebuilds the identical one."""
    sizes = tuple(int(s) for s in args.sizes.split(","))
    dag = sample_dag(args.shape, args.n, sizes=sizes, seed=args.seed)
    scenario = build_scenario(dag, f"cal-{args.shape}{args.n}-s{args.seed}", seed=args.seed)
    distinct = {node.size for node in dag.nodes}
    if len(distinct) < 2:
        print(
            f"  ! every node drew size {distinct.pop()} from sizes={sizes}: uniform "
            "output lengths make the timing model unfittable. Pick another --seed "
            "or widen --sizes.",
            file=sys.stderr,
        )
    return scenario


def _budget(args) -> Budget | None:
    if args.max_calls is None and args.max_output_tokens is None:
        return None
    kwargs = {}
    if args.max_calls is not None:
        kwargs["max_calls"] = args.max_calls
    if args.max_output_tokens is not None:
        kwargs["max_output_tokens"] = args.max_output_tokens
    return Budget(**kwargs)


# ------------------------------------------------------------------ commands


def cmd_models(args) -> int:
    for key in sorted(PRICE_SHEETS):
        p = PRICE_SHEETS[key]
        print(
            f"  {key:<22} in ${p.input_per_mtok:>5.2f}  out ${p.output_per_mtok:>6.2f}  "
            f"cache-read ${p.cache_read_per_mtok:.2f}  cache-write ${p.cache_write_per_mtok:.2f}  "
            f"per MTok, as of {p.as_of}"
        )
    print("\n  claude-sonnet-5 is the introductory sheet, valid through 2026-08-31;")
    print("  claude-sonnet-5@list restates the same model at list rates.")
    return 0


def cmd_manifest(args) -> int:
    if args.set == "core":
        manifest = CORE
        if args.seeds:
            keep = {int(s) for s in args.seeds.split(",")}
            specs = tuple(s for s in CORE.specs if s.seed in keep)
            if not specs:
                raise SystemExit(f"no core spec has a seed in {sorted(keep)}")
            manifest = Manifest(
                name=f"core-seeds-{'-'.join(str(s) for s in sorted(keep))}",
                specs=specs,
                note=f"{CORE.note} Trimmed to seeds {sorted(keep)} per shape*n cell.".strip(),
            )
    elif args.set == "anchor":
        manifest = ANCHOR
    else:
        manifest = heldout(count=args.heldout_count, offset=args.heldout_offset)
    path = write_manifest(manifest, args.out)
    print(manifest.summary())
    print(f"\n  written to {path}")
    print(f"  COMMIT THIS FILE. The fingerprint {manifest.fingerprint} is the "
          "pre-registration; uncommitted it is just an assertion.")
    return 0


def cmd_calibrate(args) -> int:
    price = price_sheet(args.model)
    scenario = calibration_scenario(args)
    out = Path(args.out)
    source = f"{price.model} calibration {out.name} @ {price.as_of}"
    print(f"  model {price.model}  endpoint {args.base_url}  max_tokens {args.max_tokens}")
    print(f"  scenario {scenario.dag.shape} n={len(scenario.dag.nodes)} -> {out}")
    if args.replay:
        result = replay(out, scenario, price, source=source)
    else:
        client = build_client(args, price)
        result = run_calibration(
            scenario, client, price, out,
            source=source,
            orderings=args.orderings,
            repeats=args.repeats,
            max_turns=args.max_turns,
            budget=_budget(args),
        )
    print()
    print(result.report())
    print(f"\n  next: --calibration {out / 'calibration.json'} on the experiment command")
    return 0


def cmd_experiment(args) -> int:
    price = price_sheet(args.model)
    manifest = load_manifest(args.manifest)
    calib = CalibrationResult.load(args.calibration)

    calibrated_model = (calib.price_sheet or {}).get("model", "")
    if calibrated_model != price.model:
        raise SystemExit(
            f"calibration {args.calibration} was measured on {calibrated_model!r}, "
            f"but --model resolves to {price.model!r}. Curves do not transfer "
            "across models or price sheets; calibrate this model first."
        )
    timing_keys = {f.name for f in fields(TimingModel)}
    timing_dict = {k: v for k, v in (calib.timing_model or {}).items() if k in timing_keys}
    if not timing_dict:
        raise SystemExit(
            "the calibration has no timing model, so the latency axis would be "
            "a guess. See its 'skipped' section for why the fit was refused, "
            "then re-run calibration on a scenario with varied node sizes."
        )
    timing = TimingModel(**timing_dict)
    cost_model = calib.to_cost_model()
    still = unmeasured(cost_model)
    if still:
        print(f"  ! {len(still)} constant(s) still placeholder ({', '.join(still)}) -- "
              "every card will be marked NOT COMPARABLE.", file=sys.stderr)

    client = build_client(args, price)
    betas = tuple(float(b) for b in args.betas.split(","))
    print(f"  model {price.model}  manifest {manifest.name} ({len(manifest)} scenarios, "
          f"fingerprint {manifest.fingerprint})")
    print(f"  betas {betas}  repeats {args.repeats}  -> {args.out}")
    results = run_experiment(
        manifest, client, price, timing, cost_model, args.out,
        calibration_source=calib.source,
        betas=betas,
        repeats=args.repeats,
        disclosed=not args.no_disclosed,
        ordering_subset=args.ordering_subset,
        max_turns=args.max_turns,
        budget=_budget(args),
    )
    print()
    print(results.report())
    return 0


# ---------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_endpoint_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", required=True, choices=sorted(PRICE_SHEETS),
                       help="a pinned model; rates live in PRICE_SHEETS")
        p.add_argument("--base-url", default=ANTHROPIC_API)
        p.add_argument("--api-key-env", default="ANTHROPIC_API_KEY")
        p.add_argument("--max-tokens", type=int, default=16000,
                       help="thinking and text share this cap on current models; "
                            "4096 truncates mid-tool-call")
        p.add_argument("--timeout", type=float, default=300.0)
        p.add_argument("--max-turns", type=int, default=60)
        p.add_argument("--max-calls", type=int, default=None,
                       help="override Budget.max_calls (default 400)")
        p.add_argument("--max-output-tokens", type=int, default=None,
                       help="override Budget.max_output_tokens (default 400000)")

    mod = sub.add_parser("models", help="print the pinned price sheets")
    mod.set_defaults(func=cmd_models)

    man = sub.add_parser("manifest", help="write a pre-registered manifest to disk")
    man.add_argument("--set", default="core", choices=("core", "anchor", "heldout"))
    man.add_argument("--seeds", help="core only: keep these seeds, e.g. '11,23' trims 36 -> 24")
    man.add_argument("--heldout-count", type=int, default=12)
    man.add_argument("--heldout-offset", type=int, default=0)
    man.add_argument("--out", required=True)
    man.set_defaults(func=cmd_manifest)

    cal = sub.add_parser("calibrate", help="run the calibration against a real endpoint")
    add_endpoint_args(cal)
    cal.add_argument("--shape", default="wide", choices=tuple(sorted(SHAPES)))
    cal.add_argument("--n", type=int, default=6)
    cal.add_argument("--sizes", default="2,3,5",
                     help="node sizes drawn per node; varied on purpose -- see docstring")
    cal.add_argument("--seed", type=int, default=11)
    cal.add_argument("--orderings", type=int, default=3)
    cal.add_argument("--repeats", type=int, default=2)
    cal.add_argument("--replay", action="store_true",
                     help="re-derive constants from traces already in --out; no runs, no key")
    cal.add_argument("--out", required=True)
    cal.set_defaults(func=cmd_calibrate)

    exp = sub.add_parser("experiment", help="run a manifest and write a stamped results file")
    add_endpoint_args(exp)
    exp.add_argument("--manifest", required=True)
    exp.add_argument("--calibration", required=True, help="path to calibration.json")
    exp.add_argument("--betas", default="0.0,1.0")
    exp.add_argument("--repeats", type=int, default=1)
    exp.add_argument("--no-disclosed", action="store_true")
    exp.add_argument("--ordering-subset", type=int, default=3)
    exp.add_argument("--out", required=True)
    exp.set_defaults(func=cmd_experiment)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
