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

The OpenAI leg landed 2026-08-27 (it was dropped Aug 24 when its go/no-go
deadline passed unmet; TASKS-AND-OPEN-ISSUES.md section 3): gpt-* models carry
their own provider-pinned price sheets below -- the price vector is
provider-specific, not model-specific -- and `build_client` dispatches them to
`OpenAIClient` via `provider_for`. When a gpt model is selected and --base-url
/ --api-key-env are left at their Anthropic defaults, `endpoint_defaults`
re-routes them to https://api.openai.com and OPENAI_API_KEY; explicit flags
always win, which is how tests point a gpt model at a local fake.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

from generator.dag import SHAPES, sample_dag
from generator.manifest import ANCHOR, CORE, Manifest, ScenarioSpec, heldout
from generator.manifest import load as load_manifest
from generator.manifest import write as write_manifest
from generator.oracle import unmeasured
from generator.scenario import Scenario, build_scenario

from .audit import audit_summary, eps_from_calibration, run_audit
from .calibrate import PriceSheet, TimingModel
from .calibration import CalibrationResult, replay, run_calibration
from .client import AnthropicClient, Client, OpenAIClient
from .experiment import run_experiment
from .runner import Budget

ANTHROPIC_API = "https://api.anthropic.com"
OPENAI_API = "https://api.openai.com"

# Dollars per million tokens, from the vendor's published rates, checked
# 2026-08-24. Cache reads are 0.1x input; cache writes carry the 2x premium of
# the ONE-HOUR TTL, because that is the TTL the harness pins (design doc
# section 7 -- constants measured under an unpinned TTL do not reproduce).
# The cache columns are live: the client breakpoints the system block and the
# conversation tail at this TTL (HARNESS_SPEC below), and the preflight's
# `cache_read_input_tokens > 0` check is what proves the writes are being read.
#
# claude-sonnet-5 is on introductory pricing through 2026-08-31. The intro
# sheet is the default because every planned run lands inside the window; the
# "@list" entry exists so any absolute dollar figure can be restated at list
# rates, which the write-up is required to do (Cost-And-Funding-Notes item 5).
#
# claude-opus-5 is the matrix leg -- lead and subagents alike, one model per
# leg (the runner shares a single client by design; design doc section 7).
# There is deliberately no fable-tier sheet: rejected Aug 25 for the matrix --
# 2x rates, tier mismatch against any workhorse-tier comparator, and its
# refusal stop reason cannot be mitigated inside a pinned-model leg, because
# the sanctioned mitigation (server-side fallbacks) swaps models mid-run.
# Reasoning in design doc section 10; a flagship-subset panel is queued for
# October, and only then does a fable sheet get added here.
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
    # gpt-5.6-sol is the OpenAI comparator leg, on promotional pricing through
    # 2026-11-21 (rates checked 2026-08-27). The promo sheet is the default
    # because every planned run lands inside the window; "@list" exists so any
    # absolute dollar figure can be restated at list rates, the same discipline
    # claude-sonnet-5 follows. Cached input bills at 0.1x like everywhere else,
    # but the cache-WRITE rate is a true zero, not a missing number: OpenAI
    # prompt caching is automatic and charges nothing to write, so there is no
    # premium to enter -- and correspondingly no TTL to pin (see HARNESS_SPEC).
    "gpt-5.6-sol": PriceSheet(
        model="gpt-5.6-sol", as_of="2026-08-27",
        input_per_mtok=4.00, output_per_mtok=20.00,
        cache_read_per_mtok=0.40, cache_write_per_mtok=0.00,
    ),
    "gpt-5.6-sol@list": PriceSheet(
        model="gpt-5.6-sol", as_of="2026-08-27",
        input_per_mtok=5.00, output_per_mtok=30.00,
        cache_read_per_mtok=0.50, cache_write_per_mtok=0.00,
    ),
}


# The pinned harness spec (TASKS-AND-OPEN-ISSUES section 2, Aug 24). These are
# published constants: every measured run uses exactly these values, and the
# report reports them. The rest of the spec lives where it is enforced --
# max_tokens 16000 is the --max-tokens default below (thinking and text share
# the cap on current models; 4096 truncates mid-tool-call), the concurrency cap
# is tools.MAX_CONCURRENCY = 4 (pinned by test), and fan-out launches
# UNSTAGGERED (concurrent spawns forfeiting cache reads is a finding to
# measure, not an inefficiency to engineer away -- open issue section 4.5).
#
# thinking "adaptive" is the only on-mode the current generation accepts, with
# depth belonging to `effort`; "high" is the API default, pinned explicitly
# because a silently inherited default is not a published constant.
# claude-haiku-4-5 is plumbing, never matrix: it predates adaptive thinking
# and the effort parameter and rejects both with a 400, so its spec sends
# neither -- the fake upstream enforces the same rejection offline.
HARNESS_SPEC: dict[str, dict] = {
    "claude-opus-5": {"thinking": {"type": "adaptive"}, "effort": "high", "cache_ttl": "1h"},
    "claude-sonnet-5": {"thinking": {"type": "adaptive"}, "effort": "high", "cache_ttl": "1h"},
    "claude-haiku-4-5": {"thinking": None, "effort": None, "cache_ttl": "1h"},
    # The OpenAI leg's spec deliberately has NO thinking key (an Anthropic-only
    # parameter; effort maps to reasoning_effort on this wire format) and NO
    # cache_ttl: OpenAI prompt caching is automatic -- unbilled on write, with
    # provider-managed retention that cannot be purchased or pinned.
    # COMPARABILITY CAVEAT, to be published with any cross-provider table: the
    # Anthropic legs run under a pinned one-hour TTL while this leg's cache
    # lifetime floats at the provider's discretion, so cache-hit rates (and the
    # realized costs downstream of them) are not measured under the same
    # retention discipline. max_tokens 16000 matches the --max-tokens default
    # every leg runs under, recorded here so this leg's constants are greppable
    # in one place like the others.
    # effort "none" is REQUIRED, not chosen: the real endpoint 400s function
    # tools on /v1/chat/completions unless reasoning_effort is explicitly
    # "none" (verified live 2026-08-27; the default is non-none, so omitting
    # the field also 400s). Comparability caveat: the opus leg runs effort
    # "high" — this leg measures gpt-5.6-sol's no-reasoning tool mode. The
    # reasoning-ON leg requires the /v1/responses wire format (October queue).
    "gpt-5.6-sol": {"max_tokens": 16000, "effort": "none"},
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


def provider_for(model: str) -> str:
    """Which wire format a pinned model speaks: "openai" for gpt-*, otherwise
    "anthropic". The prefix is the dispatch rule on purpose -- a new gpt model
    needs only its sheets and spec entered above, not a new branch."""
    return "openai" if model.startswith("gpt-") else "anthropic"


def endpoint_defaults(model: str, base_url: str, api_key_env: str) -> tuple[str, str]:
    """Re-route the two Anthropic-shaped defaults when the model is OpenAI's.

    Only the untouched defaults move: an explicit --base-url or --api-key-env
    always wins, which is both how tests point a gpt model at a local fake and
    how a real run reaches an OpenAI-compatible open-source endpoint.
    """
    if provider_for(model) == "openai":
        if base_url == ANTHROPIC_API:
            base_url = OPENAI_API
        if api_key_env == "ANTHROPIC_API_KEY":
            api_key_env = "OPENAI_API_KEY"
    return base_url, api_key_env


def client_for_model(model: str, api_key: str, base_url: str,
                     max_tokens: int, timeout: float) -> Client:
    """The provider dispatch: the ONE place a pinned model becomes a client.

    probe.py and matrix.py construct clients through here too, so the mapping
    from model to wire format -- and to its pinned harness spec -- cannot drift
    between entry points.
    """
    spec = HARNESS_SPEC[model]
    if provider_for(model) == "openai":
        return OpenAIClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            timeout=timeout,
            effort=spec["effort"],
        )
    return AnthropicClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        timeout=timeout,
        thinking=spec["thinking"],
        effort=spec["effort"],
        cache_ttl=spec["cache_ttl"],
    )


def build_client(args, price: PriceSheet) -> Client:
    base_url, key_env = endpoint_defaults(price.model, args.base_url, args.api_key_env)
    key = os.environ.get(key_env, "")
    if not key:
        raise SystemExit(
            f"{key_env} is not set. Export it, or name another "
            "variable with --api-key-env. There is no keyless mode against a "
            "real endpoint; for a keyless dry run, point --base-url at a "
            "ProtocolUpstream the way tests/test_cli.py does."
        )
    return client_for_model(price.model, key, base_url, args.max_tokens, args.timeout)


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
    print("  gpt-5.6-sol is the promotional sheet, valid through 2026-11-21;")
    print("  gpt-5.6-sol@list restates it at list rates. Its cache-write rate is a")
    print("  true zero: OpenAI caching is automatic, unbilled on write, TTL unpinnable")
    print("  (cross-provider cache-hit rates are therefore not TTL-comparable).")
    print("\n  harness spec, pinned 2026-08-24 (published constants):")
    print("    max_tokens 16000 | thinking adaptive | effort high | cache TTL 1h")
    print("    fan-out unstaggered | concurrency cap 4 (tools.MAX_CONCURRENCY)")
    print("    claude-haiku-4-5 (plumbing only) predates adaptive thinking and")
    print("    effort; its spec sends neither.")
    print("    gpt-5.6-sol (OpenAI leg, added 2026-08-27): max_tokens 16000,")
    print("    reasoning_effort none — REQUIRED for tools on chat completions")
    print("    (verified live; reasoning-on needs /v1/responses, October);")
    print("    no thinking, no cache TTL (caching automatic).")
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
        # `replay` itself is a checker and never writes. But the CLI's contract
        # is "one command from a pinned model to a file", and the rescue case --
        # a killed run whose traces survived but whose extraction never ran --
        # ends here with the report printed and nothing for `experiment` to
        # consume. So the CLI writes the re-derived result, surfacing any
        # disagreement with a previously published file first rather than
        # clobbering the evidence that the extraction changed.
        published = out / "calibration.json"
        if published.exists():
            prior = CalibrationResult.load(published)
            same = json.dumps(prior.constants, sort_keys=True, default=str) == json.dumps(
                result.constants, sort_keys=True, default=str
            )
            if not same:
                print(
                    "  ! replay DISAGREES with the published calibration.json -- the "
                    "extraction changed since it was written. Overwriting with the "
                    "re-derived constants; the traces stay the source of truth."
                )
        result.write(out)
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

    # The calibration's measured noise floors set what "the same outcome" means
    # in the Tier-A test and the implied-beta tie-break. Without them scoring
    # runs at exact float equality, which is a placeholder, not a test.
    floors = None
    fd = (calib.floors or {}).get("dollars")
    fm = (calib.floors or {}).get("minutes")
    if fd or fm:
        floors = (fd, fm)
        parts = []
        if fd is not None:
            parts.append(f"${fd:.4g}")
        if fm is not None:
            parts.append(f"{fm:.4g} min")
        print(f"  materiality floors: {' + beta * '.join(parts)}")
    else:
        print("  ! calibration carries no materiality floors; outcome ties are "
              "judged at exact float equality", file=sys.stderr)

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
        floors=floors,
    )
    print()
    print(results.report())
    return 0


def _load_calibration(args) -> tuple:
    """(calibration, timing, cost_model, floors) with the experiment command's
    refusals: wrong model, missing timing model, missing floors for an audit."""
    price = price_sheet(args.model)
    calib = CalibrationResult.load(args.calibration)
    calibrated_model = (calib.price_sheet or {}).get("model", "")
    if calibrated_model != price.model:
        raise SystemExit(
            f"calibration {args.calibration} was measured on {calibrated_model!r}, "
            f"but --model resolves to {price.model!r}. Curves do not transfer."
        )
    timing_keys = {f.name for f in fields(TimingModel)}
    timing_dict = {k: v for k, v in (calib.timing_model or {}).items() if k in timing_keys}
    if not timing_dict:
        raise SystemExit("the calibration has no timing model; see its 'skipped' section.")
    floors = ((calib.floors or {}).get("dollars"), (calib.floors or {}).get("minutes"))
    return calib, TimingModel(**timing_dict), calib.to_cost_model(), floors


def cmd_audit(args) -> int:
    """The committee audit: execute every plan that could win on one scenario.

    Epsilon is read off the calibration's own estimation gate unless overridden;
    --dry-run prices the table without a key or a dollar, which is also how the
    audit's enumeration and collapse stay reproducible by anyone.
    """
    price = price_sheet(args.model)
    calib, timing, cm, floors = _load_calibration(args)
    if floors[0] is None:
        raise SystemExit(
            "the calibration carries no dollar floor; outcome-distinct collapse "
            "needs a resolution. Re-run calibration QA first."
        )
    eps = args.eps if args.eps is not None else eps_from_calibration(calib)
    scenario = ScenarioSpec(args.shape, args.n, args.size, args.seed).build()
    client = None if args.dry_run else build_client(args, price)
    print(f"  model {price.model}  scenario {scenario.id}  beta {args.beta:g}  "
          f"eps {'-' if eps is None else f'{eps:.0%}'} "
          f"({'override' if args.eps is not None else 'from estimation gate'})")
    result = run_audit(
        scenario, client, price, timing, cm, args.out,
        beta=args.beta, eps=eps, floors=floors,
        band_only=args.band_only, dry_run=args.dry_run,
        max_turns=args.max_turns, budget=_budget(args),
    )
    print()
    print(result.report())
    return 0


def cmd_audit_summary(args) -> int:
    """Combine per-scenario audits into the report's X% -- the pre-registered
    worst-case rule, refusing a certified X when any audit cannot contribute."""
    paths = sorted(Path(args.dir).glob("*/audit.json"))
    if not paths:
        raise SystemExit(f"no audit.json under {args.dir}/*/")
    s = audit_summary(paths)
    tau = s["mean_tau"]
    print(f"  audits: {s['n_audits']}   champion retention: {s['champion_retention']}"
          f"   mean tau: {'-' if tau is None else format(tau, '+.2f')}")
    for row in s["per_scenario"]:
        x = "-" if row["x"] is None else f"{row['x']:.0%}"
        print(f"    {row['scenario']:<16} champion predicted rank {row['r']}/{row['N']}"
              f"  x={x}  in_band={row['in_band']}  complete={row['complete']}")
    for c in s["caveats"]:
        print(f"  ! {c}")
    if s["safe_filter_X"] is not None:
        print(f"\n  SAFE FILTER X = {s['safe_filter_X']:.0%} (worst case across audits; "
              "the calculation may discard this bottom fraction of its plan table "
              "without ever discarding a measured champion)")
    else:
        print("\n  no certified X: at least one audit is band-only or incomplete")
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

    aud = sub.add_parser("audit", help="committee audit: execute every plan that could win")
    add_endpoint_args(aud)
    aud.add_argument("--calibration", required=True, help="path to calibration.json")
    aud.add_argument("--shape", required=True, choices=tuple(sorted(SHAPES)))
    aud.add_argument("--n", type=int, required=True)
    aud.add_argument("--size", type=int, default=3, help="node size, matching the manifest cell")
    aud.add_argument("--seed", type=int, required=True)
    aud.add_argument("--beta", type=float, default=1.0)
    aud.add_argument("--eps", type=float, default=None,
                     help="override the estimation-gate epsilon (default: read from calibration)")
    aud.add_argument("--band-only", action="store_true",
                     help="execute only the 2-eps band; no safe-filter rate")
    aud.add_argument("--dry-run", action="store_true",
                     help="price the table and stop; no key, no spend")
    aud.add_argument("--out", required=True)
    aud.set_defaults(func=cmd_audit)

    aus = sub.add_parser("audit-summary", help="combine audits into the pre-registered X%")
    aus.add_argument("dir", help="directory containing per-scenario audit dirs")
    aus.set_defaults(func=cmd_audit_summary)

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
