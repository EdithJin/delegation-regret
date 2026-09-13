# Delegation Regret

**When should an AI agent spawn subagents, and when should it just do the work itself?**

Delegation is Amdahl's law with a spawn tax: the serial fraction bounds the speedup, and every subagent buys parallelism at a fixed price. This repo measures that decision empirically — how many dollars fan-out adds, how many minutes it saves, and the break-even price of latency `beta* = (fan-out dollars − serial dollars) / (serial minutes − fan-out minutes)` at which the trade is worth it.

![Four dependency shapes over the same six subtasks](figures/shapes.svg)

A scenario is a set of equal-size subtasks plus a dependency structure, rendered as a Python package with failing tests to repair. The structure is the experimental variable: on `wide` every subtask can run at once, on `chain` nothing overlaps and fan-out can only lose. Same work, opposite correct answers — both derived, neither asserted.

## How it works

```
generator/   DAG scenarios + executable defect payload (balanced defects, import-edge deps)
    ↓
oracle       exhaustive plan enumeration, calibrated cost model, latency simulation,
             optimal-k intervals over the β axis
    ↓
harness/     the agent loop: real subagent spawns on a capped pool, both wire formats
             over real HTTP, every run witnessed by an independent logging proxy
    ↓
scoring/     delegation regret, implied β, execution-verified success gates
    ↓
audit        committee audit executes every plan that could win; findings scripts
             recompute every reported number from saved traces, exiting nonzero on drift
```

Success is execution-verified (restored test suites in a clean subprocess — no LLM judge). Dollars come from provider usage against dated price sheets; minutes come from a calibrated analytic clock over the observed call schedule. A trace whose usage disagrees with the proxy is disqualified.

## Findings

![Executed cost, latency, break-even boundary, and allocation-shape audit for Claude Opus](figures/fig1-preview.png)

Under a controlled direct-edit strategy, Opus fan-out is dominated at node size 3, while the measured break-even value of a saved minute falls from $1.85 at size 8 to $0.55 at size 15 and $0.47 at size 25. At the audited wide-15 cell, maximal fan-out scores 2.15 versus serial's 2.53 across all 12 direct-edit allocation shapes, but an observed programmatic-serial execution scores 1.32—lower than every controlled shape—so the boundary is strategy-conditional.

![Per-run spawning and spawn-packing behavior for Opus, GPT, and K3](figures/fig2-preview.png)

On large wide tasks, Opus spawns in 6/10 blind runs versus 2/11 runs given a stated $1-per-minute objective; these are directional contrasts from small, unbalanced cells, not population rates. Execution mode differs just as sharply: Opus packs 14/14 free-choice and 6/6 forced multi-spawn runs, GPT packs 21/21 free-choice but serializes 6/6 forced runs, and K3 packs 4/4 forced ladder arms but spawns in 0/30 free-choice traces.

## Quick start

Python 3.9+, standard library only (verified on 3.9 and 3.14).

```bash
git clone https://github.com/EdithJin/delegation-regret
cd delegation-regret
python -m generator.demo               # ~2s tour: shapes, oracle decisions, the β axis
python -m unittest discover -s tests   # 371 tests, offline, no API key
```

## Running the instrument

Offline, no credentials:

```bash
python -m harness.smoke preflight      # verify the instrument; zero API calls
python -m harness.cli models           # print the pinned price sheets
python -m harness.cli manifest --set core   # write the fingerprinted scenario manifest
```

Live runs read the key from `ANTHROPIC_API_KEY` (override with `--api-key-env`; the OSS leg uses `OPENAI_API_KEY`-style env selection):

```bash
python -m harness.cli calibrate ...    # measure provider cost/timing constants, write traces
python -m harness.cli experiment ...   # manifest in → stamped results file out
python probe.py ...                    # paired serial vs max-fanout boundary probe
python matrix.py ...                   # one behavioral cell per fresh process
python -m harness.cli audit ...        # execute every plan that could win
python -m harness.cli audit-summary <dir>
```

Each command's `--help` documents its arguments; the experiment driver refuses to write a results file missing its manifest fingerprint, calibration source, or price-sheet date.

## Reproducing the findings

The tracked curated sample provides a real execution, its independent proxy ledger, and the measured calibration used to price it:

```bash
python billing_audit.py          # reconcile usage and compute billed spend
python opus_findings.py --sample # recompute the sample's cost and analytic time
```

The full trace archive stays out of history. When restored under `results/`, every headline number regenerates from artifacts:

```bash
python opus_findings.py     # Opus tallies, boundary ladder, audit assertions
python gpt_findings.py      # GPT leg
python kimi_findings.py     # Kimi K3 leg
python billing_audit.py     # recheck every archived trace/proxy pair and ledger
```

Figures rebuild with matplotlib: `python figures/make_fig1.py` (PDF output directory via `FIG_PDF_DIR`, default `figures/`).

Pre-registered run record: [`docs/run-record-gpt-packed-ladder.md`](docs/run-record-gpt-packed-ladder.md) freezes the GPT packed-ladder repetitions and no-early-stopping rule before paid execution.

## Limitations

- Core scenarios are capped at eight subtasks because exhaustive plan pricing grows rapidly.
- Figure 1 uses one executed draw per arm at each size; its materiality bars are not sampling intervals.
- Tasks are synthetic Python packages with deliberately injected defects, so transfer to organic repositories is untested.
- The enumerator knows the dependency graph that the agent must discover, and its controlled boundary omits that discovery cost.
- The full trace archive is not in Git history; only the curated, identifier-checked sample is included under `results/sample/`.

## Layout

| | |
|---|---|
| [`generator/demo.py`](generator/demo.py) | Start here — the whole instrument in five printed sections |
| [`generator/oracle.py`](generator/oracle.py) | Plan space, feasibility, schedule simulation, implied β |
| [`generator/templates.py`](generator/templates.py) | The executable payload: modules, defect injection, generated suites |
| [`harness/runner.py`](harness/runner.py) | The agent loop, real `spawn_subagent`, forced-plan baseline |
| [`harness/proxy.py`](harness/proxy.py) | The independent billing witness every run routes through |
| [`scoring/regret.py`](scoring/regret.py) | Regret, what it refuses to score, implied β |
| [`harness/audit.py`](harness/audit.py) | The committee audit and its structural cross-check |
| [`results/sample/`](results/sample/) | Curated trace/proxy pair plus the measured calibration that prices it |
| [`tests/`](tests/) | What is actually guaranteed, including regression pins for every correction |

MIT licensed. *Research project, built independently. Questions and objections are welcome — open an issue.*
