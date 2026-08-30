# GPT packed-mode ladder — confirmatory run record

**Frozen before paid execution: 2026-08-29 18:48 PDT.**

## Question

What is the measured serial-versus-max-fanout delegation boundary for
`gpt-5.6-sol` at reasoning high when the four forced spawns are explicitly
required in one assistant turn and therefore can execute concurrently?

## Fixed run set

- endpoint: `/v1/responses`;
- model and effort: `gpt-5.6-sol`, reasoning high;
- scenario family: wide-4, seed 11;
- node sizes: 3, 8, 15, and 25;
- repetitions: exactly two fresh-process paired probes per size;
- arms per probe: serial first, then max-fanout with `--pack-spawns`;
- total: 8 paired probes / 16 paid arms;
- no result-dependent repetitions or early stopping.

The exact output directories are:

```text
results/probe-gpt-hi-wide4-s3-packed-confirm-r1
results/probe-gpt-hi-wide4-s3-packed-confirm-r2
results/probe-gpt-hi-wide4-s8-packed-confirm-r1
results/probe-gpt-hi-wide4-s8-packed-confirm-r2
results/probe-gpt-hi-wide4-s15-packed-confirm-r1
results/probe-gpt-hi-wide4-s15-packed-confirm-r2
results/probe-gpt-hi-wide4-s25-packed-confirm-r1
results/probe-gpt-hi-wide4-s25-packed-confirm-r2
```

## Admission and read-off

Every arm is retained in the run record. Economic scoring requires restored-
suite success, plan compliance, and proxy reconciliation. Packing obedience is
measured independently from `SpawnRecord.batch`: all four spawns must share one
batch. `probe.py` computes paired measured

```text
beta* = (fanout dollars - serial dollars)
        / (serial analytic minutes - fanout analytic minutes)
```

when fan-out saves minutes. A serialized or slower fan-out has no packed-mode
break-even value; it is reported as an instruction-obedience or dominance
outcome, never discarded. Dollar and minute differences are interpreted
against the reasoning-high calibration floors.

## Outcome-independent report mapping

- finite, floor-clearing values: report the GPT packed-mode boundary by size;
- packed but dominated: concurrency does not cover overhead at that size;
- packing disobedience: report the instruction-reliability limit;
- mixed repeats: report instability, not a pooled success rate;
- failed or unreconciled arm: exclude from economic scoring with the reason.

The earlier size-15/25 fanout-only packed probes remain labeled pilots and are
not pooled into this paired confirmatory denominator.
