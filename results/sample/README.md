# Curated execution sample

This directory contains one real Claude Opus execution from the wide-4,
size-15 boundary probe:

- `opus-wide4-s15-fanout-trace.json` is the harness trace;
- `opus-wide4-s15-fanout-proxy.jsonl` is the independent usage ledger; and
- `calibration.json` contains the measured timing model and dated price sheet
  used to price the trace.

The trace and ledger are exact copies of the corresponding private-archive
artifacts. The calibration's source-trace list is retained for provenance, but
those full calibration traces are intentionally not published.

Before inclusion, these files were checked for API keys, authorization headers,
email addresses, local absolute paths, and account, organization, request, or
user identifiers; none were found. The artifacts contain only synthetic-task
content plus model, usage, timing, and tool metadata.

From the repository root, verify and summarize the sample with:

```bash
python billing_audit.py
python opus_findings.py --sample
```
