# r19 — Export Adapters, Determinism Benchmarks, and Case Studies

## Purpose

This roadmap delivers three things: (1) trace export adapters for external
analysis tools, (2) automated determinism benchmarks that prove AegisRT's
scheduling jitter is within bounds, and (3) the four MVP case studies that
demonstrate AegisRT's core value propositions.

## Dependencies

- r18: `AuditTrail`, `MetricsAggregator`, enhanced `TraceCollector`
- r17: `Scheduler` (source of scheduling decision events)
- r11: `MemoryOrchestrator` (for case study 3)

## Sub-phase Overview

| Sub-phase | Focus | Key Deliverable |
|-----------|-------|-----------------|
| r19-a | Export adapters | JSON, CSV, Perfetto exporters |
| r19-b | Determinism benchmarks | CV < 5% assertion in CI |
| r19-c | MVP case studies | 4 validated scenarios |

---

## Phase r19-a: Export Adapters

### JSON Schema

```json
{
  "aegisrt_trace": {
    "version": "1.0",
    "events": [
      {
        "type": "SchedulingDecision",
        "timestamp_ns": 1234567890,
        "model_id": "model_a",
        "request_id": 42,
        "decision_chain_id": 7,
        "payload": "selected by EDF, deadline=15ms, wcrt=8ms"
      }
    ]
  }
}
```

### Perfetto Format

Perfetto uses the Chrome Trace Event Format:

```json
{"ph": "X", "ts": 1234.567, "dur": 5.0, "pid": 1, "tid": 1,
 "name": "model_a::execute", "args": {"wcrt_us": 8000}}
```

This allows AegisRT traces to be visualised in `chrome://tracing` or
Perfetto UI without any additional tooling.

---

## Phase r19-b: Determinism Benchmarks

### Coefficient of Variation

```
CV = stddev(latencies) / mean(latencies)
```

A CV < 5% means the scheduling decision latency is highly consistent.
This is the quantitative definition of "deterministic" for AegisRT.

### Benchmark Setup

- 1000 invocations of `Scheduler::run_next()` with a 3-model EDF workload.
- MockBackend with fixed 1ms execution time (removes backend variance).
- Measure only the scheduling decision overhead (not execution time).
- Assert CV < 5% and p99 < 100us.

---

## Phase r19-c: MVP Case Studies

### Case Study 1: Admission Control Validation

Demonstrates that AegisRT never schedules a task set that violates
schedulability. Run 1000 scheduling cycles with 3 admitted models.
Assert zero deadline violations.

### Case Study 2: Determinism Validation

Demonstrates that scheduling decisions are consistent. Run 1000 invocations
with 2 concurrent models. Assert CV < 5%.

### Case Study 3: Memory Efficiency

Demonstrates >= 15% GPU memory reduction via orchestrated allocation.
Uses the same 3-model workload as r12 benchmarks.

### Case Study 4: Admission Rejection

Demonstrates that AegisRT explicitly rejects an inadmissible model with
a human-readable rationale. Attempt to admit a 4th model that would push
utilisation above the RMS bound. Assert rejection with rationale string
containing utilisation numbers.

## Exit Criteria

- All 3 export formats produce valid output.
- Determinism CV < 5% in CI.
- All 4 case studies pass.
- Results documented in `docs/CASE_STUDIES.md`.
- `docs/BENCHMARKS.md` written with all 5 MVP criteria results (scheduling overhead, determinism, admission correctness, memory efficiency, explainability).
