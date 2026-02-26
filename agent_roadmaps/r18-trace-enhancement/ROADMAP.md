# r18 — Trace Enhancement and Observability

## Purpose

AegisRT's observability contract requires that every scheduling decision and
resource allocation is traceable with rationale. This roadmap upgrades the
basic `TraceCollector` from r04 into a full observability system with
correlated event chains, scheduling-domain queries, and computed metrics.

## Dependencies

- r17: `Scheduler` (source of scheduling decision events)
- r04: `TraceCollector`, `TraceEvent`
- r14: `AdmissionResult`

## Sub-phase Overview

| Sub-phase | Focus | Key Deliverable |
|-----------|-------|-----------------|
| r18-a | Event correlation | `request_id`, `decision_chain_id`, extended query |
| r18-b | AuditTrail | Scheduling-domain query API |
| r18-c | MetricsAggregator | Per-model utilisation, latency, miss rate |

---

## Phase r18-a: TraceEvent Enhancement and Extended Queries

### Enhanced TraceEvent

```cpp
struct TraceEvent {
    // Existing fields (r04) -- string-based per ARCHITECTURE.md
    std::string event_id;         // unique identifier
    std::string event_type;
    std::string component;
    Timestamp   timestamp;
    Duration    duration;
    std::string model_id;
    std::string request_id;       // unique per execution request
    std::string rationale;
    std::map<std::string, std::string> attributes;

    // New fields (r18)
    std::string decision_chain_id; // links related events (admit -> schedule -> execute)
};
```

### Decision Chain Correlation

A single model execution produces a chain of events:
```
AdmissionDecision (chain_id=42) -> SchedulingDecision (chain_id=42) -> ExecutionStarted (chain_id=42) -> ExecutionCompleted (chain_id=42)
```

`reconstruct_decision_chain(42)` returns all four events in timestamp order.

---

## Phase r18-b: AuditTrail

### AuditTrail API

```cpp
class AuditTrail {
public:
    explicit AuditTrail(std::shared_ptr<TraceCollector> trace);

    std::optional<AdmissionResult> get_admission_decision(ModelID model_id) const;
    std::vector<SchedulingDecision> get_scheduling_history(
        ModelID model_id, TimeWindow window) const;
    std::vector<DeadlineMissEvent> get_deadline_misses(TimeWindow window) const;
    std::vector<AdmissionRejection> get_rejection_history(TimeWindow window) const;
};
```

`AuditTrail` is a read-only view over `TraceCollector`. It does not store
any state of its own -- all queries are computed from the trace at call time.

---

## Phase r18-c: MetricsAggregator

### LatencyStats

```cpp
struct LatencyStats {
    Duration mean;
    Duration stddev;
    Duration percentile_99;
    Duration percentile_999;
    size_t   sample_count;
};
```

### Utilisation Computation

```
utilisation(model, window) = sum(execution_time_i) / window_duration
```

This is the observed utilisation, distinct from the theoretical utilisation
`C/T` used in admission control.

### Deadline Miss Rate

```
miss_rate(model, window) = deadline_miss_count / total_execution_count
```

A miss rate > 0 in a validated workload indicates a scheduling policy problem.

## Exit Criteria

- `reconstruct_decision_chain()` returns events in correct timestamp order.
- `AuditTrail` queries return correct results on pre-populated trace.
- `MetricsAggregator` utilisation matches manual calculation.
- API headers frozen.
