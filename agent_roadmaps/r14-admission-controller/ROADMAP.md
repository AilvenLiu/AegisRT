# r14 — AdmissionController

## Purpose

`AdmissionController` is the formal schedulability analysis engine at the heart
of AegisRT's research contribution. It answers the question: "given the currently
admitted set of models, can we safely admit one more?" It uses real-time scheduling
theory (RMS and EDF) adapted for non-preemptive GPU execution.

## Dependencies

- r13: `WCETProfile` (WCET estimates for each model)
- r06: `ModelID`, `Result<T>`, `ErrorCode`
- r04: `TraceCollector`

## Sub-phase Overview

| Sub-phase | Focus | Key Deliverable |
|-----------|-------|-----------------|
| r14-a | Data structures | `TaskParameters`, `AdmissionRequest`, `AdmissionResult` |
| r14-b | RMS + EDF tests | `test_rms()`, `test_edf()`, utilisation computation |
| r14-c | `analyze()` + tracing | Full pipeline, rationale, thread safety |

---

## Phase r14-a: Task Parameters and Admission Data Structures

### TaskParameters

```cpp
struct TaskParameters {
    ModelID  model_id;
    Duration wcet;      // from WCETProfile::worst_case
    Duration period;    // invocation period
    Duration deadline;  // relative deadline (often == period)
    uint32_t priority;  // lower number = higher priority (RMS: shorter period = higher priority)
};
```

### AdmissionResult

```cpp
struct AdmissionResult {
    bool     admitted;
    std::string reason;          // human-readable rationale
    double   utilisation;        // total U after admission
    Duration wcrt;               // worst-case response time (filled by r15)
    Duration max_blocking_time;  // non-preemptive blocking (filled by r15)
    std::string analysis_detail; // full analysis dump

    static AdmissionResult make_admitted(double u, std::string reason);
    static AdmissionResult make_rejected(double u, std::string reason);
};
```

---

## Phase r14-b: RMS and EDF Schedulability Tests

### RMS Liu-Layland Bound

For n periodic tasks with RMS priority assignment:

```
U_bound(n) = n * (2^(1/n) - 1)

n=1: 1.000
n=2: 0.828
n=3: 0.780
n=4: 0.757
n=5: 0.743
...
n->inf: ln(2) = 0.693
```

If `U <= U_bound(n)`, the task set is schedulable under RMS.

### EDF Test

EDF is optimal for preemptive scheduling. For non-preemptive GPU scheduling,
EDF provides a necessary (but not sufficient) condition:

```
U = sum(C_i / T_i) <= 1.0
```

### Precomputed Bound Table

```cpp
// Precomputed for n=1..64 to avoid floating-point pow() at runtime
constexpr double kRmsBound[] = {
    1.000, 0.828, 0.780, 0.757, 0.743, 0.735, 0.729, 0.724, ...
};
```

---

## Phase r14-c: analyze() Integration

### Analysis Pipeline

```
analyze(request, existing):
  1. Compute U_new = sum(C_i/T_i for existing) + C_new/T_new
  2. Run test_rms(all_tasks) -> rms_ok
  3. Run test_edf(all_tasks) -> edf_ok
  4. admitted = rms_ok && edf_ok
  5. Generate rationale string
  6. Emit AdmissionDecision trace event
  7. Return AdmissionResult
```

### Rationale Example

```
"Admitted: U=0.742 <= U_bound(3)=0.780 [RMS], U=0.742 <= 1.0 [EDF]. "
"Adding model_d (C=5ms, T=20ms, U_contrib=0.250)."
```

## Exit Criteria

- RMS test matches Liu-Layland 1973 paper examples exactly.
- EDF test correctly admits U=0.99 and rejects U=1.01.
- `analyze()` is thread-safe under concurrent calls.
- Trace events emitted for every admission decision.
- API header frozen.
