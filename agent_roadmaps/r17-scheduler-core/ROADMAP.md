# r17 — Scheduler Core and Phase 3 Validation

## Purpose

`Scheduler` is the top-level orchestration loop for Layer 3. It ties together
`AdmissionController` (admission gate), `SchedulingPolicy` (ordering), and
`ExecutionContext` (dispatch). Every model execution in AegisRT flows through
the Scheduler.

## Dependencies

- r15: `AdmissionController` (with RTA)
- r16: `SchedulingPolicy`, `ScheduledTask`
- r07: `ExecutionContext`
- r04: `TraceCollector`
- r06: `ModelID`, `Result<T>`, `ErrorCode`

## Sub-phase Overview

| Sub-phase | Focus | Key Deliverable |
|-----------|-------|-----------------|
| r17-a | Skeleton + submission gate | `submit()`, model registry |
| r17-b | `run_next()` + deadline miss | Execution loop, latency measurement |
| r17-c | Integration tests + Phase 3 gate | 3-model workload, latency benchmark |

---

## Phase r17-a: Scheduler Skeleton and Submission Gate

### Config and Factory

```cpp
struct Config {
    size_t   max_pending_requests  = 100;
    Duration scheduling_interval   = Duration::from_micros(100);
    bool     enable_admission_control = true;
    bool     enable_tracing           = true;
};

static Result<std::unique_ptr<Scheduler>> create(
    std::unique_ptr<SchedulingPolicy> policy,
    std::unique_ptr<AdmissionController> admission,
    std::shared_ptr<DeviceMemoryPool> memory_pool,
    TraceCollector& tracer,
    const Config& config = {}
);
```

### Orchestration Thread

The Scheduler runs an internal `orchestration_loop_()` thread that:
1. Dequeues requests from `ConcurrentQueue<ExecutionRequest> pending_`.
2. Calls `policy_->select_next()`.
3. Checks deadline, dispatches to `ExecutionContext::execute()`.
4. Records trace events.

`start()` spawns the thread; `stop()` signals it to exit and joins.

### Constructor

```cpp
class Scheduler {
public:
    Scheduler(
        std::shared_ptr<AdmissionController> admission,
        std::unique_ptr<SchedulingPolicy> policy,
        std::shared_ptr<TraceCollector> trace);

    Result<AdmissionResult> submit(ModelID model_id, TaskParameters params);
    void remove_model(ModelID model_id);
    Result<ExecutionResult> run_next();

private:
    std::shared_ptr<AdmissionController> admission_;
    std::unique_ptr<SchedulingPolicy> policy_;
    std::shared_ptr<TraceCollector> trace_;
    std::unordered_map<ModelID, TaskParameters> registered_;
    mutable std::mutex mutex_;
};
```

### admit() Sequence

```
admit(context, request):
  1. Call admission_->analyze(request, existing_tasks)
  2. If rejected: return AdmissionResult with admitted=false
  3. If admitted: bind memory via MemoryOrchestrator, register context, add to policy
  4. Return ModelID
```

### evict() Sequence

```
evict(model_id):
  1. Remove from policy_->remove_model(model_id)
  2. Release memory binding via memory_orchestrator_->release(model_id)
  3. Remove from contexts_ map and admitted_tasks_
```

### submit() Sequence

```
submit(model_id, priority, deadline):
  1. Validate model_id is admitted
  2. Create ExecutionRequest with request_id, deadline, priority
  3. Enqueue to pending_ (ConcurrentQueue)
  4. Return RequestID
```

---

## Phase r17-b: run_next() Execution Loop

### run_next() Sequence

```
run_next():
  1. Call policy_->select_next() -> optional<ScheduledTask>
  2. If nullopt: return ErrorCode::QueueEmpty
  3. Check is_deadline_missed(task, now()) -> if true: emit DeadlineMiss, return error
  4. Record decision_start_time
  5. Dispatch to ExecutionContext::execute(request)
  6. Record decision_latency = execution_start - decision_start_time
  7. Emit SchedulingDecision trace event
  8. Return ExecutionResult with decision_latency
```

### Decision Latency Target

The time from `run_next()` entry to the moment execution begins on the GPU
must be < 100 microseconds. This is measured and asserted in CI.

```cpp
struct ExecutionResult {
    ModelID   model_id;
    Duration  execution_time;
    Duration  decision_latency;  // run_next() overhead
    bool      deadline_met;
};
```

---

## Phase r17-c: Integration Tests and Phase 3 Validation

### 3-Model EDF Test

```
Models: A (C=5ms, T=20ms, D=20ms), B (C=3ms, T=10ms, D=10ms), C (C=2ms, T=15ms, D=15ms)
U = 5/20 + 3/10 + 2/15 = 0.25 + 0.30 + 0.13 = 0.68 (admitted by EDF)

Expected EDF ordering (by deadline):
  t=0: B (D=10), C (D=15), A (D=20) -> select B
  t=3: C (D=15), A (D=20) -> select C
  t=5: A (D=20) -> select A
```

### Decision Latency Benchmark

```cpp
// Runs 10,000 iterations, asserts p99 < 100us
BENCHMARK(SchedulerDecisionLatency) {
    for (int i = 0; i < 10000; ++i) {
        auto start = Clock::now();
        scheduler.run_next();
        auto latency = Clock::now() - start;
        latencies.push_back(latency);
    }
    ASSERT_LT(percentile(latencies, 0.99), 100us);
}
```

### Phase 3 Exit Criteria

- [ ] Zero deadline violations in 3-model EDF test workload.
- [ ] Decision latency p99 < 100 microseconds.
- [ ] Admission correctly rejects 4th model that exceeds utilisation bound.
- [ ] All trace events emitted for every scheduling decision.
- [ ] Results documented in `docs/PHASE3_VALIDATION.md`.
- [ ] CI benchmark job is green.
