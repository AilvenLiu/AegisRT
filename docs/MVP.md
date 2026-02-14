# AegisRT MVP Definition

## Core Philosophy

The Minimum Viable Product of AegisRT is **not** a feature-complete system. It is a **conceptually complete** demonstration that validates one fundamental hypothesis:

> **Real-time scheduling theory can be adapted for GPU execution to provide provable latency bounds in multi-model inference scenarios.**

The MVP exists to answer a single question definitively: *Does this approach provide measurable, demonstrable value over existing solutions?*

If the MVP cannot demonstrate this value, the hypothesis is wrong, and the project direction should be reconsidered. This is the purpose of an MVP—validated learning, not feature delivery.

---

## MVP Success Criteria

### Primary Validation Metrics

The MVP succeeds when it can **demonstrate** the following capabilities:

| Criterion | Measurement Method | Success Threshold | Why This Matters |
|-----------|-------------------|-------------------|------------------|
| **Scheduling Overhead** | Microbenchmark | < 100 μs per decision | Proves system is practical for real-time use |
| **Determinism** | Latency distribution analysis | CV < 5% under controlled conditions | Validates formal approach works in practice |
| **Admission Correctness** | Deadline violation count | Zero violations for admitted models | Proves schedulability analysis is correct |
| **Memory Efficiency** | Memory usage comparison | > 15% reduction vs isolated allocation | Demonstrates cross-model orchestration value |
| **Explainability** | Trace reconstruction | 100% of decisions reconstructible | Validates observability as a contract |

### Validation Framework

For each criterion, the MVP must include:

1. **Test harness**: Automated test that exercises the capability
2. **Measurement script**: Reproducible measurement procedure
3. **Baseline comparison**: Comparison against "no scheduler" baseline
4. **Documentation**: How to run and interpret results

---

## MVP Scope Definition

### What the MVP MUST Demonstrate

These capabilities are **non-negotiable** for MVP completion:

1. **End-to-End Scheduling Flow**
   - Model submission → admission → execution → completion
   - All steps traced and observable

2. **Formal Admission Control**
   - Schedulability analysis with documented rationale
   - Rejection of infeasible configurations

3. **WCET Estimation**
   - Conservative bounds with statistical confidence
   - Contention-aware profiling capability

4. **Deterministic Execution**
   - Bounded latency under controlled multi-model conditions
   - No deadline violations for admitted models

5. **Full Observability**
   - Every decision traceable and explainable
   - Offline analysis capability

### What the MVP MAY Include (If Time Permits)

These are **nice-to-have** capabilities that enhance the demonstration:

- Multiple scheduling policies (FIFO + EDF)
- Real TensorRT backend (vs mock backend)
- Basic performance dashboard
- Example application with realistic models

### What the MVP MUST NOT Include

These are **explicitly out of scope** and should not be attempted:

| Out of Scope | Reason |
|--------------|--------|
| Multi-GPU support | Complicates scheduling model |
| Dynamic graph support | Destroys WCET predictability |
| Kernel optimization | Not the problem AegisRT solves |
| Distributed deployment | Single-node scope constraint |
| Production-grade error recovery | Research prototype, not production system |
| Configuration management | YAGNI for MVP |
| Deployment automation | YAGNI for MVP |

---

## MVP Architecture

### Minimal Component Set

```
+------------------------------------------------------------------+
|                       MVP Architecture                           |
+------------------------------------------------------------------+
|                                                                  |
|  +------------------------------------------------------------+  |
|  |                    Scheduler (Core)                        |  |
|  |                                                            |  |
|  |   +-------------+  +-------------+  +------------------+   |  |
|  |   | WCETProfiler|  | Admission   |  | EDFPolicy        |   |  |
|  |   |             |  | Controller  |  |                  |   |  |
|  |   | Statistical |  |             |  | Non-Preemptive   |   |  |
|  |   | Profiling   |  | Formal      |  | EDF              |   |  |
|  |   |             |  | Analysis    |  |                  |   |  |
|  |   +-------------+  +-------------+  +------------------+   |  |
|  |                                                            |  |
|  +---------------------------+--------------------------------+  |
|                              |                                   |
|  +---------------------------v--------------------------------+  |
|  |                 Resource Orchestration                     |  |
|  |                                                            |  |
|  |   +-------------+  +-------------+  +------------------+   |  |
|  |   | MemoryPool  |  | Execution   |  | MockBackend      |   |  |
|  |   |             |  | Context     |  |                  |   |  |
|  |   | Simple      |  |             |  | For development  |   |  |
|  |   | Pool        |  | Per-Model   |  | and testing      |   |  |
|  |   |             |  |             |  |                  |   |  |
|  |   +-------------+  +-------------+  +------------------+   |  |
|  |                                                            |  |
|  +---------------------------+--------------------------------+  |
|                              |                                   |
|  +---------------------------v--------------------------------+  |
|  |                    CUDA Abstraction                        |  |
|  |                                                            |  |
|  |   +-------------+  +-------------+  +------------------+   |  |
|  |   | CudaStream  |  | CudaEvent   |  | TraceCollector   |   |  |
|  |   |             |  |             |  |                  |   |  |
|  |   | RAII        |  | RAII        |  | JSON export      |   |  |
|  |   | Wrapper     |  | Wrapper     |  | Perfetto export  |   |  |
|  |   |             |  |             |  |                  |   |  |
|  |   +-------------+  +-------------+  +------------------+   |  |
|  |                                                            |  |
|  +------------------------------------------------------------+  |
|                                                                  |
+------------------------------------------------------------------+
```

### Component Justification

| Component | Why It's Necessary for MVP |
|-----------|---------------------------|
| CudaStream/Event RAII | Foundation for all GPU operations |
| MemoryPool | Required for memory orchestration demonstration |
| TraceCollector | Required for observability contract |
| ExecutionContext | Required for isolation demonstration |
| MockBackend | Enables development without real models |
| WCETProfiler | Required for admission control |
| AdmissionController | Core differentiator, must be demonstrated |
| EDFPolicy | Simplest policy with theoretical foundation |

---

## MVP Implementation Phases

### Phase A: CUDA Abstraction Layer

**Duration**: 2-3 weeks (part-time independent development)

**Goal**: Establish RAII foundation with tracing capability.

**Deliverables**:

| Deliverable | Description | Exit Criteria |
|-------------|-------------|---------------|
| `CudaStream` | RAII wrapper for cudaStream_t | No leaks (cuda-memcheck) |
| `CudaEvent` | RAII wrapper for cudaEvent_t | Timing works correctly |
| `DeviceMemoryPool` | Simple pool allocator | Allocate/deallocate works |
| `TraceCollector` | Basic event collection | JSON export works |
| Unit tests | ≥70% coverage | All tests pass |

**Implementation Notes**:

```cpp
// Key RAII pattern to implement
class CudaStream {
public:
    CudaStream() {
        cudaStreamCreate(&stream_);
        tracer_->record({"stream_create", "cuda", {}});
    }
    ~CudaStream() noexcept {
        if (stream_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
            tracer_->record({"stream_destroy", "cuda", {}});
        }
    }
    // Move-only
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }
    // ...
private:
    cudaStream_t stream_;
    TraceCollector* tracer_;
};
```

**Learning Outcomes**:
- Modern C++ RAII patterns
- CUDA stream/event semantics
- Cross-compilation (x86_64 → Jetson)
- Test-driven development

### Phase B: Execution Context

**Duration**: 2-3 weeks

**Goal**: Per-model resource isolation with fault boundaries.

**Deliverables**:

| Deliverable | Description | Exit Criteria |
|-------------|-------------|---------------|
| `ResourceBudget` | Resource constraint definition | Validation works |
| `ExecutionContext` | Per-model container | Budget enforcement works |
| `FaultBoundary` | Error isolation | Errors don't propagate |
| `MockBackend` | Test double for inference | Simulates execution |
| Integration tests | Multi-model isolation | Two models execute independently |

**Implementation Notes**:

```cpp
// Key isolation pattern
class ExecutionContext {
public:
    Result<void> execute(/* ... */) {
        if (memory_used_ + input_size > budget_.memory_limit) {
            return Result<void>::error(Error{
                .code = ErrorCode::BudgetExceeded,
                .message = "Memory budget exceeded",
                .context = "ExecutionContext::execute"
            });
        }
        // Execution within budget
        return backend_->execute(/* ... */);
    }
private:
    ResourceBudget budget_;
    std::atomic<size_t> memory_used_{0};
    // ...
};
```

**Learning Outcomes**:
- Resource isolation patterns
- Atomic operations and thread safety
- Interface design (RuntimeBackend)
- Error propagation strategies

### Phase C: WCET Profiling

**Duration**: 2-3 weeks

**Goal**: Conservative worst-case execution time estimation.

**Deliverables**:

| Deliverable | Description | Exit Criteria |
|-------------|-------------|---------------|
| `WCETProfiler` | Statistical profiler | Produces valid profiles |
| Contention profiling | Under-load profiling | Contention factor computed |
| Safety margins | Conservative estimation | 99.9% samples ≤ predicted |
| Profile persistence | Save/load profiles | Round-trip works |

**Implementation Notes**:

```cpp
// Statistical WCET estimation
Duration WCETProfiler::compute_wcet(
    const std::vector<Duration>& samples,
    double confidence_level,
    double safety_margin
) {
    // Sort for percentile computation
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());

    // Compute mean and standard deviation
    Duration mean = compute_mean(sorted);
    Duration stddev = compute_stddev(sorted, mean);

    // Compute confidence interval
    int n = sorted.size();
    double z = z_score_for_confidence(confidence_level);
    Duration ci = Duration::from_nanos(
        z * stddev.nanos() / std::sqrt(n)
    );

    // Apply safety margin
    return Duration::from_nanos(
        (mean.nanos() + ci.nanos()) * safety_margin
    );
}
```

**Learning Outcomes**:
- Statistical methods for performance analysis
- GPU timing and profiling
- Safety-critical systems thinking
- Confidence interval computation

### Phase D: Admission Control

**Duration**: 2-3 weeks

**Goal**: Formal schedulability analysis for non-preemptive GPU execution.

**Deliverables**:

| Deliverable | Description | Exit Criteria |
|-------------|-------------|---------------|
| `AdmissionController` | Schedulability analysis | Correct admit/reject decisions |
| RMS test | Rate-monotonic test | Implementation verified |
| EDF test | Earliest-deadline-first test | Implementation verified |
| Blocking analysis | Non-preemptive blocking | Blocking time computed |
| Decision logging | Rationale capture | All decisions explained |

**Implementation Notes**:

```cpp
// Non-preemptive blocking time computation
Duration AdmissionController::compute_blocking_time(
    const std::vector<TaskParameters>& tasks,
    const TaskParameters& new_task
) {
    // In non-preemptive scheduling, a task can be blocked
    // by any lower-priority task that is currently executing
    Duration max_blocking{0};
    for (const auto& task : tasks) {
        if (task.priority > new_task.priority) {
            // Lower priority task - can cause blocking
            max_blocking = std::max(max_blocking, task.wcet);
        }
    }
    return max_blocking;
}

// Response-time analysis for non-preemptive case
Duration AdmissionController::compute_response_time(
    const TaskParameters& task,
    const std::vector<TaskParameters>& all_tasks
) {
    Duration blocking = compute_blocking_time(all_tasks, task);
    Duration response = task.wcet + blocking;

    // Iterative response-time analysis
    bool converged = false;
    while (!converged) {
        Duration interference{0};
        for (const auto& other : all_tasks) {
            if (other.priority < task.priority) {
                // Higher priority - causes interference
                interference += Duration::from_nanos(
                    std::ceil(response.nanos() / other.period.nanos()) *
                    other.wcet.nanos()
                );
            }
        }

        Duration new_response = task.wcet + blocking + interference;
        if (new_response == response) {
            converged = true;
        } else if (new_response > task.deadline) {
            // Deadline miss - not schedulable
            return new_response;
        }
        response = new_response;
    }

    return response;
}
```

**Learning Outcomes**:
- Real-time scheduling theory
- Formal verification methods
- Iterative algorithms
- Documentation of decision rationale

### Phase E: EDF Scheduler

**Duration**: 2-3 weeks

**Goal**: Non-preemptive Earliest Deadline First scheduling.

**Deliverables**:

| Deliverable | Description | Exit Criteria |
|-------------|-------------|---------------|
| `EDFPolicy` | EDF implementation | Selects earliest deadline |
| `Scheduler` | Orchestration loop | Complete flow works |
| Deadline detection | Miss detection | Violations detected |
| Full trace output | Decision rationale | All decisions traceable |

**Implementation Notes**:

```cpp
// EDF policy implementation
class EDFPolicy : public SchedulingPolicy {
public:
    std::optional<ExecutionRequest> select_next(
        const std::vector<ExecutionRequest>& pending
    ) override {
        if (pending.empty()) return std::nullopt;

        // Find request with earliest absolute deadline
        auto earliest = std::min_element(
            pending.begin(), pending.end(),
            [](const ExecutionRequest& a, const ExecutionRequest& b) {
                return a.deadline < b.deadline;
            }
        );

        return *earliest;
    }
    // ...
};
```

**Learning Outcomes**:
- Scheduling policy implementation
- Orchestration patterns
- Real-time constraints handling
- Deadline semantics

### Phase F: Validation & Demonstration

**Duration**: 2-3 weeks

**Goal**: Demonstrate MVP value proposition.

**Deliverables**:

| Deliverable | Description | Exit Criteria |
|-------------|-------------|---------------|
| Performance benchmarks | Automated benchmarks | All criteria met |
| Baseline comparison | No-scheduler baseline | Measurable improvement |
| Demo application | Multi-model demo | 3+ models, bounded latency |
| Documentation | README, API docs | Understandable in < 2 hours |

**Implementation Notes**:

```cpp
// Benchmark harness
class MVPBenchmark {
public:
    void run_all() {
        benchmark_scheduling_overhead();
        benchmark_determinism();
        benchmark_admission_correctness();
        benchmark_memory_efficiency();
        benchmark_explainability();
    }

    void benchmark_determinism() {
        // Run 1000 invocations, measure latency distribution
        std::vector<Duration> latencies;
        for (int i = 0; i < 1000; ++i) {
            auto start = Clock::now();
            scheduler_->submit(model_a, Priority::normal());
            // Wait for completion...
            auto end = Clock::now();
            latencies.push_back(end - start);
        }

        // Compute coefficient of variation
        Duration mean = compute_mean(latencies);
        Duration stddev = compute_stddev(latencies, mean);
        double cv = stddev.nanos() / mean.nanos();

        std::cout << "Latency CV: " << cv << "\n";
        assert(cv < 0.05 && "Determinism criterion failed");
    }
};
```

**Learning Outcomes**:
- Benchmark design
- Statistical analysis
- Technical writing
- Demonstration skills

---

## MVP Test Scenarios

### Scenario 1: Admission Control Validation

**Purpose**: Verify admission control makes correct decisions.

**Setup**: Three models with known WCET profiles and deadlines.

```
Model A: WCET=20ms, Period=100ms, Deadline=50ms
Model B: WCET=15ms, Period=80ms, Deadline=40ms
Model C: WCET=30ms, Period=200ms, Deadline=100ms
```

**Procedure**:
1. Admit Model A → Should succeed
2. Admit Model B → Should succeed
3. Attempt to admit Model C → May succeed or fail based on analysis
4. Run all admitted models for 1000 invocations each
5. Verify: Zero deadline violations

**Pass Condition**: Admission decisions are correct, all admitted models meet deadlines.

### Scenario 2: Determinism Validation

**Purpose**: Verify latency distribution is deterministic.

**Setup**: Two models running concurrently.

**Procedure**:
1. Profile WCET for both models
2. Admit with appropriate deadlines
3. Execute 1000 invocations of each model
4. Collect latency distribution
5. Compute coefficient of variation

**Pass Condition**: Latency CV < 5% for both models.

### Scenario 3: Memory Efficiency Validation

**Purpose**: Verify cross-model memory sharing.

**Setup**: Three models with overlapping tensor lifetimes.

**Procedure**:
1. Analyze tensor lifetimes for each model
2. Compute memory plan with sharing
3. Compute memory plan without sharing (isolated)
4. Compare peak memory usage

**Pass Condition**: > 15% memory reduction with sharing.

### Scenario 4: Explainability Validation

**Purpose**: Verify all decisions are reconstructible.

**Setup**: Any execution scenario.

**Procedure**:
1. Execute multi-model workload
2. Export trace to JSON
3. Reconstruct scheduling decisions from trace
4. Verify rationale is present for each decision

**Pass Condition**: 100% of decisions reconstructible with rationale.

---

## MVP Anti-Goals

These are explicitly out of scope. **Do not implement these during MVP development.**

### Anti-Goal 1: Production Deployment

The MVP is a research prototype. Do not:
- Add crash recovery
- Implement graceful shutdown
- Create configuration management
- Build deployment automation

**Rationale**: These are engineering concerns for later phases. MVP validates the approach.

### Anti-Goal 2: Performance Optimization

The MVP validates correctness, not performance. Do not:
- Hand-tune kernels
- Implement complex memory pools
- Add speculative execution
- Optimize for specific hardware

**Rationale**: Premature optimization can mask correctness issues. Get it right first.

### Anti-Goal 3: Feature Expansion

MVP scope is fixed. Do not add:
- Additional scheduling policies (beyond EDF)
- Multi-GPU support
- Dynamic graph support
- Advanced memory management

**Rationale**: Feature creep delays validation. Scope expansion requires explicit decision.

---

## MVP as a Learning Milestone

Completing the MVP establishes capability in three domains:

### Systems Engineering

| Skill | How MVP Develops It |
|-------|---------------------|
| RAII Design | All CUDA resources through RAII wrappers |
| Concurrency | Thread-safe queues, atomic operations |
| Error Handling | Typed errors, propagation, isolation |
| Testing | Exit criteria as automated tests |

### Real-Time Systems Theory

| Skill | How MVP Develops It |
|-------|---------------------|
| Schedulability Analysis | Implementing RMS, EDF tests |
| WCET Estimation | Statistical profiling methods |
| Non-Preemptive Scheduling | Blocking time analysis |
| Formal Methods | Provably correct decisions |

### GPU Systems

| Skill | How MVP Develops It |
|-------|---------------------|
| CUDA Programming | Streams, events, memory |
| GPU Profiling | Timing, contention analysis |
| Edge Constraints | Memory pressure, isolation |
| Research Landscape | Literature review, positioning |

---

## MVP Deliverables Checklist

### Code

- [ ] `src/cuda/` — CUDA abstraction layer
- [ ] `src/context/` — Execution context management
- [ ] `src/memory/` — Memory orchestration
- [ ] `src/scheduler/` — Scheduler and policies
- [ ] `src/trace/` — Trace collection
- [ ] `tests/` — Unit and integration tests
- [ ] `benchmarks/` — Validation benchmarks

### Documentation

- [ ] `README.md` — Project overview
- [ ] `docs/ARCHITECTURE.md` — System architecture
- [ ] `docs/API.md` — API reference
- [ ] `docs/BENCHMARKS.md` — Performance results
- [ ] `docs/TUTORIAL.md` — Getting started

### Validation

- [ ] All unit tests passing (≥70% coverage)
- [ ] All integration tests passing
- [ ] All MVP criteria met
- [ ] Performance benchmarks documented
- [ ] Demo application working

---

## From MVP to Next Steps

The MVP is a **foundation**, not a destination. Upon completion:

### Immediate Next Steps (Week 1-2 Post-MVP)

1. **Validate with Real Models**: Test with actual TensorRT/ONNX models
2. **Document Findings**: Write technical blog post about MVP results
3. **Community Feedback**: Share in relevant forums (Reddit, Discord, etc.)

### Short-Term (Month 1-3 Post-MVP)

1. **Expand Policy Support**: Add RMS, priority-based scheduling
2. **Real Backend Integration**: Implement TensorRT backend
3. **Hardware Optimization**: Profile on Jetson Orin

### Medium-Term (Month 3-6 Post-MVP)

1. **Research Contribution**: Write paper or technical report
2. **Community Building**: Respond to feedback, build user base
3. **Feature Development**: Based on validated user needs

---

## Risk Mitigation

### Risk: WCET Estimation Too Conservative

**Symptom**: System rejects configurations that would work in practice.

**Mitigation**: Start conservative, refine through profiling. Document the trade-off explicitly.

### Risk: Scheduling Overhead Too High

**Symptom**: Scheduling takes longer than 100μs.

**Mitigation**: Profile, identify bottlenecks. Consider lock-free data structures.

### Risk: Determinism Not Achievable

**Symptom**: Latency CV > 5% despite formal analysis.

**Mitigation**: Investigate contention effects, thermal throttling. May require hardware-specific tuning.

### Risk: Development Takes Longer Than Expected

**Mitigation**: Each phase has minimum viable deliverables. If time-constrained, reduce to MVP, not quality.

---

**The MVP proves that real-time scheduling theory can solve the GPU orchestration problem. Everything after is engineering.**
