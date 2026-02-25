# AegisRT Development Roadmap

## Roadmap Philosophy

This roadmap transforms architectural vision into a **sequential development path** with clear exit criteria and embedded learning outcomes. The guiding principle is:

> **Each phase establishes invariants required by subsequent phases. Correctness precedes performance. Validation precedes optimization.**

### Core Principles

#### Principle 1: Sequential Dependency

Phases execute sequentially. No phase begins until the predecessor's exit criteria are satisfied. Dependency violations are architectural failures that must be corrected before proceeding.

#### Principle 2: Measurable Exit Criteria

Each phase defines specific, verifiable exit criteria. Advancement requires explicit verification, not subjective assessment. If you cannot measure it, you cannot verify it.

#### Principle 3: Learning Embedded in Development

Each phase includes explicit learning outcomes. Development is not just code delivery—it is systematic skill acquisition. The goal is not just to build AegisRT, but to become an expert in GPU systems and real-time scheduling.

#### Principle 4: Minimal Viable Deliverables

Phases deliver the minimum functionality required to satisfy exit criteria. No speculative features, no premature optimization. Scope expansion requires explicit decision and documentation.

---

## Phase Dependency Graph

```
Phase 0: CUDA Foundation
    |
    |  Establishes: RAII wrappers, memory pool, tracing
    |  Duration: 2-3 weeks
    |
    v
Phase 1: Execution Context
    |
    |  Establishes: Per-model isolation, resource budgets, fault boundaries
    |  Duration: 2-3 weeks
    |
    v
Phase 2: Memory Orchestration
    |
    |  Establishes: Cross-model sharing, lifetime analysis, pressure handling
    |  Duration: 3-4 weeks
    |
    v
Phase 3: Deterministic Scheduler (CORE DIFFERENTIATOR)
    |
    |  Establishes: WCET profiling, admission control, RT scheduling policies
    |  Duration: 4-6 weeks
    |
    v
Phase 4: Observability & Validation
    |
    |  Establishes: Full tracing, determinism validation, case studies
    |  Duration: 2-3 weeks
    |
    v
Phase 5: Integration & Polish
    |
    |  Establishes: Real backend integration, documentation, community release
    |  Duration: 3-4 weeks
    |
    v
MVP Complete → Community Feedback → Iterate
```

---

## Phase 0: CUDA Foundation

### Duration Estimate

**2-3 weeks** (part-time independent development, ~10-15 hours/week)

### Objective

Establish minimal CUDA abstraction substrate with RAII wrappers, memory pool, and build infrastructure.

### Deliverables

| Component | Description | Priority |
|-----------|-------------|----------|
| Build System | CMake configuration with cross-compilation for Jetson | P0 |
| `CudaContext` | Device selection and initialization | P0 |
| `CudaStream` | RAII wrapper for `cudaStream_t` | P0 |
| `CudaEvent` | RAII wrapper for `cudaEvent_t` | P0 |
| `DeviceMemoryPool` | Simple pool allocator with leak detection | P0 |
| `DeviceCapabilities` | Hardware capability discovery | P1 |
| `TraceCollector` | Structured event collection with JSON export | P0 |
| Unit Tests | >70% coverage for all Layer 1 components | P0 |

### Technical Specifications

```cpp
// CudaStream RAII contract
class CudaStream {
public:
    CudaStream();                           // Creates stream
    ~CudaStream() noexcept;                 // Destroys stream, syncs first
    CudaStream(CudaStream&&) noexcept;      // Move-only semantics
    CudaStream(const CudaStream&) = delete; // No copy

    cudaStream_t handle() const noexcept;
    void synchronize();
    bool is_ready() const;

    // Tracing integration
    void record_event(CudaEvent& event);
    void wait_event(const CudaEvent& event);
};

// DeviceMemoryPool contract
class DeviceMemoryPool {
public:
    explicit DeviceMemoryPool(size_t capacity);
    ~DeviceMemoryPool();

    Result<void*> allocate(size_t bytes);
    void deallocate(void* ptr);

    size_t capacity() const noexcept;
    size_t used() const noexcept;
    size_t available() const noexcept;
};
```

### Exit Criteria

| Criterion | Verification Method | Pass Condition |
|-----------|-------------------|----------------|
| All CUDA resources via RAII | Code review | No raw handles in application code |
| No memory leaks | cuda-memcheck --leak-check full | Zero leaks detected |
| All CUDA calls checked | Code review + static analysis | No unchecked API calls |
| Cross-compilation works | Build test | Build on x86_64, run on Jetson |
| Tests pass | CI run | >70% coverage, all tests green |
| Tracing works | Manual test | JSON export parseable |

### Learning Outcomes

| Skill Area | Specific Skills |
|------------|----------------|
| Modern C++ | RAII patterns, move semantics, smart pointers |
| CUDA Programming | Stream management, event synchronization, memory allocation |
| Build Systems | CMake, cross-compilation, dependency management |
| Testing | GoogleTest, coverage measurement, test design |

### References

- CUDA C++ Programming Guide: Stream Management
- "Effective Modern C++" by Scott Meyers: Items 17-22 (RAII)
- CMake documentation: Cross-compilation toolchains

---

## Phase 1: Execution Context Isolation

### Duration Estimate

**2-3 weeks**

### Objective

Provide per-model execution contexts with hard resource budgets, fault isolation, and runtime backend abstraction.

### Deliverables

| Component | Description | Priority |
|-----------|-------------|----------|
| `ResourceBudget` | Hard limits on memory, streams, compute | P0 |
| `ExecutionContext` | Per-model resource container | P0 |
| `FaultBoundary` | Error isolation between contexts | P0 |
| `RuntimeBackend` | Abstract interface for inference runtime | P0 |
| `MockBackend` | Test double for development | P0 |
| Integration Tests | Multi-model isolation scenarios | P0 |

### Technical Specifications

```cpp
struct ResourceBudget {
    size_t memory_limit;        // Maximum device memory
    int stream_limit;           // Maximum concurrent streams
    Duration compute_budget;    // Maximum compute per period

    bool is_valid() const;
    std::string validation_error() const;
};

class ExecutionContext {
public:
    static Result<std::unique_ptr<ExecutionContext>> create(
        ModelID model,
        ResourceBudget budget,
        std::unique_ptr<RuntimeBackend> backend
    );

    ModelID model_id() const noexcept;
    const ResourceBudget& budget() const noexcept;

    // Resource tracking
    size_t memory_used() const noexcept;
    int streams_in_use() const noexcept;
    bool can_allocate(size_t bytes) const noexcept;

    // Fault isolation
    bool has_error() const noexcept;
    std::optional<Error> last_error() const noexcept;
    void clear_error();

private:
    ModelID model_;
    ResourceBudget budget_;
    std::unique_ptr<RuntimeBackend> backend_;
    std::atomic<size_t> memory_used_{0};
    std::atomic<int> streams_in_use_{0};
    std::optional<Error> last_error_;
};
```

### Exit Criteria

| Criterion | Verification Method | Pass Condition |
|-----------|-------------------|----------------|
| Contexts are isolated | Integration test | Two models execute without interference |
| Budgets are enforced | Unit test | Budget violations trigger explicit error |
| Faults are contained | Integration test | One model's error does not affect others |
| Backend is abstract | Code review | New backends can be added without core changes |
| Tests pass | CI run | >70% coverage for Layer 2 components |

### Learning Outcomes

| Skill Area | Specific Skills |
|------------|----------------|
| Systems Design | Resource isolation, fault boundaries, ownership models |
| Interface Design | Abstract interfaces, dependency injection, testability |
| Concurrency | Atomic operations, thread-safe state management |
| Error Handling | Typed errors, error propagation, fault isolation |

### References

- "Systems Programming" by Bryant & O'Hallaron: Process isolation concepts
- CUDA MPS documentation: Multi-process scheduling
- GoogleTest documentation: Mock objects and dependency injection

---

## Phase 2: Memory Orchestration

### Duration Estimate

**3-4 weeks**

### Objective

Manage device memory across multiple models with lifetime-aware sharing and explicit pressure handling.

### Deliverables

| Component | Description | Priority |
|-----------|-------------|----------|
| `MemoryOrchestrator` | Cross-model memory coordination | P0 |
| `LifetimeAnalyzer` | Tensor lifetime computation | P0 |
| `MemoryPlanner` | Static allocation plan generation | P0 |
| `PressureHandler` | Explicit policies for memory pressure | P1 |
| Allocation Tests | Sharing efficiency benchmarks | P0 |

### Technical Specifications

```cpp
struct TensorLifetime {
    TensorID tensor;
    Timestamp start;    // First write
    Timestamp end;      // Last read
    size_t size;
    std::string owner_model;
};

struct AllocationPlan {
    std::map<TensorID, MemoryRegion> allocations;
    std::vector<SharingOpportunity> sharing;
    size_t peak_memory;
    size_t saved_memory;
};

class MemoryOrchestrator {
public:
    explicit MemoryOrchestrator(std::shared_ptr<DeviceMemoryPool> pool);

    // Planning (occurs at admission time)
    Result<AllocationPlan> plan(
        const std::vector<ExecutionContext*>& contexts,
        const std::vector<TensorLifetime>& lifetimes
    );

    // Execution
    Result<MemoryBinding> bind(ModelID model, const AllocationPlan& plan);
    void release(ModelID model);

    // Pressure handling
    void handle_pressure(PressurePolicy policy);
    PressureLevel current_pressure() const;

    // Observability
    size_t peak_usage() const noexcept;
    size_t current_usage() const noexcept;
};
```

### Exit Criteria

| Criterion | Verification Method | Pass Condition |
|-----------|-------------------|----------------|
| Sharing is automatic | Integration test | Non-overlapping lifetimes identified |
| Peak memory is bounded | Static analysis | Correct peak estimate computed |
| Pressure is explicit | Unit test | Policy-driven handling works |
| Efficiency is measurable | Benchmark | >15% memory reduction vs isolated |
| Tests pass | CI run | >70% coverage for Layer 3 components |

### Learning Outcomes

| Skill Area | Specific Skills |
|------------|----------------|
| Memory Management | Arena allocators, lifetime analysis, memory pooling |
| Algorithms | Graph algorithms, interval scheduling, bin packing |
| Systems | Memory pressure, OOM handling, resource quotas |
| Performance | Memory bandwidth, cache effects, allocation patterns |

### References

- "The Art of Multiprocessor Programming" by Herlihy & Shavit: Memory management
- GPU memory management research: Orbit allocator, Buddy allocator
- Bin packing algorithms: First-fit, best-fit, optimal strategies

---

## Phase 3: Deterministic Scheduler (CORE DIFFERENTIATOR)

### Duration Estimate

**4-6 weeks** (This is the most critical phase)

### Objective

Implement real-time scheduling algorithms adapted for non-preemptive GPU execution, with WCET profiling and formal admission control.

**This is the core research contribution of AegisRT.**

### Deliverables

| Component | Description | Priority |
|-----------|-------------|----------|
| `WCETProfiler` | Statistical worst-case execution time estimation | P0 |
| `AdmissionController` | Formal schedulability analysis | P0 |
| `SchedulingPolicy` | Abstract interface with FIFO, EDF implementations | P0 |
| `Scheduler` | Central orchestration with admission control | P0 |
| Scheduling Tests | Schedulability validation, deadline miss detection | P0 |

### Technical Specifications

```cpp
struct WCETProfile {
    Duration worst_case;
    Duration average_case;
    Duration best_case;
    Duration percentile_99;
    Duration percentile_999;
    double contention_factor;
    int sample_count;
    double confidence_level;
};

struct AdmissionRequest {
    ModelID model;
    WCETProfile wcet;
    Period period;
    Deadline deadline;
    Priority priority;
};

struct AdmissionResult {
    bool admitted;
    std::string reason;
    std::string analysis_detail;
    double utilisation;
    Duration worst_case_response_time;
    Duration max_blocking_time;
};

class AdmissionController {
public:
    AdmissionResult analyze(
        const AdmissionRequest& request,
        const std::vector<ExecutionContext*>& existing
    );

    // Schedulability tests
    bool test_rms(const std::vector<TaskParameters>& tasks);
    bool test_edf(const std::vector<TaskParameters>& tasks);
    Duration compute_blocking_time(
        const std::vector<Duration>& wcets,
        Duration new_task_wcet,
        Priority new_task_priority
    );
    Duration compute_response_time(
        const TaskParameters& task,
        const std::vector<TaskParameters>& all_tasks
    );
};

class EDFPolicy : public SchedulingPolicy {
public:
    std::optional<ExecutionRequest> select_next(
        const std::vector<ExecutionRequest>& pending
    ) override;

    void add_model(ModelID model, const WCETProfile& profile) override;
    void remove_model(ModelID model) override;

private:
    std::map<ModelID, WCETProfile> profiles_;
};
```

### Exit Criteria

| Criterion | Verification Method | Pass Condition |
|-----------|-------------------|----------------|
| WCET is conservative | Statistical test | Actual ≤ predicted in 99.9% of cases |
| Admission is correct | Integration test | Admitted models never miss deadlines |
| EDF outperforms FIFO | Benchmark | Measurable utilization improvement |
| Scheduling is fast | Microbenchmark | Decision latency < 100 μs |
| Decisions are traceable | Trace analysis | 100% reconstructible from trace |
| Tests pass | CI run | >80% coverage (critical path) |

### Learning Outcomes

| Skill Area | Specific Skills |
|------------|----------------|
| Real-Time Systems | RMS, EDF, schedulability analysis, response-time analysis |
| Statistics | Confidence intervals, outlier detection, safety margins |
| Algorithm Design | Priority queues, deadline sorting, feasibility testing |
| GPU Systems | Non-preemptive scheduling, contention effects |

### Research Questions to Explore

1. How does non-preemptive blocking time affect schedulability bounds?
2. What safety margins are needed for GPU WCET estimation?
3. How does thermal throttling affect formal guarantees?
4. Can we predict contention effects from model architecture?

### References

- Liu & Layland (1973): Rate-Monotonic Scheduling
- George et al. (1996): Non-preemptive scheduling analysis
- REEF (SOSP'23): GPU scheduling for real-time inference
- Clockwork (OSDI'20): Predictable inference serving

---

## Phase 4: Observability & Validation

### Duration Estimate

**2-3 weeks**

### Objective

Provide full traceability, validate determinism claims with controlled workloads, and produce performance case studies.

### Deliverables

| Component | Description | Priority |
|-----------|-------------|----------|
| `TraceCollector` enhancements | Structured event collection | P0 |
| `MetricsAggregator` | Per-model resource utilization | P1 |
| `AuditTrail` | Scheduling decision rationale | P0 |
| Export Adapters | Perfetto, JSON, CSV formats | P0 |
| Case Studies | Multi-model performance benchmarks | P0 |
| Documentation | Architecture, API, benchmarks | P0 |

### Technical Specifications

```cpp
struct TraceEvent {
    std::string event_id;
    std::string event_type;
    std::string component;
    Timestamp timestamp;
    Duration duration;
    std::string model_id;
    std::string request_id;
    std::string rationale;
    std::map<std::string, std::string> attributes;
};

class TraceCollector {
public:
    explicit TraceCollector(size_t buffer_size = 10000);

    void record(TraceEvent event);

    std::vector<TraceEvent> query(
        std::optional<Timestamp> start = std::nullopt,
        std::optional<Timestamp> end = std::nullopt,
        std::optional<std::string> event_type = std::nullopt,
        std::optional<std::string> model_id = std::nullopt
    );

    void export_json(const std::string& path) const;
    void export_perfetto(const std::string& path) const;

    std::vector<TraceEvent> reconstruct_decision_chain(
        const std::string& event_id
    ) const;
};
```

### Exit Criteria

| Criterion | Verification Method | Pass Condition |
|-----------|-------------------|----------------|
| Decisions are reconstructible | Trace analysis | 100% from trace alone |
| Determinism is validated | Benchmark | Latency CV < 5% under load |
| Case studies complete | Documentation | 3+ models with bounded latency |
| Architecture is understandable | External review | Senior engineer < 2 hours |
| Documentation is complete | Checklist | README, API docs, benchmarks |

### Learning Outcomes

| Skill Area | Specific Skills |
|------------|----------------|
| Observability | Tracing, metrics, structured logging |
| Performance Analysis | Benchmarking, profiling, regression testing |
| Technical Writing | Documentation, architecture description |
| Validation | Experimental design, statistical significance |

---

## Phase 5: Integration & Polish

### Duration Estimate

**3-4 weeks**

### Objective

Integrate real inference backends, polish the codebase, and prepare for community release.

### Deliverables

| Component | Description | Priority |
|-----------|-------------|----------|
| `TensorRTBackend` | Real TensorRT integration | P0 |
| `TVMBackend` | TVM Runtime integration | P1 |
| `ONNXBackend` | ONNX Runtime integration | P1 |
| Jetson Optimization | Orin-specific tuning | P0 |
| Example Applications | Demo workloads | P0 |
| Community Preparation | Contributing guide, issue templates | P0 |

### Exit Criteria

| Criterion | Verification Method | Pass Condition |
|-----------|-------------------|----------------|
| Real backends work | Integration test | Inference with actual models |
| Jetson validation | Hardware test | Tests pass on Orin |
| Examples run | Manual test | Demo application documented |
| Community ready | Checklist | CONTRIBUTING.md, issue templates |
| Documentation complete | Review | All docs up to date |

### Learning Outcomes

| Skill Area | Specific Skills |
|------------|----------------|
| Integration | API design, FFI, error handling |
| Platform Expertise | Jetson architecture, CUDA optimization |
| Open Source | Community building, contribution workflows |
| Project Management | Release planning, documentation |

---

## Cross-Phase Requirements

These requirements apply to **ALL phases** and are non-negotiable:

### Code Quality

| Requirement | Verification | Pass Condition |
|-------------|-------------|----------------|
| RAII for all resources | Code review | No raw handles |
| All CUDA calls checked | Static analysis | Zero unchecked calls |
| No compiler warnings | Build log | Clean build |
| Clang-tidy clean | clang-tidy run | Zero warnings |
| Cppcheck clean | cppcheck run | Zero errors |

### Testing

| Requirement | Verification | Pass Condition |
|-------------|-------------|----------------|
| Unit test coverage >70% | gcov/llvm-cov | >70% lines covered |
| Critical path coverage >80% | gcov/llvm-cov | >80% for scheduler |
| Integration tests pass | CI pipeline | All tests green |
| Memory sanitiser pass | ASan, MSan | Zero errors |

### Documentation

| Requirement | Verification | Pass Condition |
|-------------|-------------|----------------|
| API documentation | Doxygen/Sphinx | All public APIs documented |
| Architecture decisions | ADRs in docs/ | Major decisions recorded |
| Code comments | Review | Complex logic explained |
| README up to date | Manual | Accurate project description |

---

## Roadmap Completion Criteria

The roadmap is **complete** when ALL of the following are satisfied:

### Functional Completeness

- [ ] All phase exit criteria met and verified
- [ ] MVP scope delivered
- [ ] Real inference backends integrated
- [ ] Demo application working

### Architectural Integrity

- [ ] Layer separation is clean
- [ ] No cross-layer violations
- [ ] Each layer independently testable
- [ ] Design principles maintained

### Determinism Validation

- [ ] Latency CV < 5% under multi-model load
- [ ] Admission control prevents deadline violations
- [ ] Schedulability analysis is correct
- [ ] All decisions traceable

### Community Readiness

- [ ] Documentation complete
- [ ] Examples working
- [ ] Contributing guide published
- [ ] Issue templates available

### Personal Growth

- [ ] Research contribution identified
- [ ] Learning outcomes achieved
- [ ] Portfolio-ready project
- [ ] Technical writing completed

---

## Risk Mitigation

### Risk: Phase Takes Longer Than Expected

**Mitigation**: Each phase has minimum viable deliverables. If time-constrained:
1. Reduce scope to MVP, not quality
2. Document what was deferred
3. Do not skip verification

### Risk: Technical Blocker Discovered

**Mitigation**:
1. Document blocker in phase notes
2. Research solutions (papers, forums, experts)
3. Consider scope adjustment if fundamental
4. Seek community input if stuck >1 week

### Risk: Hardware Limitations

**Mitigation**:
1. Test on both server GPU and Jetson
2. Use mock backends for development
3. Cloud GPU for CI when Jetson unavailable
4. Design for portability from start

### Risk: Motivation Drops

**Mitigation**:
1. Each phase delivers visible progress
2. Document achievements at phase end
3. Share progress publicly for accountability
4. Take breaks between phases

---

## Session Continuity

For development sessions (especially important for part-time development):

### Session Start

1. Read current phase document
2. Review last session's progress (git log, notes)
3. Identify today's specific goal
4. Update TODO list

### During Session

1. Focus on exit criteria
2. Document decisions as you make them
3. Write tests alongside code
4. Commit frequently with meaningful messages

### Session End

1. Update phase notes with progress
2. Commit all changes
3. Update TODO list
4. Write brief summary of what was accomplished

---

## References

- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Vision**: [OUTLOOK.md](OUTLOOK.md)
- **MVP Definition**: [MVP.md](MVP.md)
- **Whitepaper**: [WHITEPAPER.md](WHITEPAPER.md)

---

**"The roadmap is not a schedule. It is a commitment to depth. Each phase builds capability that cannot be skipped."**
