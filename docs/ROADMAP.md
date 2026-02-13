# AegisRT Development Roadmap

## Roadmap Philosophy

This roadmap transforms the architectural vision into a **sequential development path** with clear exit criteria and embedded learning outcomes. The guiding principle is:

> **Each phase establishes invariants required by subsequent phases. Correctness precedes performance.**

---

## Roadmap Principles

### Principle 1: Sequential Dependency

Phases execute sequentially. No phase begins until the predecessor's exit criteria are satisfied. Dependency violations are architectural failures.

### Principle 2: Measurable Exit Criteria

Each phase defines specific, verifiable exit criteria. Advancement requires explicit verification, not subjective assessment.

### Principle 3: Learning Embedded in Development

Each phase includes explicit learning outcomes. Development is not just code delivery—it is systematic skill acquisition.

### Principle 4: Minimal Viable Deliverables

Phases deliver the minimum functionality required to satisfy exit criteria. No speculative features, no premature optimisation.

---

## Phase Dependency Graph

```
Phase 0: CUDA Foundation
    |
    |  Establishes: RAII wrappers, memory pool, build system
    |
    v
Phase 1: Execution Context
    |
    |  Establishes: Per-model isolation, resource budgets, fault boundaries
    |
    v
Phase 2: Memory Orchestration
    |
    |  Establishes: Cross-model sharing, lifetime analysis, pressure handling
    |
    v
Phase 3: Deterministic Scheduler (CORE DIFFERENTIATOR)
    |
    |  Establishes: WCET profiling, admission control, RT scheduling policies
    |
    v
Phase 4: Observability & Validation
    |
    |  Establishes: Full tracing, determinism validation, case studies
    |
    v
Phase 5: Integration & Polish
    |
    |  Establishes: Real backend integration, documentation, community release
    |
    v
Roadmap Complete
```

---

## Phase 0: CUDA Foundation

### Duration Estimate

2-3 weeks (part-time independent development)

### Objective

Establish minimal CUDA abstraction substrate with RAII wrappers, memory pool, and build infrastructure.

### Deliverables

| Component | Description |
|-----------|-------------|
| Build System | CMake configuration with cross-compilation support for Jetson |
| `CudaContext` | Device selection and initialisation |
| `CudaStream` | RAII wrapper for `cudaStream_t` |
| `CudaEvent` | RAII wrapper for `cudaEvent_t` |
| `DeviceMemoryPool` | Explicit allocation/deallocation with leak detection |
| `DeviceCapabilities` | Hardware capability discovery |
| `TraceCollector` | Structured event collection with JSON export |
| Unit Tests | >70% coverage for all Layer 1 components |

### Technical Specifications

```cpp
// CudaStream RAII contract
class CudaStream {
public:
    CudaStream();                           // Creates stream
    ~CudaStream();                          // Destroys stream (no leak)
    CudaStream(CudaStream&&) noexcept;      // Move-only semantics
    CudaStream(const CudaStream&) = delete; // No copy

    cudaStream_t handle() const;
    void synchronize();
    bool is_ready() const;
};

// DeviceMemoryPool contract
class DeviceMemoryPool {
public:
    explicit DeviceMemoryPool(size_t capacity);

    Result<void*> allocate(size_t bytes);
    void deallocate(void* ptr);

    size_t capacity() const;
    size_t used() const;
    size_t available() const;
};
```

### Exit Criteria

| Criterion | Verification Method |
|-----------|-------------------|
| All CUDA resources via RAII | Code review, no raw handles in application code |
| No memory leaks | cuda-memcheck with leak check enabled |
| All CUDA calls checked | Code review, no unchecked API calls |
| Cross-compilation works | Build on x86_64, deploy to Jetson |
| Tests pass | >70% coverage, all tests green |

### Learning Outcomes

| Skill Area | Specific Skills |
|------------|----------------|
| Modern C++ | RAII patterns, move semantics, smart pointers |
| CUDA Programming | Stream management, event synchronisation, memory allocation |
| Build Systems | CMake, cross-compilation, dependency management |
| Testing | GoogleTest, coverage measurement, test design |

### References

- CUDA C++ Programming Guide: Stream Management
- "Effective Modern C++" by Scott Meyers: Item 17-22 (RAII)
- CMake documentation: Cross-compilation

---

## Phase 1: Execution Context Isolation

### Duration Estimate

2-3 weeks

### Objective

Provide per-model execution contexts with hard resource budgets, fault isolation, and runtime backend abstraction.

### Deliverables

| Component | Description |
|-----------|-------------|
| `ExecutionContext` | Per-model resource container with ownership semantics |
| `ResourceBudget` | Hard limits on memory, streams, and compute time |
| `FaultBoundary` | Error isolation between contexts |
| `RuntimeBackend` | Abstract interface for inference runtime integration |
| `MockBackend` | Test double for development |
| Integration Tests | Multi-model isolation scenarios |

### Technical Specifications

```cpp
struct ResourceBudget {
    size_t memory_limit;      // Maximum device memory
    int stream_limit;         // Maximum concurrent streams
    Duration compute_budget;  // Maximum compute time per period
};

class ExecutionContext {
public:
    ExecutionContext(
        ModelID model,
        ResourceBudget budget,
        std::unique_ptr<RuntimeBackend> backend
    );

    // Resource tracking
    bool can_allocate(size_t bytes) const;
    bool can_acquire_stream() const;

    // Fault isolation
    bool has_error() const;
    std::optional<CudaError> last_error() const;
    void clear_error();

private:
    ModelID model_;
    ResourceBudget budget_;
    std::unique_ptr<RuntimeBackend> backend_;
    std::atomic<size_t> memory_used_{0};
    std::atomic<int> streams_in_use_{0};
    std::optional<CudaError> last_error_;
};
```

### Exit Criteria

| Criterion | Verification Method |
|-----------|-------------------|
| Contexts are isolated | Two models execute without interference |
| Budgets are enforced | Budget violations trigger explicit rejection |
| Faults are contained | One model's error does not affect others |
| Backend is abstract | New backends can be added without core changes |
| Tests pass | >70% coverage for Layer 2 components |

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
- GoogleTest documentation: Mock objects

---

## Phase 2: Memory Orchestration

### Duration Estimate

3-4 weeks

### Objective

Manage device memory across multiple models with lifetime-aware sharing and explicit pressure handling.

### Deliverables

| Component | Description |
|-----------|-------------|
| `MemoryOrchestrator` | Cross-model memory coordination |
| `LifetimeAnalyser` | Tensor lifetime computation from execution graphs |
| `MemoryPlanner` | Static allocation plan generation |
| `PressureHandler` | Explicit policies for memory pressure |
| Allocation Tests | Sharing efficiency benchmarks |

### Technical Specifications

```cpp
struct TensorLifetime {
    TensorID tensor;
    Timestamp start;    // First write
    Timestamp end;      // Last read
    size_t size;
};

struct AllocationPlan {
    std::map<TensorID, MemoryRegion> allocations;
    std::vector<SharingOpportunity> sharing;
    size_t peak_memory;
};

class MemoryOrchestrator {
public:
    // Planning (occurs at admission time)
    AllocationPlan plan(
        const std::vector<ExecutionContext*>& contexts,
        const std::vector<TensorLifetime>& lifetimes
    );

    // Execution
    Result<MemoryBinding> bind(ModelID model, const AllocationPlan& plan);
    void release(ModelID model);

    // Pressure handling
    void handle_pressure(PressurePolicy policy);
};
```

### Exit Criteria

| Criterion | Verification Method |
|-----------|-------------------|
| Sharing is automatic | Non-overlapping lifetimes identified and exploited |
| Peak memory is bounded | Static analysis produces correct peak estimate |
| Pressure is explicit | Policy-driven handling, no hidden heuristics |
| Efficiency is measurable | >15% memory reduction vs isolated allocation |
| Tests pass | >70% coverage for Layer 3 components |

### Learning Outcomes

| Skill Area | Specific Skills |
|------------|----------------|
| Memory Management | Arena allocators, lifetime analysis, memory pooling |
| Algorithms | Graph algorithms, interval scheduling, bin packing |
| Systems | Memory pressure, OOM handling, resource quotas |
| Performance | Memory bandwidth, cache effects, allocation patterns |

### References

- "The Art of Multiprocessor Programming" by Herlihy & Shavit: Memory management
- GPU memory management research: Orbit, Buddy allocator
- Bin packing algorithms: First-fit, best-fit, optimal

---

## Phase 3: Deterministic Scheduler (CORE DIFFERENTIATOR)

### Duration Estimate

4-6 weeks

### Objective

Implement real-time scheduling algorithms adapted for non-preemptive GPU execution, with WCET profiling and formal admission control.

**This is the core research contribution of AegisRT.**

### Deliverables

| Component | Description |
|-----------|-------------|
| `WCETProfiler` | Statistical worst-case execution time estimation |
| `AdmissionController` | Formal schedulability analysis |
| `SchedulingPolicy` | Abstract interface with FIFO, RMS, EDF implementations |
| `Scheduler` | Central orchestration with admission control |
| Scheduling Tests | Schedulability validation, deadline miss detection |

### Technical Specifications

```cpp
struct WCETProfile {
    Duration worst_case;
    Duration average_case;
    double contention_factor;
    int sample_count;
    double confidence_level;
};

struct AdmissionRequest {
    ModelID model;
    WCETProfile wcet;
    Period period;
    Deadline deadline;
};

struct AdmissionResult {
    bool admitted;
    std::string reason;
    double utilisation;
    Duration worst_case_response_time;
};

class AdmissionController {
public:
    AdmissionResult analyse(
        const AdmissionRequest& request,
        const std::vector<ExecutionContext*>& existing
    );

    // Schedulability tests
    bool test_rms(const std::vector<TaskParameters>& tasks);
    bool test_edf(const std::vector<TaskParameters>& tasks);
    Duration compute_blocking_time(const std::vector<Duration>& wcets);
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

| Criterion | Verification Method |
|-----------|-------------------|
| WCET is conservative | Actual execution ≤ predicted in 99.9% of cases |
| Admission is correct | Admitted models never miss deadlines |
| EDF outperforms FIFO | Measurable utilisation improvement |
| Scheduling is fast | Decision latency < 100 μs |
| Decisions are traceable | 100% reconstructible from trace |
| Tests pass | >80% coverage (critical path) |

### Learning Outcomes

| Skill Area | Specific Skills |
|------------|----------------|
| Real-Time Systems | RMS, EDF, schedulability analysis, response-time analysis |
| Statistics | Confidence intervals, outlier detection, safety margins |
| Algorithm Design | Priority queues, deadline sorting, feasibility testing |
| GPU Systems | Non-preemptive scheduling, contention effects |

### Research Questions to Explore

1. How does non-preemptive blocking time affect schedulability?
2. What safety margins are needed for GPU WCET?
3. How does thermal throttling affect guarantees?

### References

- Liu & Layland (1973): Rate-Monotonic Scheduling
- George et al. (1996): Non-preemptive scheduling
- REEF (SOSP'23): GPU scheduling for real-time
- Clockwork (OSDI'20): Predictable inference serving

---

## Phase 4: Observability & Validation

### Duration Estimate

2-3 weeks

### Objective

Provide full traceability, validate determinism claims with real workloads, and produce performance case studies.

### Deliverables

| Component | Description |
|-----------|-------------|
| `TraceCollector` | Structured event collection |
| `MetricsAggregator` | Per-model resource utilisation |
| `AuditTrail` | Scheduling decision rationale |
| Export Adapters | Perfetto, JSON, custom formats |
| Case Studies | Multi-model performance benchmarks |
| Documentation | Architecture, API, benchmarks |

### Technical Specifications

```cpp
struct TraceEvent {
    std::string event_type;
    std::string component;
    std::string rationale;
    Timestamp timestamp;
    Duration duration;
    std::map<std::string, std::string> attributes;
};

class TraceCollector {
public:
    void record(TraceEvent event);

    std::vector<TraceEvent> query(
        std::optional<Timestamp> start,
        std::optional<Timestamp> end,
        std::optional<std::string> event_type
    );

    void export_json(const std::string& path);
    void export_perfetto(const std::string& path);
};
```

### Exit Criteria

| Criterion | Verification Method |
|-----------|-------------------|
| Decisions are reconstructible | 100% from trace alone |
| Determinism is validated | Latency CV < 5% under multi-model load |
| Case studies complete | 3+ models with bounded latency |
| Architecture is understandable | Senior engineer review < 2 hours |
| Documentation is complete | README, API docs, benchmarks |

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

3-4 weeks

### Objective

Integrate real inference backends, polish the codebase, and prepare for community release.

### Deliverables

| Component | Description |
|-----------|-------------|
| `TensorRTBackend` | Real TensorRT integration |
| `TVMBackend` | TVM Runtime integration |
| `ONNXBackend` | ONNX Runtime integration |
| Jetson Optimisation | Orin-specific tuning |
| Example Applications | Demo workloads |
| Community Preparation | Contributing guide, issue templates |

### Exit Criteria

| Criterion | Verification Method |
|-----------|-------------------|
| Real backends work | Inference with actual models |
| Jetson validation | Tests pass on Orin hardware |
| Examples run | Demo application documented |
| Community ready | CONTRIBUTING.md, issue templates |

### Learning Outcomes

| Skill Area | Specific Skills |
|------------|----------------|
| Integration | API design, FFI, error handling |
| Platform Expertise | Jetson architecture, CUDA optimisation |
| Open Source | Community building, contribution workflows |
| Project Management | Release planning, documentation |

---

## Cross-Phase Requirements

These requirements apply to ALL phases:

### Code Quality

| Requirement | Verification |
|-------------|-------------|
| RAII for all resources | Code review |
| All CUDA calls checked | Static analysis |
| No compiler warnings | Build log |
| Clang-tidy clean | clang-tidy run |
| Cppcheck clean | cppcheck run |

### Testing

| Requirement | Verification |
|-------------|-------------|
| Unit test coverage >70% | gcov/llvm-cov |
| Critical path coverage >80% | gcov/llvm-cov |
| Integration tests pass | CI pipeline |
| Memory sanitiser pass | ASan, MSan |

### Documentation

| Requirement | Verification |
|-------------|-------------|
| API documentation | Doxygen/Sphinx |
| Architecture decisions | ADRs in docs/ |
| Code comments | Review |
| README up to date | Manual |

---

## Roadmap Completion Criteria

The roadmap is complete when ALL of the following are satisfied:

### Functional Completeness

- [ ] All phase exit criteria met and verified
- [ ] MVP scope delivered
- [ ] Real inference backends integrated

### Architectural Integrity

- [ ] Layer separation is clean
- [ ] No cross-layer violations
- [ ] Each layer independently testable

### Determinism Validation

- [ ] Latency CV < 5% under multi-model load
- [ ] Admission control prevents deadline violations
- [ ] Schedulability analysis is correct

### Community Readiness

- [ ] Documentation complete
- [ ] Examples working
- [ ] Contributing guide published

### Personal Growth

- [ ] Research contribution identified
- [ ] Learning outcomes achieved
- [ ] Portfolio-ready project

---

## Risk Mitigation

### Risk: Phase Takes Longer Than Expected

**Mitigation**: Each phase has minimum viable deliverables. If time-constrained, reduce scope to MVP, not quality.

### Risk: Technical Blocker Discovered

**Mitigation**: Document blockers in phase notes. Seek community input if stuck. Consider scope adjustment.

### Risk: Hardware Limitations

**Mitigation**: Test on both server GPU and Jetson. Use mock backends for development. Cloud GPU for CI.

---

## Session Continuity

For development sessions:

1. **Start**: Read current phase document, review last session's progress
2. **Work**: Focus on exit criteria, document decisions
3. **End**: Update phase notes, commit progress, update TODOs

---

## References

- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Vision**: [OUTLOOK.md](OUTLOOK.md)
- **MVP Definition**: [MVP.md](MVP.md)
- **Terminology**: [TERMINOLOGY.md](TERMINOLOGY.md)

---

**"The roadmap is not a schedule. It is a commitment to depth."**
