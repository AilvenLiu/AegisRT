# AegisRT MVP Definition

## Philosophy: Conceptual Completeness Over Feature Completeness

The Minimum Viable Product of AegisRT is **not** a feature-complete system. It is a **conceptually complete** demonstration that validates the core hypothesis:

> **Real-time scheduling theory can be adapted for GPU execution to provide provable latency bounds in multi-model inference scenarios.**

The MVP must answer one question definitively: *Does this approach provide measurable value over existing solutions?*

---

## MVP Success Criteria

The MVP succeeds when it can demonstrate:

| Criterion | Measurement | Success Threshold |
|-----------|-------------|-------------------|
| Scheduling Overhead | Latency of scheduling decision | < 100 μs per decision |
| Determinism | Latency distribution CV | < 5% coefficient of variation |
| Admission Control | Rejection accuracy | 100% (no admitted model misses deadline) |
| Memory Efficiency | Cross-model sharing | > 15% reduction vs isolated allocation |
| Explainability | Decision reconstruction | 100% of decisions traceable to rationale |

---

## MVP Scope Definition

### What the MVP MUST Demonstrate

1. **End-to-End Scheduling Flow**: From model submission to execution completion
2. **Formal Admission Control**: Schedulability analysis with documented rationale
3. **WCET Estimation**: Conservative bounds with statistical confidence
4. **Deterministic Execution**: Bounded latency under controlled conditions
5. **Full Observability**: Every decision traceable and explainable

### What the MVP MAY Include

- Single scheduling policy (EDF recommended)
- Limited operator set (Conv, GEMM, Elementwise)
- Single GPU context
- Mock runtime backend for testing

### What the MVP MUST NOT Include

- Multi-GPU support
- Dynamic graph support
- Kernel optimisation
- Distributed deployment
- Production-grade error recovery

---

## MVP Architecture

### Minimal Component Set

```
+--------------------------------------------------------------+
|                       MVP Architecture                       |
+--------------------------------------------------------------+
|                                                              |
|  +--------------------------------------------------------+  |
|  |                    Scheduler (Core)                    |  |
|  |                                                        |  |
|  |   +-------------+  +-------------+  +--------------+   |  |
|  |   | WCETProfiler|  | Admission   |  | EDFPolicy    |   |  |
|  |   | (Statistical|  | Controller  |  | (Non-Preempt |   |  |
|  |   |  Profiling) |  | (Analysis)  |  |  ive EDF)    |   |  |
|  |   +-------------+  +-------------+  +--------------+   |  |
|  |                                                        |  |
|  +---------------------------+----------------------------+  |
|                              |                               |
|  +---------------------------v----------------------------+  |
|  |                 Resource Orchestration                 |  |
|  |                                                        |  |
|  |   +-------------+  +-------------+  +--------------+   |  |
|  |   | MemoryPool  |  | Execution   |  | MockBackend  |   |  |
|  |   | (Simple)    |  | Context     |  | (for testing |   |  |
|  |   |             |  | (Per-Model) |  |  only)       |   |  |
|  |   +-------------+  +-------------+  +--------------+   |  |
|  |                                                        |  |
|  +---------------------------+----------------------------+  |
|                              |                               |
|  +---------------------------v----------------------------+  |
|  |                    CUDA Abstraction                    |  |
|  |                                                        |  |
|  |   +-------------+  +-------------+  +--------------+   |  |
|  |   | CudaStream  |  | CudaEvent   |  |TraceCollector|   |  |
|  |   | (RAII)      |  | (RAII)      |  |(JSON output) |   |  |
|  |   +-------------+  +-------------+  +--------------+   |  |
|  |                                                        |  |
|  +--------------------------------------------------------+  |
|                                                              |
+--------------------------------------------------------------+
```

---

## MVP Implementation Phases

### Phase A: CUDA Abstraction (Week 1-2)

**Goal**: Establish RAII foundation with tracing capability.

**Deliverables**:
- `CudaStream` and `CudaEvent` RAII wrappers
- `DeviceMemoryPool` with explicit allocation/deallocation
- `TraceCollector` with JSON export
- Basic unit tests (≥70% coverage)

**Exit Criteria**:
- All CUDA resources managed via RAII
- No resource leaks (verified by cuda-memcheck)
- Trace output parseable as JSON

### Phase B: Execution Context (Week 2-3)

**Goal**: Per-model resource isolation with fault boundaries.

**Deliverables**:
- `ExecutionContext` with resource budgets
- `ResourceBudget` enforcement
- `MockBackend` for testing
- Context isolation tests

**Exit Criteria**:
- Two mock models execute without interference
- Budget violations trigger explicit rejection
- Errors isolated to originating context

### Phase C: WCET Profiling (Week 3-4)

**Goal**: Conservative worst-case execution time estimation.

**Deliverables**:
- `WCETProfiler` with statistical methods
- Contention-aware profiling mode
- Safety margin computation
- Profile serialisation/deserialisation

**Exit Criteria**:
- WCET bounds are conservative (actual ≤ predicted in 99.9% of cases)
- Profiling converges within 100 samples
- Profiles can be saved and loaded

### Phase D: Admission Control (Week 4-5)

**Goal**: Formal schedulability analysis for non-preemptive GPU execution.

**Deliverables**:
- `AdmissionController` with EDF analysis
- Non-preemptive blocking time computation
- Response time analysis
- Admission decision logging

**Exit Criteria**:
- Admitted models never miss deadlines (under controlled conditions)
- Rejected models have documented rationale
- Schedulability analysis is verifiable by inspection

### Phase E: EDF Scheduler (Week 5-6)

**Goal**: Non-preemptive Earliest Deadline First scheduling.

**Deliverables**:
- `EDFPolicy` implementation
- `Scheduler` orchestration loop
- Deadline miss detection
- Full trace output

**Exit Criteria**:
- Scheduling overhead < 100 μs
- Latency CV < 5% under multi-model load
- Every scheduling decision traceable

### Phase F: Validation (Week 6-7)

**Goal**: Demonstrate MVP value proposition.

**Deliverables**:
- Performance benchmarks
- Comparison vs baseline (no scheduler)
- Documentation
- Demo application

**Exit Criteria**:
- All MVP success criteria met
- Architecture understandable in < 2 hours
- Demo runs 3+ models with bounded latency

---

## MVP Test Scenarios

### Scenario 1: Admission Control Validation

**Setup**: Three models with known WCET profiles and deadlines.

**Test**:
1. Admit model A (should succeed)
2. Admit model B (should succeed)
3. Attempt to admit model C (may succeed or fail based on analysis)
4. Verify: No admitted model misses deadline

**Pass Condition**: Admission decisions are correct and documented.

### Scenario 2: Determinism Validation

**Setup**: Two models running concurrently, 1000 invocations each.

**Test**:
1. Profile WCET for both models
2. Admit with appropriate deadlines
3. Execute 1000 invocations
4. Analyse latency distribution

**Pass Condition**: Latency CV < 5%, no deadline misses.

### Scenario 3: Memory Efficiency Validation

**Setup**: Three models with overlapping tensor lifetimes.

**Test**:
1. Analyse tensor lifetimes
2. Compute memory plan with sharing
3. Compare vs per-model isolated allocation

**Pass Condition**: > 15% memory reduction.

### Scenario 4: Explainability Validation

**Setup**: Any execution scenario.

**Test**:
1. Execute multi-model workload
2. Export trace to JSON
3. Reconstruct scheduling decisions

**Pass Condition**: 100% of decisions reconstructible from trace.

---

## MVP Anti-Goals

These are explicitly out of scope for MVP:

### Anti-Goal 1: Production Deployment

MVP is a research prototype, not a production system. Do not optimise for:
- Crash recovery
- Graceful shutdown
- Configuration management
- Deployment automation

### Anti-Goal 2: Performance Optimisation

MVP validates the approach; it does not optimise it. Do not:
- Hand-tune kernels
- Implement complex memory pools
- Add speculative execution
- Optimise for specific hardware

### Anti-Goal 3: Feature Expansion

MVP scope is fixed. Do not add:
- Additional scheduling policies
- Multi-GPU support
- Dynamic graph support
- Advanced memory management

---

## MVP as a Learning Milestone

Completing the MVP establishes your capability in:

### Systems Engineering

- **RAII Design**: Managing GPU resources with explicit ownership
- **Concurrency**: Thread-safe queues, atomic operations, event synchronisation
- **Error Handling**: Typed errors, propagation, isolation

### Real-Time Systems Theory

- **Schedulability Analysis**: RMS, EDF, response-time analysis
- **WCET Estimation**: Statistical methods, safety margins, contention effects
- **Non-Preemptive Scheduling**: Blocking time analysis, adapted feasibility tests

### GPU Systems

- **CUDA Programming**: Streams, events, memory management
- **GPU Scheduling Research**: Understanding gaps in existing approaches
- **Edge Constraints**: Memory pressure, thermal throttling, shared resources

### Software Engineering

- **Test-Driven Development**: Exit criteria as tests
- **Documentation**: Architecture as communication
- **Incremental Delivery**: Phase-by-phase with verification

---

## MVP Deliverables Checklist

### Code

- [ ] `src/cuda/` — CUDA abstraction layer
- [ ] `src/context/` — Execution context management
- [ ] `src/memory/` — Memory orchestration
- [ ] `src/scheduler/` — Scheduler and policies
- [ ] `src/trace/` — Trace collection
- [ ] `tests/` — Unit and integration tests
- [ ] `examples/` — Demo applications

### Documentation

- [ ] `docs/ARCHITECTURE.md` — System architecture
- [ ] `docs/API.md` — API reference
- [ ] `docs/BENCHMARKS.md` — Performance results
- [ ] `docs/TUTORIAL.md` — Getting started guide

### Validation

- [ ] All unit tests passing (≥70% coverage)
- [ ] All integration tests passing
- [ ] Performance benchmarks documented
- [ ] Code review completed

---

## From MVP to Next Steps

The MVP is a **foundation**, not a destination. Upon completion:

1. **Validate with Real Models**: Test with actual TensorRT/ONNX models
2. **Expand Policy Support**: Add RMS, priority-based scheduling
3. **Real Backend Integration**: Implement TensorRT backend
4. **Hardware Optimisation**: Profile on Jetson Orin, adapt parameters
5. **Community Engagement**: Publish, present, gather feedback

---

**The MVP proves that real-time scheduling theory can solve the GPU orchestration problem. Everything after is engineering.**
