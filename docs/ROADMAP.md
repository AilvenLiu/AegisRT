# AegisRT Development Roadmap

## Purpose and Scope

This roadmap defines the sequential development path for AegisRT from foundational substrate to a working GPU resource orchestrator with deterministic scheduling guarantees. Each phase builds upon the previous, with mandatory exit criteria preventing premature advancement.

**Roadmap Objective:** Transform architectural vision (OUTLOOK.md) into a working GPU orchestration layer that provides deterministic multi-model scheduling, cross-model memory management, execution context isolation, and full observability.

**Roadmap Constraint:** Phases execute sequentially. No phase begins until predecessor's exit criteria are satisfied.

---

## Roadmap Principles

### 1. Sequential Execution

Phases are not parallelisable. Each phase establishes invariants required by subsequent phases. Skipping or reordering phases violates architectural dependencies.

### 2. Mandatory Exit Criteria

Each phase defines measurable, verifiable exit criteria. Advancement requires explicit verification, not subjective assessment.

### 3. Minimal Viable Deliverables

Phases deliver the minimum functionality required to satisfy exit criteria. No speculative features, no premature optimisation.

### 4. Architectural Consistency

All phases respect constraints defined in CLAUDE.md, CONTRIBUTING.md, and OUTLOOK.md. No phase introduces violations for convenience.

### 5. Incremental Complexity

Early phases establish simple, correct implementations. Later phases introduce optimisation and sophistication. Correctness precedes performance.

---

## Phase Dependency Graph

```
Phase 0: Foundations (Layer 1)
    |
    | (Establishes: CUDA abstraction, memory pool, build system)
    v
Phase 1: Execution Context Isolation (Layer 2)
    |
    | (Establishes: Per-model contexts, resource budgets, runtime backends)
    v
Phase 2: Cross-Model Memory Orchestration (Layer 3)
    |
    | (Establishes: Lifetime analysis, memory sharing, pressure handling)
    v
Phase 3: Deterministic Scheduler (Layer 4 - Core Differentiator)
    |
    | (Establishes: RT scheduling, WCET profiling, admission control)
    v
Phase 4: Observability & Validation (Layer 5)
    |
    | (Establishes: Tracing, metrics, audit trail, performance case studies)
    v
Roadmap Complete
```

**Critical Path:** No phase can begin until all predecessors complete. Dependency violations are architectural failures.

---

## Phase Summaries

### Phase 0: Foundations (Layer 1: CUDA Runtime Abstraction)

**Intent:** Establish minimal CUDA abstraction substrate with RAII wrappers, memory pool, and build system.

**Key Deliverables:**
- Repository scaffolding (CMake, directory structure, CI configuration)
- CUDA context and stream abstraction with RAII wrappers
- CUDA event abstraction with RAII wrappers
- Device memory pool with explicit allocation/deallocation
- CUDA_CHECK error macro for all API calls
- Basic device capability discovery

**Exit Criteria:**
- All CUDA resources managed via RAII (no raw handles in application code)
- Memory pool allocates and deallocates without leaks (verified by cuda-memcheck)
- Build system supports cross-compilation for Jetson Orin
- All CUDA API calls checked for errors
- Unit tests with >= 70% coverage for Layer 1 components

**Architectural Significance:** Establishes resource ownership model and CUDA abstraction layer. All subsequent layers build on these primitives.

**Learning Focus:** Modern C++20, CUDA programming, RAII patterns, CMake

---

### Phase 1: Execution Context Isolation (Layer 2)

**Intent:** Provide per-model execution contexts with hard resource budgets, fault isolation, and runtime backend abstraction.

**Key Deliverables:**
- `ExecutionContext` with per-model resource ownership
- `ResourceBudget` with hard limits on memory and streams
- `FaultBoundary` for per-context error isolation
- `RuntimeBackend` abstract interface for wrapping existing runtimes
- At least one concrete backend (TensorRT or mock backend for testing)

**Exit Criteria:**
- Two models execute in isolated contexts without resource interference
- Resource budget violations trigger explicit rejection (not silent degradation)
- One model's CUDA error does not crash the other model's context
- Runtime backend interface is clean enough to add new backends without modifying core
- Unit tests with >= 70% coverage for Layer 2 components

**Architectural Significance:** Establishes the orchestration model -- AegisRT manages contexts, not kernels. This is where cuAdapter experience directly applies.

**Learning Focus:** OS concepts (process isolation, resource limits), CUDA MPS, interface design

---

### Phase 2: Cross-Model Memory Orchestration (Layer 3)

**Intent:** Manage device memory across multiple models with lifetime-aware sharing and explicit pressure handling.

**Key Deliverables:**
- `MemoryOrchestrator` for cross-model memory coordination
- `LifetimeAnalyser` for computing tensor lifetimes and sharing opportunities
- `MemoryPlanner` for static cross-model allocation plans
- `PressureHandler` with explicit policies (shed, reject, compact)
- Memory sharing between non-overlapping execution windows

**Exit Criteria:**
- Peak memory usage reduced compared to per-model isolation (measurable)
- Memory sharing correctly identified and exploited between non-overlapping models
- Pressure handling follows explicit policy (no hidden heuristics)
- All memory allocations occur during planning, not during execution
- Unit tests with >= 70% coverage for Layer 3 components

**Architectural Significance:** Demonstrates that cross-model orchestration provides measurable value over isolated execution. This is a key differentiator.

**Learning Focus:** Memory allocation algorithms, graph theory, bin-packing, lifetime analysis

---

### Phase 3: Deterministic Scheduler (Layer 4 - Core Differentiator)

**Intent:** Implement real-time scheduling algorithms adapted for non-preemptive GPU execution, with WCET profiling and formal admission control.

**Key Deliverables:**
- `WCETProfiler` for worst-case execution time estimation per model
- `AdmissionController` with schedulability analysis
- `SchedulingPolicy` implementations: FIFO baseline, Rate-Monotonic (RMS), Earliest Deadline First (EDF)
- Non-preemptive scheduling analysis (adapted for GPU constraints)
- Deadline miss detection and reporting

**Exit Criteria:**
- WCET bounds are conservative and validated (actual execution never exceeds bound)
- Admission control correctly rejects models that would violate existing guarantees
- EDF admits more models than FIFO with same guarantee level (measurable)
- Latency distributions are unimodal with < 5% coefficient of variation
- Scheduling decisions are fully traceable (can reconstruct why each decision was made)
- Unit tests with >= 80% coverage for Layer 4 components (critical path)

**Architectural Significance:** This is the core research contribution. Adapting RT scheduling theory to non-preemptive GPU execution is genuinely novel. This is what makes AegisRT not "just another runtime."

**Learning Focus:** Real-time scheduling theory (Liu & Layland, EDF), WCET analysis, formal methods, GPU scheduling research (REEF, Clockwork, Orion)

---

### Phase 4: Observability & Validation (Layer 5)

**Intent:** Provide full traceability, validate determinism claims with real workloads, and produce performance case studies.

**Key Deliverables:**
- `TraceCollector` with structured event collection
- `MetricsAggregator` with per-model resource utilisation
- `AuditTrail` with scheduling decision rationale
- Integration with Perfetto or NVIDIA Nsight
- Performance case studies on Jetson Orin (5+ models, latency analysis)
- Architecture documentation and design rationale

**Exit Criteria:**
- Every scheduling decision reconstructible from traces
- Performance case studies demonstrate determinism claims (< 5% CV)
- Architecture understandable by senior engineer in < 2 hours
- System runs 5+ models concurrently on Orin with bounded latency
- Public-facing documentation clearly communicates what AegisRT is and why

**Architectural Significance:** Validates that the system delivers on its promises. Without observability, determinism cannot be verified.

---

## Cross-Phase Constraints

These constraints apply to ALL phases:

### C1: ASCII-Only British English

All code, comments, documentation, and commit messages use ASCII-only characters with British English spelling. No exceptions.

### C2: RAII for All Resources

All CUDA resources (streams, events, device memory) managed via RAII wrappers. No raw handles in application code.

### C3: Explicit Error Handling

All CUDA API calls checked for errors. No silent failures. Errors propagated with sufficient context for diagnosis.

### C4: No Kernel Implementation

AegisRT orchestrates existing runtimes. It does not implement kernel execution, operator fusion, or model compilation. Kernel execution is delegated to runtime backends (TensorRT, TVM, ONNX Runtime).

### C5: Deterministic Execution

Execution order and resource allocation are deterministic. No heuristics, no hidden global state.

### C6: Traceability

All scheduling decisions, memory allocations, and kernel launches are traceable via structured logs.

### C7: Testing Requirements

Minimum 70% line coverage. Critical paths (scheduling, memory management) require 80%+ coverage.

### C8: Static Analysis

All code passes clang-tidy and cppcheck with project configuration. No compiler warnings.

---

## Roadmap Completion Criteria

The roadmap is complete when ALL of the following are satisfied:

1. **Functional Completeness:** All phase exit criteria met and verified
2. **Architectural Integrity:** System demonstrates clean 5-layer separation (CUDA abstraction, context isolation, memory orchestration, scheduling, observability are independently testable)
3. **Determinism Validation:** Latency distributions are unimodal with < 5% coefficient of variation under multi-model workloads
4. **Scheduling Guarantees:** Admission control correctly prevents deadline violations; schedulability analysis is provably correct
5. **Explainability Validation:** Scheduling decisions are traceable and reconstructible from logs
6. **Orchestration Value:** Cross-model memory sharing and deterministic scheduling provide measurable improvement over naive per-model execution
7. **Documentation Completeness:** Architecture understandable without code inspection
8. **Demonstrability:** System can be presented in 30-minute technical talk without apology

**Verification Approach:** Conduct formal review with checklist. Each criterion requires explicit evidence (test results, profiling data, peer review feedback).

---

## Roadmap Governance

### Authority Hierarchy

1. `CLAUDE.md` - Agent operating constraints (highest authority)
2. `CONTRIBUTING.md` - Contribution standards
3. `docs/OUTLOOK.md` - Architectural philosophy
4. This `ROADMAP.md` - Execution sequence
5. Phase-specific documents (`docs/roadmap/phase-N-*.md`) - Detailed phase specifications

### Modification Policy

Roadmap modifications require explicit justification:
- **Adding phases:** Only if new architectural requirement discovered
- **Reordering phases:** Only if dependency analysis reveals error
- **Removing phases:** Only if objective becomes redundant

Convenience is not justification. Architectural necessity is.

### Session Continuity

For agent-driven work:
- Each session begins by reading active phase document
- Session ends with progress update in phase document or session log
- No implicit memory across sessions—all state externalised

---

## References

- **Architectural Philosophy:** `docs/OUTLOOK.md`
- **Agent Constraints:** `CLAUDE.md`
- **Contribution Standards:** `CONTRIBUTING.md`
- **Phase Details:** `docs/roadmap/phase-{0,1,2,3,4}-*.md`
- **System Architecture:** `docs/ARCHITECTURE.md`
- **Terminology:** `docs/TERMINOLOGY.md`
