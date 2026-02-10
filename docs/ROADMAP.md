# AegisRT Development Roadmap

## Purpose and Scope

This roadmap defines the sequential development path for AegisRT from foundational substrate to demonstrable reference architecture. Each phase builds upon the previous, with mandatory exit criteria preventing premature advancement.

**Roadmap Objective:** Transform architectural vision (OUTLOOK.md) into working system through disciplined, incremental construction.

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
Phase 0: Foundations
    |
    | (Establishes: CUDA context, streams, memory pool)
    v
Phase 1: Graph & Memory Semantics
    |
    | (Establishes: Static DAG, tensor lifetimes, memory reuse)
    v
Phase 2: Scheduler v1
    |
    | (Establishes: Multi-model submission, scheduling policy)
    v
Phase 3: Advanced Runtime Control
    |
    | (Establishes: CUDA Graphs, preemption, memory pressure handling)
    v
Phase 4: Documentation & Expression
    |
    | (Establishes: Architecture diagrams, design rationale, case studies)
    v
Roadmap Complete
```

**Critical Path:** No phase can begin until all predecessors complete. Dependency violations are architectural failures.

---

## Phase Summaries

### Phase 0: Foundations

**Intent:** Establish minimal execution substrate capable of running a single model end-to-end without framework dependencies.

**Key Deliverables:**
- Repository scaffolding (CMake, directory structure, CI configuration)
- CUDA context and stream abstraction with RAII wrappers
- Device memory pool with explicit allocation/deallocation
- Single-model execution path (load graph, allocate memory, execute, retrieve results)

**Exit Criteria:**
- Single pre-lowered model executes successfully on target device
- All CUDA resources managed via RAII (no raw handles in application code)
- Build system supports cross-compilation for Jetson Orin
- Zero framework dependencies at runtime (PyTorch/TensorFlow only for graph export)

**Architectural Significance:** Establishes resource ownership model and CUDA abstraction layer. All subsequent phases build on these primitives.

**Detailed Specification:** See `docs/roadmap/phase-0-foundations.md`

---

### Phase 1: Graph & Memory Semantics

**Intent:** Make execution explicit by introducing static DAG representation and deterministic memory management.

**Key Deliverables:**
- Static execution graph (DAG) with explicit operator dependencies
- Tensor lifetime analysis and memory reuse strategy
- Event-based dependency tracking for asynchronous execution
- Memory allocation plan computed at graph construction time

**Exit Criteria:**
- Execution order is deterministic and traceable
- Peak memory usage reduced compared to naive allocation (measurable via profiling)
- Memory allocations occur only during graph construction, not during execution
- Execution traces include complete dependency information

**Architectural Significance:** Separates graph construction (expensive, one-time) from execution (cheap, repeated). Enables static analysis and optimisation.

**Detailed Specification:** See `docs/roadmap/phase-1-graph-memory.md`

---

### Phase 2: Scheduler v1

**Intent:** Introduce scheduling policy as first-class abstraction, enabling multi-model execution with explicit resource arbitration.

**Key Deliverables:**
- Multi-model submission interface (queue-based API)
- FIFO and priority-based scheduling policies
- Stream allocation strategy (fixed pool vs dynamic assignment)
- CPU-side orchestration loop with explicit scheduling decisions

**Exit Criteria:**
- Two models execute concurrently without resource conflicts
- Scheduling decisions are traceable (logs show why each decision was made)
- Latency impact of concurrent execution is measurable and explainable
- Scheduler interface is policy-agnostic (policy is injected, not hardcoded)

**Architectural Significance:** Elevates scheduler from utility to central orchestration component. Establishes separation between policy (scheduler) and mechanism (executor).

**Detailed Specification:** See `docs/roadmap/phase-2-scheduler-v1.md`

---

### Phase 3: Advanced Runtime Control

**Intent:** Explore deeper runtime techniques for reducing overhead and handling resource pressure.

**Key Deliverables:**
- CUDA Graph integration for reduced launch overhead
- Soft preemption points for long-running kernels
- Memory pressure handling (graceful degradation, explicit rejection)
- Detailed profiling hooks (kernel timing, memory events, scheduling decisions)

**Exit Criteria:**
- Launch overhead reduced by measurable factor (compare CUDA Graph vs stream-based execution)
- System remains stable under memory pressure (no crashes, predictable failure modes)
- Profiling data sufficient for offline analysis (can reconstruct full execution timeline)
- Preemption points enable bounded worst-case latency for high-priority models

**Architectural Significance:** Demonstrates advanced CUDA techniques without compromising architectural clarity. Validates that abstractions support optimisation.

**Detailed Specification:** See `docs/roadmap/phase-3-advanced-runtime.md`

---

### Phase 4: Documentation & Expression

**Intent:** Transform engineering artifacts into communication artifacts suitable for technical presentation and discussion.

**Key Deliverables:**
- Architecture diagrams (component relationships, data flow, control flow)
- Design rationale documents (why decisions were made, alternatives rejected)
- Performance case studies (latency analysis, memory profiling, scheduling trade-offs)
- Public-facing README with clear project positioning

**Exit Criteria:**
- Architecture understandable by senior engineer in < 2 hours (informal review)
- Design rationale documents answer "why" questions without requiring code inspection
- Performance case studies demonstrate determinism and explainability claims
- README clearly communicates what AegisRT is, is not, and why it exists

**Architectural Significance:** Validates that architecture is presentable and defensible. Ensures project achieves clarity objective.

**Detailed Specification:** See `docs/roadmap/phase-4-documentation.md`

---

## Cross-Phase Constraints

These constraints apply to ALL phases:

### C1: ASCII-Only British English

All code, comments, documentation, and commit messages use ASCII-only characters with British English spelling. No exceptions.

### C2: RAII for All Resources

All CUDA resources (streams, events, device memory) managed via RAII wrappers. No raw handles in application code.

### C3: Explicit Error Handling

All CUDA API calls checked for errors. No silent failures. Errors propagated with sufficient context for diagnosis.

### C4: No Framework Coupling

PyTorch, TensorFlow, ONNX Runtime are build-time dependencies only. No runtime coupling.

### C5: Deterministic Execution

Execution order and resource allocation are deterministic. No heuristics, no hidden global state.

### C6: Traceability

All scheduling decisions, memory allocations, and kernel launches are traceable via structured logs.

### C7: Testing Requirements

Minimum 70% line coverage. Critical paths (scheduling, memory management) require 90%+ coverage.

### C8: Static Analysis

All code passes clang-tidy and cppcheck with project configuration. No compiler warnings.

---

## Roadmap Completion Criteria

The roadmap is complete when ALL of the following are satisfied:

1. **Functional Completeness:** All phase exit criteria met and verified
2. **Architectural Integrity:** System demonstrates clean separation of concerns (scheduler, executor, memory manager are independently testable)
3. **Determinism Validation:** Latency distributions are unimodal with < 5% coefficient of variation
4. **Explainability Validation:** Scheduling decisions are traceable and reconstructible from logs
5. **Documentation Completeness:** Architecture understandable without code inspection
6. **Demonstrability:** System can be presented in 30-minute technical talk without apology

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
- **System Architecture:** `docs/ARCHITECTURE.md` (created in Phase 0)
- **Terminology:** `docs/TERMINOLOGY.md` (created in Phase 0)
