# r20 — Documentation

## Purpose

Complete, accurate documentation is a first-class deliverable for AegisRT v1.0-alpha.
The goal is that a new engineer with C++ and CUDA background can understand the
architecture, build the project, run the demo, and add a new model -- all within
2 hours of reading the docs.

## Dependencies

- r19: All case studies complete (docs reference real results)
- All r01-r19 APIs frozen

## Sub-phase Overview

| Sub-phase | Focus | Key Deliverable |
|-----------|-------|-----------------|
| r20-a | API reference + tutorial | `docs/API.md`, `docs/TUTORIAL.md` |
| r20-b | Doxygen + inline docs | Doxyfile, CI coverage check |
| r20-c | ADRs + README + review | 4 ADRs, updated README |

---

## Phase r20-a: API Reference and Tutorial

### docs/API.md Structure

```
# AegisRT Public API Reference

## Layer 1: CUDA Abstraction
### CudaStream
### CudaEvent
### DeviceMemoryPool
### TraceCollector

## Layer 2: Resource Orchestration
### ExecutionContext
### MemoryOrchestrator
### RuntimeBackend (interface)

## Layer 3: Deterministic Scheduler
### WCETProfiler
### AdmissionController
### Scheduler
```

Each class entry includes: purpose, constructor, all public methods with
parameter types and return types, and a minimal usage example.

### docs/TUTORIAL.md Structure

```
# Getting Started with AegisRT

## 1. Build
## 2. Run the Multi-Model Demo
## 3. Interpret the Trace Output
## 4. Adding a New Model
   4.1 Profile the model (WCETProfiler)
   4.2 Submit for admission (AdmissionController)
   4.3 Schedule execution (Scheduler)
   4.4 Observe results (AuditTrail)
## 5. Understanding Admission Rejection
```

---

## Phase r20-b: Doxygen and Inline Documentation

### Doxygen Comment Style

```cpp
/// @brief Create a new ExecutionContext for the given model.
///
/// @param model_id  Unique identifier for the model.
/// @param budget    Hard resource limits enforced during execution.
/// @param backend   Inference backend to dispatch execution to.
/// @return          Owning pointer to the context, or an error.
static Result<std::unique_ptr<ExecutionContext>> create(
    ModelID model_id,
    ResourceBudget budget,
    std::shared_ptr<RuntimeBackend> backend);
```

### Coverage Requirement

Every public symbol in `include/aegisrt/` must have at minimum a `@brief`.
The CI step runs `doxygen --check` and fails if any public symbol is undocumented.

---

## Phase r20-c: Architecture Decision Records

### ADR Format

```markdown
# ADR-001: RAII CUDA Wrappers

## Status: Accepted

## Context
Raw CUDA handles (cudaStream_t, cudaEvent_t) require manual cleanup.
In complex control flow, leaks are easy to introduce.

## Decision
Wrap all CUDA handles in RAII classes (CudaStream, CudaEvent).
All classes are move-only (no copy constructor or copy assignment).

## Consequences
+ No CUDA resource leaks possible in application code.
+ Ownership semantics are explicit.
- Slightly more verbose than raw handles.
- Cannot use CUDA handles in C-style APIs without .get().
```

### External Review Simulation

Walk through the documentation as a hypothetical new engineer:
1. Can I build the project from README alone? (no prior knowledge assumed)
2. Can I run the demo and see output?
3. Can I add a new model by following TUTORIAL.md?
4. Can I find the API for any class in API.md?
5. Do the ADRs explain why the key design decisions were made?

Document any gaps found and fix them before marking this roadmap complete.

## Exit Criteria

- `docs/API.md` covers all public APIs in all three layers.
- `docs/API.md` covers all public APIs in all three layers.
- `docs/TUTORIAL.md` guides a new engineer through the full workflow.
- `docs/BENCHMARKS.md` is complete and accurate (verified from r19 results).
- Doxygen CI step is green (no undocumented public symbols).
- 4 ADRs written and committed.
- README.md reflects current project status.
- External review simulation completed with no unresolved gaps.
