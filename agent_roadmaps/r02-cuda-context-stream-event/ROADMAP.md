# ROADMAP -- r02-cuda-context-stream-event

> This document describes a single-session roadmap task.
> It is written to be read by an AI agent with no prior context.
> Verbosity is intentional to prevent ambiguity.

---

## 1. Background and Motivation

AegisRT requires three core CUDA RAII primitives before any higher-layer code can be written.
CudaContext, CudaStream, and CudaEvent are the foundation of all GPU resource management.
Every component in Layer 2 and Layer 3 depends on these being correct and leak-free.

Without this roadmap:
- No GPU resource management is possible.
- Memory pool (r03) cannot be implemented.
- Scheduler cannot dispatch work to GPU streams.

---

## 2. Overall Objective

By the end of this roadmap, the following MUST be true:

- CudaContext, CudaStream, CudaEvent are implemented as move-only RAII classes.
- All unit tests pass.
- cuda-memcheck --leak-check full reports zero leaks.
- Move semantics verified: moved-from objects are in valid null state.
- clang-tidy clean on all new files.

These objectives are **contractual**.

---

## 3. Explicit Non-Goals

The following are **explicitly excluded** from this roadmap:

- No DeviceMemoryPool (r03).
- No TraceCollector (r04).
- No higher-layer integration.
- No kernel implementations.

---

## 4. High-Level Strategy

Implement each primitive as a move-only RAII class. Use deleted copy constructor and
copy assignment to enforce move-only semantics. Verify with unit tests and cuda-memcheck.
CudaStream destructor must synchronise before destroy to prevent use-after-free.

---

## 5. Phase Overview

This roadmap has a single phase: **phase-r02 -- CudaContext, CudaStream, CudaEvent**.

---

## 6. Phase Details

### phase-r02 -- CudaContext, CudaStream, CudaEvent

#### Objective

After this phase:
- Three RAII primitives are implemented and tested.
- Zero memory leaks under cuda-memcheck.
- Move semantics are correct.

#### Inputs / Preconditions

- r01 complete: build system operational, CUDA_CHECK macro available.

#### Constraints (Re-affirmed)

- Refer to INVARIANTS.md for all constraints.
- CudaStream and CudaEvent MUST be move-only.
- CudaStream destructor MUST synchronise before destroy.

#### Execution Guidance

Implement in order: CudaContext first (device selection), then CudaStream (stream management),
then CudaEvent (event management and timing). Write unit tests for each before moving to the next.

Typical failure modes:
- Forgetting to set handle to nullptr after move: causes double-destroy.
- CudaStream destructor not synchronising: causes use-after-free in CUDA runtime.
- CudaEvent destructor throwing: violates noexcept contract.

#### Deliverables

- `src/cuda/cuda_context.hpp / .cpp`
- `src/cuda/cuda_stream.hpp / .cpp`
- `src/cuda/cuda_event.hpp / .cpp`
- `tests/test_cuda_context.cpp`
- `tests/test_cuda_stream.cpp`
- `tests/test_cuda_event.cpp`

#### Exit Criteria

This phase is complete when:

- All unit tests pass.
- cuda-memcheck --leak-check full reports zero leaks.
- Move semantics verified: moved-from object is in valid null state.
- clang-tidy clean on all new files.

---

## 7. Risk and Rollback Considerations

- Risk: CUDA device not available in CI. Mitigation: skip GPU tests in CI, run locally.
- Risk: Move semantics subtle bugs. Mitigation: explicit test for null state after move.
- Rollback: Revert commits; no persistent state outside source files.

---

## 8. Completion Definition

The roadmap is considered complete when:

- All tasks in phase-r02 are marked `completed` in roadmap.yml.
- All exit criteria above are verified.
- A session handoff file exists in `sessions/`.
- No open blockers remain.

---

## 9. Final Execution Rule

> Follow this document literally.
> Do not infer intent beyond what is written.
