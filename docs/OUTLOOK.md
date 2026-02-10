## 1. Vision

AegisRT is envisioned as a **reference‑quality edge AI runtime architecture**, comparable in conceptual weight (but not scale) to projects such as:

- TensorRT Runtime (kernel‑centric)
- TVM Runtime (compiler‑centric)
- CUDA Graphs (launch‑centric)

but from a *system‑oriented* angle.

The long‑term vision is **not adoption**, but **clarity**:

- Clarity of how GPU execution *should* be managed under strict resource constraints
- Clarity of scheduling trade‑offs between latency, throughput, and isolation
- Clarity of memory lifetime semantics in multi‑model inference

This makes AegisRT valuable as:

- A personal flagship project
- A discussion artifact with senior infra / compiler engineers
- A demonstrable proof of deep system thinking

---

## 2. What AegisRT Explicitly Is

- A **single‑node** AI execution runtime
- Targeting **edge GPUs / SoCs**
- Focused on **inference**, not training
- Implemented primarily in **C++ / CUDA**
- Designed around **explicit control**, not heuristics hidden in frameworks

---

## 3. What AegisRT Explicitly Is *Not*

- *Not* a full ML framework
- *Not* a PyTorch replacement
- *Not* a compiler (it consumes pre‑lowered graphs)
- *Not* a production‑ready SDK
- *Not* a benchmark‑only toy

AegisRT intentionally lives **between** frameworks and kernels.

---

## 4. Conceptual Pillars

### 4.1 Determinism over Peak Throughput

Edge systems care more about *worst‑case latency* than average throughput. AegisRT treats determinism as a first‑class design constraint.

**Architectural Implication:** Scheduling decisions prioritise latency predictability over utilisation maximisation. This manifests in explicit stream allocation, bounded queue depths, and rejection of work rather than unbounded buffering.

**Code Manifestation:** Schedulers expose worst‑case latency metrics. Memory allocators fail fast rather than triggering hidden compaction. Execution traces include timing distributions, not just averages.

### 4.2 Explicit Resource Ownership

No hidden global allocators, no implicit streams. Every resource has an owner, lifetime, and release semantics.

**Architectural Implication:** RAII principles extend to GPU resources. CUDA streams, events, and device memory are wrapped in C++ objects with clear ownership semantics. No resource outlives its owner.

**Code Manifestation:** All CUDA resources are managed via `std::unique_ptr` with custom deleters or dedicated RAII wrappers. Lifetimes are statically analysable. No global state beyond the CUDA runtime itself.

### 4.3 Scheduler as a First‑Class Citizen

Scheduling policy is not a side effect of execution—it *is* the execution model.

**Architectural Implication:** The scheduler is not a utility layer—it is the central orchestration component. Execution graphs are submitted to the scheduler, not executed directly. Policy (FIFO, priority, fairness) is explicit and swappable.

**Code Manifestation:** Scheduler interface accepts execution graphs and returns scheduling decisions. Execution engine consumes scheduler output. Policy is injected, not hardcoded.

### 4.4 Hardware Awareness Without Hardware Lock‑In

The system exposes Orin‑specific properties (SM count, memory limits), but avoids baking assumptions that prevent portability.

**Architectural Implication:** Hardware capabilities are queried at runtime and exposed as configuration parameters. Algorithms adapt to available resources rather than assuming fixed hardware profiles.

**Code Manifestation:** Device properties are encapsulated in a `DeviceContext` object. Scheduling and memory allocation strategies accept device constraints as parameters. No `#ifdef JETSON_ORIN` in core logic.

---

## 5. Intended Audience

- AI Infra engineers
- Runtime / compiler engineers
- Systems engineers working on edge AI
- Myself, six months from now, explaining my thinking to others

---

## 6. Architectural Philosophy

AegisRT's architecture is governed by separation of concerns across three primary domains:

### 6.1 Graph Representation vs Execution vs Scheduling

**Graph Representation:** Static, immutable DAG of operators with explicit data dependencies. Constructed once, executed many times. No runtime graph mutation.

**Execution:** Stateless kernel invocation layer. Accepts a graph, device context, and memory bindings. Produces execution traces. Does not make scheduling decisions.

**Scheduling:** Policy layer that decides *when* and *on which stream* to execute submitted graphs. Owns resource allocation strategy. Decoupled from execution mechanics.

**Rationale:** Separation enables independent evolution of scheduling policy without touching execution logic. Execution can be tested deterministically. Scheduling can be swapped or A/B tested.

### 6.2 Explicit State Machines over Implicit Heuristics

AegisRT prefers explicit, inspectable state machines over heuristic-driven behaviour.

**Example:** Memory allocation does not use hidden heuristics to decide when to compact or evict. Instead, allocation requests either succeed immediately or fail with explicit error codes. The caller decides how to respond.

**Rationale:** Heuristics obscure system behaviour and make debugging non-deterministic. Explicit state machines are testable, traceable, and understandable.

### 6.3 Compile-Time vs Runtime Decisions

**Compile-Time (Graph Construction):** Operator fusion, memory layout selection, kernel selection. These are expensive decisions made once during graph lowering.

**Runtime (Execution):** Stream assignment, memory binding, synchronisation. These are cheap decisions made per invocation.

**Rationale:** Separating expensive optimisation from cheap orchestration keeps the runtime fast and predictable. Graph construction can take seconds; execution must take microseconds.

### 6.4 Layered Abstraction Model

```
+---------------------------+
| Scheduling Policy Layer   |  (FIFO, Priority, Fairness)
+---------------------------+
| Execution Orchestration   |  (Stream management, synchronisation)
+---------------------------+
| Memory Management Layer   |  (Allocation, lifetime tracking)
+---------------------------+
| CUDA Runtime Abstraction  |  (Streams, events, kernels)
+---------------------------+
| CUDA Driver API           |
+---------------------------+
```

Each layer depends only on the layer below. No layer reaches across boundaries.

### 6.5 Observability as First-Class Constraint

Every scheduling decision, memory allocation, and kernel launch is traceable. Execution produces structured logs and metrics suitable for offline analysis.

**Rationale:** Without observability, determinism cannot be verified. Tracing is not optional—it is part of the contract.

### 6.6 Failure Domains and Error Propagation

Errors are categorised by recoverability:

- **Transient Errors:** Temporary resource exhaustion. Caller can retry.
- **Permanent Errors:** Invalid graph, unsupported operator. Caller must abort.
- **Fatal Errors:** CUDA driver failure, device lost. Process must terminate.

Each layer propagates errors upward with sufficient context for diagnosis. No silent failures.

---

## 7. Systems Constraints and Trade-offs

### 7.1 Edge SoC Constraints

**Constraint:** Limited VRAM (8-16 GB on Jetson Orin), thermal throttling, shared memory with CPU.

**Rationale:** Edge devices cannot rely on datacenter-class cooling or memory capacity. Memory pressure is the norm, not the exception.

**Implication:** Memory allocation must be predictable and bounded. No unbounded caching. Explicit memory budgets per model.

**Alternative Rejected:** Dynamic memory management with hidden eviction policies. Too unpredictable for real-time constraints.

### 7.2 Single-Node Constraint

**Constraint:** AegisRT targets single-GPU execution. No distributed scheduling, no multi-node communication.

**Rationale:** Distributed scheduling introduces complexity (network latency, failure modes, consistency) orthogonal to core runtime concerns.

**Implication:** Scheduling decisions are local and synchronous. No need for distributed consensus or fault tolerance.

**Alternative Rejected:** Multi-node support. Adds 10x complexity for use cases outside project scope.

### 7.3 Static Graph Constraint

**Constraint:** Execution graphs are static DAGs. No dynamic control flow (if/while), no runtime graph mutation.

**Rationale:** Static graphs enable ahead-of-time optimisation (memory planning, kernel fusion) and deterministic execution.

**Implication:** Dynamic models (RNNs with variable sequence length, dynamic batching) require graph recompilation or padding.

**Alternative Rejected:** Dynamic graphs with runtime recompilation. Sacrifices determinism and adds compilation overhead to critical path.

### 7.4 Inference-Only Constraint

**Constraint:** AegisRT is designed for inference, not training. No gradient computation, no backpropagation, no optimiser state.

**Rationale:** Training and inference have fundamentally different resource profiles. Training requires large memory for activations and gradients; inference requires low latency.

**Implication:** Memory allocator optimises for small, frequent allocations. No need to track computation history.

**Alternative Rejected:** Unified training/inference runtime. Training requirements dominate design, making inference suboptimal.

### 7.5 Determinism vs Throughput Trade-off

**Trade-off:** Deterministic scheduling (fixed stream assignment, bounded queues) reduces peak throughput compared to opportunistic scheduling.

**Choice:** Prioritise determinism. Edge systems value predictability over peak performance.

**Consequence:** Under light load, GPU utilisation may be suboptimal. This is acceptable.

### 7.6 Memory Predictability vs Flexibility Trade-off

**Trade-off:** Static memory planning (allocate all buffers upfront) is predictable but inflexible. Dynamic allocation is flexible but unpredictable.

**Choice:** Static memory planning with explicit budgets. Flexibility is achieved through graph recompilation, not runtime allocation.

**Consequence:** Models with highly variable memory requirements may require multiple graph variants.

---

## 8. Anti-Patterns and Anti-Goals

### 8.1 Anti-Pattern: Hidden Global State

**Description:** Global allocators, implicit default streams, singleton device contexts.

**Why Forbidden:** Global state makes testing impossible, introduces hidden dependencies, and prevents concurrent execution of independent workloads.

**Correct Approach:** All resources are explicitly passed as parameters or owned by context objects with clear lifetimes.

### 8.2 Anti-Pattern: Heuristic-Driven Resource Management

**Description:** Allocators that "intelligently" decide when to compact, evict, or prefetch based on usage patterns.

**Why Forbidden:** Heuristics are non-deterministic, difficult to test, and fail unpredictably under novel workloads.

**Correct Approach:** Explicit allocation policies with predictable failure modes. Caller decides retry strategy.

### 8.3 Anti-Pattern: Framework Coupling

**Description:** Tight integration with PyTorch, TensorFlow, or ONNX Runtime internals.

**Why Forbidden:** Framework coupling creates maintenance burden and limits architectural freedom.

**Correct Approach:** Consume pre-lowered, framework-agnostic graph representations. Frameworks are build-time dependencies, not runtime dependencies.

### 8.4 Anti-Pattern: Premature Generalisation

**Description:** Designing for hypothetical future requirements (multi-GPU, distributed, training) not in current scope.

**Why Forbidden:** Generalisation adds complexity without delivering value. Abstractions should emerge from concrete needs, not speculation.

**Correct Approach:** Solve the single-GPU inference problem completely. Generalise only when multiple concrete use cases demand it.

### 8.5 Anti-Goal: Production Readiness

**Goal:** AegisRT is a reference architecture and learning artifact, not a production SDK.

**Implication:** No need for enterprise features (telemetry integration, A/B testing infrastructure, gradual rollout). Focus on clarity and correctness.

### 8.6 Anti-Goal: Benchmark Optimisation

**Goal:** AegisRT prioritises understandability over leaderboard performance.

**Implication:** If a technique improves performance but obscures architecture (e.g., hand-tuned assembly kernels), reject it. Use cuBLAS/cuDNN for kernels; focus on orchestration.

---

## 9. System Boundaries and Interfaces

### 9.1 Input Boundary: Pre-Lowered Computation Graphs

**Format:** Framework-agnostic graph representation (e.g., ONNX, serialised protobuf, or custom IR).

**Assumption:** Graphs are already optimised (operator fusion, layout selection). AegisRT does not perform high-level optimisations.

**Contract:** Graphs are immutable after submission. No runtime mutation.

### 9.2 Output Boundary: Execution Traces and Metrics

**Format:** Structured logs (JSON, protobuf) containing:
- Kernel launch timestamps
- Memory allocation events
- Scheduling decisions
- Synchronisation points

**Purpose:** Offline analysis, debugging, performance profiling.

**Contract:** Tracing does not affect execution correctness, only performance.

### 9.3 Hardware Boundary: CUDA Runtime API

**Abstraction Level:** AegisRT wraps CUDA runtime API (streams, events, memory), not driver API.

**Rationale:** Runtime API provides sufficient control for scheduling and memory management without low-level complexity.

**Contract:** All CUDA calls are wrapped in RAII objects. No raw CUDA handles escape AegisRT interfaces.

### 9.4 Scheduling Boundary: CPU Orchestration, GPU Execution

**CPU Responsibilities:** Graph submission, scheduling decisions, memory allocation, synchronisation.

**GPU Responsibilities:** Kernel execution only. No GPU-side scheduling or memory management.

**Rationale:** CPU has full system visibility and can make globally optimal decisions. GPU-side scheduling adds complexity without clear benefit for single-node inference.

### 9.5 Memory Boundary: Device Memory Only

**Scope:** AegisRT manages device (GPU) memory only. Host (CPU) memory is caller's responsibility.

**Rationale:** Host memory management is well-solved by standard C++ allocators. Device memory requires specialised lifetime tracking.

**Contract:** Caller provides host buffers for input/output. AegisRT handles device-side allocation and transfers.

---

## 10. Success Metrics and Evaluation Criteria

### 10.1 Clarity Metric: Understandable in < 2 Hours

**Test:** Can a senior systems engineer understand the full architecture by reading documentation and skimming code in under 2 hours?

**Measurement:** Conduct informal reviews with peers. Track questions and confusion points.

### 10.2 Determinism Metric: Identical Latency Distributions

**Test:** Run the same model 1000 times. Latency distribution should be unimodal with low variance (< 5% coefficient of variation).

**Measurement:** Collect per-invocation latency. Plot histogram. Verify no long tail.

### 10.3 Explainability Metric: Traceable Scheduling Decisions

**Test:** For any execution, can we reconstruct why the scheduler made specific decisions (stream assignment, priority, rejection)?

**Measurement:** Execution traces include decision rationale. Offline analysis tool can replay scheduling logic.

### 10.4 Demonstrability Metric: Presentable Architecture

**Test:** Can the architecture be explained in a 30-minute technical presentation without apology or caveats?

**Measurement:** Prepare presentation. Deliver to technical audience. Collect feedback on clarity.

### 10.5 Architectural Metric: Clean Separation of Concerns

**Test:** Can scheduling policy be swapped without modifying execution engine? Can memory allocator be replaced without touching scheduler?

**Measurement:** Implement alternative scheduler (e.g., priority-based). Verify no changes to execution layer required.

---

## 11. Intended Audience

- AI Infra engineers
- Runtime / compiler engineers
- Systems engineers working on edge AI
- Myself, six months from now, explaining my thinking to others

