# AegisRT Terminology

## Purpose

This document defines domain-specific terms used throughout AegisRT documentation and codebase. Consistent terminology prevents ambiguity and ensures clear communication across sessions.

---

## Core Concepts

### Execution Graph

**Definition:** Immutable directed acyclic graph (DAG) representing a computation. Nodes are operators (e.g., Conv2D, MatMul), edges are tensors (data dependencies).

**Contrast with Computation Graph:** Computation graphs (PyTorch, TensorFlow) support dynamic control flow and runtime mutation. Execution graphs are static and immutable.

**Usage:** "The execution graph is constructed once during model loading and executed many times."

---

### Operator

**Definition:** Single computational unit in an execution graph (e.g., matrix multiplication, convolution, activation function).

**Contrast with Kernel:** An operator may invoke one or more CUDA kernels. For example, a Conv2D operator may invoke cuDNN kernels.

**Contrast with Node:** "Operator" refers to the computational operation. "Node" refers to the graph representation.

**Usage:** "Each operator in the graph has explicit input and output tensors."

---

### Kernel

**Definition:** GPU function executed on CUDA device. Invoked via `<<<grid, block>>>` syntax or cuBLAS/cuDNN APIs.

**Contrast with Operator:** Operators are high-level abstractions. Kernels are low-level GPU functions.

**Usage:** "The MatMul operator invokes a cuBLAS kernel for matrix multiplication."

---

### Tensor

**Definition:** Multi-dimensional array of data (e.g., 4D tensor for image batches: [batch, channels, height, width]).

**Contrast with Buffer:** Tensors have shape and data type. Buffers are raw memory allocations.

**Usage:** "Tensor lifetimes are computed statically to enable memory reuse."

---

### Buffer

**Definition:** Contiguous block of device memory. May hold one or more tensors.

**Contrast with Tensor:** Buffers are memory allocations. Tensors are logical data structures.

**Usage:** "Multiple tensors with non-overlapping lifetimes share the same buffer."

---

### Allocation

**Definition:** Act of reserving device memory via `cudaMalloc` or memory pool.

**Contrast with Buffer:** Allocation is the process. Buffer is the result.

**Usage:** "Memory allocations occur only during graph construction, not during execution."

---

## Execution Concepts

### Stream

**Definition:** CUDA stream, a sequence of operations that execute in order on the GPU. Multiple streams can execute concurrently.

**Contrast with Context:** Streams are execution queues. Contexts are device-level state containers.

**Usage:** "Each model execution is assigned to a dedicated stream from the stream pool."

---

### Context

**Definition:** CUDA context, encapsulating device state (memory allocations, streams, events). One context per device.

**Contrast with Stream:** Context is device-level. Stream is execution-level.

**Usage:** "The CUDA context is initialised once at startup and shared across all executions."

---

### Event

**Definition:** CUDA event, a synchronisation primitive. Used to record completion of operations and wait for dependencies.

**Contrast with Stream:** Events synchronise across streams. Streams execute operations sequentially.

**Usage:** "An event is recorded after the producer operator and waited by consumer operators."

---

## Scheduling Concepts

### Scheduler

**Definition:** Central orchestration component that decides when and on which stream to execute submitted graphs.

**Contrast with Executor:** Scheduler makes policy decisions (when, where). Executor performs execution (how).

**Usage:** "The scheduler dequeues requests based on the configured scheduling policy."

---

### Executor

**Definition:** Stateless component that executes a graph on a given stream with given memory bindings.

**Contrast with Scheduler:** Executor is mechanism. Scheduler is policy.

**Usage:** "The executor invokes kernels in topological order without making scheduling decisions."

---

### Orchestrator

**Definition:** Synonym for scheduler. Emphasises coordination role.

**Usage:** "The orchestrator manages the submission queue and dispatches execution requests."

---

### Scheduling Policy

**Definition:** Algorithm that selects the next execution request from the submission queue (e.g., FIFO, priority-based, fairness).

**Contrast with Scheduler:** Policy is the algorithm. Scheduler is the component that applies the policy.

**Usage:** "The scheduling policy is injected into the scheduler at construction time."

---

### Submission Queue

**Definition:** Bounded, thread-safe queue holding pending execution requests.

**Contrast with Scheduler:** Queue is data structure. Scheduler is orchestration component.

**Usage:** "Submissions are rejected if the queue is full, preventing unbounded memory growth."

---

## Memory Concepts

### Lifetime

**Definition:** Time interval during which a tensor is live (allocated and potentially accessed).

**Contrast with Allocation:** Lifetime is temporal. Allocation is spatial.

**Usage:** "Tensor lifetimes are computed via liveness analysis on the execution graph."

---

### Liveness Analysis

**Definition:** Dataflow analysis algorithm that computes when each tensor is allocated (first use) and deallocated (last use).

**Contrast with Lifetime:** Liveness analysis is the algorithm. Lifetime is the result.

**Usage:** "Liveness analysis enables static memory planning by identifying non-overlapping lifetimes."

---

### Memory Reuse

**Definition:** Strategy where multiple tensors with non-overlapping lifetimes share the same memory buffer.

**Contrast with Memory Pooling:** Reuse is within a single execution. Pooling is across executions.

**Usage:** "Memory reuse reduces peak memory usage by 30% compared to naive allocation."

---

### Memory Planning

**Definition:** Process of computing a static allocation plan that maps tensors to memory offsets.

**Contrast with Memory Allocation:** Planning is compile-time. Allocation is runtime.

**Usage:** "Memory planning occurs once during graph construction, not during execution."

---

### Peak Memory

**Definition:** Maximum device memory usage during execution of a graph.

**Contrast with Total Memory:** Peak is maximum at any point in time. Total is sum of all allocations.

**Usage:** "Peak memory is reduced through memory reuse, enabling larger models on edge devices."

---

## Performance Concepts

### Determinism

**Definition:** Property where identical inputs produce identical outputs with identical execution order and timing.

**Contrast with Predictability:** Determinism is exact repeatability. Predictability is bounded variance.

**Usage:** "AegisRT prioritises determinism over peak throughput for edge inference."

---

### Predictability

**Definition:** Property where execution behaviour (latency, memory usage) is bounded and explainable.

**Contrast with Determinism:** Predictability allows variance within bounds. Determinism requires exact repeatability.

**Usage:** "Memory allocation is predictable because it occurs only during graph construction."

---

### Latency

**Definition:** Time from submission of execution request to completion (end-to-end time).

**Contrast with Throughput:** Latency is per-request time. Throughput is requests per unit time.

**Usage:** "High-priority requests have bounded latency even under concurrent load."

---

### Throughput

**Definition:** Number of execution requests completed per unit time.

**Contrast with Latency:** Throughput is aggregate metric. Latency is per-request metric.

**Usage:** "AegisRT prioritises latency predictability over peak throughput."

---

### Launch Overhead

**Definition:** CPU time required to submit a kernel to the GPU (includes API call overhead, driver processing).

**Contrast with Kernel Execution Time:** Launch overhead is CPU-side. Kernel execution is GPU-side.

**Usage:** "CUDA Graphs reduce launch overhead by batching kernel submissions."

---

## Advanced Concepts

### CUDA Graph

**Definition:** CUDA feature that captures a sequence of kernel launches as a graph, enabling batched submission with reduced overhead.

**Contrast with Execution Graph:** CUDA Graphs are CUDA runtime feature. Execution graphs are AegisRT abstraction.

**Usage:** "CUDA Graphs reduce launch overhead by 30% compared to stream-based execution."

---

### Preemption

**Definition:** Interrupting execution of a low-priority request to allow a high-priority request to execute.

**Contrast with Scheduling:** Scheduling decides order before execution. Preemption interrupts during execution.

**Usage:** "Soft preemption points enable bounded latency for high-priority requests."

---

### Soft Preemption

**Definition:** Cooperative preemption where execution voluntarily yields at designated points (e.g., operator boundaries).

**Contrast with Hard Preemption:** Soft preemption requires cooperation. Hard preemption is enforced by hardware.

**Usage:** "Soft preemption is simpler than hardware preemption and sufficient for edge inference."

---

### Memory Pressure

**Definition:** State where available device memory is insufficient to satisfy allocation requests.

**Contrast with Memory Exhaustion:** Pressure is approaching limit. Exhaustion is reaching limit.

**Usage:** "Under memory pressure, the scheduler rejects new submissions to prevent crashes."

---

### Admission Control

**Definition:** Policy that decides whether to accept or reject a submission based on resource availability.

**Contrast with Scheduling:** Admission control is accept/reject decision. Scheduling is ordering decision.

**Usage:** "Admission control rejects submissions when memory budget exceeds available memory."

---

## Observability Concepts

### Traceability

**Definition:** Property where all decisions (scheduling, allocation, execution) are logged with sufficient context for reconstruction.

**Contrast with Observability:** Traceability is decision logging. Observability is system-wide visibility.

**Usage:** "Scheduling decisions are traceable via structured logs that include rationale."

---

### Observability

**Definition:** Property where system behaviour is visible through metrics, logs, and traces.

**Contrast with Traceability:** Observability is broader (includes metrics, health). Traceability is narrower (decision logs).

**Usage:** "Observability is a first-class constraint in AegisRT architecture."

---

### Profiling

**Definition:** Process of collecting detailed execution metrics (kernel timing, memory usage, synchronisation events).

**Contrast with Tracing:** Profiling is performance-focused. Tracing is decision-focused.

**Usage:** "Profiling hooks capture kernel launch timestamps for offline analysis."

---

## Architectural Concepts

### Layer

**Definition:** Horizontal slice of architecture with specific responsibility (e.g., CUDA Abstraction, Memory Management, Scheduling).

**Contrast with Component:** Layers are architectural divisions. Components are implementation units.

**Usage:** "Each layer depends only on the layer below, preventing circular dependencies."

---

### Component

**Definition:** Cohesive unit of implementation with well-defined interface (e.g., Scheduler, Executor, MemoryPlanner).

**Contrast with Module:** Components are logical units. Modules are file-level units.

**Usage:** "The Scheduler component orchestrates execution requests."

---

### Abstraction

**Definition:** Interface that hides implementation details and exposes only essential operations.

**Contrast with Implementation:** Abstraction is interface. Implementation is realisation.

**Usage:** "CUDA streams are wrapped in RAII abstractions to enforce resource ownership."

---

### Invariant

**Definition:** Property that must always hold true (e.g., "graphs are immutable after construction").

**Contrast with Constraint:** Invariants are internal properties. Constraints are external requirements.

**Usage:** "The scheduler maintains the invariant that at most one request executes per stream."

---

## References

- **Architecture:** `docs/ARCHITECTURE.md`
- **Vision and Philosophy:** `docs/OUTLOOK.md`
- **Development Roadmap:** `docs/ROADMAP.md`
