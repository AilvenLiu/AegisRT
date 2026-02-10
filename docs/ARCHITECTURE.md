# AegisRT Architecture

## Overview

AegisRT is a layered edge AI runtime architecture designed for deterministic, explainable, and resource-constrained inference. The system decomposes into four primary layers, each with explicit responsibilities and minimal cross-layer coupling.

**Design Philosophy:** Separation of concerns, explicit resource ownership, and observability as first-class constraint.

---

## System Decomposition

### Layer 1: CUDA Runtime Abstraction

**Responsibility:** Wrap raw CUDA API in RAII objects with explicit ownership semantics.

**Components:**
- `CudaContext`: Device selection and initialisation
- `CudaStream`: RAII wrapper for `cudaStream_t`
- `CudaEvent`: RAII wrapper for `cudaEvent_t`
- `CUDA_CHECK` macro: Error checking for all CUDA API calls

**Key Invariants:**
- No raw CUDA handles escape this layer
- All resources managed via RAII (no manual cleanup)
- All CUDA calls checked for errors (no silent failures)

**Dependencies:** CUDA Runtime API only

---

### Layer 2: Memory Management

**Responsibility:** Provide explicit device memory allocation with deterministic lifetimes.

**Components:**
- `DeviceMemoryPool`: Allocates and tracks device memory
- `DeviceBuffer`: RAII wrapper for device memory allocation
- `MemoryPlanner`: Computes static memory allocation plan from tensor lifetimes
- `LifetimeAnalyser`: Computes tensor lifetimes from execution graph

**Key Invariants:**
- Memory allocation occurs only during graph construction (not during execution)
- Allocation failures are explicit (no hidden retry or compaction)
- Tensor lifetimes are statically analysable

**Dependencies:** Layer 1 (CUDA Runtime Abstraction)

---

### Layer 3: Execution Orchestration

**Responsibility:** Execute computation graphs on GPU with explicit dependency tracking.

**Components:**
- `ExecutionGraph`: Immutable DAG representation of computation
- `Executor`: Stateless kernel invocation layer
- `DependencyTracker`: Manages CUDA events for synchronisation
- `GraphCapture`: Captures execution as CUDA Graph (optional optimisation)

**Key Invariants:**
- Graphs are immutable after construction
- Execution is deterministic (same graph always executes identically)
- Dependencies are explicit (event-based synchronisation)

**Dependencies:** Layer 1 (CUDA Runtime Abstraction), Layer 2 (Memory Management)

---

### Layer 4: Scheduling Policy

**Responsibility:** Decide when and on which stream to execute submitted graphs.

**Components:**
- `Scheduler`: Central orchestration component
- `SchedulingPolicy`: Abstract interface for policies (FIFO, Priority, Fairness)
- `StreamPool`: Manages fixed pool of CUDA streams
- `OrchestrationLoop`: Main scheduling loop (runs on dedicated thread)

**Key Invariants:**
- Scheduling policy is explicit and swappable
- Stream allocation is deterministic
- Scheduling decisions are traceable

**Dependencies:** Layer 1 (CUDA Runtime Abstraction), Layer 2 (Memory Management), Layer 3 (Execution Orchestration)

---

## Component Diagram

```
+---------------------------------------------------------------+
|                    Layer 4: Scheduling Policy                 |
|                                                               |
|  +------------------+  +------------------+  +--------------+ |
|  | Scheduler        |  | SchedulingPolicy |  | StreamPool   | |
|  | (Orchestration)  |  | (FIFO, Priority) |  | (Fixed Pool) | |
|  +------------------+  +------------------+  +--------------+ |
+---------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------+
|                 Layer 3: Execution Orchestration              |
|                                                               |
|  +------------------+  +------------------+  +--------------+ |
|  | ExecutionGraph   |  | Executor         |  | Dependency   | |
|  | (Immutable DAG)  |  | (Stateless)      |  | Tracker      | |
|  +------------------+  +------------------+  +--------------+ |
+---------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------+
|                   Layer 2: Memory Management                  |
|                                                               |
|  +------------------+  +------------------+  +--------------+ |
|  | MemoryPlanner    |  | LifetimeAnalyser |  | DeviceBuffer | |
|  | (Static Plan)    |  | (Liveness)       |  | (RAII)       | |
|  +------------------+  +------------------+  +--------------+ |
+---------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------+
|                Layer 1: CUDA Runtime Abstraction              |
|                                                               |
|  +------------------+  +------------------+  +--------------+ |
|  | CudaContext      |  | CudaStream       |  | CudaEvent    | |
|  | (Device Init)    |  | (RAII Wrapper)   |  | (RAII)       | |
|  +------------------+  +------------------+  +--------------+ |
+---------------------------------------------------------------+
                              |
                              v
                    +---------------------+
                    | CUDA Runtime API    |
                    +---------------------+
```

---

## Data Flow

### Model Input to Execution Output

```
1. Model File (ONNX, Custom IR)
        |
        v
2. ModelLoader
        |
        v
3. ExecutionGraph (Immutable DAG)
        |
        v
4. LifetimeAnalyser
        |
        v
5. MemoryPlanner (Static Allocation Plan)
        |
        v
6. Scheduler (Submission Queue)
        |
        v
7. StreamPool (Allocate Stream)
        |
        v
8. Executor (Kernel Invocation)
        |
        v
9. DependencyTracker (Event Synchronisation)
        |
        v
10. Output Tensors (Device Memory)
        |
        v
11. Host Memory (cudaMemcpy)
```

**Key Observation:** Steps 1-5 occur once (graph construction). Steps 6-11 occur per invocation (execution).

---

## Control Flow

### Submission to Completion

```
1. User submits ExecutionRequest (graph, priority, deadline)
        |
        v
2. Scheduler enqueues request (bounded queue)
        |
        v
3. OrchestrationLoop dequeues request (policy-driven)
        |
        v
4. SchedulingPolicy selects next request (FIFO, Priority)
        |
        v
5. StreamPool allocates stream
        |
        v
6. MemoryPlanner binds tensors to memory offsets
        |
        v
7. Executor executes graph on stream
        |
        v
8. DependencyTracker inserts/waits events
        |
        v
9. Completion callback invoked
        |
        v
10. StreamPool releases stream
```

**Key Observation:** Scheduling decisions (steps 3-4) are decoupled from execution mechanics (steps 7-8).

---

## Key Abstractions

### ExecutionGraph

**Purpose:** Immutable DAG representation of computation.

**Interface:**
```cpp
class ExecutionGraph {
public:
    const std::vector<OperatorNode>& operators() const;
    const std::vector<TensorEdge>& tensors() const;
    const std::vector<size_t>& topologicalOrder() const;
    size_t memoryBudget() const;
};
```

**Invariants:**
- Constructed once, executed many times
- No runtime mutation
- Topological order is deterministic

---

### Scheduler

**Purpose:** Central orchestration component.

**Interface:**
```cpp
class Scheduler {
public:
    Scheduler(std::unique_ptr<SchedulingPolicy> policy,
              std::unique_ptr<StreamPool> streams);

    RequestID submit(const ExecutionGraph& graph,
                     Priority priority,
                     Deadline deadline);

    void start();  // Start orchestration loop
    void stop();   // Stop orchestration loop
};
```

**Invariants:**
- Policy is injected (not hardcoded)
- Submission is non-blocking
- Orchestration loop runs on dedicated thread

---

### MemoryPlanner

**Purpose:** Compute static memory allocation plan.

**Interface:**
```cpp
class MemoryPlanner {
public:
    AllocationPlan plan(const ExecutionGraph& graph);
};

struct AllocationPlan {
    std::unordered_map<TensorID, size_t> offsets;
    size_t peakMemory;
};
```

**Invariants:**
- Planning occurs at graph construction (not execution)
- Plan is deterministic (same graph produces same plan)
- Peak memory is statically computable

---

## Dependency Graph

```
ModelLoader
    |
    v
ExecutionGraph
    |
    +---> LifetimeAnalyser
    |           |
    |           v
    |     MemoryPlanner
    |           |
    +-----------|
                |
                v
            Scheduler
                |
                +---> SchedulingPolicy
                |
                +---> StreamPool
                |
                v
            Executor
                |
                +---> DependencyTracker
                |
                v
            CudaStream, CudaEvent
                |
                v
            CUDA Runtime API
```

**Key Observation:** Dependencies flow downward. No layer depends on layers above.

---

## Concurrency Model

### Thread Responsibilities

**Main Thread:**
- Model loading and graph construction
- Submission of execution requests
- Retrieval of results

**Orchestration Thread:**
- Dequeuing requests from submission queue
- Scheduling decisions (policy-driven)
- Stream allocation and release
- Dispatching execution to GPU

**GPU Threads:**
- Kernel execution (asynchronous)
- Event-based synchronisation

**Key Invariants:**
- No shared mutable state between threads (except submission queue, which is thread-safe)
- GPU execution is asynchronous (no CPU-GPU synchronisation unless required)

---

## Error Propagation

### Error Categories

**Transient Errors:**
- Temporary resource exhaustion (e.g., stream pool full)
- Caller can retry

**Permanent Errors:**
- Invalid graph (e.g., unsupported operator)
- Caller must abort

**Fatal Errors:**
- CUDA driver failure (e.g., device lost)
- Process must terminate

### Error Handling Pattern

```cpp
enum class ErrorCategory { Transient, Permanent, Fatal };

struct Error {
    ErrorCategory category;
    std::string message;
    std::string context;  // File, line, function
};

// Example usage
Result<RequestID> Scheduler::submit(const ExecutionGraph& graph) {
    if (queue_.full()) {
        return Error{ErrorCategory::Transient, "Queue full", __CONTEXT__};
    }
    if (!graph.isValid()) {
        return Error{ErrorCategory::Permanent, "Invalid graph", __CONTEXT__};
    }
    // Submit...
}
```

---

## Observability

### Tracing Points

**Scheduling Decisions:**
- Request submission (ID, priority, deadline)
- Policy selection (which request chosen, why)
- Stream allocation (which stream assigned)

**Execution Events:**
- Kernel launch (operator ID, stream ID, timestamp)
- Event record/wait (tensor ID, event ID, timestamp)
- Completion (request ID, latency, timestamp)

**Memory Events:**
- Allocation (tensor ID, size, offset, timestamp)
- Deallocation (tensor ID, timestamp)

### Trace Format

Structured logs (JSON) suitable for offline analysis:

```json
{
  "event": "schedule_decision",
  "request_id": "req-123",
  "policy": "Priority",
  "priority": 10,
  "queue_depth": 5,
  "stream_id": "stream-2",
  "timestamp": 1234567890
}
```

---

## Performance Characteristics

### Time Complexity

**Graph Construction:** O(n log n) where n = number of operators (topological sort)

**Memory Planning:** O(n^2) where n = number of tensors (bin-packing)

**Execution:** O(n) where n = number of operators (sequential kernel launches)

**Scheduling:** O(log n) where n = queue depth (priority queue)

### Space Complexity

**Graph Representation:** O(n + m) where n = operators, m = tensors

**Memory Plan:** O(n) where n = number of tensors

**Execution State:** O(1) (stateless executor)

---

## Design Trade-offs

### Static vs Dynamic Graphs

**Choice:** Static graphs only.

**Trade-off:** Flexibility sacrificed for determinism and predictability.

**Rationale:** Edge systems prioritise predictability over flexibility. Dynamic graphs require runtime recompilation, adding latency and complexity.

### Fail-Fast vs Graceful Degradation

**Choice:** Fail-fast on resource exhaustion.

**Trade-off:** Robustness sacrificed for predictability.

**Rationale:** Hidden retry and eviction policies are non-deterministic. Explicit failures enable caller to decide retry strategy.

### Policy Abstraction vs Hardcoded Policy

**Choice:** Policy abstraction (injected, not hardcoded).

**Trade-off:** Complexity increased for flexibility.

**Rationale:** Scheduling policy is domain-specific. Abstraction enables experimentation and A/B testing without modifying core system.

---

## References

- **Vision and Philosophy:** `docs/OUTLOOK.md`
- **Development Roadmap:** `docs/ROADMAP.md`
- **Terminology:** `docs/TERMINOLOGY.md`
- **Agent Constraints:** `CLAUDE.md`
- **Contribution Standards:** `CONTRIBUTING.md`
