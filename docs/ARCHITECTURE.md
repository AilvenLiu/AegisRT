# AegisRT Architecture

## Overview

AegisRT is a **GPU resource orchestrator** for deterministic, multi-model edge AI inference. It sits above existing inference runtimes (TensorRT, TVM Runtime, ONNX Runtime) and manages the GPU resources they compete for: CUDA streams, device memory, compute time, and scheduling priority.

The system decomposes into five primary layers, each with explicit responsibilities and minimal cross-layer coupling.

**Design Philosophy:** Separation of concerns, explicit resource ownership, real-time scheduling theory, and observability as first-class constraint.

**Key Distinction:** AegisRT does not execute kernels or compile models. It orchestrates existing runtimes, providing deterministic scheduling, memory management, context isolation, and observability that no individual runtime provides on its own.

---

## System Position

```
+---------------------------------------------------------------+
|                    Application Layer                           |
|  (Autonomous Driving Pipeline, Robotics, Edge AI Services)    |
+---------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------+
|                    AegisRT Orchestrator                        |
|  Layer 5: Observability (tracing, metrics, audit)             |
|  Layer 4: Scheduling Policy (RMS, EDF, admission control)     |
|  Layer 3: Cross-Model Memory Orchestration                    |
|  Layer 2: Execution Context Isolation                         |
|  Layer 1: CUDA Runtime Abstraction                            |
+---------------------------------------------------------------+
                         |         |
                    +----v---+ +---v------+
                    |  TRT   | | TVM RT   |  <-- Existing Runtimes
                    +--------+ +----------+
                         |         |
                    +----v---------v-----+
                    |   CUDA Runtime     |
                    +--------------------+
```

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

### Layer 2: Execution Context Isolation

**Responsibility:** Provide per-model execution contexts with hard resource budgets and fault isolation.

**Components:**
- `ExecutionContext`: Per-model resource container (streams, memory budget, fault boundary)
- `ResourceBudget`: Hard limits on memory, stream count, and compute time per model
- `FaultBoundary`: Isolates CUDA errors to the originating context
- `RuntimeBackend`: Abstract interface wrapping existing runtimes (TensorRT, TVM, etc.)

**Key Invariants:**
- Each model executes within its own context (no shared mutable state)
- Resource budgets are hard limits (exceeding triggers explicit rejection)
- One model's failure does not propagate to other models
- Runtime backends are opaque (AegisRT controls resources, not execution internals)

**Dependencies:** Layer 1 (CUDA Runtime Abstraction)

---

### Layer 3: Cross-Model Memory Orchestration

**Responsibility:** Manage device memory across multiple models with lifetime-aware sharing and explicit pressure handling.

**Components:**
- `MemoryOrchestrator`: Coordinates memory allocation across all execution contexts
- `LifetimeAnalyser`: Computes tensor lifetimes and identifies sharing opportunities
- `MemoryPlanner`: Computes static memory allocation plans from cross-model analysis
- `PressureHandler`: Explicit policies for memory pressure (shed, reject, compact)

**Key Invariants:**
- Memory allocation occurs only during planning (not during execution)
- Cross-model sharing is computed statically (no runtime heuristics)
- Pressure handling uses explicit policies (no hidden eviction)
- Peak memory usage is statically computable and bounded

**Dependencies:** Layer 1 (CUDA Runtime Abstraction), Layer 2 (Execution Context Isolation)

---

### Layer 4: Scheduling Policy (Core Differentiator)

**Responsibility:** Decide when and with what resources to execute submitted workloads, grounded in real-time scheduling theory.

**Components:**
- `Scheduler`: Central orchestration component with admission control
- `SchedulingPolicy`: Abstract interface for RT policies (RMS, EDF, Priority)
- `AdmissionController`: Determines if a new model can be admitted without violating existing guarantees
- `WCETProfiler`: Maintains worst-case execution time profiles per model
- `StreamPool`: Manages fixed pool of CUDA streams
- `OrchestrationLoop`: Main scheduling loop (runs on dedicated thread)

**Key Invariants:**
- Scheduling policy is explicit, swappable, and grounded in RT theory
- Admission control runs schedulability analysis before accepting new workloads
- WCET bounds are maintained and updated via profiling
- All scheduling decisions are traceable with rationale
- Stream allocation is deterministic

**Dependencies:** Layer 1 (CUDA Runtime Abstraction), Layer 2 (Execution Context Isolation), Layer 3 (Cross-Model Memory Orchestration)

---

### Layer 5: Observability

**Responsibility:** Provide full traceability of all scheduling decisions, resource allocations, and execution events.

**Components:**
- `TraceCollector`: Structured event collection for all layers
- `MetricsAggregator`: Per-model, per-stream, per-pool resource utilisation
- `AuditTrail`: Scheduling decision log with rationale (why was model X delayed?)
- `ExportAdapter`: Integration with standard tooling (Perfetto, NVIDIA Nsight)

**Key Invariants:**
- Every scheduling decision is logged with sufficient context for reconstruction
- Tracing does not affect execution correctness (only performance)
- Metrics are exportable in standard formats
- Observability is not optional -- it is part of the system contract

**Dependencies:** All layers (cross-cutting concern)

---

## Component Diagram

```
+---------------------------------------------------------------+
|                    Layer 5: Observability                      |
|                                                               |
|  +------------------+  +------------------+  +--------------+ |
|  | TraceCollector   |  | MetricsAggregator|  | AuditTrail   | |
|  | (Structured)     |  | (Per-Model)      |  | (Decisions)  | |
|  +------------------+  +------------------+  +--------------+ |
+---------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------+
|              Layer 4: Scheduling Policy (RT Theory)           |
|                                                               |
|  +------------------+  +------------------+  +--------------+ |
|  | Scheduler        |  | SchedulingPolicy |  | Admission    | |
|  | (Orchestration)  |  | (RMS, EDF)       |  | Controller   | |
|  +------------------+  +------------------+  +--------------+ |
|  +------------------+  +------------------+                   |
|  | WCETProfiler     |  | StreamPool       |                   |
|  | (Profiling)      |  | (Fixed Pool)     |                   |
|  +------------------+  +------------------+                   |
+---------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------+
|           Layer 3: Cross-Model Memory Orchestration           |
|                                                               |
|  +------------------+  +------------------+  +--------------+ |
|  | MemoryOrchestrator| | LifetimeAnalyser |  | Pressure     | |
|  | (Cross-Model)    |  | (Sharing)        |  | Handler      | |
|  +------------------+  +------------------+  +--------------+ |
+---------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------+
|             Layer 2: Execution Context Isolation              |
|                                                               |
|  +------------------+  +------------------+  +--------------+ |
|  | ExecutionContext  |  | ResourceBudget   |  | Runtime      | |
|  | (Per-Model)      |  | (Hard Limits)    |  | Backend      | |
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
|  +------------------+  +------------------+                   |
|  | DeviceMemoryPool |  | DeviceBuffer     |                   |
|  | (Allocation)     |  | (RAII Wrapper)   |                   |
|  +------------------+  +------------------+                   |
+---------------------------------------------------------------+
                              |
                              v
              +-------------------------------+
              | Existing Runtimes             |
              | (TensorRT, TVM RT, ONNX RT)   |
              +-------------------------------+
                              |
                              v
                    +---------------------+
                    | CUDA Runtime API    |
                    +---------------------+
```

---

## Data Flow

### Model Registration to Execution

```
1. Pre-Compiled Model (TensorRT engine, TVM module, etc.)
        |
        v
2. RuntimeBackend wraps model (opaque execution handle)
        |
        v
3. ExecutionContext created (resource budget, fault boundary)
        |
        v
4. WCETProfiler profiles model (execution time distribution)
        |
        v
5. AdmissionController checks schedulability (can this model be added?)
        |
        v
6. MemoryOrchestrator plans cross-model memory allocation
        |
        v
7. Scheduler admits model to scheduling pool
```

### Per-Invocation Execution

```
1. Execution request arrives (model ID, priority, deadline)
        |
        v
2. Scheduler selects next request (policy-driven: RMS, EDF)
        |
        v
3. StreamPool allocates stream within context budget
        |
        v
4. MemoryOrchestrator binds memory within context budget
        |
        v
5. RuntimeBackend executes model on allocated resources
        |
        v
6. TraceCollector records execution events
        |
        v
7. Completion callback invoked, stream released
```

**Key Observation:** Model registration (expensive, once) is separated from per-invocation execution (cheap, repeated). Admission control ensures that once a model is admitted, its latency guarantees are maintained.

---

## Control Flow

### Submission to Completion

```
1. User submits ExecutionRequest (model ID, priority, deadline)
        |
        v
2. Scheduler enqueues request (bounded queue, per-context)
        |
        v
3. OrchestrationLoop dequeues request (RT policy-driven)
        |
        v
4. SchedulingPolicy selects next request (RMS, EDF, Priority)
        |
        v
5. AdmissionController verifies schedulability (if new model)
        |
        v
6. StreamPool allocates stream within context budget
        |
        v
7. MemoryOrchestrator binds memory within context budget
        |
        v
8. RuntimeBackend executes model on allocated resources
        |
        v
9. TraceCollector records events (scheduling decision, execution, memory)
        |
        v
10. Completion callback invoked
        |
        v
11. StreamPool releases stream, context budget updated
```

**Key Observation:** Scheduling decisions (steps 3-5) are decoupled from execution mechanics (step 8). The RuntimeBackend is opaque -- AegisRT controls resources, not execution internals.

---

## Key Abstractions

### ExecutionContext

**Purpose:** Per-model resource container with hard budgets and fault isolation.

**Interface:**
```cpp
class ExecutionContext {
public:
    ExecutionContext(ResourceBudget budget,
                    std::unique_ptr<RuntimeBackend> backend);

    const ResourceBudget& budget() const;
    RuntimeBackend& backend();
    bool withinBudget(size_t memoryRequest, size_t streamRequest) const;
};
```

**Invariants:**
- Each model has exactly one context
- Resource budgets are immutable after creation
- Faults are scoped to the originating context

---

### Scheduler

**Purpose:** Central orchestration component with RT scheduling and admission control.

**Interface:**
```cpp
class Scheduler {
public:
    Scheduler(std::unique_ptr<SchedulingPolicy> policy,
              std::unique_ptr<StreamPool> streams,
              std::unique_ptr<AdmissionController> admission);

    Result<ModelID> admit(std::unique_ptr<ExecutionContext> context,
                          WCETProfile profile,
                          Deadline deadline);

    RequestID submit(ModelID model,
                     Priority priority);

    void start();  // Start orchestration loop
    void stop();   // Stop orchestration loop
};
```

**Invariants:**
- Admission control runs schedulability analysis before accepting
- Policy is injected (not hardcoded)
- Submission is non-blocking
- Orchestration loop runs on dedicated thread

---

### MemoryOrchestrator

**Purpose:** Cross-model memory coordination with lifetime-aware sharing.

**Interface:**
```cpp
class MemoryOrchestrator {
public:
    AllocationPlan plan(const std::vector<ExecutionContext*>& contexts);
    Result<MemoryBinding> bind(ModelID model, const AllocationPlan& plan);
    void handlePressure(PressurePolicy policy);
};

struct AllocationPlan {
    std::unordered_map<ModelID, std::vector<MemoryRegion>> allocations;
    std::vector<SharingOpportunity> sharing;
    size_t peakMemory;
};
```

**Invariants:**
- Planning occurs at admission time (not execution time)
- Plan is deterministic (same inputs produce same plan)
- Peak memory is statically computable
- Sharing opportunities are identified and exploited automatically

---

## Dependency Graph

```
TraceCollector, MetricsAggregator, AuditTrail  (Layer 5 - cross-cutting)
    |
    v (observes all layers)

Scheduler
    |
    +---> SchedulingPolicy (RMS, EDF, Priority)
    |
    +---> AdmissionController
    |           |
    |           +---> WCETProfiler
    |
    +---> StreamPool
    |
    v
MemoryOrchestrator
    |
    +---> LifetimeAnalyser
    |
    +---> PressureHandler
    |
    v
ExecutionContext
    |
    +---> ResourceBudget
    |
    +---> RuntimeBackend (TensorRT, TVM, ONNX RT)
    |
    +---> FaultBoundary
    |
    v
CudaContext, CudaStream, CudaEvent, DeviceMemoryPool
    |
    v
CUDA Runtime API
```

**Key Observation:** Dependencies flow downward. No layer depends on layers above. Existing runtimes are wrapped as opaque backends at Layer 2.

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
- Model admission (model ID, WCET profile, schedulability result)
- Request submission (request ID, model ID, priority, deadline)
- Policy selection (which request chosen, why, alternative considered)
- Stream allocation (which stream assigned, within which context)

**Execution Events:**
- RuntimeBackend invocation (model ID, stream ID, timestamp)
- Completion (request ID, actual latency vs deadline, timestamp)
- Deadline miss (request ID, expected vs actual, severity)

**Memory Events:**
- Context allocation (model ID, budget, actual usage)
- Cross-model sharing (which models share, memory saved)
- Pressure event (trigger, policy applied, models affected)

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

**Model Admission:** O(n) where n = number of admitted models (schedulability analysis)

**Memory Planning:** O(n * m) where n = models, m = memory regions (cross-model sharing)

**Per-Invocation Scheduling:** O(log n) where n = queue depth (priority queue / EDF)

**Execution Dispatch:** O(1) (delegate to RuntimeBackend)

### Space Complexity

**Per-Model State:** O(1) (execution context, WCET profile, resource budget)

**Memory Plan:** O(n * m) where n = models, m = memory regions

**Scheduling State:** O(n) where n = admitted models (WCET profiles, deadlines)

**Trace Buffer:** O(k) where k = configurable trace buffer size

---

## Design Trade-offs

### Orchestration vs Execution

**Choice:** Orchestrate existing runtimes rather than implementing kernel execution.

**Trade-off:** Cannot optimise kernel-level execution, but avoids competing with mature runtimes.

**Rationale:** TensorRT, TVM, and ONNX Runtime are excellent at kernel execution. The unsolved problem is multi-model orchestration under resource constraints. Focus on the gap, not the solved problem.

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
