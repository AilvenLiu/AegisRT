# AegisRT Architecture

## Design Philosophy

AegisRT is built on three core principles that differentiate it from existing runtimes:

### Principle 1: Orchestration Over Execution

AegisRT does not execute kernels. It orchestrates existing runtimes (TensorRT, TVM, ONNX Runtime) by controlling **when** and **with what resources** they execute. This separation enables:

- Independent evolution of scheduling policy without touching execution
- Leveraging mature runtimes for what they do best (kernel optimisation)
- Focusing development effort on the unsolved problem (deterministic orchestration)

### Principle 2: Formal Guarantees Over Best-Effort

AegisRT provides **provable** latency bounds through:

- Formal schedulability analysis derived from real-time systems theory
- Conservative WCET (Worst-Case Execution Time) estimation with contention awareness
- Admission control that rejects work rather than degrading silently

### Principle 3: Observability as a Contract

Every scheduling decision, resource allocation, and execution event is traceable. Without observability, determinism claims cannot be verified. Tracing is not optional—it is part of the system contract.

---

## System Position in the Stack

```
+---------------------------------------------------------------------+
|                        Application Layer                            |
|        (Autonomous Driving Pipeline, Robotics, Edge AI Services)    |
+-------------------------------+-------------------------------------+
                                |
+-------------------------------v-------------------------------------+
|                           AegisRT                                   |
|                                                                     |
|  +---------------------------------------------------------------+  |
|  |               Layer 3: Deterministic Scheduler                |  |
|  |                                                               |  |
|  |  +-------------+ +-------------+ +-------------------------+  |  |
|  |  | WCETProfiler| | Admission   | | SchedulingPolicy        |  |  |
|  |  |             | | Controller  | | (FIFO, RMS, EDF, Custom)|  |  |
|  |  | Profiling   | | Formal      | |                         |  |  |
|  |  | Statistical | | Analysis    | | Priority/Deadline-based |  |  |
|  |  +-------------+ +-------------+ +-------------------------+  |  |
|  +-----------------------------+---------------------------------+  |
|                                |                                    |
|  +-----------------------------v---------------------------------+  |
|  |             Layer 2: Resource Orchestration                   |  |
|  |                                                               |  |
|  |  +-------------+ +-------------+ +-------------------------+  |  |
|  |  | Memory      | | Execution   | |    RuntimeBackend       |  |  |
|  |  | Orchestrator| | Context     | | (Abstract Interface)    |  |  |
|  |  |             | |             | |                         |  |  |
|  |  | Lifetime-   | | Per-Model   | | +-----+ +-----+ +-----+ |  |  |
|  |  | Aware       | | Budgets &   | | | TRT | | TVM | |ONNX | |  |  |
|  |  | Sharing     | | Isolation   | | +-----+ +-----+ +-----+ |  |  |
|  |  +-------------+ +-------------+ +-------------------------+  |  |
|  +-----------------------------+---------------------------------+  |
|                                |                                    |
|  +-----------------------------v---------------------------------+  |
|  |          Layer 1: CUDA Abstraction & Observability            |  |
|  |                                                               |  |
|  |   +-------------+ +-------------+ +------------------------+  |  |
|  |   | CUDA RAII   | | Device      | | TraceCollector         |  |  |
|  |   | Wrappers    | | Capability  | |                        |  |  |
|  |   |             | | Discovery   | | Structured Event Logs  |  |  |
|  |   | Stream      | |             | | Decision Rationale     |  |  |
|  |   | Event       | | SM Count    | | Performance Metrics    |  |  |
|  |   | Memory      | | Memory BW   | |                        |  |  |
|  |   +-------------+ +-------------+ +------------------------+  |  |
|  +---------------------------------------------------------------+  |
+--------------------------------+-----------------------------------+
                                 |
                     +-----------v-----------+
                     |    CUDA Runtime API   |
                     +-----------------------+
```

---

## Layer 1: CUDA Abstraction & Observability

### Purpose

Provide safe, RAII-managed access to CUDA resources with full traceability of all operations.

### Components

#### CUDA RAII Wrappers

```cpp
// All CUDA resources are managed through RAII
class CudaStream {
public:
    CudaStream();                           // Creates stream
    ~CudaStream();                          // Destroys stream
    CudaStream(CudaStream&&) noexcept;      // Move-only
    CudaStream& operator=(CudaStream&&);
    
    cudaStream_t handle() const;
    void synchronize();
    
private:
    cudaStream_t stream_;
};

class CudaEvent {
public:
    CudaEvent(unsigned int flags = 0);
    ~CudaEvent();
    
    void record(const CudaStream& stream);
    void synchronize();
    float elapsedSince(const CudaEvent& start) const;
    
private:
    cudaEvent_t event_;
};
```

#### Device Capability Discovery

```cpp
struct DeviceCapabilities {
    int sm_count;
    size_t total_memory;
    size_t shared_memory_per_block;
    int max_threads_per_block;
    int compute_capability_major;
    int compute_capability_minor;
    int memory_clock_khz;
    int memory_bus_width;
    
    // Derived metrics
    size_t theoretical_memory_bandwidth() const;
    int theoretical_flops() const;
};

class DeviceContext {
public:
    static DeviceCapabilities query(int device_id = 0);
    static std::vector<DeviceCapabilities> query_all();
};
```

#### TraceCollector

```cpp
struct TraceEvent {
    std::string event_type;      // "schedule", "execute", "allocate", etc.
    std::string component;       // "scheduler", "memory", "context"
    std::string rationale;       // Why this decision was made
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

### Invariants

- **No raw CUDA handles escape this layer** — All resources managed through RAII wrappers
- **All CUDA calls are checked** — No silent failures; errors propagate with context
- **All operations are traceable** — TraceCollector receives events from all components

---

## Layer 2: Resource Orchestration

### Purpose

Manage execution contexts, memory allocation, and runtime backends with clear ownership and isolation.

### Components

#### Execution Context

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
    
    ModelID model() const;
    const ResourceBudget& budget() const;
    RuntimeBackend& backend();
    
    // Resource tracking
    size_t memory_used() const;
    int streams_in_use() const;
    bool can_allocate(size_t bytes) const;
    
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

#### Memory Orchestrator

```cpp
struct TensorLifetime {
    TensorID tensor;
    Timestamp start;    // When tensor is first written
    Timestamp end;      // When tensor is last read
    size_t size;
};

struct SharingOpportunity {
    TensorID tensor_a;
    TensorID tensor_b;
    size_t shared_memory;  // Memory saved by sharing
};

class MemoryOrchestrator {
public:
    // Planning phase (occurs at model admission)
    AllocationPlan plan(
        const std::vector<ExecutionContext*>& contexts,
        const std::vector<TensorLifetime>& lifetimes
    );
    
    // Execution phase
    Result<MemoryBinding> bind(
        ModelID model,
        const AllocationPlan& plan
    );
    
    void release(ModelID model);
    
    // Pressure handling
    void handle_pressure(PressurePolicy policy);
    
    // Observability
    size_t peak_usage() const;
    size_t current_usage() const;

private:
    std::unordered_map<ModelID, MemoryBinding> bindings_;
    std::unique_ptr<DeviceMemoryPool> pool_;
};
```

#### Runtime Backend (Abstract Interface)

```cpp
class RuntimeBackend {
public:
    virtual ~RuntimeBackend() = default;
    
    // Resource estimation
    virtual size_t estimate_memory() const = 0;
    virtual std::vector<TensorLifetime> estimate_lifetimes() const = 0;
    
    // Execution
    virtual Result<void> execute(
        const CudaStream& stream,
        const MemoryBinding& memory
    ) = 0;
    
    // Status
    virtual bool is_valid() const = 0;
    virtual std::string runtime_name() const = 0;
};

// Concrete implementations
class TensorRTBackend : public RuntimeBackend { /* ... */ };
class TVMBackend : public RuntimeBackend { /* ... */ };
class ONNXBackend : public RuntimeBackend { /* ... */ };
```

### Invariants

- **Each model has exactly one execution context** — No shared mutable state between models
- **Resource budgets are hard limits** — Exceeding triggers explicit rejection, not silent degradation
- **One model's failure does not propagate** — Fault boundaries isolate errors
- **Memory allocation occurs at planning time** — No runtime allocation surprises

---

## Layer 3: Deterministic Scheduler (Core Contribution)

### Purpose

Provide real-time scheduling with formal admission control, adapting classical RT theory for GPU execution constraints.

### The GPU Scheduling Challenge

Classical real-time scheduling makes assumptions that GPUs violate:

```
+---------------------------------------------------------------------+
|                    Classical RT Assumptions                         |
+---------------------------------------------------------------------+
| 1. Tasks are preemptible           -> GPUs: Kernels are NOT         |
| 2. WCET is static and known        -> GPUs: Varies with contention  |
| 3. Single sequential processor     -> GPUs: Massively parallel      |
| 4. Tasks are independent           -> GPUs: Share memory bandwidth  |
+---------------------------------------------------------------------+
```

AegisRT addresses each violation:

| Violation | AegisRT Adaptation |
|-----------|-------------------|
| Non-preemptible kernels | Non-preemptive EDF scheduling analysis |
| Variable WCET | Conservative statistical profiling with safety margins |
| Parallel execution | Model sequentialisation with resource isolation |
| Shared resources | Contention-aware WCET estimation |

### Components

#### WCET Profiler

```cpp
struct WCETProfile {
    Duration worst_case;       // Conservative upper bound
    Duration average_case;     // For capacity planning
    Duration best_case;        // Lower bound
    double contention_factor;  // How much slower under load
    int sample_count;          // Statistical confidence
};

class WCETProfiler {
public:
    // Profiling
    WCETProfile profile(
        ExecutionContext& context,
        int min_samples = 100,
        double confidence_level = 0.99
    );
    
    // Contention-aware profiling
    WCETProfile profile_under_load(
        ExecutionContext& context,
        const std::vector<ExecutionContext*>& background_contexts
    );
    
    // Statistical methods
    Duration compute_wcet(
        const std::vector<Duration>& samples,
        double confidence_level,
        double safety_margin = 1.5  // 50% safety margin
    );
};
```

#### Admission Controller

```cpp
struct AdmissionRequest {
    ModelID model;
    WCETProfile wcet;
    Period period;          // For periodic tasks
    Deadline deadline;      // Relative deadline
    Priority priority;      // For priority-based scheduling
};

struct AdmissionResult {
    bool admitted;
    std::string reason;     // If rejected, why
    double utilisation;     // System utilisation after admission
    Duration worst_case_response_time;  // Predicted WCRT
};

class AdmissionController {
public:
    AdmissionResult analyse(
        const AdmissionRequest& request,
        const std::vector<ExecutionContext*>& existing_contexts
    );
    
    // Schedulability tests
    bool test_rms_schedulability(
        const std::vector<std::pair<Duration, Period>>& tasks
    );
    
    bool test_edf_schedulability(
        const std::vector<std::pair<Duration, Deadline>>& tasks
    );
    
    // Non-preemptive analysis
    Duration compute_blocking_time(
        const std::vector<Duration>& all_wcets,
        Duration new_task_wcet
    );
};
```

#### Scheduling Policy

```cpp
struct ExecutionRequest {
    ModelID model;
    Priority priority;
    Deadline deadline;
    std::optional<Timestamp> earliest_start;
};

class SchedulingPolicy {
public:
    virtual ~SchedulingPolicy() = default;
    
    virtual void add_model(ModelID model, const WCETProfile& profile) = 0;
    virtual void remove_model(ModelID model) = 0;
    
    virtual std::optional<ExecutionRequest> select_next(
        const std::vector<ExecutionRequest>& pending
    ) = 0;
    
    virtual std::string policy_name() const = 0;
};

// FIFO baseline
class FIFOPolicy : public SchedulingPolicy { /* ... */ };

// Rate-Monotonic Scheduling
class RMSPolicy : public SchedulingPolicy { /* ... */ };

// Earliest Deadline First
class EDFPolicy : public SchedulingPolicy { /* ... */ };

// Priority-based
class PriorityPolicy : public SchedulingPolicy { /* ... */ };
```

#### Scheduler (Orchestration)

```cpp
class Scheduler {
public:
    Scheduler(
        std::unique_ptr<SchedulingPolicy> policy,
        std::unique_ptr<AdmissionController> admission,
        std::unique_ptr<StreamPool> streams,
        TraceCollector& tracer
    );
    
    // Model management
    Result<ModelID> admit(
        std::unique_ptr<ExecutionContext> context,
        const AdmissionRequest& request
    );
    
    void evict(ModelID model);
    
    // Execution
    RequestID submit(
        ModelID model,
        Priority priority = Priority::normal(),
        std::optional<Deadline> deadline = std::nullopt
    );
    
    // Lifecycle
    void start();   // Start orchestration loop
    void stop();    // Stop orchestration loop
    
    // Observability
    SchedulerMetrics metrics() const;

private:
    void orchestration_loop_();
    
    std::unique_ptr<SchedulingPolicy> policy_;
    std::unique_ptr<AdmissionController> admission_;
    std::unique_ptr<StreamPool> streams_;
    std::unique_ptr<MemoryOrchestrator> memory_;
    TraceCollector& tracer_;
    
    ConcurrentQueue<ExecutionRequest> pending_;
    std::unordered_map<ModelID, ExecutionContext> contexts_;
};
```

### Schedulability Analysis

#### Rate-Monotonic Scheduling (RMS)

For periodic tasks with implicit deadlines (deadline = period):

Utilisation Bound: $U \leq n (2^{\frac{1}{n}} - 1)$.

Where:
- $U = \sum{\frac{C_i}{T_i}}$ for all tasks
- $C_i$ = Worst-case execution time
- $T_i$ = Period
- $n$ = Number of tasks

For $n \rightarrow \infty : U \leq ln(2) \approx 0.693$

#### Earliest Deadline First (EDF)

For tasks with arbitrary deadlines:


Utilisation Bound: $U = \sum{\frac{C_i}{D_i}} \leq 1$

Where:
- $D_i$ = Relative deadline

EDF can achieve 100% utilisation, but requires:
- Preemption (not available on GPU)
- Accurate WCET bounds

#### Non-Preemptive Adaptation

For non-preemptive GPU execution, we must account for **blocking time**:


Blocking Time: $B_i = \max(C_j)$ for all $j$ with lower priority

Response Time Analysis:
$R_i = C_i + B_i + \sum{\frac{R_i}{T_j}} \cdot C_j$ (for all $j$ with higher priority)

Iterate until $R_i$ converges or $R_i > D_i$ (deadline miss)


---

## Data Flow

### Model Admission Flow

```
1. Load pre-compiled model (TensorRT engine, TVM module, etc.)
        |
        v
2. Create RuntimeBackend wrapper
        |
        v
3. WCETProfiler profiles model under various contention levels
        |
        v
4. Construct AdmissionRequest with WCET profile and timing requirements
        |
        v
5. AdmissionController runs schedulability analysis
        |
        +-- Rejected -> Return reason to caller
        |
        v
6. Create ExecutionContext with resource budget
        |
        v
7. MemoryOrchestrator plans cross-model memory allocation
        |
        v
8. Scheduler admits model to scheduling pool
        |
        v
9. TraceCollector records admission event with rationale
```

### Per-Invocation Execution Flow

```
1. Application submits ExecutionRequest (model, priority, deadline)
        |
        v
2. Scheduler enqueues request in pending queue
        |
        v
3. OrchestrationLoop dequeues request (blocking)
        |
        v
4. SchedulingPolicy selects next request to execute
        |
        v
5. StreamPool allocates stream within context budget
        |
        v
6. MemoryOrchestrator binds memory within context budget
        |
        v
7. RuntimeBackend.execute() called with stream and memory
        |
        v
8. TraceCollector records execution event
        |
        v
9. Completion callback invoked
        |
        v
10. StreamPool releases stream
        |
        v
11. TraceCollector records completion event
```

---

## Concurrency Model

### Thread Architecture

```
+-------------------------------------------+
| Main Thread                               |
|  - Model admission/eviction               |
|  - Execution request submission           |
|  - Result retrieval                       |
+-------------------------------------------+
                    |
                    | (thread-safe queue)
                    v
+-------------------------------------------+
| Orchestration Thread                      |
|  - Dequeue pending requests               |
|  - Scheduling policy evaluation           |
|  - Stream allocation                      |
|  - Execution dispatch                     |
|  - Completion handling                    |
+-------------------------------------------+
                    |
                    | (async CUDA operations)
                    v
+-------------------------------------------+
| GPU Execution                             |
|  - Kernel execution (asynchronous)        |
|  - Memory transfers (asynchronous)        |
|  - Event-based synchronisation            |
+-------------------------------------------+
```

### Synchronisation Strategy

- **Submission Queue**: Lock-free MPMC queue for pending requests
- **GPU Synchronisation**: CUDA events, not CPU barriers
- **State Access**: Atomic operations for counters, mutex for complex state
- **No Shared Mutable State**: Each thread owns its state; communication via queues

---

## Error Propagation

### Error Categories

| Category | Example | Recovery |
|----------|---------|----------|
| Transient | Stream pool exhausted | Retry with backoff |
| Permanent | Invalid model format | Abort and report |
| Fatal | CUDA device lost | Process termination |

### Error Handling Pattern

```cpp
template<typename T>
class Result {
public:
    bool is_ok() const;
    bool is_error() const;
    const T& value() const;
    const Error& error() const;
    
private:
    std::variant<T, Error> data_;
};

struct Error {
    ErrorCategory category;
    std::string message;
    std::string component;
    std::string context;  // File:line:function
};
```

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Model Admission | O(n) | n = number of admitted models |
| Memory Planning | O(n × m) | n = models, m = tensors |
| Per-Invocation Scheduling | O(log n) | n = queue depth |
| Execution Dispatch | O(1) | Delegates to RuntimeBackend |

### Space Complexity

| Storage | Complexity | Notes |
|---------|------------|-------|
| Per-Model State | O(1) | Context, WCET profile, budget |
| Memory Plan | O(n × m) | n = models, m = tensors |
| Trace Buffer | O(k) | k = configurable buffer size |

---

## Design Trade-offs

### Trade-off 1: Determinism vs Throughput

**Decision**: Prioritise deterministic latency over peak throughput.

**Rationale**: Edge autonomous systems value predictability. A 10% throughput reduction is acceptable if it eliminates tail latency spikes.

**Manifestation**: Bounded queue depths, explicit admission control, non-work-conserving scheduler.

### Trade-off 2: Static vs Dynamic Graphs

**Decision**: Support only static execution graphs.

**Rationale**: Dynamic graphs require runtime recompilation, which destroys latency predictability. Static graphs enable ahead-of-time memory planning and WCET estimation.

**Manifestation**: No support for dynamic control flow; variable-length inputs require padding or multiple graph variants.

### Trade-off 3: Admission vs Graceful Degradation

**Decision**: Fail-fast on resource exhaustion.

**Rationale**: Hidden degradation is non-deterministic. Explicit rejection enables the caller to implement application-specific fallback strategies.

**Manifestation**: Admission control rejects models that would violate guarantees; no background eviction or compaction.

---

## References

### Academic Papers

- Liu, C. L., & Layland, J. W. (1973). Scheduling algorithms for multiprogramming in a hard-real-time environment. *Journal of the ACM*.
- George, L., Rivierre, N., & Spuri, M. (1996). Preemptive and non-preemptive real-time uniprocessor scheduling. *INRIA Research Report*.
- Zhang, Q., et al. (REEF, SOSP'23). Reactive GPU execution for real-time AI services.
- Gujarati, A., et al. (Clockwork, OSDI'20). Serving DNNs in real-time at datacenter scale.

### Related Projects

- TensorRT: https://developer.nvidia.com/tensorrt
- TVM: https://tvm.apache.org/
- ONNX Runtime: https://onnxruntime.ai/
- PREEMPT-RT Linux: https://wiki.linuxfoundation.org/realtime/

---

## Document References

- **Vision and Philosophy**: [OUTLOOK.md](OUTLOOK.md)
- **Development Roadmap**: [ROADMAP.md](ROADMAP.md)
- **MVP Definition**: [MVP.md](MVP.md)
- **Terminology**: [TERMINOLOGY.md](TERMINOLOGY.md)
