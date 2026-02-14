# AegisRT Architecture

## Architectural Philosophy

AegisRT is built on four foundational principles that inform every design decision. These principles are not merely guidelines—they are architectural invariants that cannot be violated without compromising the project's core value.

### Principle 1: Orchestration Over Execution

**Core Tenet**: AegisRT does not execute GPU kernels. It orchestrates existing runtimes by controlling **when** and **with what resources** they execute.

**Rationale**: TensorRT, TVM, and ONNX Runtime have invested decades of engineering in kernel optimization. Competing with them is both futile and unnecessary. The unsolved problem is not "how to run a kernel efficiently" but "how to run multiple kernels predictably."

**Manifestation**:
- All kernel execution delegated to `RuntimeBackend` implementations
- AegisRT manages resources (streams, memory, events) and scheduling (when, what priority)
- No kernel implementations within AegisRT core

### Principle 2: Formal Guarantees Over Best-Effort

**Core Tenet**: Scheduling decisions must be provably correct, not empirically tuned.

**Rationale**: Edge autonomous systems require worst-case guarantees, not average-case performance. A system that works 99.9% of the time but fails unpredictably is worse than a system that explicitly rejects unworkable configurations.

**Manifestation**:
- Admission control based on formal schedulability analysis
- Conservative WCET estimation with statistical confidence bounds
- Explicit rejection over silent degradation

### Principle 3: Observability as a Contract

**Core Tenet**: Every scheduling decision, resource allocation, and execution event must be traceable.

**Rationale**: Without observability, determinism claims cannot be verified. In production systems, "why did this happen?" is often more valuable than "what happened?" AegisRT treats traceability as a functional requirement, not a debugging feature.

**Manifestation**:
- Structured traces for all decisions
- Rationale logging for admission/rejection
- Offline analysis tools for reconstruction

### Principle 4: Composition Over Competition

**Core Tenet**: AegisRT complements existing ecosystems; it does not compete with them.

**Rationale**: The AI inference ecosystem is rich with excellent tools. AegisRT's value lies in addressing what these tools do not provide, not in replacing them.

**Manifestation**:
- `RuntimeBackend` abstraction for seamless integration
- Focus on what runtimes cannot do (scheduling, isolation, transparency)
- No proprietary model formats or compilation pipelines

---

## System Positioning

### Where AegisRT Lives in the Stack

```
+----------------------------------------------------------------------+
|                        Application Layer                             |
|     (Autonomous Driving Pipeline, Robotics Control, Edge Services)   |
+--------------------------------+-------------------------------------+
                                 |
                                 | Application API
                                 v
+----------------------------------------------------------------------+
|                           AegisRT Core                               |
|                                                                      |
|  +---------------------------------------------------------------+   |
|  |                  Layer 3: Deterministic Scheduler             |   |
|  |                                                               |   |
|  |  +-------------+  +-------------+  +----------------------+   |   |
|  |  | WCETProfiler|  | Admission   |  | SchedulingPolicy     |   |   |
|  |  |             |  | Controller  |  | (FIFO/RMS/EDF/Custom)|   |   |
|  |  | Statistical |  |             |  |                      |   |   |
|  |  | Profiling   |  | Formal      |  | Priority & Deadline  |   |   |
|  |  | Contention  |  | Analysis    |  | Based Selection      |   |   |
|  |  | Awareness   |  |             |  |                      |   |   |
|  |  +-------------+  +-------------+  +----------------------+   |   |
|  +-----------------------------+---------------------------------+   |
|                                |                                     |
|                                v                                     |
|  +---------------------------------------------------------------+   |
|  |               Layer 2: Resource Orchestration                 |   |
|  |                                                               |   |
|  |  +-------------+  +-------------+  +----------------------+   |   |
|  |  | Memory      |  | Execution   |  | RuntimeBackend       |   |   |
|  |  | Orchestrator|  | Context     |  | (Abstract Interface) |   |   |
|  |  |             |  |             |  |                      |   |   |
|  |  | Lifetime    |  | Per-Model   |  | +--------+ +-------+ |   |   |
|  |  | Analysis    |  | Budgets     |  | |  TRT   | |  TVM  | |   |   |
|  |  | Sharing     |  | Isolation   |  | |Backend | |Backend| |   |   |
|  |  | Planning    |  | FaultBounds |  | +--------+ +-------+ |   |   |
|  |  +-------------+  +-------------+  +----------------------+   |   |
|  +-----------------------------+---------------------------------+   |
|                                |                                     |
|                                v                                     |
|  +---------------------------------------------------------------+   |
|  |            Layer 1: CUDA Abstraction & Observability          |   |
|  |                                                               |   |
|  |  +-------------+  +-------------+  +----------------------+   |   |
|  |  | CUDA RAII   |  | Device      |  | TraceCollector       |   |   |
|  |  | Wrappers    |  | Capability  |  |                      |   |   |
|  |  |             |  | Discovery   |  | Structured Events    |   |   |
|  |  | Stream      |  |             |  | Decision Rationale   |   |   |
|  |  | Event       |  | SM Count    |  | Performance Metrics  |   |   |
|  |  | Memory      |  | Memory BW   |  | Export Formats       |   |   |
|  |  +-------------+  +-------------+  +----------------------+   |   |
|  +---------------------------------------------------------------+   |
+---------------------------------+------------------------------------+
                                  |
                                  | CUDA Runtime API
                                  v
                      +-----------------------+
                      |   CUDA Driver/API     |
                      |   (NVIDIA Hardware)   |
                      +-----------------------+
```

### Layer Responsibilities

| Layer | Primary Responsibility | Key Invariant |
|-------|----------------------|---------------|
| Layer 1 | Safe resource access with traceability | No raw CUDA handles escape |
| Layer 2 | Resource ownership and isolation | One context per model, no shared mutable state |
| Layer 3 | Scheduling decisions with formal guarantees | All decisions provably correct |

---

## Layer 1: CUDA Abstraction & Observability

### Purpose

Provide safe, RAII-managed access to CUDA resources with comprehensive traceability. This layer is the foundation upon which all higher-level functionality is built.

### Design Rationale

**Why RAII?**: CUDA resources (streams, events, memory) are scarce and expensive. RAII ensures deterministic resource lifecycle management, eliminating entire classes of bugs (leaks, double-free, use-after-free).

**Why Tracing at This Level?**: By embedding tracing at the lowest level, we guarantee that no operation goes unobserved. This enables complete audit trails without requiring explicit logging in upper layers.

### Component Specifications

#### 1.1 CudaStream RAII Wrapper

```cpp
namespace aegis::cuda {

class CudaStream {
public:
    // Construction & Destruction
    CudaStream();                              // Creates stream with default flags
    explicit CudaStream(unsigned int flags);   // Creates stream with custom flags
    ~CudaStream() noexcept;                    // Destroys stream, asserts no pending ops

    // Move-only semantics
    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    // Access
    cudaStream_t handle() const noexcept;
    bool is_valid() const noexcept;

    // Operations
    void synchronize();                        // Blocks until stream completes
    bool is_ready() const;                     // Non-blocking check

    // Event management
    void record_event(CudaEvent& event);
    void wait_event(const CudaEvent& event);

private:
    cudaStream_t stream_{nullptr};
    TraceCollector* tracer_;  // Non-owning reference for tracing
};

// Stream creation flags
enum class StreamFlags : unsigned int {
    Default = 0,
    NonBlocking = cudaStreamNonBlock,
    HighPriority = 0x100,  // Custom flag for priority tracking
};

} // namespace aegis::cuda
```

**Invariants**:
- `stream_` is either `nullptr` or a valid CUDA stream
- Destructor blocks until all pending operations complete
- All operations are traced with timing information

#### 1.2 CudaEvent RAII Wrapper

```cpp
namespace aegis::cuda {

class CudaEvent {
public:
    explicit CudaEvent(unsigned int flags = cudaEventDefault);
    ~CudaEvent() noexcept;

    CudaEvent(CudaEvent&& other) noexcept;
    CudaEvent& operator=(CudaEvent&& other) noexcept;
    CudaEvent(const CudaEvent&) = delete;

    // Access
    cudaEvent_t handle() const noexcept;

    // Operations
    void record(const CudaStream& stream);
    void synchronize();                        // Blocks until event completes
    bool is_ready() const;                     // Non-blocking check

    // Timing
    float elapsed_ms(const CudaEvent& start) const;  // Time since start event

private:
    cudaEvent_t event_{nullptr};
    bool recorded_{false};
};

enum class EventFlags : unsigned int {
    Default = cudaEventDefault,
    DisableTiming = cudaEventDisableTiming,
    BlockingSync = cudaEventBlockingSync,
    Interprocess = cudaEventInterprocess,
};

} // namespace aegis::cuda
```

#### 1.3 DeviceMemoryPool

```cpp
namespace aegis::cuda {

class DeviceMemoryPool {
public:
    explicit DeviceMemoryPool(size_t capacity);
    ~DeviceMemoryPool();

    // Non-copyable, non-movable (owns device memory)
    DeviceMemoryPool(const DeviceMemoryPool&) = delete;
    DeviceMemoryPool& operator=(const DeviceMemoryPool&) = delete;

    // Allocation
    Result<void*> allocate(size_t bytes);
    void deallocate(void* ptr);

    // Query
    size_t capacity() const noexcept;
    size_t used() const noexcept;
    size_t available() const noexcept;

    // Pressure handling
    void set_pressure_callback(std::function<void(PressureLevel)> callback);

private:
    void* base_address_{nullptr};
    size_t capacity_{0};
    std::atomic<size_t> used_{0};
    std::mutex allocation_mutex_;
    std::map<void*, size_t> allocations_;  // ptr -> size for tracking
    TraceCollector* tracer_;
};

enum class PressureLevel {
    None,       // < 50% capacity used
    Moderate,   // 50-75% capacity used
    High,       // 75-90% capacity used
    Critical,   // > 90% capacity used
};

} // namespace aegis::cuda
```

#### 1.4 DeviceCapabilityDiscovery

```cpp
namespace aegis::cuda {

struct DeviceCapabilities {
    // Identification
    int device_id;
    std::string name;
    int compute_capability_major;
    int compute_capability_minor;

    // Compute resources
    int sm_count;
    int max_threads_per_block;
    int max_threads_per_sm;
    int warp_size;
    size_t shared_memory_per_block;
    size_t shared_memory_per_sm;

    // Memory resources
    size_t total_memory;
    size_t free_memory;
    int memory_clock_khz;
    int memory_bus_width;

    // Derived metrics
    size_t theoretical_memory_bandwidth_gbps() const {
        return (memory_clock_khz * 1000 * (memory_bus_width / 8) * 2) / 1'000'000'000;
    }

    int theoretical_flops() const {
        // Approximate: SM count * max threads per SM * ops per clock
        // Exact formula depends on architecture
        return sm_count * max_threads_per_sm * 2;  // Simplified
    }

    // Platform detection
    bool is_jetson() const;
    bool is_integrated_gpu() const;
};

class DeviceContext {
public:
    static DeviceCapabilities query(int device_id = 0);
    static std::vector<DeviceCapabilities> query_all();
    static int device_count();

    // Resource reservation
    static Result<void> reserve_memory(size_t bytes);
    static void release_reserved_memory();
};

} // namespace aegis::cuda
```

#### 1.5 TraceCollector

```cpp
namespace aegis::trace {

struct TraceEvent {
    // Identification
    std::string event_id;      // Unique identifier
    std::string event_type;    // "allocate", "execute", "schedule", etc.
    std::string component;     // "memory", "scheduler", "context"

    // Timing
    Timestamp timestamp;
    Duration duration;

    // Context
    std::string model_id;      // Associated model (if any)
    std::string request_id;    // Associated request (if any)

    // Decision rationale
    std::string rationale;     // Human-readable explanation

    // Attributes (flexible key-value storage)
    std::map<std::string, std::string> attributes;
};

class TraceCollector {
public:
    explicit TraceCollector(size_t buffer_size = 10000);

    // Recording
    void record(TraceEvent event);

    // Querying
    std::vector<TraceEvent> query(
        std::optional<Timestamp> start = std::nullopt,
        std::optional<Timestamp> end = std::nullopt,
        std::optional<std::string> event_type = std::nullopt,
        std::optional<std::string> model_id = std::nullopt
    );

    // Export
    void export_json(const std::string& path) const;
    void export_perfetto(const std::string& path) const;
    void export_csv(const std::string& path) const;

    // Analysis
    std::vector<TraceEvent> reconstruct_decision_chain(const std::string& event_id) const;

private:
    RingBuffer<TraceEvent> buffer_;
    std::mutex buffer_mutex_;
};

} // namespace aegis::trace
```

### Layer 1 Invariants

| Invariant | Enforcement |
|-----------|-------------|
| No raw CUDA handles escape | Private members, handle() returns const |
| All CUDA calls checked | Wrapper functions throw on error |
| All operations traced | Tracing in constructor/destructor/methods |
| RAII ownership | Delete copy constructors/assignment |

---

## Layer 2: Resource Orchestration

### Purpose

Manage execution contexts, memory allocation, and runtime backends with clear ownership, strict isolation, and explicit resource budgets.

### Design Rationale

**Why Per-Model Contexts?**: Each model has different resource requirements and failure modes. Isolation prevents one model's misbehavior from corrupting others.

**Why Ahead-of-Time Memory Planning?**: Runtime memory allocation is unpredictable. Planning allocation statically enables provable memory bounds.

**Why Abstract Runtime Backend?**: Different deployment scenarios use different runtimes (TensorRT for performance, ONNX for flexibility). Abstraction enables runtime-agnostic orchestration.

### Component Specifications

#### 2.1 ResourceBudget

```cpp
namespace aegis::context {

struct ResourceBudget {
    // Memory constraints
    size_t memory_limit;           // Maximum device memory (bytes)
    size_t host_memory_limit;      // Maximum host memory (bytes)

    // Execution constraints
    int stream_limit;              // Maximum concurrent streams
    Duration compute_budget;       // Maximum compute time per period
    int max_concurrent_inferences; // Maximum parallel invocations

    // QoS parameters
    Priority priority;             // Scheduling priority
    Duration deadline;             // Relative deadline for each invocation
    Period period;                 // For periodic models

    // Validation
    bool is_valid() const;
    std::string validation_error() const;
};

class Priority {
public:
    static Priority real_time() { return Priority(0); }
    static Priority high() { return Priority(1); }
    static Priority normal() { return Priority(2); }
    static Priority low() { return Priority(3); }
    static Priority background() { return Priority(4); }

    int value() const noexcept { return value_; }
    bool operator<(const Priority& other) const { return value_ < other.value_; }

private:
    explicit Priority(int value) : value_(value) {}
    int value_;
};

} // namespace aegis::context
```

#### 2.2 ExecutionContext

```cpp
namespace aegis::context {

class ExecutionContext {
public:
    // Factory method
    static Result<std::unique_ptr<ExecutionContext>> create(
        ModelID model,
        ResourceBudget budget,
        std::unique_ptr<RuntimeBackend> backend
    );

    ~ExecutionContext();

    // Non-copyable, non-movable (owns resources)
    ExecutionContext(const ExecutionContext&) = delete;
    ExecutionContext& operator=(const ExecutionContext&) = delete;

    // Identification
    ModelID model_id() const noexcept;
    const ResourceBudget& budget() const noexcept;

    // Resource tracking
    size_t memory_used() const noexcept;
    size_t memory_available() const noexcept;
    int streams_in_use() const noexcept;
    bool can_allocate(size_t bytes) const noexcept;
    bool can_acquire_stream() const noexcept;

    // Execution
    Result<void> execute(
        const std::vector<Tensor>& inputs,
        std::vector<Tensor>& outputs,
        const CudaStream& stream
    );

    // Fault management
    bool has_error() const noexcept;
    std::optional<Error> last_error() const noexcept;
    void clear_error();

    // Metrics
    ExecutionContextMetrics metrics() const;

private:
    ExecutionContext(
        ModelID model,
        ResourceBudget budget,
        std::unique_ptr<RuntimeBackend> backend
    );

    ModelID model_;
    ResourceBudget budget_;
    std::unique_ptr<RuntimeBackend> backend_;

    // Resource tracking
    std::atomic<size_t> memory_used_{0};
    std::atomic<int> streams_in_use_{0};
    std::atomic<int> active_inferences_{0};

    // Fault state
    std::optional<Error> last_error_;
    std::mutex error_mutex_;

    // Metrics
    std::atomic<uint64_t> total_inferences_{0};
    std::atomic<uint64_t> total_errors_{0};
};

struct ExecutionContextMetrics {
    uint64_t total_inferences;
    uint64_t total_errors;
    Duration total_execution_time;
    Duration avg_execution_time;
    size_t peak_memory_used;
    int peak_streams_used;
};

} // namespace aegis::context
```

#### 2.3 MemoryOrchestrator

```cpp
namespace aegis::memory {

struct TensorLifetime {
    TensorID tensor;
    Timestamp start;        // First write timestamp
    Timestamp end;          // Last read timestamp
    size_t size;
    std::string owner_model;
};

struct SharingOpportunity {
    TensorID tensor_a;
    TensorID tensor_b;
    size_t shared_memory;   // Memory saved by sharing
    std::string reason;     // "non-overlapping-lifetimes"
};

struct AllocationPlan {
    std::map<TensorID, MemoryRegion> allocations;
    std::vector<SharingOpportunity> sharing;
    size_t peak_memory;
    size_t saved_memory;    // Memory saved through sharing
};

class MemoryOrchestrator {
public:
    explicit MemoryOrchestrator(std::shared_ptr<DeviceMemoryPool> pool);

    // Planning phase (at admission time)
    Result<AllocationPlan> plan(
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
    PressureLevel current_pressure() const;

    // Observability
    size_t peak_usage() const noexcept;
    size_t current_usage() const noexcept;
    std::vector<MemoryRegion> active_regions() const;

private:
    std::shared_ptr<DeviceMemoryPool> pool_;
    std::unordered_map<ModelID, MemoryBinding> bindings_;
    std::vector<AllocationPlan> plans_;
    TraceCollector* tracer_;

    // Lifetime analysis
    std::vector<SharingOpportunity> analyze_sharing_opportunities(
        const std::vector<TensorLifetime>& lifetimes
    );
};

enum class PressurePolicy {
    Reject,         // Reject new allocations
    Evict,          // Evict least-recently-used bindings
    Compact,        // Attempt memory compaction
    Degrade         // Reduce quality guarantees
};

struct MemoryRegion {
    void* base_address;
    size_t size;
    std::vector<TensorID> tensors;  // Tensors mapped to this region
};

struct MemoryBinding {
    ModelID model;
    std::map<TensorID, MemoryRegion> tensor_regions;
    Timestamp bound_at;
};

} // namespace aegis::memory
```

#### 2.4 RuntimeBackend (Abstract Interface)

```cpp
namespace aegis::runtime {

class RuntimeBackend {
public:
    virtual ~RuntimeBackend() = default;

    // Metadata
    virtual std::string runtime_name() const = 0;
    virtual std::string model_name() const = 0;

    // Resource estimation (static analysis)
    virtual size_t estimate_memory() const = 0;
    virtual std::vector<TensorLifetime> estimate_lifetimes() const = 0;
    virtual std::vector<TensorSpec> input_specs() const = 0;
    virtual std::vector<TensorSpec> output_specs() const = 0;

    // Execution
    virtual Result<void> execute(
        const CudaStream& stream,
        const std::vector<Tensor>& inputs,
        std::vector<Tensor>& outputs,
        const MemoryBinding& memory
    ) = 0;

    // Status
    virtual bool is_valid() const = 0;
    virtual std::string status_message() const = 0;

    // Warmup (optional optimization)
    virtual Result<void> warmup(const CudaStream& stream) {
        return Result<void>::ok();
    }
};

// Concrete implementations
class TensorRTBackend : public RuntimeBackend { /* ... */ };
class TVMBackend : public RuntimeBackend { /* ... */ };
class ONNXBackend : public RuntimeBackend { /* ... */ };
class MockBackend : public RuntimeBackend { /* for testing */ };

struct TensorSpec {
    std::string name;
    std::vector<int64_t> shape;
    DataType dtype;
    MemoryLayout layout;
};

class Tensor {
public:
    Tensor(const TensorSpec& spec, void* data);

    const TensorSpec& spec() const noexcept;
    void* data() noexcept;
    const void* data() const noexcept;
    size_t size_bytes() const noexcept;

private:
    TensorSpec spec_;
    void* data_;
};

} // namespace aegis::runtime
```

### Layer 2 Invariants

| Invariant | Enforcement |
|-----------|-------------|
| One context per model | Factory method, unique_ptr ownership |
| Budgets are hard limits | Checked before allocation, explicit error on violation |
| Faults are isolated | Each context maintains independent error state |
| Memory is planned ahead | AllocationPlan computed at admission, not runtime |

---

## Layer 3: Deterministic Scheduler (Core Contribution)

### Purpose

Provide real-time scheduling with formal admission control, adapting classical real-time theory for GPU execution constraints.

### The GPU Scheduling Challenge: A Deeper Look

Classical real-time scheduling theory makes assumptions that GPUs fundamentally violate. Understanding these violations is essential for appreciating AegisRT's contribution.

#### Violation 1: Non-Preemptibility

**Classical Assumption**: Tasks can be interrupted at any point and resumed later.

**GPU Reality**: Once a kernel is launched, it runs to completion. There is no GPU-level preemption (with limited exceptions on newer architectures).

**Impact**: A high-priority task may have to wait for a long-running low-priority kernel to finish. This "blocking time" must be accounted for in schedulability analysis.

**AegisRT Adaptation**: Non-preemptive EDF/RMS analysis with explicit blocking time computation.

#### Violation 2: Variable WCET

**Classical Assumption**: Worst-case execution time is a known, static parameter.

**GPU Reality**: GPU execution time varies dramatically based on:
- Co-running kernels (SM contention)
- Memory bandwidth usage (DRAM contention)
- Thermal state (throttling)
- Clock frequency (dynamic frequency scaling)

**Impact**: A task that takes 10ms alone might take 30ms under load.

**AegisRT Adaptation**: Contention-aware WCET profiling with statistical safety margins.

#### Violation 3: Resource Interference

**Classical Assumption**: Tasks are independent and do not affect each other's execution.

**GPU Reality**: All tasks share:
- L2 cache
- Memory bandwidth
- SM schedulers
- Power/thermal envelope

**Impact**: Task A's execution time depends on what Task B is doing.

**AegisRT Adaptation**: Resource interference modeling in WCET estimation.

### Component Specifications

#### 3.1 WCETProfiler

```cpp
namespace aegis::scheduler {

struct WCETProfile {
    Duration worst_case;           // Conservative upper bound
    Duration average_case;         // Typical execution time
    Duration best_case;            // Lower bound
    Duration percentile_99;        // 99th percentile
    Duration percentile_999;       // 99.9th percentile

    double contention_factor;      // How much slower under load
    int sample_count;              // Statistical confidence
    double confidence_level;       // e.g., 0.99 for 99% confidence

    // Metadata
    ProfileMethod method;
    std::chrono::system_clock::time_point profiled_at;
    std::string hardware_id;
};

enum class ProfileMethod {
    Statistical,           // Statistical sampling
    Analytical,           // Static analysis (future)
    Hybrid,               // Combined approach
};

class WCETProfiler {
public:
    explicit WCETProfiler(TraceCollector& tracer);

    // Basic profiling
    Result<WCETProfile> profile(
        ExecutionContext& context,
        int min_samples = 100,
        double confidence_level = 0.99
    );

    // Contention-aware profiling
    Result<WCETProfile> profile_under_load(
        ExecutionContext& context,
        const std::vector<ExecutionContext*>& background_contexts,
        int min_samples = 100
    );

    // Statistical methods
    Duration compute_wcet(
        const std::vector<Duration>& samples,
        double confidence_level,
        double safety_margin = 1.5
    );

    // Profile persistence
    Result<void> save_profile(
        ModelID model,
        const WCETProfile& profile,
        const std::string& path
    );
    Result<WCETProfile> load_profile(
        ModelID model,
        const std::string& path
    );

private:
    TraceCollector& tracer_;

    // Statistical utilities
    Duration compute_mean(const std::vector<Duration>& samples);
    Duration compute_stddev(const std::vector<Duration>& samples, Duration mean);
    Duration compute_percentile(const std::vector<Duration>& samples, double p);
    Duration compute_confidence_interval(
        Duration mean, Duration stddev, int n, double confidence
    );
};

} // namespace aegis::scheduler
```

#### 3.2 AdmissionController

```cpp
namespace aegis::scheduler {

struct AdmissionRequest {
    ModelID model;
    WCETProfile wcet;
    Period period;              // For periodic tasks
    Deadline deadline;          // Relative deadline
    Priority priority;
    ResourceBudget budget;
};

struct AdmissionResult {
    bool admitted;
    std::string reason;         // If rejected, why
    std::string analysis_detail;// Detailed analysis

    // Metrics
    double utilisation;         // System utilisation after admission
    Duration worst_case_response_time;  // Predicted WCRT

    // Blocking analysis
    Duration max_blocking_time; // Maximum blocking from lower-priority tasks
};

class AdmissionController {
public:
    explicit AdmissionController(TraceCollector& tracer);

    // Admission analysis
    AdmissionResult analyze(
        const AdmissionRequest& request,
        const std::vector<ExecutionContext*>& existing_contexts
    );

    // Schedulability tests
    bool test_rms_schedulability(
        const std::vector<TaskParameters>& tasks
    );
    bool test_edf_schedulability(
        const std::vector<TaskParameters>& tasks
    );

    // Non-preemptive analysis
    Duration compute_blocking_time(
        const std::vector<Duration>& all_wcets,
        Duration new_task_wcet,
        Priority new_task_priority
    );

    // Response-time analysis
    Duration compute_response_time(
        const TaskParameters& task,
        const std::vector<TaskParameters>& all_tasks
    );

private:
    TraceCollector& tracer_;
    std::vector<TaskParameters> admitted_tasks_;

    // Internal analysis methods
    bool test_utilization_bound(
        const std::vector<TaskParameters>& tasks,
        double bound
    );
    Duration response_time_analysis_iteration(
        const TaskParameters& task,
        const std::vector<TaskParameters>& higher_priority,
        Duration initial_estimate
    );
};

struct TaskParameters {
    ModelID model;
    Duration wcet;
    Period period;
    Deadline deadline;
    Priority priority;
};

} // namespace aegis::scheduler
```

#### 3.3 SchedulingPolicy

```cpp
namespace aegis::scheduler {

struct ExecutionRequest {
    RequestID request_id;
    ModelID model;
    Priority priority;
    Deadline deadline;
    std::optional<Timestamp> earliest_start;
    std::optional<Timestamp> latest_finish;
    std::map<std::string, std::string> metadata;
};

class SchedulingPolicy {
public:
    virtual ~SchedulingPolicy() = default;

    // Model management
    virtual void add_model(ModelID model, const WCETProfile& profile) = 0;
    virtual void remove_model(ModelID model) = 0;
    virtual void update_profile(ModelID model, const WCETProfile& profile) = 0;

    // Scheduling decision
    virtual std::optional<ExecutionRequest> select_next(
        const std::vector<ExecutionRequest>& pending
    ) = 0;

    // Metadata
    virtual std::string policy_name() const = 0;
    virtual std::string policy_description() const = 0;

    // Metrics
    virtual PolicyMetrics metrics() const = 0;
};

struct PolicyMetrics {
    uint64_t total_decisions;
    uint64_t deadline_misses;
    Duration avg_decision_time;
    std::map<std::string, uint64_t> decisions_by_model;
};

// FIFO (baseline)
class FIFOPolicy : public SchedulingPolicy { /* ... */ };

// Rate-Monotonic Scheduling
class RMSPolicy : public SchedulingPolicy { /* ... */ };

// Earliest Deadline First
class EDFPolicy : public SchedulingPolicy { /* ... */ };

// Priority-based
class PriorityPolicy : public SchedulingPolicy { /* ... */ };

} // namespace aegis::scheduler
```

#### 3.4 Scheduler (Central Orchestration)

```cpp
namespace aegis::scheduler {

class Scheduler {
public:
    struct Config {
        size_t max_pending_requests = 100;
        Duration scheduling_interval = Duration::from_micros(100);
        bool enable_admission_control = true;
        bool enable_tracing = true;
    };

    static Result<std::unique_ptr<Scheduler>> create(
        std::unique_ptr<SchedulingPolicy> policy,
        std::unique_ptr<AdmissionController> admission,
        std::shared_ptr<DeviceMemoryPool> memory_pool,
        TraceCollector& tracer,
        const Config& config = {}
    );

    ~Scheduler();

    // Model management
    Result<ModelID> admit(
        std::unique_ptr<ExecutionContext> context,
        const AdmissionRequest& request
    );
    void evict(ModelID model);

    // Request submission
    Result<RequestID> submit(
        ModelID model,
        Priority priority = Priority::normal(),
        std::optional<Deadline> deadline = std::nullopt
    );

    // Lifecycle
    void start();
    void stop();
    bool is_running() const;

    // Observability
    SchedulerMetrics metrics() const;
    std::vector<ModelID> admitted_models() const;

private:
    Scheduler(/* ... */);

    void orchestration_loop_();
    void execute_request_(const ExecutionRequest& request);

    std::unique_ptr<SchedulingPolicy> policy_;
    std::unique_ptr<AdmissionController> admission_;
    std::shared_ptr<DeviceMemoryPool> memory_pool_;
    std::unique_ptr<MemoryOrchestrator> memory_orchestrator_;
    TraceCollector& tracer_;

    ConcurrentQueue<ExecutionRequest> pending_;
    std::unordered_map<ModelID, std::unique_ptr<ExecutionContext>> contexts_;
    std::unordered_map<ModelID, WCETProfile> profiles_;

    std::thread orchestration_thread_;
    std::atomic<bool> running_{false};
    Config config_;
};

struct SchedulerMetrics {
    uint64_t total_requests;
    uint64_t completed_requests;
    uint64_t rejected_requests;
    uint64_t deadline_misses;
    Duration avg_latency;
    Duration p99_latency;
    size_t peak_pending_requests;
    size_t current_pending_requests;
};

} // namespace aegis::scheduler
```

### Schedulability Analysis Theory

#### Rate-Monotonic Scheduling (RMS)

For periodic tasks with implicit deadlines (deadline = period):

**Utilization Bound**: $U \leq n \cdot (2^{1/n} - 1)$

Where:
- $U = \sum_{i=1}^{n} \frac{C_i}{T_i}$ (total utilization)
- $C_i$ = Worst-case execution time of task $i$
- $T_i$ = Period of task $i$
- $n$ = Number of tasks

As $n \to \infty$: $U \leq \ln(2) \approx 0.693$

**Non-Preemptive Extension**:
Blocking time must be added to execution time:
$R_i = C_i + B_i + \sum_{j \in hp(i)} \lceil \frac{R_i}{T_j} \rceil \cdot C_j$

Where:
- $B_i = \max_{j \in lp(i)} C_j$ (max blocking from lower-priority tasks)
- $hp(i)$ = tasks with higher priority than $i$
- $lp(i)$ = tasks with lower priority than $i$

#### Earliest Deadline First (EDF)

For tasks with arbitrary deadlines:

**Utilization Bound**: $U = \sum_{i=1}^{n} \frac{C_i}{D_i} \leq 1$

Where $D_i$ = relative deadline of task $i$.

EDF can achieve 100% utilization but requires:
- Preemption (not available on GPU)
- Accurate WCET bounds

**Non-Preemptive EDF**:
Must check two conditions:
1. Utilization bound: $U \leq 1$
2. Maximum busy period analysis

---

## Data Flow

### Model Admission Flow

```
+----------------+     +----------------+     +----------------+
|   Application  |---->|   Load Model   |---->| Create Backend |
+----------------+     +----------------+     +----------------+
                                                      |
                                                      v
+----------------+     +----------------+     +----------------+
|   Admit Model  |<----|  WCET Profile  |<----| Profile Model  |
+----------------+     +----------------+     +----------------+
        |
        v
+----------------+     +----------------+     +----------------+
|   Admission    |---->|  Memory Plan   |---->|   Bind Memory  |
|   Analysis     |     +----------------+     +----------------+
+----------------+              |
        |                       v
        |              +----------------+
        +------------->|  Admit/Reject  |
                       +----------------+
                               |
        +----------------------+----------------------+
        |                                             |
        v                                             v
+----------------+                            +----------------+
| Create Context |                            | Return Error   |
|  Add to Pool   |                            | with Reason    |
+----------------+                            +----------------+
```

### Execution Flow

```
+----------------+     +----------------+     +----------------+
|   Application  |---->| Submit Request |---->|     Enqueue    |
+----------------+     +----------------+     +----------------+
                                                      |
                                                      v
                       +----------------+     +----------------+
                       | Orchestration  |<----|     Dequeue    |
                       |     Loop       |     +----------------+
                       +----------------+
                              |
                              v
                       +----------------+
                       |   Select Next  |
                       |    (Policy)    |
                       +----------------+
                              |
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
+----------------+    +----------------+    +----------------+
| Acquire Stream |    |   Bind Memory  |    |  Check Budget  |
+----------------+    +----------------+    +----------------+
        |                     |                     |
        +---------------------+---------------------+
                              |
                              v
                      +----------------+
                      |    Execute     |
                      |   (Backend)    |
                      +----------------+
                              |
                              v
                      +----------------+
                      | Record Trace   |
                      | Release Stream |
                      +----------------+
                              |
                              v
                      +----------------+
                      |   Complete     |
                      |   Callback     |
                      +----------------+
```

---

## Concurrency Model

### Thread Architecture

```
+-------------------------------------------+
| Main Thread                               |
|  - Model admission/eviction               |
|  - Request submission                     |
|  - Result retrieval                       |
+-------------------------------------------+
                    |
                    | (lock-free queue)
                    v
+-------------------------------------------+
| Orchestration Thread                      |
|  - Dequeue pending requests               |
|  - Policy evaluation                      |
|  - Resource allocation                    |
|  - Execution dispatch                     |
+-------------------------------------------+
                    |
                    | (async CUDA operations)
                    v
+-------------------------------------------+
| GPU Execution                             |
|  - Kernel execution (asynchronous)        |
|  - Memory transfers (asynchronous)        |
|  - Event-based synchronization            |
+-------------------------------------------+
```

### Synchronization Strategy

| Resource | Synchronization Method |
|----------|----------------------|
| Pending queue | Lock-free MPMC queue |
| GPU synchronization | CUDA events |
| Counter updates | Atomic operations |
| Complex state | Mutex with minimal critical section |

---

## Error Handling

### Error Categories

| Category | Example | Recovery Strategy |
|----------|---------|-------------------|
| Transient | Stream pool exhausted | Retry with backoff |
| Permanent | Invalid model format | Abort, report to caller |
| Resource | Memory allocation failed | Apply pressure policy |
| Execution | CUDA error during inference | Isolate to context, report |

### Error Propagation Pattern

```cpp
template<typename T>
class Result {
public:
    static Result ok(T value) { return Result(std::move(value)); }
    static Result error(Error err) { return Result(std::move(err)); }

    bool is_ok() const { return std::holds_alternative<T>(data_); }
    bool is_error() const { return std::holds_alternative<Error>(data_); }

    const T& value() const& { return std::get<T>(data_); }
    T&& value() && { return std::get<T>(std::move(data_)); }

    const Error& error() const& { return std::get<Error>(data_); }

private:
    explicit Result(T value) : data_(std::move(value)) {}
    explicit Result(Error error) : data_(std::move(error)) {}

    std::variant<T, Error> data_;
};

struct Error {
    ErrorCode code;
    std::string message;
    std::string component;
    std::string context;
    std::optional<std::string> model_id;
    std::optional<std::string> request_id;
};
```

---

## Design Trade-offs

### Trade-off 1: Determinism vs Throughput

**Decision**: Prioritize deterministic latency over peak throughput.

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

### Trade-off 4: Complexity vs Comprehensibility

**Decision**: Favor simple, understandable implementations over complex optimizations.

**Rationale**: For a research-oriented project, comprehensibility is essential for validation and contribution.

**Manifestation**: Clear component boundaries, extensive documentation, traceable decision rationale.

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Model Admission | O(n) | n = admitted models |
| Memory Planning | O(n × m) | n = models, m = tensors |
| Scheduling Decision | O(log n) | n = queue depth (priority queue) |
| Execution Dispatch | O(1) | Delegates to backend |

### Space Complexity

| Storage | Complexity | Notes |
|---------|------------|-------|
| Per-Model State | O(1) | Context, profile, budget |
| Memory Plan | O(n × m) | n = models, m = tensors |
| Trace Buffer | O(k) | k = configurable buffer size |

---

## References

### Foundational Papers

- Liu, C. L., & Layland, J. W. (1973). Scheduling algorithms for multiprogramming in a hard-real-time environment. *Journal of the ACM*.
- George, L., Rivierre, N., & Spuri, M. (1996). Preemptive and non-preemptive real-time uniprocessor scheduling. *INRIA Research Report*.
- Liu, J. W. S. (2000). *Real-Time Systems*. Prentice Hall.

### GPU Systems Research

- Zhang, Q., et al. (REEF, SOSP 2023). Reactive GPU execution for real-time AI services.
- Gujarati, A., et al. (Clockwork, OSDI 2020). Serving DNNs in real-time at datacenter scale.
- Yu, G., et al. (Orion, ATC 2023). Accelerating DNN inference with GPU-aware memory management.
- Chen, J., et al. (Prem, ASPLOS 2023). Prem: Predictable GPU execution for real-time systems.

### Related Projects

- NVIDIA TensorRT: https://developer.nvidia.com/tensorrt
- Apache TVM: https://tvm.apache.org/
- ONNX Runtime: https://onnxruntime.ai/
- PREEMPT-RT Linux: https://wiki.linuxfoundation.org/realtime/

---

## Document References

- **Vision and Philosophy**: [OUTLOOK.md](OUTLOOK.md)
- **Development Roadmap**: [ROADMAP.md](ROADMAP.md)
- **MVP Definition**: [MVP.md](MVP.md)
- **Whitepaper**: [WHITEPAPER.md](WHITEPAPER.md)
