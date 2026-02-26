# r07 — ExecutionContext and FaultBoundary

## Purpose

ExecutionContext is the central per-model execution container for Layer 2.
Every model that runs through AegisRT owns exactly one ExecutionContext.
It enforces hard resource budgets, captures faults in isolation, and emits
structured trace events for every lifecycle transition.

## Dependencies

- r06: `Result<T>`, `ResourceBudget`, `RuntimeBackend`, `ModelID`, `ErrorCode`
- r04: `TraceCollector`
- r02: `CudaStream` (used by backend)

## Sub-phase Overview

| Sub-phase | Focus | Key Deliverable |
|-----------|-------|-----------------|
| r07-a | Context lifecycle + state machine | `ExecutionContext::create()` factory |
| r07-b | Budget enforcement + resource tracking | Thread-safe `allocate_memory()` / `execute()` |
| r07-c | FaultBoundary + tracing integration | Fault isolation, trace events |

---

## Phase r07-a: Context Lifecycle and State Machine

### State Machine

```
Created --> Ready --> Running --> Ready (on success)
                  \-> Faulted (on backend error)
Faulted --> Ready (via reset())
Any --> Destroyed (destructor)
```

### API Contract

```cpp
// include/aegisrt/layer2/execution_context.hpp
namespace aegisrt::layer2 {

enum class ContextState { Created, Ready, Running, Faulted, Destroyed };

class ExecutionContext {
public:
    // Factory -- only way to construct
    static Result<std::unique_ptr<ExecutionContext>> create(
        ModelID model_id,
        ResourceBudget budget,
        std::shared_ptr<RuntimeBackend> backend);

    // Accessors
    ModelID model_id() const noexcept;
    const ResourceBudget& budget() const noexcept;
    ContextState state() const noexcept;

    // Non-copyable, movable
    ExecutionContext(const ExecutionContext&) = delete;
    ExecutionContext& operator=(const ExecutionContext&) = delete;
    ExecutionContext(ExecutionContext&&) noexcept;
    ExecutionContext& operator=(ExecutionContext&&) noexcept;

    ~ExecutionContext();

private:
    ExecutionContext(ModelID, ResourceBudget, std::shared_ptr<RuntimeBackend>);
    // ...
};

} // namespace aegisrt::layer2
```

### Tasks

- **task-r07-a-0**: Write design doc covering state transitions and invariants.
- **task-r07-a-1**: Define `ContextState` enum.
- **task-r07-a-2**: Write class skeleton with private members.
- **task-r07-a-3**: Implement `create()` factory with argument validation.
- **task-r07-a-4**: Implement const accessors.
- **task-r07-a-5**: Implement destructor (release memory, assert not Running).
- **task-r07-a-6**: Unit tests for factory and state transitions.

---

## Phase r07-b: Budget Enforcement and Resource Tracking

### Design

All resource counters use `std::atomic` for lock-free thread safety.
`allocate_memory()` uses a compare-and-swap loop to prevent over-allocation
under concurrent access.

```cpp
// Budget enforcement
Result<void> allocate_memory(size_t bytes);
void release_memory(size_t bytes) noexcept;
bool can_allocate(size_t bytes) const noexcept;

// Execution
Result<ExecutionResult> execute(const ExecutionRequest& request);

// Recovery
void reset();  // Faulted -> Ready, clears counters
```

### ExecutionContextMetrics

```cpp
struct ExecutionContextMetrics {
    uint64_t total_inferences;
    uint64_t total_errors;
    Duration total_execution_time;
    Duration avg_execution_time;
    size_t   peak_memory_used;
    int      peak_streams_used;
};
```

`ExecutionContext::metrics()` returns an atomic snapshot of all counters.
Used by `MetricsAggregator` (r18) to compute per-model utilisation.

### execute() Sequence

1. Assert state == Ready; transition to Running.
2. Call `can_allocate()` for request memory requirement.
3. Call `backend_->execute(request)`.
4. On success: update counters, transition to Ready, return result.
5. On failure: capture fault, transition to Faulted, return error.

### Tasks

- **task-r07-b-0**: Add atomic counters to class.
- **task-r07-b-1**: Implement `can_allocate()` and `can_acquire_stream()`.
- **task-r07-b-2**: Implement `allocate_memory()` with CAS loop.
- **task-r07-b-3**: Implement `release_memory()` with underflow assertion.
- **task-r07-b-4**: Implement `execute()` with full sequence above.
- **task-r07-b-5**: Implement `reset()`.
- **task-r07-b-6**: Unit tests including concurrent allocation stress test.

---

## Phase r07-c: FaultBoundary Integration and Tracing

### FaultBoundary Design

```cpp
class FaultBoundary {
public:
    bool has_error() const noexcept;
    const Error& last_error() const;   // Precondition: has_error()
    void record_error(Error e) noexcept;
    void clear_error() noexcept;
private:
    std::optional<Error> last_error_;
};
```

`FaultBoundary` is composed into `ExecutionContext` (not inherited).
Each context has its own independent fault state.

### Trace Events Emitted

| Event | When |
|-------|------|
| `ContextCreated` | `create()` succeeds |
| `ExecutionStarted` | `execute()` begins |
| `ExecutionCompleted` | `execute()` succeeds |
| `FaultRecorded` | backend returns error |
| `ContextDestroyed` | destructor runs |

### Tasks

- **task-r07-c-0**: Implement `FaultBoundary` class.
- **task-r07-c-1**: Embed `FaultBoundary` in `ExecutionContext`.
- **task-r07-c-2**: Wire fault capture into `execute()`.
- **task-r07-c-3**: Add `has_fault()`, `last_fault()`, `clear_fault()` delegation.
- **task-r07-c-4**: Emit trace events at each lifecycle point.
- **task-r07-c-5**: Unit tests for fault isolation between two contexts.
- **task-r07-c-6**: Verify TSan clean, freeze API header.

## Exit Criteria

- All unit tests pass (including concurrent stress tests).
- TSan reports no data races.
- `ExecutionContext` API header is frozen (no breaking changes in later roadmaps).
- Trace events appear in `TraceCollector` for all lifecycle transitions.
