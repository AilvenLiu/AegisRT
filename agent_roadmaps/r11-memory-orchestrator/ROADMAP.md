# r11 — MemoryOrchestrator and PressureHandler

## Purpose

`MemoryOrchestrator` is the top-level coordinator for Layer 2 memory management.
It owns the lifecycle of memory bindings across all active models, responds to
memory pressure, and emits structured trace events for every allocation decision.

## Dependencies

- r10: `AllocationPlan`, `MemoryBinding`, `MemoryPlanner`
- r09: `LifetimeAnalyzer`, `TensorLifetime`, `SharingOpportunity`
- r03: `DeviceMemoryPool`
- r04: `TraceCollector`
- r06: `ModelID`, `Result<T>`

## Sub-phase Overview

| Sub-phase | Focus | Key Deliverable |
|-----------|-------|-----------------|
| r11-a | PressurePolicy + skeleton | `current_pressure()`, threshold constants |
| r11-b | Plan/bind/release | Full lifecycle with guards |
| r11-c | Pressure handling + tracing | All 4 policies, trace events |

---

## Phase r11-a: PressurePolicy and Orchestrator Skeleton

### Pressure Thresholds

```cpp
namespace aegisrt::layer2 {

enum class PressureLevel { Normal, Elevated, Critical };
enum class PressurePolicy { Evict, Reject, Compact, Log };

// Thresholds (fraction of pool capacity)
constexpr double kPressureElevatedThreshold = 0.70;
constexpr double kPressureCriticalThreshold = 0.90;

} // namespace
```

### Orchestrator Constructor

```cpp
class MemoryOrchestrator {
public:
    MemoryOrchestrator(
        std::shared_ptr<DeviceMemoryPool> pool,
        std::shared_ptr<TraceCollector> trace);

    PressureLevel current_pressure() const noexcept;
    // ...
private:
    std::shared_ptr<DeviceMemoryPool> pool_;
    std::shared_ptr<TraceCollector> trace_;
    std::unordered_map<ModelID, MemoryBinding> bindings_;
    std::unordered_map<ModelID, Timestamp> last_used_;
};
```

---

## Phase r11-b: Plan, Bind, and Release

### Lifecycle Sequence

```
plan(contexts)   -> AllocationPlan  (static analysis, no GPU allocation)
bind(model, plan) -> MemoryBinding  (actual GPU memory allocation from pool)
[model executes using binding.get_ptr(tensor_id)]
release(model)                      (return GPU memory to pool)
```

### Guards

- `bind()` on an already-bound model returns `ErrorCode::AlreadyBound`.
- `release()` on an unbound model is a silent no-op (idempotent).

---

## Phase r11-c: Pressure Handling

### Policy Dispatch

| Policy | Action |
|--------|--------|
| `Reject` | Return `ErrorCode::MemoryPressure` to the caller |
| `Evict` | Release the least-recently-used binding to free memory |
| `Compact` | Re-run `plan()` on remaining contexts, rebind with new plan |
| `Degrade` | Reduce quality guarantees and emit `PressureHandled` trace event |

### Trace Events

| Event | When |
|-------|------|
| `PlanGenerated` | `plan()` completes |
| `BindingCreated` | `bind()` succeeds |
| `BindingReleased` | `release()` runs |
| `PressureHandled` | `handle_pressure()` runs |

## Exit Criteria

- All lifecycle tests pass (plan -> bind -> release).
- All 4 pressure policies tested.
- LRU eviction order verified.
- Trace events appear in `TraceCollector` for all operations.
- API header frozen.
