# r09 — Tensor Lifetime Analysis

## Purpose

Before AegisRT can share GPU memory between models, it must know precisely
when each tensor is alive. This roadmap builds the data structures and
algorithms that answer: "which tensors from different models are never
simultaneously live, and therefore can share the same memory region?"

## Dependencies

- r07: `ExecutionContext`, `ModelID`
- r04: `TraceCollector` (source of alloc/free timestamps)

## Sub-phase Overview

| Sub-phase | Focus | Key Deliverable |
|-----------|-------|-----------------|
| r09-a | Core data structures | `TensorLifetime`, `MemoryRegion`, `SharingOpportunity` |
| r09-b | LifetimeAnalyzer algorithms | Overlap detection, sharing opportunity finder |
| r09-c | Validation + API freeze | Property tests, performance test, frozen headers |

---

## Phase r09-a: Core Data Structures

### TensorID

```cpp
// Strong typedef -- prevents accidental mixing with other IDs
struct TensorID {
    uint64_t value;
    bool operator==(const TensorID&) const noexcept = default;
};

// std::hash specialisation for use in unordered_map/set
template<> struct std::hash<TensorID> {
    size_t operator()(TensorID id) const noexcept {
        return std::hash<uint64_t>{}(id.value);
    }
};
```

### TensorLifetime

```cpp
struct TensorLifetime {
    TensorID    tensor_id;
    Timestamp   alloc_time;   // nanoseconds since epoch
    Timestamp   free_time;    // nanoseconds since epoch
    size_t      size_bytes;
    size_t      alignment;    // must be power of 2
    ModelID     owner_model_id;

    bool is_live_at(Timestamp t) const noexcept {
        return t >= alloc_time && t < free_time;
    }
    bool overlaps(const TensorLifetime& other) const noexcept {
        return alloc_time < other.free_time && free_time > other.alloc_time;
    }
};
```

### MemoryRegion

```cpp
struct MemoryRegion {
    size_t offset_bytes;
    size_t size_bytes;
    size_t alignment;

    bool overlaps(const MemoryRegion& other) const noexcept {
        return offset_bytes < other.offset_bytes + other.size_bytes
            && other.offset_bytes < offset_bytes + size_bytes;
    }
};
```

---

## Phase r09-b: LifetimeAnalyzer Core Algorithms

### Overlap Detection

Two tensors overlap if their live intervals intersect:

```
Tensor A: [alloc_a -------- free_a)
Tensor B:          [alloc_b -------- free_b)
                   ^-- overlap here
```

Non-overlapping tensors are candidates for memory sharing.

### Algorithm Complexity

- `find_non_overlapping()`: O(n^2) naive, acceptable for n < 10,000 tensors.
- Future optimisation: sweep-line algorithm for O(n log n).

### Cross-Model Constraint

Only tensors from **different** models are sharing candidates.
Tensors from the same model may have aliasing semantics that prevent sharing.

```cpp
bool is_sharing_candidate(const TensorLifetime& a, const TensorLifetime& b) {
    return a.owner_model_id != b.owner_model_id  // different models
        && !a.overlaps(b);                        // non-overlapping lifetimes
}
```

---

## Phase r09-c: Validation and API Freeze

### Property-Based Tests

1. For all pairs (a, b) where `a.overlaps(b)` is true:
   `find_sharing_opportunities()` must NOT include (a, b).

2. For all `SharingOpportunity` (a, b, shared_size):
   `shared_size <= min(a.size_bytes, b.size_bytes)`.

### Performance Target

`find_sharing_opportunities(1000 tensors)` must complete in < 10ms on
a development machine. This ensures the planner does not become a bottleneck.

## Exit Criteria

- All unit and property-based tests pass.
- Performance target met (< 10ms for 1000 tensors).
- `LifetimeAnalyzer` API header frozen.
- Integration with `TraceCollector` verified.
