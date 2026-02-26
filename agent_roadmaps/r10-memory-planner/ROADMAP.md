# r10 — MemoryPlanner

## Purpose

`MemoryPlanner` takes the output of `LifetimeAnalyzer` (tensor lifetimes and
sharing opportunities) and produces an `AllocationPlan` that assigns each
tensor to a `MemoryRegion` within the pool. The goal is to minimise peak
memory usage by reusing regions for tensors that are never simultaneously live.

## Dependencies

- r09: `TensorLifetime`, `SharingOpportunity`, `MemoryRegion`, `TensorID`
- r06: `ModelID`, `Result<T>`

## Sub-phase Overview

| Sub-phase | Focus | Key Deliverable |
|-----------|-------|-----------------|
| r10-a | Data structures | `AllocationPlan`, `MemoryBinding` |
| r10-b | Bin-packing algorithm | `MemoryPlanner::generate_plan()` |
| r10-c | Correctness validation | `is_valid()` tests, alignment tests |

---

## Phase r10-a: AllocationPlan and MemoryBinding

### AllocationPlan

```cpp
struct AllocationPlan {
    // TensorID -> assigned MemoryRegion within the pool
    std::unordered_map<TensorID, MemoryRegion> allocations;

    // Which tensor pairs share a region
    std::vector<std::pair<TensorID, TensorID>> sharing_pairs;

    size_t peak_bytes;   // max concurrent live bytes
    size_t saved_bytes;  // isolated_peak - shared_peak

    // Invariant: no two simultaneously-live tensors share overlapping regions
    bool is_valid(const std::vector<TensorLifetime>& lifetimes) const;
};
```

### MemoryBinding

`MemoryBinding` is the runtime counterpart to `AllocationPlan`. It holds
actual device pointers after the plan has been applied to a `DeviceMemoryPool`.

```cpp
struct MemoryBinding {
    ModelID model_id;
    const AllocationPlan& plan;
    std::unordered_map<TensorID, void*> bound_regions;

    bool is_bound(TensorID id) const noexcept;
    void* get_ptr(TensorID id) const;  // Precondition: is_bound(id)
};
```

---

## Phase r10-b: Bin-Packing Algorithm

### Algorithm: First-Fit Decreasing (FFD)

1. Sort tensors by `size_bytes` descending.
2. For each tensor, check if it can share a region with an already-placed tensor
   (non-overlapping lifetimes, compatible alignment).
3. If sharing is possible, assign the same `MemoryRegion`.
4. Otherwise, allocate a new region at the next aligned offset.

### Alignment-Aware Offset Computation

```cpp
size_t align_up(size_t offset, size_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
}
```

All `MemoryRegion::offset_bytes` values must satisfy:
`offset_bytes % alignment == 0`.

### Peak Memory Computation

Sweep through all timestamps where allocations or frees occur.
At each timestamp, sum the sizes of all live tensors.
The maximum sum is `peak_bytes`.

---

## Phase r10-c: Validation and Correctness Proofs

### is_valid() Implementation

```cpp
bool AllocationPlan::is_valid(const std::vector<TensorLifetime>& lifetimes) const {
    // For every pair of tensors that share a MemoryRegion:
    // their lifetimes must NOT overlap
    for (auto& [id_a, region_a] : allocations) {
        for (auto& [id_b, region_b] : allocations) {
            if (id_a >= id_b) continue;
            if (!region_a.overlaps(region_b)) continue;
            // Regions overlap -- lifetimes must not
            auto& lt_a = find_lifetime(lifetimes, id_a);
            auto& lt_b = find_lifetime(lifetimes, id_b);
            if (lt_a.overlaps(lt_b)) return false;  // INVALID
        }
    }
    return true;
}
```

## Exit Criteria

- `generate_plan()` always produces plans where `is_valid()` returns true.
- Shared plans have `peak_bytes <= isolated_peak_bytes`.
- All offsets satisfy alignment requirements.
- Performance target met (< 5ms for 500 tensors).
- API header frozen.
