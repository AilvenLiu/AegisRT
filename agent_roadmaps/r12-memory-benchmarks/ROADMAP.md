# r12 — Memory Benchmarks and Phase 2 Validation

## Purpose

This roadmap validates the Phase 2 memory orchestration system with quantitative
benchmarks. The primary claim to validate is: AegisRT achieves >= 15% GPU memory
reduction compared to isolated per-model allocation, without violating any
correctness invariants.

## Dependencies

- r11: `MemoryOrchestrator`, `PressureHandler` (complete)
- r10: `AllocationPlan`, `MemoryPlanner`
- r09: `LifetimeAnalyzer`, `TensorLifetime`
- r03: `DeviceMemoryPool`

## Sub-phase Overview

| Sub-phase | Focus | Key Deliverable |
|-----------|-------|-----------------|
| r12-a | Benchmark harness + baseline | Isolated allocation measurement |
| r12-b | Orchestrated measurement | >= 15% reduction assertion |
| r12-c | Pressure handling + Phase 2 gate | All policies tested, CI integration |

---

## Phase r12-a: Benchmark Harness and Baseline

### 3-Model Workload Specification

The benchmark uses a deterministic workload with known sharing opportunities:

```
Model A: tensors [T1(100MB, t=0..10), T2(50MB, t=5..15), T3(200MB, t=10..20)]
Model B: tensors [T4(100MB, t=12..22), T5(50MB, t=18..28)]
Model C: tensors [T6(200MB, t=22..32), T7(100MB, t=25..35)]

Sharing opportunities:
  T1 (A, t=0..10) <-> T4 (B, t=12..22): non-overlapping, 100MB shareable
  T3 (A, t=10..20) <-> T6 (C, t=22..32): non-overlapping, 200MB shareable
```

### Isolated Baseline

```
Isolated peak = max(A_peak, B_peak, C_peak) summed = 350 + 150 + 300 = 800MB
```

### Orchestrated Target

```
Orchestrated peak <= 800MB * 0.85 = 680MB  (>= 15% reduction)
```

---

## Phase r12-b: Orchestrated Allocation and Reduction

### Reduction Calculation

```cpp
double reduction_pct =
    (isolated_peak_bytes - orchestrated_peak_bytes)
    / static_cast<double>(isolated_peak_bytes);

ASSERT_GE(reduction_pct, 0.15) << "Memory reduction below 15% target";
```

### Correctness Invariants

After every benchmark run:
1. `plan.is_valid(lifetimes)` must return true.
2. `pool.used()` must return 0 after all `release()` calls.
3. No tensor pointer aliasing for simultaneously-live tensors.

---

## Phase r12-c: Pressure Handling and Phase 2 Exit

### Pressure Simulation

```cpp
// Fill pool to 85% to trigger Elevated pressure
fill_pool_to_fraction(pool, 0.85);
EXPECT_EQ(orchestrator.current_pressure(), PressureLevel::Elevated);
orchestrator.handle_pressure(PressurePolicy::Evict);
EXPECT_LT(pool.used_fraction(), 0.85);  // eviction freed memory
```

### Phase 2 Exit Criteria

- [ ] >= 15% memory reduction demonstrated on 3-model workload.
- [ ] All 4 pressure policies tested and verified.
- [ ] `AllocationPlan::is_valid()` passes for all generated plans.
- [ ] Pool returns to 0 usage after all releases.
- [ ] Benchmark results documented in `docs/PHASE2_VALIDATION.md`.
- [ ] CI benchmark job is green.
