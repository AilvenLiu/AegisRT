# r22 — Jetson Orin Optimisation, Examples, and v1.0-alpha Release

## Purpose

This is the final roadmap. It delivers three things: (1) Jetson Orin-specific
optimisations that make AegisRT practical on the target edge hardware,
(2) example applications that demonstrate AegisRT's value to new users,
and (3) the v1.0-alpha release that marks the completion of the development
arc from scratch.

## Dependencies

- r21: All backends integrated
- r20: All documentation complete
- r19: All case studies validated

## Sub-phase Overview

| Sub-phase | Focus | Key Deliverable |
|-----------|-------|-----------------|
| r22-a | Jetson Orin optimisation | Unified memory, power mode WCET adjustment |
| r22-b | Example applications | 3 examples with MockBackend |
| r22-c | v1.0-alpha release | CHANGELOG, version bump, git tag |

---

## Phase r22-a: Jetson Orin Optimisation

### Unified Memory

Jetson Orin uses a unified memory architecture: CPU and GPU share the same
physical DRAM. This means `cudaMemcpy` between host and device is a no-op
(the data is already in the right place). AegisRT must detect this and skip
unnecessary copies.

```cpp
bool DeviceMemoryPool::is_unified_memory() const noexcept {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id_);
    return prop.integrated != 0;
}
```

### Power Mode WCET Adjustment

Jetson Orin supports multiple power modes (via `nvpmodel`). Lower power modes
reduce clock frequencies, which increases execution time. The `WCETProfiler`
must store separate profiles per power mode.

```cpp
enum class JetsonPowerMode {
    MaxN,    // all cores, max frequency
    Mode15W, // 15W TDP
    Mode10W, // 10W TDP
    Mode5W,  // 5W TDP
};

// Query current power mode
JetsonPowerMode query_power_mode();
```

The `AdmissionController` uses the profile for the current power mode when
computing schedulability. Switching power modes requires re-running admission
analysis.

---

## Phase r22-b: Example Applications

### examples/multi_model_demo

Demonstrates the core AegisRT workflow:
1. Create 3 `ExecutionContext` instances with different budgets.
2. Profile each model with `WCETProfiler` (using MockBackend).
3. Submit all 3 models to `Scheduler` with EDF policy.
4. Run 100 scheduling cycles.
5. Export trace to JSON.
6. Print summary: utilisation, deadline miss rate, memory usage.

### examples/autonomous_driving

Simulates a 100ms autonomous driving cycle:
- Perception model: C=20ms, T=100ms, D=100ms (highest priority)
- Prediction model: C=15ms, T=100ms, D=80ms
- Planning model: C=10ms, T=100ms, D=60ms

Uses RMS scheduling. Verifies all deadlines met over 1000 cycles.

### examples/admission_demo

Interactive demonstration of admission control:
1. Admit 3 models (succeeds).
2. Attempt to admit a 4th model that exceeds the RMS bound (rejected).
3. Print the rejection rationale.
4. Reduce the 4th model's period to make it admissible.
5. Re-submit (succeeds).

---

## Phase r22-c: v1.0-alpha Release

### CHANGELOG.md Entry

```markdown
## v1.0-alpha (2026-XX-XX)

### What Works
- Full three-layer architecture (CUDA abstraction, resource orchestration, scheduler)
- RMS and EDF admission control with iterative response-time analysis
- Cross-model GPU memory sharing (>= 15% reduction demonstrated)
- Deterministic scheduling (CV < 5% for decision latency)
- TensorRT, TVM, and ONNX Runtime backend integration
- Jetson Orin support (unified memory, power mode awareness)
- Complete observability (AuditTrail, MetricsAggregator, JSON/CSV/Perfetto export)

### Known Limitations
- No preemption support (non-preemptive scheduling only)
- Single GPU device only (multi-GPU not yet supported)
- Static memory planning only (no dynamic reallocation during execution)

### Future Roadmap
- Multi-GPU support
- Dynamic memory reallocation
- NVDLA backend for Jetson
```

### Version Bump

```cmake
# CMakeLists.txt
project(AegisRT VERSION 1.0.0 LANGUAGES CXX CUDA)
```

```cpp
// include/aegisrt/version.hpp
#define AEGISRT_VERSION_MAJOR 1
#define AEGISRT_VERSION_MINOR 0
#define AEGISRT_VERSION_PATCH 0
#define AEGISRT_VERSION_SUFFIX "-alpha"
#define AEGISRT_VERSION_STRING "1.0.0-alpha"
```

## Exit Criteria (v1.0-alpha Gate)

- [ ] All tests pass on x86_64 (CI).
- [ ] All tests pass on Jetson Orin (hardware).
- [ ] All 3 examples build and run with MockBackend.
- [ ] CHANGELOG.md written.
- [ ] Version bumped to 1.0.0-alpha.
- [ ] `v1.0-alpha` git tag created on master.
- [ ] Final CI run is fully green.
