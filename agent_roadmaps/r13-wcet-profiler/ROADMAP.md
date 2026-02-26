# r13 — WCETProfiler

## Purpose

`WCETProfiler` provides statistically-grounded worst-case execution time estimates
for each model. These estimates are the primary input to `AdmissionController`.
Conservative WCET estimates (with safety margins) are what enable AegisRT to make
formal schedulability guarantees.

## Dependencies

- r04: `TraceCollector` (source of execution duration samples)
- r06: `ModelID`, `Result<T>`, `ErrorCode`

## Sub-phase Overview

| Sub-phase | Focus | Key Deliverable |
|-----------|-------|-----------------|
| r13-a | Data structures + statistics | `WCETProfile`, `stats.hpp` |
| r13-b | Profiler measurement | `compute_profile()` with confidence intervals |
| r13-c | Contention + serialisation | `measure_contention_factor()`, JSON save/load |

---

## Phase r13-a: WCETProfile and Statistical Primitives

### ProfileMethod

```cpp
enum class ProfileMethod {
    Statistical,  // Statistical sampling (default)
    Analytical,   // Static analysis (future work)
    Hybrid,       // Combined approach
};
```

### WCETProfile

```cpp
struct WCETProfile {
    Duration worst_case;       // max observed + safety margin
    Duration average_case;     // mean of samples
    Duration best_case;        // min observed
    Duration percentile_99;    // 99th percentile
    Duration percentile_999;   // 99.9th percentile
    double   contention_factor; // ratio under concurrent load
    size_t   sample_count;
    double   confidence_level;  // e.g. 0.99 for 99%
    ProfileMethod method;         // how the profile was computed
    std::string hardware_id;      // device name for portability checks

    bool is_valid() const noexcept;
};
```

### Statistical Primitives (stats.hpp)

```cpp
namespace aegisrt::stats {
    double mean(std::span<const double> samples);
    double stddev(std::span<const double> samples);
    double percentile(std::span<double> samples, double p);  // modifies order
    double z_score(double confidence);  // e.g. z_score(0.99) = 2.576
}
```

---

## Phase r13-b: WCETProfiler Measurement

### Confidence Interval Formula

```
upper_bound = mean + z(confidence) * stddev / sqrt(n)
wcet = upper_bound * safety_margin
```

Where:
- `z(0.99) = 2.576`
- `safety_margin >= 1.0` (typically 1.1 to 1.3 for GPU workloads)
- `n >= 30` (minimum sample count for CLT to apply)

### Contention-Aware Profiling

```cpp
// Profile under concurrent load from other models
Result<WCETProfile> WCETProfiler::profile_under_load(
    ExecutionContext& context,
    const std::vector<ExecutionContext*>& background_contexts,
    int min_samples = 100
);
```

This method runs `context` while `background_contexts` are executing concurrently,
measuring the actual contention factor. The resulting `WCETProfile::contention_factor`
reflects real-world multi-model interference.

### Minimum Sample Enforcement

```cpp
if (samples.size() < min_samples_) {
    return Result<WCETProfile>::error(
        ErrorCode::InsufficientSamples,
        "Need at least " + std::to_string(min_samples_) + " samples");
}
```

---

## Phase r13-c: Contention Factor and Serialisation

### Contention Factor

GPU kernels run slower under contention from other models sharing the device.
The contention factor captures this:

```
contention_factor = p99_under_concurrent_load / p99_isolated
```

A contention factor of 1.3 means the model runs 30% slower under load.
`AdmissionController` multiplies WCET by `contention_factor` when computing
schedulability under multi-model scenarios.

### JSON Schema

```json
{
  "model_id": "model_a",
  "worst_case_ns": 15000000,
  "average_case_ns": 8000000,
  "best_case_ns": 6000000,
  "percentile_99_ns": 12000000,
  "percentile_999_ns": 14000000,
  "contention_factor": 1.25,
  "sample_count": 500,
  "confidence_level": 0.99
}
```

## Exit Criteria

- `compute_profile()` produces valid `WCETProfile` for >= 30 samples.
- p99/p999 accuracy within 5% of true percentile on known distributions.
- Contention factor > 1.0 under concurrent load (verified with MockBackend).
- JSON round-trip preserves all fields exactly.
- API header frozen.
