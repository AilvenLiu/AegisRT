# r15 — Response-Time Analysis

## Purpose

The RMS utilisation test (r14) is a sufficient but not necessary condition for
schedulability. Response-Time Analysis (RTA) is exact for fixed-priority
non-preemptive scheduling. This roadmap extends `AdmissionController` with
iterative RTA, giving AegisRT a tighter admission bound.

## Dependencies

- r14: `AdmissionController`, `TaskParameters`, `AdmissionResult`
- r06: `Result<T>`, `ErrorCode`

## Sub-phase Overview

| Sub-phase | Focus | Key Deliverable |
|-----------|-------|-----------------|
| r15-a | Blocking time | `compute_blocking_time()` |
| r15-b | Iterative RTA | `compute_response_time()` with convergence |
| r15-c | Integration | RTA wired into `analyze()`, all tasks checked |

---

## Phase r15-a: Non-Preemptive Blocking Time

### Theory

In non-preemptive scheduling, a high-priority task can be blocked by a
lower-priority task that has already started executing (and cannot be
preempted). The worst-case blocking time is:

```
B_i = max { C_j : priority(j) < priority(i) }
```

That is, the maximum WCET among all lower-priority tasks.

### Implementation

```cpp
Duration AdmissionController::compute_blocking_time(
    const std::vector<TaskParameters>& tasks,
    const TaskParameters& new_task) const
{
    Duration max_blocking{0};
    for (const auto& t : tasks) {
        if (t.priority > new_task.priority) {  // lower priority
            max_blocking = std::max(max_blocking, t.wcet);
        }
    }
    return max_blocking;
}
```

---

## Phase r15-b: Iterative RTA Algorithm

### Formula (George et al. 1996)

```
R_i^(0) = C_i + B_i

R_i^(k+1) = C_i + B_i + sum_{j: priority(j) < priority(i)} ceil(R_i^k / T_j) * C_j

Converged when R_i^(k+1) == R_i^k
Infeasible when R_i^k > D_i (deadline)
```

### Implementation Sketch

```cpp
Result<Duration> AdmissionController::compute_response_time(
    const TaskParameters& task,
    const std::vector<TaskParameters>& all_tasks) const
{
    Duration B = compute_blocking_time(all_tasks, task);
    Duration R = task.wcet + B;  // initial estimate

    for (int iter = 0; iter < kMaxRtaIterations; ++iter) {
        Duration R_next = task.wcet + B;
        for (const auto& hp : higher_priority_tasks(all_tasks, task)) {
            R_next += ceil_div(R, hp.period) * hp.wcet;
        }
        if (R_next == R) return R;  // converged
        if (R_next > task.deadline) return ErrorCode::DeadlineMissed;
        R = R_next;
    }
    return ErrorCode::RtaNoConvergence;
}
```

### Integer Arithmetic

All durations are `int64_t` nanoseconds. Use integer ceiling division:

```cpp
int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}
```

---

## Phase r15-c: Integration into analyze()

### Updated Admission Condition

After RTA integration, a task set is admissible only if:
1. RMS utilisation test passes (necessary condition check).
2. EDF utilisation test passes (necessary condition check).
3. RTA converges for ALL tasks in the set (sufficient condition).
4. WCRT <= deadline for ALL tasks.

This is stricter than r14 alone and may reject task sets that pass the
utilisation tests but fail the exact RTA check.

### Rationale Example

```
"Admitted: U=0.742 [RMS/EDF ok]. WCRT: model_a=8ms (D=10ms), "
"model_b=12ms (D=15ms), model_c=6ms (D=8ms). Blocking: 3ms."
```

## Exit Criteria

- RTA results match George et al. 1996 Table 1 examples exactly.
- Infeasible task sets return `ErrorCode::DeadlineMissed`.
- RTA rejects at least one task set that passes the RMS utilisation test.
- All r14 regression tests still pass.
- API header frozen.
