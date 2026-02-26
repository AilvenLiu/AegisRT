# r16 — Scheduling Policies

## Purpose

`SchedulingPolicy` is the strategy interface that decouples the Scheduler's
orchestration loop from the specific ordering algorithm. Three concrete policies
are implemented: FIFO (baseline), EDF (optimal for soft real-time), and RMS
(optimal for hard real-time with fixed priorities).

## Dependencies

- r14: `TaskParameters`, `AdmissionResult`
- r06: `ModelID`, `Result<T>`

## Sub-phase Overview

| Sub-phase | Focus | Key Deliverable |
|-----------|-------|-----------------|
| r16-a | Interface + types | `SchedulingPolicy`, `ScheduledTask` |
| r16-b | FIFO + EDF + RMS | Three concrete policy implementations |
| r16-c | Comparative tests | Verify policies produce different orderings |

---

## Phase r16-a: SchedulingPolicy Interface

### ScheduledTask

```cpp
struct ScheduledTask {
    ModelID    model_id;
    Timestamp  deadline;        // absolute deadline
    uint32_t   priority;        // lower = higher priority (RMS convention)
    Duration   wcet;            // from WCETProfile
    Timestamp  submission_time; // when enqueue() was called
    uint64_t   request_id;      // unique per submission
};
```

### SchedulingPolicy Interface

```cpp
class SchedulingPolicy {
public:
    virtual ~SchedulingPolicy() = default;

    // Select the next task to execute. Returns nullopt if queue is empty.
    virtual std::optional<ScheduledTask> select_next() = 0;

    // Add a task to the scheduling queue.
    virtual void enqueue(ScheduledTask task) = 0;

    // Remove all pending tasks for a model (e.g. on model deregistration).
    virtual void remove_model(ModelID model_id) = 0;

    // Number of tasks currently queued.
    virtual size_t size() const noexcept = 0;
};
```

---

## Phase r16-b: Concrete Policy Implementations

### FIFOPolicy

Simplest policy: tasks execute in arrival order. Useful as a baseline and
for workloads where all tasks have equal importance.

```cpp
class FIFOPolicy : public SchedulingPolicy {
    std::queue<ScheduledTask> queue_;
public:
    std::optional<ScheduledTask> select_next() override {
        if (queue_.empty()) return std::nullopt;
        auto task = queue_.front();
        queue_.pop();
        return task;
    }
};
```

### EDFPolicy

Earliest Deadline First: optimal for utilisation under preemptive scheduling.
For non-preemptive GPU scheduling, EDF minimises deadline misses in practice.

```cpp
// Comparator: earlier deadline = higher priority
struct EDFComparator {
    bool operator()(const ScheduledTask& a, const ScheduledTask& b) const {
        if (a.deadline != b.deadline) return a.deadline > b.deadline;
        return a.submission_time > b.submission_time;  // tie-break: FIFO
    }
};
```

### RMSPolicy

Rate Monotonic Scheduling: optimal for fixed-priority preemptive scheduling.
Shorter period = higher priority (lower priority number).

### PolicyMetrics

```cpp
struct PolicyMetrics {
    uint64_t total_decisions;
    uint64_t deadline_misses;
    Duration avg_decision_time;
    std::map<std::string, uint64_t> decisions_by_model;
};
```

Every `SchedulingPolicy` implementation must track these metrics and return them
via `metrics()`. Used by `MetricsAggregator` (r18) for per-policy analysis.

### PriorityPolicy

A fourth concrete policy that selects by explicit priority number (lower = higher priority),
with configurable tie-breaking (by deadline or by submission time).

```cpp
class PriorityPolicy : public SchedulingPolicy {
    // select_next: lowest priority.value() wins
    // tie-break: configurable (deadline or submission_time)
};
```

---

## Phase r16-c: Comparative Tests

### Key Scenario

Given tasks:
- Task A: submitted at t=0, deadline=t=20, period=20ms
- Task B: submitted at t=1, deadline=t=10, period=10ms

Expected selections:
- FIFO: selects A (arrived first)
- EDF: selects B (earlier deadline)
- RMS: selects B (shorter period = higher priority)

This scenario verifies that all three policies are correctly implemented
and produce meaningfully different orderings.

## Exit Criteria

- All three policies pass their unit tests.
- Comparative test verifies FIFO != EDF for the scenario above.
- Empty queue returns `nullopt` without crashing.
- API header frozen.
