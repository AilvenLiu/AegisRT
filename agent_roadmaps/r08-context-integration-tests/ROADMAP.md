# r08 — Context Integration Tests

## Purpose

This roadmap contains no new production code. Its sole purpose is to prove
that the Phase 1 isolation guarantees hold under realistic multi-model scenarios.
A passing r08 suite is the gate to beginning Phase 2 (memory orchestration).

## Dependencies

- r07: `ExecutionContext`, `FaultBoundary` (complete)
- r06: `MockBackend`, `Result<T>`, `ResourceBudget`
- r04: `TraceCollector`

## Isolation Guarantees Under Test

1. Budget isolation: one model's budget exhaustion does not affect another.
2. Fault isolation: one model's backend failure does not propagate to another.
3. Resource counter isolation: per-model counters are independent.
4. Concurrency safety: concurrent execution on separate contexts is data-race-free.

## Sub-phase Overview

| Sub-phase | Focus | Key Deliverable |
|-----------|-------|-----------------|
| r08-a | Test infrastructure | `IntegrationTestFixture`, helpers |
| r08-b | Budget + fault isolation | 6 isolation test cases |
| r08-c | Concurrency + trace validation | TSan-clean concurrent tests, CI integration |

---

## Phase r08-a: Test Infrastructure and Fixture Design

### Test Matrix

| Axis | Scenarios |
|------|-----------|
| Budget | tight, generous, exactly-at-limit, one-byte-over |
| Fault | no fault, fault on first call, fault after N calls |
| Resource | independent counters, shared TraceCollector |
| Concurrency | sequential, 2-thread, 8-thread stress |

### Fixture Design

```cpp
class IntegrationTestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        trace_ = std::make_shared<TraceCollector>(1024);
        backend_a_ = std::make_shared<MockBackend>();
        backend_b_ = std::make_shared<MockBackend>();
        ctx_a_ = ExecutionContext::create(
            ModelID{"model_a"}, make_tight_budget(), backend_a_).value();
        ctx_b_ = ExecutionContext::create(
            ModelID{"model_b"}, make_generous_budget(), backend_b_).value();
    }

    std::shared_ptr<TraceCollector> trace_;
    std::shared_ptr<MockBackend> backend_a_, backend_b_;
    std::unique_ptr<ExecutionContext> ctx_a_, ctx_b_;
};
```

### Tasks

- **task-r08-a-0**: Write test matrix design doc.
- **task-r08-a-1**: Implement `IntegrationTestFixture`.
- **task-r08-a-2**: Add error injection helpers to `MockBackend`.
- **task-r08-a-3**: Add budget configuration helpers.
- **task-r08-a-4**: Add trace assertion helpers.

---

## Phase r08-b: Budget and Fault Isolation Tests

### Key Test Cases

```cpp
TEST_F(IntegrationTestFixture, BudgetViolationDoesNotAffectOtherModel) {
    // Exhaust model_a budget
    auto result_a = ctx_a_->allocate_memory(budget_a.memory_bytes + 1);
    EXPECT_FALSE(result_a.has_value());

    // model_b should still execute normally
    auto result_b = ctx_b_->execute(make_request());
    EXPECT_TRUE(result_b.has_value());
}

TEST_F(IntegrationTestFixture, FaultInModelADoesNotAffectModelB) {
    backend_a_->inject_error_on_next(ErrorCode::BackendFailure);
    ctx_a_->execute(make_request());  // triggers fault

    EXPECT_TRUE(ctx_a_->has_fault());
    EXPECT_FALSE(ctx_b_->has_fault());  // isolation guarantee
}
```

### Tasks

- **task-r08-b-0** through **task-r08-b-5**: See roadmap.yml for full list.

---

## Phase r08-c: Concurrency and Trace Validation Tests

### Concurrency Test Pattern

```cpp
TEST_F(IntegrationTestFixture, ConcurrentExecutionNoDataRaces) {
    std::thread t_a([&]{ ctx_a_->execute(make_request()); });
    std::thread t_b([&]{ ctx_b_->execute(make_request()); });
    t_a.join();
    t_b.join();
    // TSan will detect any races
}
```

### CI Integration

The integration test binary must be added to the CI matrix with:
- `ASAN_OPTIONS=detect_leaks=1`
- `TSAN_OPTIONS=halt_on_error=1`

### Tasks

- **task-r08-c-0** through **task-r08-c-5**: See roadmap.yml for full list.

## Exit Criteria (Phase 1 Gate)

- All 16 integration tests pass.
- TSan reports zero data races.
- ASAN reports zero memory errors.
- Phase 1 isolation guarantees documented in `docs/PHASE1_VALIDATION.md`.
- CI integration test job is green.
