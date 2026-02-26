# ROADMAP -- r06-resource-budget-backend

> Long-form execution manual. Written for an AI agent with no prior context.

---

## 1. Background and Motivation

Layer 2 requires a resource contract (ResourceBudget), an abstract execution interface
(RuntimeBackend), and a test double (MockBackend). These three components are the
foundation of ExecutionContext (r07). Without them, no per-model isolation is possible.

Additionally, the common types (ModelID, Result<T>, Error) are needed by every component
in Layer 2 and Layer 3. They must be defined here, before ExecutionContext, to avoid
circular dependencies.

Without this roadmap:
- ExecutionContext (r07) cannot be implemented.
- No per-model resource isolation is possible.
- Tests cannot simulate inference without real models.

---

## 2. Overall Objective

By the end of this roadmap, ALL of the following MUST be true:

- ModelID is a strong typedef with equality, hash, and comparison operators.
- Result<T> holds either a value or an Error; never throws.
- ResourceBudget validates itself and describes validation errors.
- RuntimeBackend is a pure abstract interface with no default implementations.
- MockBackend implements RuntimeBackend with configurable latency and error injection.
- MockBackend lives in tests/mocks/ (never in src/).
- All unit tests pass.

---

## 3. Explicit Non-Goals

- No ExecutionContext (r07).
- No FaultBoundary (r07).
- No real inference backend (r21).
- No CUDA dependency in ResourceBudget or RuntimeBackend headers.

---

## 4. High-Level Strategy

### Result<T> Design

Result<T> is the error handling primitive for all Layer 2+ operations. It is inspired
by Rust's Result type and C++23's std::expected. The design goals are:
- Never throw exceptions (CUDA code cannot use exceptions reliably).
- Make error handling explicit and visible at call sites.
- Support monadic chaining for clean error propagation.

Implementation: use std::variant<T, Error> internally.

### TensorSpec and Tensor

```cpp
struct TensorSpec {
    std::string name;
    std::vector<int64_t> shape;
    DataType dtype;
    MemoryLayout layout;
};

class Tensor {
public:
    Tensor(const TensorSpec& spec, void* data);
    const TensorSpec& spec() const noexcept;
    void* data() noexcept;
    size_t size_bytes() const noexcept;
};
```

### Extended RuntimeBackend Interface

```cpp
class RuntimeBackend {
public:
    // Static analysis (called at admission time)
    virtual size_t estimate_memory() const = 0;
    virtual std::vector<TensorLifetime> estimate_lifetimes() const = 0;
    virtual std::vector<TensorSpec> input_specs() const = 0;
    virtual std::vector<TensorSpec> output_specs() const = 0;

    // Optional warmup (default no-op)
    virtual Result<void> warmup(const CudaStream& stream) {
        return Result<void>::ok();
    }
    // ...
};
```

### MockBackend Design

MockBackend is a test double that simulates inference without real models. It must be
flexible enough to test all scenarios:
- Normal execution with configurable latency.
- Error injection for fault boundary testing.
- Call tracking for verifying execution sequences.

MockBackend lives in tests/mocks/ because it is a test utility, not production code.
It must never be linked into the production binary.

---

## 5. Sub-Phase A: Common Types

### Objective

Establish ModelID, ErrorCode, Error, and Result<T> before any other Layer 2 code.
These types are used throughout the entire codebase.

### Task Execution Guidance

task-r06-a-0 (ModelID):
struct ModelID {
    std::string value;
    explicit ModelID(std::string v) : value(std::move(v)) {}
    bool operator==(const ModelID& other) const { return value == other.value; }
    bool operator<(const ModelID& other) const { return value < other.value; }
    const std::string& to_string() const { return value; }
};
// std::hash specialisation for use in unordered_map
namespace std {
    template<> struct hash<ModelID> {
        size_t operator()(const ModelID& id) const {
            return std::hash<std::string>{}(id.value);
        }
    };
}

task-r06-a-1 (ErrorCode):
enum class ErrorCode {
    BudgetExceeded,
    BackendError,
    InvalidArgument,
    NotFound,
    Timeout,
    InternalError,
    DeadlineMissed,
    NotSchedulable
};

task-r06-a-2 (Error):
struct Error {
    ErrorCode code;
    std::string message;
    std::string context;  // e.g. "ExecutionContext::execute"
    std::string to_string() const;
};

task-r06-a-3 (Result<T>):
template<typename T>
class Result {
public:
    static Result<T> ok(T value);
    static Result<T> error(Error err);
    bool is_ok() const;
    bool is_error() const;
    const T& value() const;  // asserts is_ok()
    T& value();
    T unwrap();  // moves value out, asserts is_ok()
    const Error& error() const;  // asserts is_error()
private:
    std::variant<T, Error> data_;
};
// Result<void> specialisation for operations with no return value

task-r06-a-4 (Monadic operations):
- map(f): if ok, return Result<U>::ok(f(value())); else return error
- and_then(f): if ok, return f(value()); else return error (f returns Result<U>)
- or_else(f): if error, return f(error()); else return value (f returns Result<T>)

task-r06-a-5 (Unit tests):
- ModelID equality, hash (use in unordered_map), comparison
- Result<T>: ok path, error path, monadic chaining
- Result<void>: ok() and error() paths

### Exit Criteria for Sub-Phase A

- ModelID, ErrorCode, Error, Result<T> compile with zero warnings.
- Unit tests pass.
- No CUDA dependency.

---

## 6. Sub-Phase B: ResourceBudget and Execution Types

### Objective

Define the resource contract and the execution request/result types.

### Task Execution Guidance

task-r06-b-0 (ResourceBudget):
struct ResourceBudget {
    size_t memory_limit_bytes;
    int stream_limit;
    Duration compute_budget;

    bool is_valid() const {
        return memory_limit_bytes > 0 && stream_limit > 0 && compute_budget.nanos > 0;
    }
    std::string validation_error() const;  // describes first violation
};

task-r06-b-1 (default_budget):
static ResourceBudget default_budget() {
    return ResourceBudget{
        .memory_limit_bytes = 512ULL * 1024 * 1024,  // 512 MB
        .stream_limit = 4,
        .compute_budget = Duration::from_millis(100)
    };
}

task-r06-b-2 (TensorDescriptor):
enum class DType { Float32, Float16, Int8, Int32, Bool };
struct TensorDescriptor {
    std::string name;
    std::vector<int64_t> shape;
    DType dtype;
    size_t size_bytes() const;  // product of shape * sizeof(dtype)
};

task-r06-b-3 (ExecutionRequest):
struct ExecutionRequest {
    ModelID model_id;
    std::string request_id;  // UUID
    std::vector<TensorDescriptor> input_tensors;
    Timestamp deadline;
    Timestamp submission_time;
};

task-r06-b-4 (ExecutionResult):
struct ExecutionResult {
    ModelID model_id;
    std::string request_id;
    std::vector<TensorDescriptor> output_tensors;
    Duration actual_latency;
    bool deadline_met;
};

task-r06-b-5 (Unit tests):
- ResourceBudget: valid budget, zero memory_limit, zero stream_limit, zero compute_budget
- ResourceBudget: validation_error() describes correct field
- TensorDescriptor: size_bytes() computation

### Exit Criteria for Sub-Phase B

- ResourceBudget validates correctly.
- TensorDescriptor, ExecutionRequest, ExecutionResult compile.
- Unit tests pass.

---

## 7. Sub-Phase C: RuntimeBackend Interface and MockBackend

### Objective

Define the abstract backend interface and implement a test double.

### Task Execution Guidance

task-r06-c-0 (BackendCapabilities):
struct BackendCapabilities {
    std::string backend_name;
    int max_batch_size;
    std::vector<DType> supported_dtypes;
    bool requires_pinned_memory;
    bool supports_async_execution;
};

task-r06-c-1 (RuntimeBackend):
class RuntimeBackend {
public:
    virtual ~RuntimeBackend() = default;
    virtual Result<void> load(const std::string& model_path) = 0;
    virtual Result<void> unload() = 0;
    virtual Result<ExecutionResult> execute(const ExecutionRequest& request) = 0;
    virtual BackendCapabilities capabilities() const = 0;
    virtual bool is_loaded() const = 0;
};
No default implementations. Pure abstract.

task-r06-c-2 (MockBackend):
class MockBackend : public RuntimeBackend {
public:
    explicit MockBackend(Duration fixed_latency = Duration::from_millis(10));
    // Implements all RuntimeBackend methods
    // execute() sleeps for fixed_latency, then returns success
};
Lives in tests/mocks/mock_backend.hpp and tests/mocks/mock_backend.cpp.

task-r06-c-3 (Error injection):
- void inject_error_on_call(int n): next n-th call to execute() returns error
- void inject_error_always(): all subsequent execute() calls return error
- void clear_error_injection(): remove all injected errors

task-r06-c-4 (Call tracking):
- int call_count() const: number of execute() calls
- const ExecutionRequest& last_request() const: most recent request
- const std::vector<ExecutionRequest>& execution_history() const
- void reset(): clear call count, history, error injection

task-r06-c-5 (Latency simulation):
- void set_latency(Duration d): set fixed latency for all execute() calls
- void set_random_latency(Duration mean, Duration stddev, uint64_t seed): random latency
- Uses std::this_thread::sleep_for for latency simulation

task-r06-c-6 (Unit tests):
- MockBackend: fixed latency, random latency, error injection, call tracking
- MockBackend: reset() clears all state
- RuntimeBackend: verify MockBackend satisfies interface (compile-time check)

### Exit Criteria for Sub-Phase C

- RuntimeBackend compiles as pure abstract class.
- MockBackend implements all RuntimeBackend methods.
- Error injection and call tracking work correctly.
- MockBackend lives in tests/mocks/, not src/.

---

## 8. Completion Definition

This roadmap is complete when:
- All tasks in all three sub-phases are marked completed in roadmap.yml.
- All exit criteria above are verified.
- A session handoff file exists in sessions/.
- agent_roadmaps/README.md updated to reflect r06 completed and r07 active.
