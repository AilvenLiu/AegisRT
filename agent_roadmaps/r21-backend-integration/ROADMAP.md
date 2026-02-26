# r21 — Backend Integration

## Purpose

AegisRT's `RuntimeBackend` interface (r06) was designed to be pluggable.
This roadmap implements three real backends: TensorRT (NVIDIA's primary
inference runtime), TVM (Apache's compiler-based runtime), and ONNX Runtime
(cross-platform inference). All three are optional dependencies -- AegisRT
builds and tests correctly without any of them installed.

## Dependencies

- r06: `RuntimeBackend` interface, `ExecutionRequest`, `ExecutionResult`
- r07: `ExecutionContext` (dispatches to backend)
- r02: `CudaStream` (used for async execution)

## Sub-phase Overview

| Sub-phase | Focus | Key Deliverable |
|-----------|-------|-----------------|
| r21-a | TensorRT backend | `TensorRTBackend` with load/execute/unload |
| r21-b | TVM + ONNX backends | `TVMBackend`, `ONNXBackend` |
| r21-c | Conformance + CI | Backend conformance suite, CI matrix |

---

## Phase r21-a: TensorRT Backend

### Optional Dependency Pattern

```cmake
# cmake/FindTensorRT.cmake
find_path(TensorRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/x86_64-linux-gnu /usr/local/cuda/include)
find_library(TensorRT_LIBRARY nvinfer
    HINTS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)

if(TensorRT_INCLUDE_DIR AND TensorRT_LIBRARY)
    set(TensorRT_FOUND TRUE)
    add_library(TensorRT::TensorRT IMPORTED SHARED)
    # ...
endif()
```

In `CMakeLists.txt`:
```cmake
if(TensorRT_FOUND)
    target_compile_definitions(aegisrt PRIVATE AEGISRT_HAVE_TENSORRT)
    target_link_libraries(aegisrt PRIVATE TensorRT::TensorRT)
endif()
```

### TensorRTBackend Lifecycle

```cpp
class TensorRTBackend : public RuntimeBackend {
public:
    Result<void> load(const std::filesystem::path& engine_path);
    Result<ExecutionResult> execute(const ExecutionRequest& request) override;
    void unload() override;

private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> trt_context_;
};
```

Destruction order matters: `trt_context_` before `engine_` before `runtime_`.

---

## Phase r21-b: TVM and ONNX Runtime Backends

### TVMBackend

TVM compiles models to shared libraries (`.so`). The backend loads the
library and calls the `default_function` entry point.

```cpp
Result<void> TVMBackend::load(const std::filesystem::path& lib_path) {
    module_ = tvm::runtime::Module::LoadFromFile(lib_path.string());
    run_func_ = module_.GetFunction("default");
    return Result<void>::ok();
}
```

### ONNXBackend

ONNX Runtime uses a session-based API with provider configuration:

```cpp
Result<void> ONNXBackend::load(const std::filesystem::path& model_path) {
    Ort::SessionOptions opts;
    OrtCUDAProviderOptions cuda_opts;
    cuda_opts.device_id = device_id_;
    opts.AppendExecutionProvider_CUDA(cuda_opts);
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);
    return Result<void>::ok();
}
```

---

## Phase r21-c: Backend Conformance Suite

### Conformance Tests

Any `RuntimeBackend` implementation must pass:

1. `load()` succeeds with a valid model file.
2. `execute()` returns a valid `ExecutionResult` after `load()`.
3. `execute()` returns `ErrorCode::NotLoaded` before `load()`.
4. `unload()` can be called multiple times without crashing.
5. `load()` after `unload()` succeeds (reload).
6. Error from backend is propagated as `Result` error, not exception.

### CI Matrix

```yaml
jobs:
  unit-tests:
    # Always runs -- uses MockBackend only
    runs-on: ubuntu-latest

  integration-trt:
    # Only runs if TRT is available
    runs-on: [self-hosted, gpu, tensorrt]
    if: github.event_name == 'push' && github.ref == 'refs/heads/master'

  integration-onnx:
    runs-on: [self-hosted, gpu]
    if: github.event_name == 'push' && github.ref == 'refs/heads/master'
```

## Exit Criteria

- `TensorRTBackend` passes conformance suite (on TRT-equipped runner).
- `TVMBackend` and `ONNXBackend` pass conformance suite.
- `MockBackend` passes conformance suite (always, in CI).
- Unit tests build and pass without any backend installed.
- `docs/BACKENDS.md` written.
