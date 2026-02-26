# ROADMAP -- r05-layer1-validation

> Long-form execution manual. Written for an AI agent with no prior context.

---

## 1. Background and Motivation

Before Layer 2 begins, Layer 1 must be production-quality. This roadmap is a dedicated
validation pass -- no new features, only verification. It ensures coverage, memory safety,
static analysis, and cross-compilation all pass before the next phase builds on top.

Without this roadmap:
- Layer 2 would build on a foundation with unknown bugs.
- Memory leaks in Layer 1 would corrupt Layer 2 tests.
- API instability would cause cascading changes across all roadmaps.

---

## 2. Overall Objective

By the end of this roadmap, ALL of the following MUST be true:

- Coverage >70% for every Layer 1 source file individually.
- cuda-memcheck reports zero leaks and zero errors.
- ASAN reports zero errors.
- UBSAN reports zero errors.
- TSan reports zero data races.
- Cross-compilation succeeds without errors or warnings.
- clang-tidy reports zero warnings on all Layer 1 files.
- cppcheck reports zero errors on all Layer 1 files.
- Integration smoke test passes.
- Layer 1 public API is documented and marked stable.

---

## 3. Explicit Non-Goals

- No new production features.
- No Layer 2 code.
- No new public APIs.
- No performance optimisation.

---

## 4. High-Level Strategy

Systematic validation in three sub-phases:
1. Coverage: measure, identify gaps, add targeted tests.
2. Memory safety: cuda-memcheck, ASAN, UBSAN, TSan.
3. Static analysis, cross-compile, API freeze.

Each gate must pass before the next is attempted. Do not skip gates.

---

## 5. Sub-Phase A: Coverage Measurement and Gap Closure

### Objective

Achieve >70% line coverage for every Layer 1 source file.

### Task Execution Guidance

task-r05-a-0 (Coverage toolchain):
- Add coverage preset to CMakePresets.json:
  CMAKE_CXX_FLAGS: "-fprofile-arcs -ftest-coverage"
  CMAKE_EXE_LINKER_FLAGS: "-lgcov"
- Add CMake custom target coverage-report:
  lcov --capture --directory . --output-file coverage.info
  genhtml coverage.info --output-directory coverage-html
- CI job: build with coverage preset, run tests, generate report, fail if any file < 70%

task-r05-a-1 (Baseline measurement):
- Run: cmake --preset coverage && cmake --build --preset coverage && ctest
- Run: lcov and genhtml
- Identify all files below 70% threshold
- Document gaps in session handoff

task-r05-a-2 (Coverage gap closure for src/cuda/):
- For each uncovered branch, add a targeted test
- Focus on: error paths (invalid device ID, CUDA errors), destructor paths, edge cases
- Do NOT add tests that test implementation details; test observable behaviour

task-r05-a-3 (Coverage gap closure for src/trace/):
- Focus on: overflow handling, filter edge cases, empty buffer queries
- Verify reconstruct_decision_chain() with empty and single-event chains

task-r05-a-4 (Coverage CI step):
- Add coverage job to .github/workflows/ci.yml
- Job fails if any source file has < 70% line coverage
- Upload HTML report as CI artifact

### Exit Criteria for Sub-Phase A

- All Layer 1 source files have >70% line coverage.
- Coverage CI job passes.

---

## 6. Sub-Phase B: Memory Safety and Sanitizer Integration

### Objective

Verify zero memory errors, undefined behaviour, and data races.

### Task Execution Guidance

task-r05-b-0 (cuda-memcheck):
- Run: cuda-memcheck --leak-check full ./test_binary for each CUDA test binary
- Add CI job: cuda-memcheck-check
- Zero leaks and zero invalid accesses required

task-r05-b-1 (ASAN):
- Build with asan preset
- Run all tests
- Zero ASAN errors required
- Common ASAN errors to watch for: heap-use-after-free, heap-buffer-overflow

task-r05-b-2 (UBSAN):
- Build with ubsan preset (add -fsanitize=undefined to asan preset or separate preset)
- Run all tests
- Zero UBSAN errors required
- Common UBSAN errors: signed integer overflow, null pointer dereference

task-r05-b-3 (TSan):
- Build with tsan preset
- Run TraceCollector concurrency tests specifically
- Zero data races required

task-r05-b-4 (Fix sanitizer errors):
- For each error found: document root cause, fix, verify fix
- Do NOT suppress errors with __attribute__((no_sanitize(...)))
- Fix the underlying issue

### Exit Criteria for Sub-Phase B

- cuda-memcheck: zero leaks and zero errors.
- ASAN: zero errors.
- UBSAN: zero errors.
- TSan: zero data races.

---

## 7. Sub-Phase C: Static Analysis, Cross-Compile, and API Freeze

### Objective

Final quality gates before Layer 2 begins.

### Task Execution Guidance

task-r05-c-0 (clang-tidy full pass):
- Run clang-tidy on all files in src/cuda/ and src/trace/
- Zero warnings required
- Fix any warnings found (do not suppress)

task-r05-c-1 (cppcheck full pass):
- Run cppcheck --enable=all on all files in src/cuda/ and src/trace/
- Zero errors required

task-r05-c-2 (Cross-compilation):
- Build all Layer 1 tests with debug-aarch64 preset
- Verify no compilation errors or warnings
- Tests do not need to run (require Jetson hardware)

task-r05-c-3 (Integration smoke test):
- tests/integration/layer1_smoke_test.cpp
- Creates CudaContext(0)
- Creates CudaStream and CudaEvent
- Allocates memory from DeviceMemoryPool
- Records events to TraceCollector
- Exports trace to JSON
- Verifies JSON is parseable
- Verifies no leaks

task-r05-c-4 (API freeze):
- Review all public headers in include/aegisrt/cuda/ and include/aegisrt/trace/
- Add comment "// API STABLE -- do not change without updating all dependents"
- Document any known API limitations in docs/API_NOTES.md

task-r05-c-5 (Exit criteria checklist):
- Document each criterion: criterion, verification method, result
- Update roadmap.yml: all tasks completed, roadmap completed
- Write session handoff

### Exit Criteria for Sub-Phase C

- clang-tidy: zero warnings on all Layer 1 files.
- cppcheck: zero errors on all Layer 1 files.
- Cross-compilation: succeeds without errors.
- Integration smoke test: passes.
- API freeze: documented.

---

## 8. Completion Definition

This roadmap is complete when:
- All tasks in all three sub-phases are marked completed in roadmap.yml.
- All exit criteria above are verified.
- A session handoff file exists in sessions/.
- agent_roadmaps/README.md updated to reflect r05 completed and r06 active.
