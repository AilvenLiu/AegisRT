# ROADMAP -- r01-build-infrastructure

> Long-form execution manual. Written for an AI agent with no prior context.
> Verbosity is intentional. Do not infer intent beyond what is written.

---

## 1. Background and Motivation

AegisRT is a GPU resource orchestration framework for deterministic multi-model edge AI
inference. The project starts from an empty repository. Before any C++ or CUDA code can
be written, a complete, reproducible build environment must exist.

Without this roadmap:
- No subsequent roadmap can compile, test, or cross-compile.
- Dependency management will be inconsistent across machines.
- Code quality enforcement (clang-format, clang-tidy, cppcheck) will be absent.
- Sanitizer builds (ASAN, TSAN, UBSAN) will not be configured.
- CI will not catch regressions automatically.
- The Jetson Orin cross-compilation path will be untested.

---

## 2. Overall Objective

By the end of this roadmap, ALL of the following MUST be true:

- `cmake --preset debug-x86 && cmake --build --preset debug-x86` succeeds with zero warnings.
- `cmake --preset debug-aarch64 && cmake --build --preset debug-aarch64` succeeds (cross-compile).
- CI pipeline runs on every push: build, test, lint, cross-compile jobs all pass.
- `clang-format --dry-run -Werror` reports no changes needed on all source files.
- `clang-tidy` reports zero warnings on the smoke test.
- `cppcheck` reports zero errors on all src/ files.
- CUDA_CHECK macro is defined, used in the smoke test, and aborts on error.
- ASAN + UBSAN build preset exists and the smoke test passes under it.
- Version header is generated correctly from CMakeLists.txt VERSION field.

These objectives are contractual. Do not mark this roadmap complete until all are verified.

---

## 3. Explicit Non-Goals

- No C++ or CUDA implementation code beyond the smoke test.
- No RAII wrappers, no memory management, no runtime logic.
- No tests beyond the smoke test (tests/ scaffolding only).
- No documentation beyond inline comments in CMake files and the CI README.
- No TSAN build (TSAN requires thread-aware code; smoke test has no threads).
- No Perfetto, no benchmarks, no profiling infrastructure.

---

## 4. High-Level Strategy

### Build System

Use CMake 3.25+ with native CUDA language support (enable_language(CUDA)).
CMake 3.25 is the minimum because it introduced significant improvements to CUDA
language support. Use CMakePresets.json to define reproducible build configurations.
Presets eliminate the need for manual cmake invocations with long flag lists.

### Dependency Management

Use Conan 2.x via conan_provider.cmake for reproducible dependency management.
Conan is mandatory per CLAUDE.md. vcpkg is explicitly rejected. apt/yum/brew for
C++ libraries is absolutely forbidden.

The conan_provider.cmake approach allows find_package() calls in CMakeLists.txt
to transparently resolve via Conan without requiring a separate conan install step.

### Code Quality

Three complementary tools enforced as CI gates:
1. clang-format: Style enforcement. Zero tolerance for formatting violations.
2. clang-tidy: Static analysis and modernisation. Zero tolerance for warnings.
3. cppcheck: Additional static analysis. Zero tolerance for errors.

### Cross-Compilation

The Jetson Orin toolchain file sets up aarch64-linux-gnu cross-compilation.
The cross-compile CI job verifies the build succeeds but does not run tests
(tests require actual Jetson hardware).

---

## 5. Sub-Phase A: CMake Project Skeleton and Conan Integration

### Objective

After this sub-phase:
- The project builds from scratch on x86_64 with zero warnings.
- Conan dependencies (GTest, spdlog) are available via find_package().
- Build presets cover all required configurations.
- Version header is generated.

### Task Execution Guidance

task-r01-a-0 (Root CMakeLists.txt):
- cmake_minimum_required(VERSION 3.25)
- project(AegisRT VERSION 0.1.0 LANGUAGES CXX CUDA)
- set(CMAKE_CXX_STANDARD 20), set(CMAKE_CUDA_STANDARD 20)
- set(CMAKE_EXPORT_COMPILE_COMMANDS ON) for clang-tidy
- Guard against in-source builds with FATAL_ERROR
- include(AegisRTHelpers) from cmake/modules/
- add_subdirectory(src) and add_subdirectory(tests)
- Do NOT set global compiler flags here.

task-r01-a-1 (CMake module macros):
- cmake/modules/AegisRTHelpers.cmake
- add_aegisrt_library(name [CUDA] sources...): adds library with -Wall -Wextra -Werror
- add_aegisrt_test(name sources...): adds GTest executable, registers with ctest

task-r01-a-2 (Conan integration):
- conanfile.txt with [requires] gtest/1.14.0 and spdlog/1.13.0
- [generators] CMakeDeps and CMakeToolchain
- cmake/conan_provider.cmake from https://github.com/conan-io/cmake-conan

task-r01-a-3 (Build presets):
- CMakePresets.json with: debug-x86, release-x86, asan, tsan, debug-aarch64
- ASAN preset: -fsanitize=address,undefined
- TSAN preset: -fsanitize=thread
- debug-aarch64 preset: uses cmake/toolchains/jetson-orin-aarch64.cmake

task-r01-a-4 (src/ and tests/ scaffolding):
- src/CMakeLists.txt: aegisrt_core as INTERFACE library with include/ directory
- tests/CMakeLists.txt: add_subdirectory(smoke)

task-r01-a-5 (Version header):
- cmake/version.hpp.in -> include/aegisrt/version.hpp via configure_file()
- Must contain AEGISRT_VERSION_MAJOR, MINOR, PATCH, STRING macros

### Exit Criteria for Sub-Phase A

- cmake --preset debug-x86 succeeds.
- cmake --build --preset debug-x86 succeeds with zero warnings.
- find_package(GTest) and find_package(spdlog) resolve correctly.
- Version header exists and contains correct version numbers.

---

## 6. Sub-Phase B: Code Quality Toolchain

### Objective

After this sub-phase:
- clang-format, clang-tidy, cppcheck are configured and enforced.
- CUDA_CHECK macro is defined and tested.
- Pre-commit hook is available.

### Task Execution Guidance

task-r01-b-0 (.clang-format):
- BasedOnStyle: LLVM
- ColumnLimit: 100
- IndentWidth: 4
- PointerAlignment: Left
- SortIncludes: CaseSensitive
- AllowShortFunctionsOnASingleLine: None
- AllowShortIfStatementsOnASingleLine: Never
- BraceWrapping: AfterClass true, AfterFunction true

task-r01-b-1 (.clang-tidy):
- Enable: modernize-*, readability-*, cppcoreguidelines-*, performance-*, bugprone-*
- Disable: modernize-use-trailing-return-type, readability-magic-numbers
- WarningsAsErrors: "*"
- HeaderFilterRegex: "src/.*"

task-r01-b-2 (cppcheck):
- .cppcheck config: --enable=all --suppress=missingInclude --error-exitcode=1
- cmake/cppcheck.cmake: custom target running cppcheck on all src/ files

task-r01-b-3 (CUDA_CHECK macro):
- include/aegisrt/cuda/cuda_error.hpp
- CUDA_CHECK(expr): calls check_cuda_error() with file/line/func info
- CUDA_CHECK_LAST(): calls CUDA_CHECK(cudaGetLastError())
- check_cuda_error(): prints to stderr and calls std::abort() on error
- Must be in namespace aegisrt::cuda
- Must be ASCII-only, British English comments

task-r01-b-4 (Pre-commit hook):
- scripts/pre-commit.sh: runs clang-format --dry-run -Werror on staged files
- cmake/install_hooks.cmake: custom target that installs the hook

### Exit Criteria for Sub-Phase B

- clang-format --dry-run -Werror on all source files: zero changes needed.
- clang-tidy on smoke test: zero warnings.
- cppcheck on src/: zero errors.
- CUDA_CHECK macro compiles and links correctly.

---

## 7. Sub-Phase C: CI Pipeline and Cross-Compilation

### Objective

After this sub-phase:
- CI runs on every push with 4 jobs: build, test, lint, cross-compile.
- ASAN+UBSAN job runs and passes.
- Jetson cross-compilation succeeds.
- CUDA smoke test passes.

### Task Execution Guidance

task-r01-c-0 (Jetson toolchain):
- cmake/toolchains/jetson-orin-aarch64.cmake
- CMAKE_SYSTEM_NAME Linux, CMAKE_SYSTEM_PROCESSOR aarch64
- CMAKE_C_COMPILER aarch64-linux-gnu-gcc
- CMAKE_CXX_COMPILER aarch64-linux-gnu-g++
- CMAKE_CUDA_ARCHITECTURES 87 (Jetson Orin Ampere)
- CMAKE_SYSROOT /usr/aarch64-linux-gnu
- CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY

task-r01-c-1 (CI workflow):
- .github/workflows/ci.yml
- job build-x86: cmake --preset debug-x86 && cmake --build --preset debug-x86
- job test-x86: cd build-debug-x86 && ctest --output-on-failure
- job lint: clang-format --dry-run + clang-tidy + cppcheck
- job cross-compile-aarch64: cmake --preset debug-aarch64 && cmake --build

task-r01-c-2 (CI sanitizer job):
- Separate job sanitize: cmake --preset asan && cmake --build && ctest
- Must pass before phase is complete.

task-r01-c-3 (CI documentation):
- .github/workflows/README.md explaining required packages and local reproduction

task-r01-c-4 (CUDA smoke test):
- tests/smoke/smoke_test.cu
- Includes aegisrt/cuda/cuda_error.hpp and aegisrt/version.hpp
- Allocates device memory with CUDA_CHECK(cudaMalloc(...))
- Launches trivial kernel that writes 42 to output
- Verifies output is 42, frees memory, prints "Smoke test PASSED"

task-r01-c-5 (Exit criteria verification):
- Run all CI jobs locally. Document each criterion met.
- Update roadmap.yml to completed. Write session handoff file.

### Exit Criteria for Sub-Phase C

- All 4 CI jobs pass on every push.
- ASAN+UBSAN job passes.
- Cross-compile for aarch64 succeeds without errors.
- Smoke test passes on x86_64 with cuda-memcheck.

---

## 8. Risk and Rollback Considerations

- Risk: CUDA toolkit version mismatch. Mitigation: pin CUDA version in CMakeLists.txt.
- Risk: Conan 1.x vs 2.x API differences. Mitigation: document required Conan version.
- Risk: Cross-compile toolchain not in CI. Mitigation: Docker image with aarch64 toolchain.
- Risk: clang-tidy version differences. Mitigation: pin clang-tidy version in CI.
- Rollback: All changes are in CMake/config files. Rollback by reverting commits.

---

## 9. Completion Definition

This roadmap is complete when:
- All tasks in all three sub-phases are marked completed in roadmap.yml.
- All exit criteria above are verified and documented.
- A session handoff file exists in sessions/.
- No open blockers remain.
- agent_roadmaps/README.md is updated to reflect r01 completed and r02 active.

---

## 10. Final Execution Rule

Follow this document literally.
Do not infer intent beyond what is written.
When in doubt, STOP and ask the user.
