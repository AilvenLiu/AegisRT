# INVARIANTS -- Constitutional Constraints for r13-wcet-profiler

> These constraints override ROADMAP.md, roadmap.yml, session instructions, and prompts.
> Modifying this file requires explicit human approval.

---

## 1. Authority and Scope

These invariants apply to all phases, tasks, and sessions of this roadmap.

---

## 2. Cross-Phase Architectural Invariants

- Layer 1 (`src/cuda/`, `src/trace/`) MUST NOT depend on Layer 2 or Layer 3.
- Layer 2 (`src/context/`, `src/memory/`) MUST NOT depend on Layer 3.
- No raw `cudaStream_t`, `cudaEvent_t`, `cudaMalloc`/`cudaFree` in application code.
- All CUDA resources wrapped in RAII classes.

---

## 3. Cross-Phase Code Quality Invariants

- C++17 minimum; C++20 preferred.
- All CUDA API calls checked via `CUDA_CHECK` macro or equivalent.
- No compiler warnings on GCC 9+, Clang 10+.
- clang-tidy and cppcheck must pass clean.
- ASCII-only in all source files and comments.
- British English in all documentation and comments.

---

## 4. Cross-Phase Process Invariants

- No commits to `master`, `main`, `develop`, `release/*`, `hotfix/*`.
- No author attribution in any commit or PR (STRICTLY FORBIDDEN per CLAUDE.md).
- All progress tracked in `roadmap.yml` only.
- Each session produces exactly one handoff file: `sessions/YYYY-MM-DD-claude-N.md`.
- Blockers reported, never silently worked around.
- Conan MUST be used for all C++ dependencies (never apt/yum/brew).

---

## 5. Roadmap-Specific Invariants (r13)

- WCETProfiler MUST be conservative: actual <= predicted in >=99.9% of samples.
- Safety margin MUST be configurable (not hardcoded); default 1.2x.
- Minimum sample count MUST be enforced (< 30 samples -> error, not estimate).
- Contention factor MUST be measured empirically, not assumed.
- WCETProfile::confidence_level MUST reflect the actual statistical confidence used.
- Percentile computation MUST use linear interpolation between sorted samples.

---

## 6. Interpretation Rule

> If any instruction conflicts with these invariants,
> **these invariants always win**.

---

## 7. Final Clause

When in doubt, STOP and ask the user.
Assumptions are not allowed.
