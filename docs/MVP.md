## 1. MVP Definition

The Minimum Viable Product of AegisRT is **not feature completeness**, but **conceptual completeness**.

The MVP must demonstrate:

> *A full path from model graph → execution plan → scheduled GPU execution → controlled memory usage.*

---

## 2. MVP Functional Scope

### 2.1 Input Model

- ONNX model
- Limited operator set (Conv, GEMM, Elementwise, Activation)
- Static shapes only

---

### 2.2 Execution Graph

- DAG with explicit dependencies
- Nodes map to executable kernels
- Buffers are first‑class objects

---

### 2.3 Runtime Core

- Single CUDA context
- Explicit stream pool
- Event‑based synchronization
- No implicit global state

---

### 2.4 Memory System

- Unified device memory pool
- Lifetime‑aware allocation
- Buffer reuse
- Deterministic allocation behavior

---

### 2.5 Scheduler (MVP Level)

- Single‑node scheduler
- FIFO and static priority modes
- Non‑preemptive execution
- Explicit submission and completion API

---

## 3. MVP Non‑Goals

- Dynamic shapes
- Auto‑tuning
- Kernel generation
- Multi‑GPU
- Multi‑process isolation

---

## 4. MVP Success Criteria

The MVP is considered *complete* when:

- A reader can understand the full execution lifecycle
- Performance trade‑offs are measurable
- Design decisions are documented and defensible
- The system can be reasoned about as a whole

---

## 5. MVP as a Personal Milestone

Completing this MVP means you have:

- Designed a GPU runtime *end‑to‑end*
- Made scheduling a concrete, testable artifact
- Transformed scattered expertise into a coherent system

This is the foundation upon which depth is built.
