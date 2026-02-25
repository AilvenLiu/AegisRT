# AegisRT

**Transparent GPU Resource Orchestration for Multi-Model Edge AI**

[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE)

---

## The Problem Space

### What Every Edge AI Engineer Eventually Discovers

You've deployed your first model successfully. TensorRT optimization, CUDA streams, memory pools—all working beautifully. Then reality hits: your edge device needs to run **fifteen models simultaneously**, and suddenly nothing makes sense anymore.

- Why does Model A's latency spike when Model B is loaded?
- How much memory can I safely allocate to Model C without breaking Model D?
- What's consuming my GPU memory? The profiler shows fragmented numbers.
- Why did my system miss a critical deadline? There's no audit trail.

**The uncomfortable truth**: Existing runtimes (TensorRT, TVM, ONNX Runtime) solve single-model execution brilliantly. They were never designed to answer these questions. When you run multiple models concurrently on a resource-constrained GPU, you face an **orchestration problem**, not an execution problem.

### The Gap No One Has Filled

| Question | TensorRT | TVM Runtime | ONNX Runtime | AegisRT |
|----------|----------|-------------|--------------|---------|
| Why did Model A slow down? | Unknown | Unknown | Unknown | **Traceable cause** |
| Can I safely add this model? | Try and see | Try and see | Try and see | **Formal analysis** |
| What's my worst-case latency? | No guarantee | No guarantee | No guarantee | **Provable bound** |
| How much memory will I need? | Per-model only | Per-model only | Per-model only | **Cross-model plan** |
| Can I trace every decision? | No | No | No | **Full audit trail** |

**Key Insight**: This is not a performance optimization problem. This is a **resource transparency and predictability problem** that requires a fundamentally different approach.

---

## What AegisRT Provides

### Core Value Proposition

AegisRT is a **GPU resource orchestration framework** that sits above existing inference runtimes and provides four fundamental capabilities:

```
+-----------------------------------------------------------------+
|                        Application Layer                        |
|       (Autonomous Driving, Robotics, Edge AI Services)          |
+-------------------------------+---------------------------------+
                                |
+-------------------------------v---------------------------------+
|                            AegisRT                              |
|                                                                 |
|  +-----------------+ +-----------------+ +-------------------+  |
|  | TRANSPARENCY    | | PREDICTABILITY  | | ISOLATION         |  |
|  |                 | |                 | |                   |  |
|  | Resource        | | Formal          | | Per-Model         |  |
|  | Visibility      | | Admission       | | Fault             |  |
|  | Decision Audit  | | Control         | | Boundaries        |  |
|  | Trace Replay    | | Latency Bounds  | | Resource Budgets  |  |
|  +-----------------+ +-----------------+ +-------------------+  |
|                                                                 |
+-------------------------------+---------------------------------+
                                |
              +-----------------+-----------------+
              |                 |                 |
        +-----v-----+     +-----v-----+     +-----v-----+
        | TensorRT  |     |TVM Runtime|     |   ONNX    |
        | "Execute" |     | "Execute" |     | "Execute" |
        +-----------+     +-----------+     +-----------+
```

### Four Pillars of AegisRT

#### Pillar 1: Resource Transparency

Every GPU resource allocation, every scheduling decision, every execution event is traced and explainable.

```cpp
// Example: Query why Model A's latency increased
auto trace = tracer.query("model_a", start_time, end_time);
for (const auto& event : trace) {
    std::cout << event.timestamp << ": " << event.rationale << "\n";
}
// Output:
// 14:23:45.123: Model A scheduled (deadline: 14:23:45.200)
// 14:23:45.125: Blocked by Model B (wcet: 45ms, non-preemptive)
// 14:23:45.170: Model A execution started
// 14:23:45.195: Model A completed (latency: 72ms)
```

#### Pillar 2: Predictable Performance

Instead of "best-effort" execution, AegisRT provides **provable latency bounds** through formal admission control derived from real-time scheduling theory.

```cpp
// Before accepting a model, AegisRT answers:
AdmissionResult result = admission.analyze(request);
if (result.admitted) {
    std::cout << "Worst-case response time: " << result.wcrt << "\n";
    std::cout << "Confidence: " << result.confidence << "\n";
} else {
    std::cout << "Rejected: " << result.reason << "\n";
}
```

#### Pillar 3: Execution Isolation

Each model operates within a resource budget. One model's misbehavior cannot corrupt another's execution.

```cpp
ResourceBudget budget = {
    .memory_limit = 512_MB,
    .stream_limit = 2,
    .compute_budget = Duration::from_millis(50)
};

auto ctx = ExecutionContext::create(model, budget, backend);
// If ctx exceeds budget → explicit error, not silent degradation
```

#### Pillar 4: Cross-Model Memory Orchestration

Memory is planned globally, not per-model. Non-overlapping tensor lifetimes are automatically identified and shared.

```
+------------------+
|   Model A        |  [===tensor_a===]        [===tensor_c===]
+------------------+
+------------------+
|   Model B        |      [===tensor_b===]
+------------------+
+------------------+
|   Timeline       |  t0         t1         t2         t3
+------------------+

Memory Plan: tensor_a and tensor_b can share the same region (non-overlapping)
```

---

## What AegisRT Is NOT

To understand AegisRT's value, it's equally important to understand what it does NOT do:

| AegisRT Does NOT | Why This Matters |
|------------------|------------------|
| Execute kernels | Delegates to TensorRT/TVM/ONNX Runtime (they do this better) |
| Compile models | Consumes pre-compiled models from existing toolchains |
| Optimize single-model throughput | Focuses on multi-model predictability |
| Provide distributed inference | Single-node, single-GPU scope |
| Replace CUDA | Uses CUDA as the underlying substrate |

**The Complementarity Principle**: AegisRT enables what existing runtimes cannot provide—orchestration, transparency, and predictability for multi-model scenarios. It does not compete with them; it completes them.

---

## Core Technical Differentiator: GPU-Aware Real-Time Scheduling

### The Theoretical Challenge

Classical real-time scheduling theory (Liu & Layland, 1973) assumes conditions that GPUs fundamentally violate:

| Classical RT Assumption | GPU Reality |
|------------------------|-------------|
| Tasks are preemptible | GPU kernels run to completion |
| WCET is static | Execution time varies with contention |
| Single sequential processor | Massively parallel with shared resources |
| Independent tasks | Models share memory bandwidth, L2 cache |

### AegisRT's Research Contribution

AegisRT bridges real-time systems theory and GPU execution reality through:

1. **Non-Preemptive Scheduling Analysis**: Adapting EDF/RMS for kernels that cannot be interrupted
2. **Contention-Aware WCET Profiling**: Statistical estimation of worst-case execution time under realistic load
3. **Formal Admission Control**: Schedulability tests that prove—or disprove—feasibility before execution
4. **Resource Interference Modeling**: Accounting for shared memory bandwidth and cache effects

This positions AegisRT at the intersection of **real-time systems research** and **AI infrastructure engineering**—an area with significant open questions and growing industrial relevance.

---

## Target Hardware and Use Cases

### Primary Target Platform

- **NVIDIA Jetson Orin** (8-64GB unified memory)
- Edge autonomous systems with strict latency requirements
- Multi-model inference pipelines (perception, prediction, planning)

### Secondary Target Platform

- NVIDIA server GPUs (for development, CI, and profiling)
- Cloud-edge hybrid scenarios where edge behavior must be predictable

### Ideal Use Cases

1. **Autonomous Driving Pipelines**: 15-30 models running concurrently with safety-critical deadlines
2. **Industrial Robotics**: Multi-camera, multi-model perception with deterministic response times
3. **Edge AI Services**: Multi-tenant inference on shared edge infrastructure with SLA guarantees

---

## Architecture Overview

AegisRT is organized into three layers, with the scheduling layer as the core differentiator:

```
+-------------------------------------------------------+
|          Layer 3: Deterministic Scheduler             |
|                                                       |
|  +--------------+ +--------------+ +----------------+ |
|  | WCETProfiler | | Admission    | | Scheduling     | |
|  | (Statistical)| | Controller   | | Policy         | |
|  |              | | (Formal      | | (FIFO, RMS,    | |
|  |              | |  Analysis)   | |  EDF)          | |
|  +--------------+ +--------------+ +----------------+ |
+---------------------------+---------------------------+
                            |
+---------------------------v---------------------------+
|             Layer 2: Resource Orchestration           |
|                                                       |
|  +--------------+ +--------------+ +----------------+ |
|  | Memory       | | Execution    | | Runtime        | |
|  | Orchestrator | | Context      | | Backend        | |
|  | (Cross-Model)| | (Isolation)  | | (TensorRT...)  | |
|  +--------------+ +--------------+ +----------------+ |
+---------------------------+---------------------------+
                            |
+---------------------------v---------------------------+
|        Layer 1: CUDA Abstraction & Observability      |
|                                                       |
|  +--------------+ +--------------+ +----------------+ |
|  | CUDA RAII    | | Device       | | Trace          | |
|  | Wrappers     | | Discovery    | | Collector      | |
|  +--------------+ +--------------+ +----------------+ |
+-------------------------------------------------------+
```

**Design Principle**: Layer 3 is the research contribution. Layers 1-2 are necessary infrastructure for Layer 3 to function.

---

## Project Status

### Current Phase

**Phase 0: Foundation** — CUDA abstraction layer and build infrastructure.

### Development Roadmap

| Phase | Focus | Key Deliverable |
|-------|-------|-----------------|
| 0 | CUDA Abstraction | RAII wrappers, memory pool, tracing |
| 1 | Context Isolation | Per-model budgets, fault boundaries |
| 2 | Memory Orchestration | Cross-model sharing, pressure handling |
| 3 | **Deterministic Scheduler** | WCET profiling, admission control, RT policies |
| 4 | Observability | Full tracing, determinism validation |
| 5 | Integration | TensorRT/TVM/ONNX backends, Jetson optimization |

See [ROADMAP.md](docs/ROADMAP.md) for detailed development phases.

---

## Documentation

| Document | Purpose |
|----------|---------|
| [WHITEPAPER.md](docs/WHITEPAPER.md) | Project vision, motivation, and strategic positioning |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture and component design |
| [MVP.md](docs/MVP.md) | Minimum Viable Product definition and success criteria |
| [ROADMAP.md](docs/ROADMAP.md) | Sequential development phases with exit criteria |
| [OUTLOOK.md](docs/OUTLOOK.md) | Research foundations and long-term vision |

---

## Who Should Use AegisRT

### Target Users

- **Edge AI Engineers**: Running multi-model workloads on constrained GPUs
- **Autonomous Systems Developers**: Building perception pipelines with hard deadlines
- **Real-Time Systems Researchers**: Investigating GPU scheduling and WCET analysis
- **AI Infrastructure Engineers**: Designing deployment pipelines for edge AI

### Prerequisites

- Familiarity with CUDA programming concepts
- Understanding of real-time systems fundamentals (helpful but not required)
- Experience with TensorRT, TVM, or ONNX Runtime

---

## Strategic Positioning for Independent Developers

AegisRT is designed with specific constraints in mind:

| Constraint | AegisRT's Response |
|------------|-------------------|
| Limited hardware (one Jetson Orin) | Single-node, single-GPU focus |
| No production traffic | Research-quality over production-quality initially |
| Limited time | Incremental phases with clear exit criteria |
| Need for systematic growth | Embedded learning outcomes in each phase |

**Why This Works for Independent Developers**:

1. **Clear scope**: No distributed systems complexity
2. **Self-contained**: No dependencies on proprietary infrastructure
3. **Research value**: Novel enough to contribute to the field
4. **Portfolio-worthy**: Demonstrates depth, not breadth
5. **Extensible**: Can grow into a larger ecosystem over time

---

## Contributing

AegisRT welcomes contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

AegisRT is released under CC BY-NC-SA 4.0. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

AegisRT draws inspiration from:
- Real-time systems research community (Liu & Layland, George et al.)
- GPU systems research (REEF, Clockwork, Orion)
- NVIDIA Jetson platform and TensorRT teams
- Apache TVM and ONNX Runtime communities

---

**"The problem is not how to run a model fast. The problem is how to run many models predictably—and understand why."**
