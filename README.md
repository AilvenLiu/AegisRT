# AegisRT

**Deterministic GPU Orchestration Framework for Multi-Model Edge AI**

[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE)

---

## The Gap AegisRT Fills

### The Problem That No Runtime Solves

Running a single neural network on a GPU is a solved problem. TensorRT, TVM Runtime, and ONNX Runtime have mastered this challenge with impressive performance. However, a fundamentally different problem remains unsolved in the open-source ecosystem:

> **How do you run 20+ models concurrently on an 8GB edge GPU with provable worst-case latency guarantees?**

This is not a performance optimisation problem—it is a **real-time scheduling problem** with unique GPU constraints that classical real-time theory does not address.

### Why Existing Runtimes Fall Short

| Capability | TensorRT | TVM Runtime | ONNX Runtime | AegisRT |
|-----------|----------|-------------|--------------|---------|
| Single-model optimisation | Excellent | Excellent | Good | N/A (delegates) |
| Multi-model concurrency | Best-effort | Best-effort | Best-effort | **Deterministic** |
| Worst-case latency bounds | No | No | No | **Provable** |
| Admission control | No | No | No | **Formal analysis** |
| Cross-model memory sharing | Manual | Manual | No | **Automatic** |
| Scheduling decision audit | No | No | No | **Full traceability** |

**Key Insight:** Existing runtimes optimise for *average-case throughput*. Edge autonomous systems require *worst-case latency guarantees*. This is an orthogonal problem.

---

## What AegisRT Is (And Is Not)

### What AegisRT Is

AegisRT is a **GPU orchestration layer** that sits **above** existing inference runtimes and provides:

1. **Deterministic Scheduling**: Real-time scheduling theory (RMS, EDF) adapted for non-preemptive GPU execution
2. **Formal Admission Control**: Schedulability analysis that proves latency bounds before accepting workloads
3. **Cross-Model Memory Orchestration**: Lifetime-aware memory sharing with statically computable peak usage
4. **Execution Context Isolation**: Per-model resource budgets with fault boundaries
5. **Full Observability**: Every scheduling decision traceable with rationale

### What AegisRT Is Not

- **Not a kernel executor** — Delegates to TensorRT/TVM/ONNX Runtime for actual inference
- **Not a model compiler** — Consumes pre-compiled models from existing toolchains
- **Not a replacement for TensorRT/TVM** — Complements them by solving what they do not address
- **Not a distributed inference framework** — Single-node, single-GPU focus

### The Complementarity Principle

```
+-----------------------------------------------------------------+
|                         Application Layer                       |
|         (Autonomous Driving, Robotics, Edge AI Services)        |
+-------------------------------+---------------------------------+
                                |
+-------------------------------v---------------------------------+
|                            AegisRT                              |
|           "What existing runtimes do NOT provide"               |
|                                                                 |
|   +-----------------+ +-----------------+ +-----------------+   |
|   | Deterministic   | | Admission       | | Cross-Model     |   |
|   | Scheduling      | | Control         | | Memory Orch.    |   |
|   | (RMS, EDF, WCET)| | (Formal)        | | (Lifetime-aware)|   |
|   +-----------------+ +-----------------+ +-----------------+   |
+-------------------------------+---------------------------------+
                                |
              +-----------------+-----------------+
              |                 |                 |
        +-----v-----+     +-----v-----+     +-----v-----+
        | TensorRT  |     |TVM Runtime|     |   ONNX    |
        |           |     |           |     |  Runtime  |
        | "What they|     | "What they|     | "What they|
        |  DO best" |     |  DO best" |     |  DO best" |
        +-----+-----+     +-----+-----+     +-----+-----+
              |                 |                 |
              +-----------------+-----------------+
                                |
                        +-------v-------+
                        |  CUDA Driver  |
                        +---------------+
```

AegisRT does not compete with TensorRT or TVM. It **enables their safe coexistence** under resource constraints.

---

## Core Technical Contribution: GPU-Aware Real-Time Scheduling

### The Research Gap

Classical real-time scheduling theory (Liu & Layland, 1973) makes assumptions that GPUs fundamentally violate:

| Classical RT Assumption | GPU Reality | Implication |
|------------------------|-------------|-------------|
| Tasks are preemptible | GPU kernels are non-preemptible | Cannot interrupt long kernels |
| WCET is static | WCET varies with contention | Need contention-aware profiling |
| Single processor | Massively parallel with shared resources | Resource interference not modelled |
| Independent tasks | Models share memory bandwidth, cache | Coupled execution behaviour |

**AegisRT's research contribution**: Adapting real-time scheduling theory for GPU execution constraints through:

1. **Conservative WCET Profiling**: Statistical methods for worst-case execution time estimation under contention
2. **Non-Preemptive EDF Scheduling**: Adapted schedulability analysis for non-preemptive task models
3. **Contention-Aware Admission Control**: Schedulability tests that account for shared resource interference
4. **Thermal-Aware Adaptation**: Dynamic adjustment of guarantees under thermal throttling

### Academic Foundations

AegisRT draws from and contributes to:

- **Real-Time Systems**: Rate-Monotonic Scheduling (Liu & Layland, 1973), EDF (Liu, 1969), Non-preemptive scheduling (George et al., 1996)
- **GPU Systems Research**: REEF (SOSP'23), Clockwork (OSDI'20), Orion (ATC'23), PREEMPT-RT Linux
- **Memory Management**: Buffer pool management, lifetime analysis, arena allocators

---

## Target Hardware and Use Cases

### Primary Target

- **NVIDIA Jetson Orin** (8-64GB unified memory)
- Edge autonomous systems with strict latency requirements
- Multi-model inference pipelines (perception, planning, control)

### Secondary Target

- NVIDIA server GPUs (for development, CI, and profiling)
- Cloud-edge hybrid scenarios where edge behaviour must be predictable

### Ideal Use Cases

1. **Autonomous Driving Pipelines**: 15-30 models running concurrently with hard real-time constraints
2. **Industrial Robotics**: Multi-camera, multi-model perception with safety-critical deadlines
3. **Edge AI Services**: Multi-tenant inference on shared edge infrastructure

---

## Architecture at a Glance

AegisRT decomposes into three primary layers:

```
+------------------------------------------------------+
|          Layer 3: Deterministic Scheduler            |
|                                                      |
|  +--------------+ +--------------+ +--------------+  |
|  | WCETProfiler | | Admission    | | Scheduling   |  |
|  | (Profiling)  | | Controller   | | Policy       |  |
|  |              | | (Analysis)   | | (RMS, EDF)   |  |
|  +--------------+ +--------------+ +--------------+  |
+---------------------------+--------------------------+
                            |
+---------------------------v--------------------------+
|             Layer 2: Resource Orchestration          |
|                                                      |
|  +--------------+ +--------------+ +--------------+  |
|  | Memory       | | Execution    | | Runtime      |  |
|  | Orchestrator | | Context      | | Backend      |  |
|  | (Cross-Model)| | (Isolation)  | | (TensorRT...)|  |
|  +--------------+ +--------------+ +--------------+  |
+------------------------------------------------------+
                            |
+------------------------------------------------------+
|        Layer 1: CUDA Abstraction & Observability     |
|                                                      |
|  +--------------+ +--------------+ +--------------+  |
|  | CUDA RAII    | | Device       | | Trace        |  |
|  | Wrappers     | | Discovery    | | Collector    |  |
|  +--------------+ +--------------+ +--------------+  |
+------------------------------------------------------+
```

**Key Principle**: Layer 3 (Scheduler) is the core contribution. Layers 1-2 are necessary infrastructure but not the differentiation.

---

## Project Status and Roadmap

AegisRT is in active development. See [ROADMAP.md](docs/ROADMAP.md) for the detailed development plan.

### Current Phase

**Phase 0: Foundations** — CUDA abstraction layer and build infrastructure.

### Milestone Goals

| Phase | Focus | Exit Criteria |
|-------|-------|---------------|
| 0 | CUDA Abstraction | RAII wrappers, memory pool, cross-compilation |
| 1 | Context Isolation | Per-model budgets, fault boundaries |
| 2 | Memory Orchestration | Cross-model sharing, pressure handling |
| 3 | **Deterministic Scheduler** | WCET profiling, admission control, RT policies |
| 4 | Observability & Validation | Full tracing, determinism validation |

---

## Documentation

| Document | Purpose |
|----------|---------|
| [OUTLOOK.md](docs/OUTLOOK.md) | Vision, research foundations, design philosophy |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture and component design |
| [ROADMAP.md](docs/ROADMAP.md) | Sequential development phases with exit criteria |
| [MVP.md](docs/MVP.md) | Minimum Viable Product definition |
| [TERMINOLOGY.md](docs/TERMINOLOGY.md) | Domain terminology glossary |

---

## Who Should Use AegisRT

### Target Users

- **Edge AI Engineers**: Running multi-model workloads on constrained GPUs
- **Autonomous Systems Developers**: Building perception pipelines with hard deadlines
- **Real-Time Systems Researchers**: Investigating GPU scheduling theory
- **AI Infrastructure Engineers**: Designing deployment pipelines for edge AI

### Prerequisites

- Familiarity with CUDA programming concepts
- Understanding of real-time systems fundamentals (helpful but not required)
- Experience with TensorRT, TVM, or ONNX Runtime

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

AegisRT is released under CC BY-NC-SA 4.0. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

AegisRT draws inspiration from:
- Real-time systems research community
- NVIDIA Jetson platform team
- TensorRT, TVM, and ONNX Runtime projects
- REEF, Clockwork, and Orion research papers

---

**"The problem is not how to run a model fast. The problem is how to run many models predictably."**
