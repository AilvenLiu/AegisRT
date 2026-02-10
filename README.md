# AegisRT

**GPU Resource Orchestrator for Deterministic Edge AI Execution**

AegisRT is a GPU resource orchestration layer for multi-model inference on
resource-constrained edge devices. It sits above existing inference runtimes
(TensorRT, TVM Runtime, ONNX Runtime) and manages the resources they compete
for: CUDA streams, device memory, compute time, and scheduling priority.

## The Problem

Running a single model on a GPU is a solved problem. Running **20+ models
concurrently on a single 8 GB edge GPU with deterministic latency guarantees**
is not.

Existing runtimes (TensorRT, TVM, ONNX Runtime) are excellent at kernel
execution. But when multiple models share a constrained GPU, you need:

- **Deterministic scheduling** with provable worst-case latency bounds
- **Cross-model memory orchestration** with lifetime-aware sharing
- **Execution context isolation** so one model's failure does not crash others
- **Admission control** that rejects work rather than degrading silently
- **Full observability** of every scheduling decision and resource allocation

No open-source project provides this today. AegisRT fills that gap.

## Architecture

```
Application (Autonomous Driving, Robotics, Edge AI Services)
                             |
    +------------------------v------------------------------+
    |                     AegisRT                           |
    |            GPU Resource Orchestrator                  |
    |                                                       |
    |  L5: Observability (tracing, metrics, audit trail)    |
    |  L4: Scheduling (RMS, EDF, admission control)         |
    |  L3: Memory Orchestration (cross-model sharing)       |
    |  L2: Context Isolation (per-model budgets)            |
    |  L1: CUDA Abstraction (RAII wrappers)                 |
    |                                                       |
    +-------------+----------------------------+------------+
                  |                            |
    +-------------v-----------+  +-------------v------------+
    |            TRT          |  |          TVM RT          |
    |                         |  |     Existing runtimes    |
    +-------------------------+  +--------------------------+
                  |                            |
    +-------------v----------------------------v------------+
    |                      CUDA Runtime                     |
    +-------------------------------------------------------+
```

**Key insight:** AegisRT does not compete with TensorRT or TVM. It
complements them by solving the orchestration problem they do not address.

## Core Differentiator: Real-Time Scheduling for GPUs

Traditional real-time scheduling (Rate-Monotonic, EDF) assumes preemptible
tasks on a sequential processor. GPUs violate these assumptions: kernels are
non-preemptible, worst-case execution time varies with contention, and the
processor is massively parallel with shared resources.

AegisRT adapts RT scheduling theory for GPU execution:

- **WCET Profiling:** Conservative worst-case execution time bounds per model
- **Admission Control:** Formal schedulability analysis before accepting new
  workloads
- **Non-Preemptive EDF:** Adapted for GPU kernel execution constraints
- **Thermal-Aware Adaptation:** Adjusts guarantees under thermal throttling

## What AegisRT Is Not

- Not a replacement for TensorRT, TVM, or ONNX Runtime
- Not a model compiler or kernel library
- Not a distributed inference framework
- Not a training framework

## Target Hardware

- **Primary:** NVIDIA Jetson Orin SoCs (8-16 GB shared memory)
- **Secondary:** NVIDIA server GPUs (for development and CI)
- **Future:** Extensible to NPUs and other accelerators

## Status

AegisRT is in active development. See [docs/ROADMAP.md](docs/ROADMAP.md) for
the development plan.

## Documentation

- [OUTLOOK.md](docs/OUTLOOK.md) -- Vision, philosophy, and design pillars
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) -- 5-layer system architecture
- [ROADMAP.md](docs/ROADMAP.md) -- Sequential development phases
- [TERMINOLOGY.md](docs/TERMINOLOGY.md) -- Domain terminology

## Licence

CC BY-NC-SA 4.0. See [LICENCE](LICENSE) for details.
