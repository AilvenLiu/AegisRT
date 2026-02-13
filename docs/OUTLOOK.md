# AegisRT Outlook

## Vision Statement

AegisRT is a **deterministic GPU orchestration framework** that brings real-time systems theory to the domain of multi-model edge AI inference. Its mission is to solve a problem that existing runtimes do not address:

> **How do you run multiple neural networks concurrently on a resource-constrained GPU with provable worst-case latency guarantees?**

This is not a performance optimisation problem. This is a **real-time scheduling problem** with unique GPU constraints that classical theory does not address. AegisRT fills this gap.

---

## Why This Problem Matters

### The Rise of Edge AI

The AI industry is experiencing a fundamental shift from cloud-centric to edge-centric deployment. Autonomous vehicles, industrial robots, and intelligent edge services require:

- **Multiple models running concurrently**: A typical autonomous driving pipeline runs 15-30 models for perception, prediction, and planning
- **Strict latency requirements**: Perception latency directly impacts safety; 10ms can be the difference between collision and avoidance
- **Resource constraints**: Edge devices have limited memory, compute, and thermal headroom
- **Predictable behaviour**: Latency spikes are unacceptable in safety-critical systems

### The Gap in Existing Solutions

Current inference runtimes optimise for **single-model throughput**, not multi-model determinism:

| Metric | What Runtimes Optimise | What Edge Systems Need |
|--------|----------------------|----------------------|
| Primary Goal | Throughput (inferences/second) | Worst-case latency |
| Resource Model | Single model owns GPU | Multiple models share GPU |
| Behaviour | Best-effort with tail latency | Deterministic with bounded latency |
| Admission | Accept all work | Reject work that violates guarantees |

**The market need is clear**: As edge AI deployment scales, the ability to guarantee latency under resource constraints becomes a competitive differentiator, not a nice-to-have.

### The Academic Opportunity

Real-time scheduling theory is well-established for CPU systems. GPU scheduling, however, remains an open research area:

```
+----------------------------------------------------+
|                 Research Landscape                 |
+----------------------------------------------------+
|                         |                          |
|     CPU Scheduling      |     GPU Scheduling       |
+-------------------------+--------------------------+
|   Mature theory         |   Open research area     |
|   Industrial standards  |   No standard approaches |
|   POSIX RT, AUTOSAR     |   No equivalent for GPU  |
|   WCET tools exist      |   GPU WCET is unsolved   |
+----------------------------------------------------|
|           AegisRT addresses this gap               |
+----------------------------------------------------+
```

By contributing to GPU scheduling research, AegisRT positions itself at the intersection of two growing fields: real-time systems and AI infrastructure.

---

## Core Differentiator: GPU-Aware Real-Time Scheduling

### The Theoretical Challenge

Classical real-time scheduling theory (Liu & Layland, 1973) assumes:

1. **Tasks are preemptible**: The scheduler can interrupt a running task at any time
2. **WCET is known and static**: Worst-case execution time is a fixed parameter
3. **Single sequential processor**: One CPU core executes tasks one at a time
4. **Independent tasks**: Tasks do not interfere with each other

**GPUs violate every single assumption.**

### How AegisRT Adapts

| Classical Assumption | GPU Reality | AegisRT Adaptation |
|---------------------|-------------|-------------------|
| Preemptible tasks | Kernels run to completion | Non-preemptive scheduling analysis |
| Static WCET | WCET varies with contention | Contention-aware WCET profiling |
| Sequential processor | Massively parallel | Model sequentialisation with resource partitioning |
| Independent tasks | Shared memory bandwidth | Resource-aware schedulability tests |

### Research Contributions

AegisRT aims to contribute to the following research directions:

#### 1. Non-Preemptive GPU Scheduling

Adapting EDF and RMS theory for non-preemptive execution requires computing **blocking time** — the maximum time a high-priority task must wait for a lower-priority task to complete. This is well-understood for CPUs but under-explored for GPUs.

**Open Questions**:
- How does kernel size distribution affect blocking time bounds?
- Can we predict blocking time from model architecture?
- What is the optimal task granularity for GPU scheduling?

#### 2. Contention-Aware WCET Estimation

GPU execution time varies significantly based on co-running workloads. Memory bandwidth, cache, and SM contention can double or triple execution time.

**Open Questions**:
- What contention factors are predictable?
- How do we build safety margins for unknown contention?
- Can we profile contention effects efficiently?

#### 3. Thermal-Aware Scheduling

Edge devices experience thermal throttling, which affects both performance and latency guarantees.

**Open Questions**:
- How do we adapt guarantees under thermal throttling?
- Can we predict thermal headroom from workload characteristics?
- What scheduling policies minimise thermal impact?

---

## Design Philosophy

### Pillar 1: Determinism Over Throughput

**Principle**: Edge autonomous systems value predictability over peak performance.

**Implication**: AegisRT prioritises worst-case latency over average throughput. This manifests as:
- Bounded queue depths (no unbounded buffering)
- Explicit admission control (reject work rather than degrade)
- Non-work-conserving scheduling (idle GPU is acceptable if guarantees are maintained)

**Trade-off**: Under light load, GPU utilisation may be suboptimal. This is acceptable.

### Pillar 2: Formal Over Heuristic

**Principle**: Scheduling decisions should be provably correct, not empirically tuned.

**Implication**: AegisRT uses formal schedulability analysis derived from real-time theory. This manifests as:
- Admission control based on utilisation bounds
- Response-time analysis for deadline guarantees
- Conservative WCET estimates with statistical confidence

**Trade-off**: Formal analysis may reject configurations that would work in practice. This is acceptable.

### Pillar 3: Observable Over Opaque

**Principle**: Every scheduling decision should be explainable.

**Implication**: AegisRT treats observability as a contract, not a feature. This manifests as:
- Structured traces for every decision
- Rationale logging for admission/rejection
- Offline analysis tools for reconstruction

**Trade-off**: Tracing adds overhead. This is acceptable for correctness.

### Pillar 4: Composition Over Competition

**Principle**: AegisRT complements existing runtimes; it does not compete with them.

**Implication**: AegisRT delegates kernel execution to TensorRT/TVM/ONNX Runtime. This manifests as:
- RuntimeBackend abstraction for integration
- Focus on what runtimes do NOT provide
- No kernel implementation or optimisation

**Trade-off**: AegisRT cannot optimise kernel-level execution. This is acceptable.

---

## Market Position and Competitive Landscape

### What AegisRT Is Not Competing With

| Category | Representative Projects | Why Not Competition |
|----------|------------------------|---------------------|
| Inference Runtimes | TensorRT, TVM, ONNX Runtime | AegisRT orchestrates them, not replaces |
| ML Frameworks | PyTorch, TensorFlow, JAX | AegisRT is deployment-focused, not training |
| Distributed Inference | Ray Serve, Triton, vLLM | AegisRT is single-node, single-GPU |
| Kernel Libraries | cuDNN, cutlass, Tensor Cores | AegisRT delegates kernel execution |

### AegisRT's Unique Position

```
+--------------------------------------------------------------+
|                  Inference Stack Positioning                 |
+--------------------------------------------------------------+
|                                                              |
|           High-Level Frameworks (Training & Serving)         |
|             PyTorch, TensorFlow, Ray Serve, vLLM             |
|                              |                               |
|   +--------------------------v---------------------------+   |
|   |                       AegisRT                        |   |
|   |                 "The Missing Layer"                  |   |
|   |                                                      |   |
|   |   Deterministic Scheduling + Admission Control +     |   |
|   |      Cross-Model Memory + Execution Isolation        |   |
|   |                                                      |   |
|   +--------------------------+---------------------------+   |
|                              |                               |
|                              v                               |
|            Inference Runtimes (Kernel Execution)             |
|             TensorRT, TVM Runtime, ONNX Runtime              |
|                              |                               |
|                              v                               |
|           GPU Hardware (CUDA, NVIDIA, AMD, Intel)            |
|                                                              |
+--------------------------------------------------------------+
```

### Target Users

1. **Edge AI Engineers**: Building multi-model pipelines on Jetson,Qualcomm, or similar platforms
2. **Autonomous Systems Developers**: Safety-critical perception systems with hard deadlines
3. **Real-Time Systems Researchers**: Investigating GPU scheduling and WCET analysis
4. **AI Infrastructure Engineers**: Designing deployment pipelines for edge AI

---

## Long-Term Vision

### Phase 1: Foundation (Current)

- Establish GPU scheduling framework
- Validate real-time theory adaptation
- Build community around deterministic edge AI

### Phase 2: Integration (Year 1-2)

- TensorRT, TVM, ONNX Runtime backend support
- Jetson Orin optimisation
- Industry pilot projects

### Phase 3: Ecosystem (Year 2-3)

- Hardware abstraction for other accelerators (NPU, DSP)
- Integration with edge orchestration platforms
- Academic collaborations and publications

### Phase 4: Standardisation (Year 3+)

- Contribute to industry standards for edge AI deployment
- Reference implementation for GPU real-time scheduling
- Educational resources and training

---

## Personal Growth Dimensions

### Why This Project Serves Your Development

AegisRT is designed to address the specific gap you identified: **busy work without systematic growth**.

#### Dimension 1: Depth Over Breadth

Instead of jumping between disparate problems (DLB, GPU Manager, CUDA operators, TVM), AegisRT focuses on a single, deep technical challenge. This enables:

- **Systematic knowledge building**: Each component builds on the previous
- **Research contribution**: Novel problems yield novel solutions
- **Portfolio depth**: One deep project > many shallow projects

#### Dimension 2: Theory Meets Practice

AegisRT bridges real-time systems theory and GPU systems practice:

- **Theoretical foundation**: RT scheduling, WCET analysis, formal methods
- **Practical application**: CUDA programming, edge deployment, performance engineering
- **Research contribution**: Adapting theory for new constraints

#### Dimension 3: Independent Viability

AegisRT is designed for **independent development**:

- **Hardware requirements**: Single Jetson Orin or cloud GPU
- **No proprietary dependencies**: All tools and frameworks are open source
- **Self-contained scope**: Single-node, single-GPU focus

#### Dimension 4: Market Relevance

The skills developed through AegisRT are increasingly valuable:

- **Edge AI deployment**: Growing market with talent shortage
- **Real-time systems**: Critical for autonomous systems, industrial IoT
- **GPU systems**: Foundation for all modern AI infrastructure

### Learning Path Embedded in Development

| Phase | Technical Skills | Research Skills | Engineering Skills |
|-------|-----------------|-----------------|-------------------|
| 0-1 | CUDA, RAII, Build Systems | Literature Review | Testing, CI/CD |
| 2-3 | Memory Management, Isolation | Problem Formulation | API Design |
| 4-5 | Scheduling Algorithms | Algorithm Design | Performance Testing |
| 6+ | Optimisation, Integration | Publication Writing | Documentation |

---

## What Success Looks Like

### Technical Success

- AegisRT provides measurable value over baseline (no scheduler)
- Admission control is provably correct
- Latency distributions are deterministic (CV < 5%)

### Community Success

- Other engineers find AegisRT useful for their edge AI projects
- Researchers cite AegisRT in GPU scheduling papers
- Contributors join the project

### Personal Success

- Deep expertise in GPU systems and real-time scheduling
- Published research or technical blog posts
- Portfolio project that demonstrates systematic thinking
- Foundation for future independent work

---

## Risks and Mitigations

### Risk: Runtimes Are Commodity

**Mitigation**: AegisRT is explicitly NOT a runtime. It is an **orchestration layer** that addresses what runtimes do not. Focus messaging on scheduling, not execution.

### Risk: GPU Scheduling Is Too Hard

**Mitigation**: Start with simplified assumptions (single GPU, static graphs, known models). Expand scope incrementally as understanding deepens.

### Risk: No Community Interest

**Mitigation**: Target a specific niche (edge AI engineers with multi-model latency requirements). Build value before building community.

### Risk: Hardware Dependencies

**Mitigation**: Maintain CUDA abstraction layer. Design for portability from the start. Test on both Jetson and server GPUs.

---

## References

### Foundational Papers

- Liu, C. L., & Layland, J. W. (1973). Scheduling algorithms for multiprogramming in a hard-real-time environment. *JACM*.
- Liu, J. W. S. (2000). *Real-Time Systems*. Prentice Hall.
- George, L., Rivierre, N., & Spuri, M. (1996). Preemptive and non-preemptive real-time uniprocessor scheduling. *INRIA*.

### GPU Systems Research

- Zhang, Q., et al. (REEF, SOSP 2023). Reactive GPU execution for real-time AI services.
- Gujarati, A., et al. (Clockwork, OSDI 2020). Serving DNNs in real-time at datacenter scale.
- Yu, G., et al. (Orion, ATC 2023). Accelerating DNN inference with GPU-aware memory management.
- Chen, J., et al. (Prem, ASPLOS 2023). Prem: Predictable GPU execution for real-time systems.

### Related Projects

- NVIDIA TensorRT: https://developer.nvidia.com/tensorrt
- Apache TVM: https://tvm.apache.org/
- ONNX Runtime: https://onnxruntime.ai/
- PREEMPT-RT Linux: https://wiki.linuxfoundation.org/realtime/

---

**"The goal is not to build another runtime. The goal is to solve the problem that runtimes do not."**
