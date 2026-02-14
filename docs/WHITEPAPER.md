# AegisRT Whitepaper

## Transparent GPU Resource Orchestration for Multi-Model Edge AI

**Version**: 1.0    
**Status**: Draft for Review    
**Date**: 2025    

---

## Executive Summary

AegisRT is a GPU resource orchestration framework designed to address a critical gap in the AI inference ecosystem: the challenge of running multiple neural networks concurrently on resource-constrained edge GPUs with provable latency guarantees and full decision transparency.

While existing inference runtimes (TensorRT, TVM, ONNX Runtime) excel at single-model execution optimization, they provide no mechanism for multi-model coordination, resource admission control, or execution transparency. As edge AI deployments grow in complexity—autonomous vehicles running 15-30 models simultaneously—this gap becomes increasingly problematic.

AegisRT positions itself as an **orchestration layer above existing runtimes**, providing four core capabilities they do not offer:

1. **Resource Transparency**: Every allocation and scheduling decision is traced and explainable
2. **Predictable Performance**: Formal admission control with provable worst-case latency bounds
3. **Execution Isolation**: Per-model resource budgets with fault boundaries
4. **Cross-Model Memory Orchestration**: Lifetime-aware memory sharing across models

This whitepaper presents the motivation, design philosophy, technical approach, and strategic positioning of AegisRT, with particular focus on its viability as an independent developer project with long-term research and engineering value.

---

## Part I: The Problem Space

### 1.1 The Edge AI Convergence

The AI industry is undergoing a structural transformation from cloud-centric to edge-centric deployment. Three converging forces drive this shift:

**Latency Imperatives**: Autonomous vehicles, industrial robots, and real-time perception systems cannot tolerate cloud round-trip latency. A vehicle traveling at 60 mph covers 88 feet per second—a 100ms latency budget is a physical constraint, not an optimization target.

**Privacy and Bandwidth**: Streaming sensor data to the cloud raises privacy concerns and requires massive bandwidth. Processing on-device eliminates both issues.

**Cost and Reliability**: Edge compute amortizes costs over device lifetime and functions without network connectivity. The unit economics favor local processing.

### 1.2 The Complexity Explosion

Modern edge AI deployments are not single-model systems. A typical autonomous driving perception stack runs:

- **15-30 neural networks** for object detection, segmentation, depth estimation, lane detection, traffic sign recognition, sensor fusion, and prediction
- **Multiple camera streams** processed concurrently at different frequencies
- **Fusion models** combining camera, lidar, radar inputs
- **Planning and control models** with different latency requirements

All of this executes on a **single edge GPU** with **limited memory** (8-64GB unified memory) and **strict latency requirements** (often 50-200ms per model).

### 1.3 The Gap in Existing Solutions

Current inference runtimes were designed for a fundamentally different problem: maximizing single-model throughput.

| What Runtimes Optimize | What Edge Systems Need |
|------------------------|------------------------|
| Throughput (inferences/second) | Worst-case latency |
| Single model owns GPU | Multiple models share GPU |
| Best-effort with tail latency | Deterministic with bounded latency |
| Accept all work | Reject work that violates guarantees |
| Opaque execution | Explainable decisions |
| Per-model memory | Cross-model memory coordination |

**The uncomfortable truth**: When you deploy your fifteenth model to an edge GPU and latency spikes unpredictably, existing tools cannot tell you why. There is no admission control, no resource visibility, no audit trail.

### 1.4 The Orchestration Problem

Running a single neural network on a GPU is a solved problem. Running multiple networks concurrently with resource constraints and latency guarantees is an **orchestration problem** that existing runtimes do not address.

This problem has two dimensions:

**Technical Dimension**: How do you schedule GPU kernels from multiple models with non-preemptible execution, variable execution times, and resource interference?

**Operational Dimension**: How do you understand what your system is doing when something goes wrong? Why did Model A's latency spike when Model B was loaded?

AegisRT addresses both dimensions through a unified framework built on real-time scheduling theory and systematic observability.

---

## Part II: Design Philosophy

### 2.1 Core Design Principles

AegisRT is built on four foundational principles that inform every design decision:

#### Principle 1: Orchestration Over Execution

**Tenet**: AegisRT does not execute GPU kernels. It orchestrates existing runtimes by controlling **when** and **with what resources** they execute.

**Rationale**: TensorRT, TVM, and ONNX Runtime have invested decades of engineering in kernel optimization. Competing with them is futile. The unsolved problem is orchestration, not execution.

**Manifestation**: All kernel execution is delegated to `RuntimeBackend` implementations. AegisRT manages streams, memory, events, and scheduling—but never kernels.

#### Principle 2: Formal Guarantees Over Best-Effort

**Tenet**: Scheduling decisions must be provably correct, not empirically tuned.

**Rationale**: Edge autonomous systems require worst-case guarantees. A system that works 99.9% of the time but fails unpredictably is worse than one that explicitly rejects unworkable configurations.

**Manifestation**: Admission control based on formal schedulability analysis derived from real-time systems theory. Conservative WCET estimation with statistical confidence bounds.

#### Principle 3: Observability as a Contract

**Tenet**: Every scheduling decision, resource allocation, and execution event must be traceable.

**Rationale**: Without observability, determinism claims cannot be verified. In production systems, "why did this happen?" is often more valuable than "what happened?"

**Manifestation**: Structured traces for all decisions. Rationale logging for admission/rejection. Offline analysis tools for reconstruction.

#### Principle 4: Composition Over Competition

**Tenet**: AegisRT complements existing ecosystems; it does not compete with them.

**Rationale**: The AI inference ecosystem is rich with excellent tools. AegisRT's value lies in addressing what these tools do not provide.

**Manifestation**: `RuntimeBackend` abstraction for seamless integration. No proprietary formats or compilation pipelines.

### 2.2 What AegisRT Is NOT

Understanding AegisRT requires understanding what it explicitly does not do:

| AegisRT Does NOT | Why |
|------------------|-----|
| Execute kernels | Delegates to TensorRT/TVM/ONNX Runtime |
| Compile models | Consumes pre-compiled models |
| Optimize single-model throughput | Focuses on multi-model predictability |
| Provide distributed inference | Single-node, single-GPU scope |
| Replace CUDA | Uses CUDA as the underlying substrate |

This explicit non-scope defines AegisRT's boundary and prevents scope creep.

### 2.3 Theoretical Foundation

AegisRT's core contribution is adapting classical real-time scheduling theory for GPU execution constraints. This requires addressing fundamental violations:

| Classical RT Assumption | GPU Reality | AegisRT Adaptation |
|------------------------|-------------|-------------------|
| Tasks are preemptible | Kernels run to completion | Non-preemptive EDF/RMS analysis |
| WCET is static | Varies with contention | Contention-aware profiling |
| Single processor | Massively parallel | Model sequentialization |
| Independent tasks | Share memory bandwidth | Resource interference modeling |

These adaptations position AegisRT at the intersection of **real-time systems research** and **AI infrastructure engineering**—an area with significant open questions.

---

## Part III: Technical Architecture

### 3.1 System Positioning

```
+-------------------------------------------------------------------+
|                        Application Layer                          |
|       (Autonomous Driving, Robotics, Edge AI Services)            |
+-------------------------------+-----------------------------------+
                                |
+-------------------------------v-----------------------------------+
|                            AegisRT                                |
|                                                                   |
|  +-----------------+ +-----------------+ +---------------------+  |
|  | TRANSPARENCY    | | PREDICTABILITY  | | ISOLATION           |  |
|  |                 | |                 | |                     |  |
|  | Resource        | | Formal          | | Per-Model           |  |
|  | Visibility      | | Admission       | | Fault               |  |
|  | Decision Audit  | | Control         | | Boundaries          |  |
|  | Trace Replay    | | Latency Bounds  | | Resource Budgets    |  |
|  +-----------------+ +-----------------+ +---------------------+  |
|                                                                   |
+-------------------------------+-----------------------------------+
                                |
              +-----------------+-----------------+
              |                 |                 |
        +-----v-----+     +-----v-----+     +-----v-----+
        |  TensorRT |     |TVM Runtime|     |   ONNX    |
        | "Execute" |     | "Execute" |     | "Execute" |
        +-----------+     +-----------+     +-----------+
```

### 3.2 Three-Layer Architecture

#### Layer 1: CUDA Abstraction & Observability

**Purpose**: Safe, RAII-managed access to CUDA resources with comprehensive traceability.

**Key Components**:
- `CudaStream`, `CudaEvent`: RAII wrappers eliminating resource leaks
- `DeviceMemoryPool`: Explicit allocation with pressure awareness
- `TraceCollector`: Structured event collection for all operations

**Invariants**:
- No raw CUDA handles escape this layer
- All CUDA calls are checked
- All operations are traced

#### Layer 2: Resource Orchestration

**Purpose**: Manage execution contexts, memory allocation, and runtime backends with clear ownership and isolation.

**Key Components**:
- `ExecutionContext`: Per-model resource container with budget enforcement
- `MemoryOrchestrator`: Cross-model memory planning with lifetime-aware sharing
- `RuntimeBackend`: Abstract interface for TensorRT/TVM/ONNX integration

**Invariants**:
- One context per model
- Budgets are hard limits
- Faults are isolated

#### Layer 3: Deterministic Scheduler

**Purpose**: Real-time scheduling with formal admission control, adapting classical theory for GPU constraints.

**Key Components**:
- `WCETProfiler`: Statistical worst-case execution time estimation
- `AdmissionController`: Formal schedulability analysis (RMS, EDF)
- `Scheduler`: Central orchestration with policy-based selection

**Invariants**:
- All decisions provably correct
- All decisions traceable
- No deadline violations for admitted models

### 3.3 Key Algorithms

#### Non-Preemptive EDF Scheduling

In non-preemptive scheduling, a high-priority task may be blocked by a lower-priority task currently executing. This blocking time must be explicitly computed:

Blocking Time: $B_i = \max(C_j)$ for all $j$ with lower priority .

Response Time: $R_i = C_i + B_i + \sum{\frac{R_i}{T_j}} \times C_j$, 
              (for all j with higher priority) .

Deadline Met: $R_i \leq D_i$. 

#### Contention-Aware WCET Estimation

GPU execution time varies with co-running workloads. AegisRT profiles execution under controlled contention and applies statistical safety margins:

$WCET = (mean + confidence\_interval) \times safety\_margin$

where:   
  confidence_interval = $z \cdot \frac{\sigma}{\sqrt{n}}$,    
  safety_margin = 1.5 x (default, 50% headroom),   
  z = z-score for desired confidence level.   

#### Lifetime-Aware Memory Sharing

Memory regions can be shared between tensors with non-overlapping lifetimes:

```
1. Compute tensor lifetimes from execution graph
2. Identify non-overlapping lifetime pairs
3. Build memory plan using interval scheduling
4. Allocate shared regions for compatible tensors
```

---

## Part IV: Research and Engineering Value

### 4.1 Research Contributions

AegisRT addresses open research questions in GPU scheduling:

#### 1. Non-Preemptive GPU Scheduling

- How does kernel size distribution affect blocking time bounds?
- Can we predict blocking time from model architecture?
- What is optimal task granularity for GPU scheduling?

#### 2. Contention-Aware WCET Estimation

- What contention factors are predictable?
- How do we build safety margins for unknown contention?
- Can we profile contention effects efficiently?

#### 3. Resource Interference Modeling

- How do we model memory bandwidth contention?
- What is the relationship between SM utilization and latency?
- Can we predict cross-model interference from profiles?

#### 4. Thermal-Aware Scheduling (Future)

- How do we adapt guarantees under thermal throttling?
- Can we predict thermal headroom from workload characteristics?

### 4.2 Engineering Value

For practitioners, AegisRT provides:

| Capability | Practical Value |
|------------|-----------------|
| Admission Control | Know before deployment if configuration is feasible |
| Latency Bounds | Guarantee worst-case response time |
| Resource Transparency | Understand why latency spikes occur |
| Memory Efficiency | Reduce memory footprint through sharing |
| Decision Audit | Debug production issues with full context |

### 4.3 Academic Foundations

AegisRT draws from established research:

| Domain | Key References |
|--------|---------------|
| Real-Time Systems | Liu & Layland (1973), George et al. (1996) |
| GPU Systems | REEF (SOSP'23), Clockwork (OSDI'20), Orion (ATC'23) |
| Memory Management | Buffer pools, arena allocators, lifetime analysis |

---

## Part V: Strategic Positioning

### 5.1 Market Landscape

AegisRT operates in a strategic gap:

| Category | Projects | Relationship to AegisRT |
|----------|----------|------------------------|
| Inference Runtimes | TensorRT, TVM, ONNX | Orchestration layer above |
| ML Frameworks | PyTorch, TensorFlow | Deployment tool, not training |
| Distributed Inference | Ray Serve, Triton, vLLM | Single-node complement |
| Kernel Libraries | cuDNN, cutlass | Delegate execution |

### 5.2 Target Users

1. **Edge AI Engineers**: Running multi-model workloads on constrained GPUs, struggling with resource contention
2. **Autonomous Systems Developers**: Building safety-critical perception with hard deadlines
3. **Real-Time Systems Researchers**: Investigating GPU scheduling and WCET analysis
4. **AI Infrastructure Engineers**: Designing deployment pipelines for edge AI

### 5.3 Competitive Differentiation

| What Others Provide | What AegisRT Provides |
|--------------------|----------------------|
| Fast single-model execution | Predictable multi-model execution |
| Best-effort scheduling | Formal admission control |
| Opaque resource usage | Transparent resource visibility |
| Per-model memory | Cross-model memory orchestration |
| Trial-and-error deployment | Provably correct configurations |

---

## Part VI: Independent Developer Viability

### 6.1 Design Constraints

AegisRT is designed for independent development with specific constraints:

| Constraint | AegisRT Response |
|------------|-----------------|
| Limited hardware (one Jetson Orin) | Single-node, single-GPU focus |
| No production traffic | Research-quality before production-quality |
| Limited time (part-time) | Incremental phases with clear exit criteria |
| Need for systematic growth | Embedded learning outcomes |

### 6.2 Why This Works for Independent Developers

1. **Clear Scope**: No distributed systems complexity
2. **Self-Contained**: No dependencies on proprietary infrastructure
3. **Research Value**: Novel enough to contribute to the field
4. **Portfolio-Worthy**: Demonstrates depth, not breadth
5. **Extensible**: Can grow into larger ecosystem over time

### 6.3 Personal Growth Dimensions

#### Depth Over Breadth

Instead of jumping between disparate problems, AegisRT focuses on a single deep challenge:

- **Systematic knowledge building**: Each component builds on previous
- **Research contribution**: Novel problems yield novel solutions
- **Portfolio depth**: One deep project beats many shallow projects

#### Theory Meets Practice

- **Theoretical foundation**: RT scheduling, WCET analysis, formal methods
- **Practical application**: CUDA programming, edge deployment, performance engineering
- **Research contribution**: Adapting theory for new constraints

#### Market-Relevant Skills

| Skill Area | Market Value |
|------------|-------------|
| Edge AI deployment | Growing market with talent shortage |
| Real-time systems | Critical for autonomous systems, industrial IoT |
| GPU systems | Foundation for modern AI infrastructure |
| Open-source contribution | Demonstrates sustained engineering capability |

### 6.4 Learning Path Embedded in Development

| Phase | Technical Skills | Research Skills | Engineering Skills |
|-------|-----------------|-----------------|-------------------|
| 0-1 | CUDA, RAII, Build Systems | Literature Review | Testing, CI/CD |
| 2-3 | Memory Management, Isolation | Problem Formulation | API Design |
| 4-5 | Scheduling Algorithms | Algorithm Design | Performance Testing |
| 6+ | Optimization, Integration | Publication Writing | Documentation |

---

## Part VII: Risk Assessment

### 7.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| WCET too conservative | Medium | Low | Accept trade-off, document clearly |
| Scheduling overhead high | Low | Medium | Profile, optimize critical path |
| Determinism not achievable | Low | High | Investigate, may require hardware tuning |
| Hardware dependencies | Medium | Medium | Maintain abstraction layer |

### 7.2 Strategic Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| No community interest | Medium | Medium | Target specific niche, build value first |
| Runtimes add features | Medium | Low | Focus on orchestration, not execution |
| Market shifts | Low | Medium | Maintain flexibility, monitor trends |

### 7.3 Personal Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Burnout | Medium | High | Phased approach, sustainable pace |
| Motivation drop | Medium | Medium | Visible progress, public accountability |
| Life changes | Medium | Medium | Document thoroughly, enable continuation |

---

## Part VIII: Success Metrics

### 8.1 Technical Success

- [ ] AegisRT provides measurable value over baseline
- [ ] Admission control is provably correct
- [ ] Latency CV < 5% under multi-model load
- [ ] All decisions explainable from trace

### 8.2 Community Success

- [ ] Other engineers find AegisRT useful
- [ ] Researchers cite AegisRT in papers
- [ ] Contributors join the project
- [ ] Project fills recognized ecosystem gap

### 8.3 Personal Success

- [ ] Deep expertise in GPU systems and real-time scheduling
- [ ] Published research or influential technical writing
- [ ] Portfolio project demonstrating systematic thinking
- [ ] Foundation for future independent work

---

## Conclusion

AegisRT addresses a real, growing problem in the AI inference ecosystem: the challenge of orchestrating multiple models on resource-constrained edge GPUs with predictability and transparency.

The project is strategically positioned:
- **Not competing** with established runtimes, but **completing** them
- **Addressing a gap** that grows as edge AI deployments become more complex
- **Grounded in theory** (real-time scheduling) with **practical application** (edge deployment)
- **Viable for independent development** with clear scope and embedded learning

For the independent developer, AegisRT offers:
- A path to **deep expertise** rather than superficial breadth
- A **portfolio-worthy project** demonstrating systematic engineering
- **Research contribution potential** in an active area
- **Market-relevant skills** in a growing field

The journey from concept to MVP is designed to be achievable in 3-4 months of part-time development, with each phase delivering tangible value and systematic learning.

---

**"The goal is not to build another runtime. The goal is to solve the problem that runtimes do not—the problem of orchestrating multiple models predictably and transparently on resource-constrained edge GPUs."**

---

## References

### Foundational Papers

- Liu, C. L., & Layland, J. W. (1973). Scheduling algorithms for multiprogramming in a hard-real-time environment. *Journal of the ACM*.
- Liu, J. W. S. (2000). *Real-Time Systems*. Prentice Hall.
- George, L., Rivierre, N., & Spuri, M. (1996). Preemptive and non-preemptive real-time uniprocessor scheduling. *INRIA Research Report*.

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
