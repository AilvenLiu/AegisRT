# ROADMAP -- r03-device-memory-pool

> Long-form execution manual. Written for an AI agent with no prior context.

---

## 1. Background and Motivation

AegisRT requires a single, tracked allocation point for all GPU device memory.
Without a memory pool:
- Allocations are scattered, leaks go undetected.
- Memory pressure cannot be measured or responded to.
- The MemoryOrchestrator (r11) has no substrate to plan against.
- Alignment requirements (CUDA requires 256-byte alignment) are not enforced.

DeviceCapabilities provides hardware introspection needed by the memory orchestrator
and scheduler (e.g., total memory determines budget limits; Jetson detection enables
unified memory optimisations).

---

## 2. Overall Objective

By the end of this roadmap, ALL of the following MUST be true:

- DeviceMemoryPool allocates and deallocates device memory with full tracking.
- All allocations are aligned to at least 256 bytes.
- Leak detection fires on unreleased allocations at destructor time.
- Double-free detection works and logs an error (does not crash).
- PressureLevel is derived from usage ratio with correct thresholds.
- PoolStats provides accurate fragmentation and peak usage data.
- DeviceCapabilities returns valid hardware data for device 0.
- Jetson detection works correctly.
- All unit tests pass. cuda-memcheck reports zero leaks.

---

## 3. Explicit Non-Goals

- No memory orchestration or sharing analysis (r09-r12).
- No integration with ExecutionContext (r07).
- No TraceCollector integration (added in r11 when orchestrator is built).
- No memory compaction or defragmentation.

---

## 4. High-Level Strategy

### Memory Pool Design

Use a simple bump allocator with a free list for the pool. The pool pre-allocates
a contiguous block of device memory at construction time. Allocations are tracked
in a std::map<void*, AllocationRecord> for O(log n) lookup.

Alignment: all allocations must be aligned to at least 256 bytes (CUDA requirement).
The allocate() method rounds up the requested size to the next alignment boundary.

### Pressure Levels

Pressure levels are derived from the used/capacity ratio:
- None: < 50%
- Low: 50-70%
- Medium: 70-85%
- High: 85-95%
- Critical: > 95%

These thresholds are fixed constants. The MemoryOrchestrator (r11) uses pressure
levels to trigger eviction or rejection policies.

---

## 5. Sub-Phase A: DeviceCapabilities and Hardware Introspection

### Objective

Implement hardware introspection before the memory pool, because the pool constructor
needs to know the device's total memory to validate the requested capacity.

### Task Execution Guidance

task-r03-a-0 (DeviceCapabilities struct):
Fields to include:
- int device_id
- int compute_capability_major, compute_capability_minor
- size_t total_memory_bytes
- int multiprocessor_count
- int max_threads_per_block
- int memory_bus_width_bits
- size_t l2_cache_size_bytes
- bool unified_addressing
- bool concurrent_kernels
- std::string device_name

task-r03-a-1 (query factory):
- static Result<DeviceCapabilities> query(int device_id)
- Validates device_id < cudaGetDeviceCount()
- Calls cudaGetDeviceProperties(&props, device_id)
- Maps cudaDeviceProp fields to DeviceCapabilities fields

task-r03-a-2 (query_all factory):
- static std::vector<DeviceCapabilities> query_all()
- Iterates 0..cudaGetDeviceCount()-1, calls query() for each

task-r03-a-3 (is_jetson detection):
- bool is_jetson() const
- Returns true if: props.integrated == 1 AND (name starts with "Orin" OR "Xavier" OR "Tegra")
- Note: cudaDeviceProp.integrated is 1 for Jetson (unified memory architecture)

task-r03-a-4 (computed fields):
- bool is_integrated_gpu() const: returns props.integrated == 1
- double memory_bandwidth_gb_s() const: computed from memory_bus_width and clock rate

task-r03-a-5 (unit tests):
- Test: query(0) succeeds on test hardware
- Test: query(-1) returns error
- Test: query(cudaGetDeviceCount()) returns error
- Test: is_jetson() returns false on x86_64 test hardware
- Test: all fields are non-zero for valid device

### Exit Criteria for Sub-Phase A

- DeviceCapabilities::query(0) returns valid data on test hardware.
- Invalid device IDs return errors.
- Unit tests pass.

---

## 6. Sub-Phase B: DeviceMemoryPool Core

### Objective

Implement the core allocation/deallocation logic with full tracking.

### Task Execution Guidance

task-r03-b-0 (AllocationRecord):
struct AllocationRecord {
    void* ptr;
    size_t size_bytes;
    size_t alignment;
    std::chrono::steady_clock::time_point allocation_time;
    std::string tag;  // optional, for debugging
};

task-r03-b-1 (Constructor):
- explicit DeviceMemoryPool(size_t capacity_bytes, int device_id = 0)
- Validates capacity_bytes > 0
- Validates capacity_bytes <= DeviceCapabilities::query(device_id).total_memory_bytes
- CUDA_CHECK(cudaMalloc(&base_ptr_, capacity_bytes))
- Initialises free list with single block: [base_ptr_, capacity_bytes]

task-r03-b-2 (allocate):
- Result<void*> allocate(size_t bytes, size_t alignment = 256)
- Validates bytes > 0, alignment is power of 2
- Finds first free block >= bytes + alignment_padding
- Computes aligned pointer within block
- Inserts AllocationRecord into allocations_ map
- Updates used_ counter
- Returns aligned pointer

task-r03-b-3 (deallocate):
- void deallocate(void* ptr)
- Looks up ptr in allocations_ map
- If not found: log error "double-free or invalid pointer", return (do NOT abort)
- If found: remove from map, return block to free list, update used_

task-r03-b-4 (allocate_tagged):
- Result<void*> allocate_tagged(size_t bytes, std::string_view tag, size_t alignment = 256)
- Calls allocate(), then sets AllocationRecord::tag

task-r03-b-5 (Destructor):
- If allocations_ map is not empty: log each entry as error with tag and size
- CUDA_CHECK(cudaFree(base_ptr_))

task-r03-b-6 (Unit tests):
- Test: allocate/deallocate round-trip
- Test: capacity arithmetic (used() + available() == capacity())
- Test: alignment enforcement (returned pointer % 256 == 0)
- Test: double-free logs error but does not crash
- Test: destructor logs leak for unreleased allocation

### Exit Criteria for Sub-Phase B

- allocate/deallocate round-trip works correctly.
- Alignment is enforced.
- Double-free detection works.
- Leak detection fires in destructor.
- Unit tests pass.

---

## 7. Sub-Phase C: Pool Observability and Pressure Tracking

### Objective

Add pressure levels, statistics, and a reset mechanism for testing.

### Task Execution Guidance

task-r03-c-0 (PressureLevel):
enum class PressureLevel { None, Low, Medium, High, Critical };
Thresholds: None <50%, Low 50-70%, Medium 70-85%, High 85-95%, Critical >95%

task-r03-c-1 (current_pressure):
- PressureLevel current_pressure() const
- Computes ratio = used_ / capacity_
- Returns appropriate PressureLevel

task-r03-c-2 (PoolStats):
struct PoolStats {
    size_t capacity_bytes;
    size_t used_bytes;
    size_t available_bytes;
    size_t allocation_count;
    size_t peak_used_bytes;
    double fragmentation_ratio;  // 0.0 = no fragmentation, 1.0 = fully fragmented
};

task-r03-c-3 (stats() and peak tracking):
- PoolStats stats() const
- peak_used_bytes: high-water mark updated on every allocate()
- fragmentation_ratio: 1.0 - (largest_free_block_size / available_bytes)

task-r03-c-4 (reset()):
- void reset()
- Logs warning if called with active allocations
- Clears allocations_ map
- Resets free list to single block covering entire pool
- Resets used_ to 0 (but NOT peak_used_bytes)
- Used in test fixtures only

task-r03-c-5 (Unit tests):
- Test: pressure levels at boundary values (49%, 50%, 70%, 85%, 95%)
- Test: PoolStats accuracy after multiple allocations
- Test: peak_used_bytes tracks high-water mark correctly
- Test: reset() clears state correctly

task-r03-c-6 (Validation):
- cuda-memcheck on all tests: zero leaks
- clang-tidy on all new files: zero warnings
- All public APIs have Doxygen comments

### Exit Criteria for Sub-Phase C

- Pressure levels computed correctly at all thresholds.
- PoolStats accurate.
- reset() works for test isolation.
- cuda-memcheck clean.

---

## 8. Completion Definition

This roadmap is complete when:
- All tasks in all three sub-phases are marked completed in roadmap.yml.
- All exit criteria above are verified.
- cuda-memcheck reports zero leaks.
- A session handoff file exists in sessions/.
- agent_roadmaps/README.md updated to reflect r03 completed and r04 active.
