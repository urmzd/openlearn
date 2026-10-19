# NVIDIA

Everything at NVIDIA revolves around parallel computation on GPUs, and its interviews assume that even "software" roles understand GPU architecture. The process is highly domain-specific: roles range from driver development to AI platform engineering, and the interview content varies significantly by team. What they share is a performance obsession. Memory access patterns, cache utilisation, and instruction-level parallelism are the vocabulary of every round, and microseconds matter.

This chapter prepares you for NVIDIA engineering roles with a focus on GPU computing, AI infrastructure, and systems programming. It begins with the GPU architecture you must understand for any NVIDIA interview, then works through the coding rounds (naive and tiled CUDA matrix multiplication, parallel reduction, a lock-free stack, and a GPU memory allocator), the system design round on multi-GPU training, inference optimisation, cluster scheduling, and GPU cloud, the core performance optimisations, and the hiring manager round.

> **Learning objectives**
> - Describe the NVIDIA GPU hierarchy from GPC to warp, and the latency of each level of GPU memory.
> - Write naive and shared-memory tiled matrix multiplication kernels and explain why tiling helps.
> - Implement a parallel sum reduction and explain the warp-level step and its synchronisation requirements.
> - Implement a first-fit GPU memory allocator with alignment and coalescing, and measure its fragmentation.
> - Design multi-GPU training, a TensorRT-style optimiser, and a topology-aware GPU scheduler.

## Before you begin

You should be comfortable in C and C++ (pointers, atomics, and memory ordering). The [Hudson River Trading](../hrt/README.md) chapter covers CPU caches and lock-free queues, which this chapter extends to GPUs, and [ML infrastructure](../shared-concepts/ml-infrastructure.md) covers the distributed training concepts behind the design round.

## The interview process

Expect three to six weeks and four to six rounds:

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, team matching |
| Phone Screen | Technical | 45-60 min | Coding + domain knowledge |
| Onsite 1 | Coding | 60 min | Systems programming, C++/CUDA |
| Onsite 2 | Domain Deep Dive | 60 min | GPU architecture, parallel computing |
| Onsite 3 | System Design | 60 min | GPU infrastructure, ML systems |
| Onsite 4 | Hiring Manager | 45 min | Culture, experience, motivation |

For a senior engineer in the US, base salary is $180-250K, RSUs are about $200-500K per year, and the bonus is 10-15% of base. Total compensation is about $400-750K at senior and $700K-1.2M or more at principal and staff. NVIDIA's stock has been explosive, and that equity appreciation has made it one of the highest-paying companies.

## What NVIDIA looks for

GPU-centric thinking comes first: even software roles require understanding GPU architecture. For systems and infrastructure roles, CUDA is the lingua franca, expected or at least valued. AI infrastructure is the growth engine; data centre GPUs (H100, B200), DGX, NVIDIA AI Enterprise, and TensorRT are the profit centres. Interviewers are performance-obsessed, thinking in memory access patterns, cache utilisation, and instruction-level parallelism. Systems programming depth matters, in heavy C and C++, with an understanding of memory hierarchies, SIMD, and vectorisation. And NVIDIA covers the full stack from silicon to cloud: hardware, drivers, libraries, frameworks, and cloud services.

## GPU architecture

You must understand this material for any NVIDIA interview. A GPU is a hierarchy of clusters. The chip contains several graphics processing clusters (GPCs), each with texture processing clusters (TPCs), each holding two streaming multiprocessors (SMs). The SM is where execution happens: it has CUDA cores for FP32 and INT32 work, Tensor Cores for matrix operations, shared memory and L1 cache, a register file, and warp schedulers that issue instructions to warps of 32 threads. Outside the SMs sit the L2 cache, the memory controllers, and HBM.

```
[GPU]
├── [GPC (Graphics Processing Cluster)] x N
│   ├── [TPC (Texture Processing Cluster)] x M
│   │   ├── [SM (Streaming Multiprocessor)] x 2
│   │   │   ├── CUDA Cores (FP32/INT32)
│   │   │   ├── Tensor Cores (Matrix ops)
│   │   │   ├── Shared Memory / L1 Cache
│   │   │   ├── Register File
│   │   │   └── Warp Scheduler
│   │   │       └── Warps (32 threads each)
│   │   └── [SM]
│   └── [TPC]
├── [L2 Cache]
├── [Memory Controllers]
└── [HBM (High Bandwidth Memory)]
```

The programming model maps onto that hardware through a handful of concepts:

| Concept | Description |
|---------|-------------|
| **Warp** | 32 threads executing in lockstep (SIMT) |
| **SM** | Streaming Multiprocessor, the basic execution unit |
| **Thread Block** | Group of threads (up to 1024) sharing shared memory |
| **Grid** | Collection of thread blocks launched by a kernel |
| **Shared Memory** | Fast on-chip memory shared within a thread block (~48-164 KB) |
| **Global Memory** | HBM, high bandwidth but high latency (~80 GB on H100) |
| **Tensor Core** | Specialized unit for matrix multiply-accumulate (FP16/BF16/INT8) |
| **NVLink** | High-bandwidth GPU-to-GPU interconnect (900 GB/s on H100) |
| **PCIe** | GPU-to-CPU interconnect (64 GB/s Gen5) |

The memory hierarchy is where performance is won or lost. Each step down costs roughly an order of magnitude in latency:

```
Register File:    ~256 KB/SM,  0 cycles latency
Shared Memory:    ~164 KB/SM,  ~5 cycles latency
L1 Cache:         ~128 KB/SM,  ~30 cycles latency
L2 Cache:         ~50 MB,      ~200 cycles latency
HBM (Global):     ~80 GB,      ~400+ cycles latency
CPU Memory:       ~TBs,        ~10,000+ cycles (over PCIe)
```

The optimisation principle follows directly: move data up the hierarchy, minimise global memory access, and maximise reuse of shared memory. Most of the code in the next section is that principle applied.

## Coding rounds

Systems roles focus on C and C++, while AI platform roles use Python. Expect GPU-aware algorithm design, memory management and pointer manipulation, parallel algorithm implementation, and possibly actual CUDA kernel writing. Four problem types are reported.

### Matrix multiplication in CUDA

The canonical CUDA question is matrix multiplication, first naively and then with shared memory tiling. In the naive kernel each thread computes one element of `C` by reading a full row of `A` and a full column of `B` from global memory, so every input element is read $N$ times from HBM. The tiled kernel has each block cooperatively load a `TILE_SIZE` by `TILE_SIZE` tile of `A` and of `B` into shared memory, synchronise, accumulate the partial products from the fast tiles, and synchronise again before loading the next pair. Each global element is now read once per tile rather than once per thread, cutting global traffic by a factor of `TILE_SIZE`. Out-of-range elements are loaded as zero, so `N` need not be a multiple of the tile size.

```cuda
// Naive matrix multiply
__global__ void matmul_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled matrix multiply (shared memory optimization)
#define TILE_SIZE 16

__global__ void matmul_tiled(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < N && t * TILE_SIZE + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}
```

The follow-up discussion usually covers three topics: shared memory bank conflicts (when threads in a warp hit the same bank at different addresses and serialise), occupancy tuning (how many blocks fit on an SM given their registers and shared memory), and coalesced memory access (adjacent threads in `threadIdx.x` reading adjacent addresses, which both loads here do).

### Parallel reduction

The second classic is summing an array in parallel. Each thread first adds two input elements, halving the work before shared memory is touched. A tree reduction in shared memory then halves the active threads at each step, with a `__syncthreads()` between steps, until 64 values remain. The last steps run within a single warp and are unrolled.

```cuda
__global__ void reduce_sum(float* input, float* output, int n) {
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load and do first reduction
    sdata[tid] = (i < n ? input[i] : 0) + (i + blockDim.x < n ? input[i + blockDim.x] : 0);
    __syncthreads();

    // Tree reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction (no sync needed within a warp)
    if (tid < 32) {
        volatile float* smem = sdata;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

Each block writes one partial sum to `output[blockIdx.x]`, so a second launch (or an atomic add) combines the blocks. The kernel assumes a block size of 256 to match `sdata`.

> **Warning:** The comment "no sync needed within a warp" was true when warps executed in strict lockstep. Since Volta, threads in a warp can be scheduled independently, and the `volatile` trick alone is no longer guaranteed correct. Modern code inserts `__syncwarp()` between the unrolled steps, or better, uses warp shuffles (`__shfl_down_sync`) and avoids shared memory for the last 32 values entirely. Knowing this is a strong signal in an NVIDIA interview.

### A lock-free stack

Systems roles also ask CPU-side concurrency. A Treiber stack pushes by pointing a new node at the current head and swinging `head` to the new node with compare-and-swap, retrying if another thread got there first; `compare_exchange_weak` updates `new_node->next` with the current head on failure, so the loop body is empty. Pop swings `head` to `old_head->next` the same way.

```cpp
#include <atomic>
#include <memory>

// Lock-free stack using CAS
template<typename T>
class LockFreeStack {
    struct Node {
        T data;
        Node* next;
        Node(T val) : data(std::move(val)), next(nullptr) {}
    };

    std::atomic<Node*> head{nullptr};

public:
    void push(T value) {
        Node* new_node = new Node(std::move(value));
        new_node->next = head.load(std::memory_order_relaxed);
        while (!head.compare_exchange_weak(
            new_node->next, new_node,
            std::memory_order_release,
            std::memory_order_relaxed));
    }

    bool pop(T& result) {
        Node* old_head = head.load(std::memory_order_relaxed);
        while (old_head &&
               !head.compare_exchange_weak(
                   old_head, old_head->next,
                   std::memory_order_acquire,
                   std::memory_order_relaxed));
        if (old_head) {
            result = std::move(old_head->data);
            delete old_head;  // Note: in production, use hazard pointers
            return true;
        }
        return false;
    }
};
```

The comment on `delete` hides the interesting part. Between loading `old_head` and reading `old_head->next`, another thread may pop and free that node, so `pop` can read freed memory; and if the freed address is reused for a new node, the CAS can succeed on a stale view of the stack, which is the ABA problem. Hazard pointers, epoch-based reclamation, or tagged pointers fix it.

### A GPU memory allocator

The fourth problem, in Python, is a simple GPU memory allocator. Memory is a list of blocks ordered by offset. Allocation is first fit: find the first free block that can hold the request after aligning its start, then split it into an optional padding block, the allocated block, and an optional remainder. The alignment arithmetic `(offset + alignment - 1) & ~(alignment - 1)` rounds up to the next multiple of a power-of-two alignment. Freeing marks the block free and coalesces adjacent free blocks so the space can be reused for larger requests.

```python
from dataclasses import dataclass
from typing import Optional
import heapq

@dataclass
class MemoryBlock:
    offset: int
    size: int
    allocated: bool = False

class GPUMemoryAllocator:
    """Simple GPU memory allocator with coalescing free blocks."""

    def __init__(self, total_memory: int):
        self.total = total_memory
        self.blocks: list[MemoryBlock] = [MemoryBlock(0, total_memory)]
        self.allocated: dict[int, MemoryBlock] = {}  # offset -> block

    def allocate(self, size: int, alignment: int = 256) -> Optional[int]:
        """Allocate memory with alignment. Returns offset or None."""
        for i, block in enumerate(self.blocks):
            if block.allocated:
                continue
            # Align the offset
            aligned_offset = (block.offset + alignment - 1) & ~(alignment - 1)
            padding = aligned_offset - block.offset
            if block.size - padding >= size:
                # Split the block
                self.blocks.pop(i)
                if padding > 0:
                    self.blocks.insert(i, MemoryBlock(block.offset, padding, False))
                    i += 1
                alloc_block = MemoryBlock(aligned_offset, size, True)
                self.blocks.insert(i, alloc_block)
                remaining = block.size - padding - size
                if remaining > 0:
                    self.blocks.insert(i + 1, MemoryBlock(aligned_offset + size, remaining, False))
                self.allocated[aligned_offset] = alloc_block
                return aligned_offset
        return None  # Out of memory

    def free(self, offset: int) -> bool:
        if offset not in self.allocated:
            return False
        block = self.allocated.pop(offset)
        block.allocated = False
        self._coalesce()
        return True

    def _coalesce(self):
        """Merge adjacent free blocks."""
        i = 0
        while i < len(self.blocks) - 1:
            curr = self.blocks[i]
            next_block = self.blocks[i + 1]
            if not curr.allocated and not next_block.allocated:
                curr.size += next_block.size
                self.blocks.pop(i + 1)
            else:
                i += 1

    def fragmentation_ratio(self) -> float:
        """Return ratio of fragmented free memory to total free memory."""
        free_blocks = [b for b in self.blocks if not b.allocated]
        if not free_blocks:
            return 0.0
        total_free = sum(b.size for b in free_blocks)
        largest_free = max(b.size for b in free_blocks)
        return 1.0 - (largest_free / total_free) if total_free > 0 else 0.0
```

The fragmentation ratio is $1 - \text{largest free block} / \text{total free}$: zero when all free memory is one contiguous block, and approaching one as free memory splinters into small pieces that cannot satisfy a large request. Allocation and coalescing both scan the block list, so they are $O(n)$ in the number of blocks; size-segregated free lists or a balanced tree keyed by size are the usual improvements. (The `heapq` import is unused, a hint that a heap of free blocks was the intended next step.)

## The system design round

Design questions at NVIDIA are about GPU infrastructure and ML systems. Four are common.

### Multi-GPU training

```
[Training Coordinator]
        |
   [Data Loader] --> [GPU 0] [GPU 1] [GPU 2] [GPU 3]
                          |      |      |      |
                     [AllReduce Ring / Tree]
                          |
                     [Gradient Sync]
                          |
                     [Parameter Update]
                          |
                     [Checkpoint Manager]
```

Cover the three kinds of parallelism. Data parallelism distributes batches across GPUs and synchronises gradients; model parallelism splits large layers across GPUs; and pipeline parallelism assigns layers to stages and uses micro-batching to reduce pipeline bubbles. Communication runs over NCCL, using NVLink within a node and InfiniBand across nodes. Mixed precision (FP16 or BF16 forward, FP32 accumulation) halves the bytes that move, which is the simplest form of gradient compression. For fault tolerance, discuss checkpointing, elastic training, and resilience to spot instance loss.

### A TensorRT-style inference optimiser

An inference optimiser rewrites a trained model for speed. Graph optimisation applies layer fusion, constant folding, and dead code elimination. Precision calibration determines the best precision (FP32, FP16, or INT8) for each layer. Kernel auto-tuning profiles multiple implementations of each operation and picks the fastest. Memory planning optimises allocation across the execution graph, and the optimiser must also handle dynamic shapes, meaning variable batch sizes and sequence lengths.

### A GPU cluster scheduler

```
[Job Queue] --> [Scheduler] --> [GPU Cluster]
                    |               |
              [Topology Aware] [NVLink/NVSwitch Topology]
                    |               |
              [Job Placement]  [GPU Health Monitor]
```

Scheduling must be topology-aware, placing multi-GPU jobs on GPUs connected by NVLink. Training jobs need gang scheduling, so that all their GPUs start simultaneously. High-priority inference workloads may preempt training. The scheduler monitors GPU health for ECC errors, thermal throttling, and hangs. And fractional GPUs via MIG (Multi-Instance GPU) let several jobs share one GPU.

### DGX Cloud

A GPU cloud adds multi-tenancy: GPUs are shared through MIG, time-slicing, and Kubernetes. Storage is a high-throughput distributed filesystem with GPUDirect Storage. Inter-node networking uses RDMA, RoCE, or InfiniBand. Monitoring tracks GPU metrics (SM utilisation, memory bandwidth, Tensor Core usage), and cost is optimised by running training on spot instances and inference on reserved capacity.

### Performance optimisations

Whatever the design, expect to discuss the standard GPU optimisations and roughly what each buys:

| Optimization | Description | Impact |
|-------------|-------------|--------|
| Memory coalescing | Adjacent threads access adjacent memory | 10-100x bandwidth improvement |
| Shared memory tiling | Cache data in shared memory | 5-10x for memory-bound kernels |
| Occupancy tuning | Balance threads/registers/shared mem | 20-50% throughput improvement |
| Warp divergence minimization | Avoid branching within warps | Up to 32x for divergent code |
| Tensor Core utilization | Use TC-friendly shapes (multiples of 16) | 5-10x for matrix ops |
| Kernel fusion | Combine multiple kernels | Eliminate memory round-trips |
| Asynchronous data transfer | Overlap compute and memory transfer | Hide transfer latency |

## The hiring manager round

NVIDIA's culture is innovation-driven ("the way we do the impossible is by imagining the unimaginable"), fast (Jensen Huang is famous for fast iteration and pivoting), technically deep, and collaborative across hardware and software teams. Common questions include "What excites you about GPU computing?", "Tell me about a performance optimization you're proud of.", "Describe a time you had to learn a new domain quickly.", "How do you approach debugging a performance regression?", and "What's your view on the future of AI hardware?"

## Preparing

Learn the CUDA basics even if you will not write CUDA daily: the execution model, the memory hierarchy, and the common patterns. For systems roles, be proficient in modern C++ (C++17/20, smart pointers, move semantics, templates). Know the GPU architecture (SM, warp, thread block, memory hierarchy) and read the H100 whitepaper. Cultivate a performance mindset in terms of memory bandwidth, compute utilisation, and latency hiding. Read the NVIDIA developer blog at developer.nvidia.com/blog, which has excellent technical content. Know NCCL, NVIDIA's multi-GPU communication library, and its collectives (AllReduce, broadcast, and so on), and understand TensorRT for model optimisation and inference serving. Finally, practise systems problems: memory allocators, lock-free structures, and cache-friendly algorithms.

## Summary

- NVIDIA interviews are domain-specific, but all assume you understand GPU architecture and the memory hierarchy.
- Tiling into shared memory and coalescing global accesses are the two most important kernel optimisations.
- Know the classic kernels (matmul, reduction) and their modern caveats, such as warp synchronisation since Volta.
- CPU-side systems questions (lock-free structures, allocators) probe memory reclamation and fragmentation.
- Design questions centre on multi-GPU training, inference optimisation, and topology-aware scheduling.

## Further reading

- [NVIDIA Developer Blog](https://developer.nvidia.com/blog)
- [NVIDIA H100 Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Glassdoor: NVIDIA SWE Interview Questions](https://www.glassdoor.com/Interview/NVIDIA-Software-Engineer-Interview-Questions-EI_IE7633.0,6_KO7,24.htm)
- [levels.fyi: NVIDIA Compensation Data](https://www.levels.fyi/companies/nvidia/salaries/software-engineer)
- Community reports on r/cscareerquestions, r/ExperiencedDevs, and Blind.
