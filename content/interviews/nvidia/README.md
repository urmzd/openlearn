# NVIDIA Software Engineer Interview Guide

Comprehensive preparation for NVIDIA engineering roles, with focus on GPU computing, AI infrastructure, and systems programming.

## Interview Process Overview

Timeline: **3-6 weeks**, **4-6 rounds**

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, team matching |
| Phone Screen | Technical | 45-60 min | Coding + domain knowledge |
| Onsite 1 | Coding | 60 min | Systems programming, C++/CUDA |
| Onsite 2 | Domain Deep Dive | 60 min | GPU architecture, parallel computing |
| Onsite 3 | System Design | 60 min | GPU infrastructure, ML systems |
| Onsite 4 | Hiring Manager | 45 min | Culture, experience, motivation |

NVIDIA's process is **highly domain-specific**. Roles range from driver development to AI platform engineering. The interview content varies significantly by team.

## Compensation (Senior, US)

- **Base**: $180-250K
- **Equity**: RSUs, ~$200-500K/yr (NVIDIA stock has been explosive)
- **Bonus**: 10-15% of base
- **Total Comp**: ~$400-750K (Senior), ~$700K-1.2M+ (Principal/Staff)
- NVIDIA's equity appreciation has made it one of the highest-paying companies

## Key Themes

1. **GPU-centric thinking** -- Everything at NVIDIA revolves around parallel computation on GPUs. Even "software" roles require understanding GPU architecture.
2. **CUDA is the lingua franca** -- For systems/infra roles, CUDA knowledge is expected or at least valued
3. **AI infrastructure is the growth engine** -- Data center GPU (H100, B200), DGX, NVIDIA AI Enterprise, TensorRT are the profit centers
4. **Performance obsessed** -- Microseconds matter. Memory access patterns, cache utilization, instruction-level parallelism
5. **Systems programming depth** -- C/C++ heavy. Understanding of memory hierarchies, SIMD, vectorization
6. **Full stack from silicon to cloud** -- NVIDIA covers hardware, drivers, libraries, frameworks, and cloud services

## GPU Architecture Fundamentals

**You must understand this for any NVIDIA interview.**

### NVIDIA GPU Architecture (Simplified)

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

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Warp** | 32 threads executing in lockstep (SIMT) |
| **SM** | Streaming Multiprocessor -- the basic execution unit |
| **Thread Block** | Group of threads (up to 1024) sharing shared memory |
| **Grid** | Collection of thread blocks launched by a kernel |
| **Shared Memory** | Fast on-chip memory shared within a thread block (~48-164 KB) |
| **Global Memory** | HBM, high bandwidth but high latency (~80 GB on H100) |
| **Tensor Core** | Specialized unit for matrix multiply-accumulate (FP16/BF16/INT8) |
| **NVLink** | High-bandwidth GPU-to-GPU interconnect (900 GB/s on H100) |
| **PCIe** | GPU-to-CPU interconnect (64 GB/s Gen5) |

### Memory Hierarchy

```
Register File:    ~256 KB/SM,  0 cycles latency
Shared Memory:    ~164 KB/SM,  ~5 cycles latency
L1 Cache:         ~128 KB/SM,  ~30 cycles latency
L2 Cache:         ~50 MB,      ~200 cycles latency
HBM (Global):     ~80 GB,      ~400+ cycles latency
CPU Memory:       ~TBs,        ~10,000+ cycles (over PCIe)
```

**Optimization principle**: Move data up the hierarchy. Minimize global memory access. Maximize shared memory reuse.

## Coding Rounds

### What to Expect

- **C/C++ focused** for systems roles (Python for AI platform roles)
- GPU-aware algorithm design
- Memory management, pointer manipulation
- Parallel algorithm implementation
- May include actual CUDA kernel writing

### Reported Problem Types

#### 1. CUDA Kernel: Matrix Multiplication

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

**Discussion points**: Shared memory bank conflicts, occupancy tuning, coalesced memory access.

#### 2. Parallel Reduction

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

#### 3. Memory-Efficient Data Structures (C++)

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

#### 4. Python: GPU Memory Manager

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

## System Design Round

### Common Topics

#### Design a Multi-GPU Training System

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

Key considerations:
- **Data parallelism**: Distribute batches across GPUs, synchronize gradients
- **Model parallelism**: Split large model layers across GPUs
- **Pipeline parallelism**: Micro-batching to reduce pipeline bubbles
- **Communication**: NCCL for GPU-to-GPU communication, NVLink within node, InfiniBand across nodes
- **Gradient compression**: Mixed precision (FP16/BF16 forward, FP32 accumulation)
- **Fault tolerance**: Checkpointing, elastic training, spot instance resilience

#### Design TensorRT Inference Optimizer

- **Graph optimization**: Layer fusion, constant folding, dead code elimination
- **Precision calibration**: Determine optimal precision (FP32/FP16/INT8) per layer
- **Kernel auto-tuning**: Profile multiple implementations per operation, select fastest
- **Memory planning**: Optimize memory allocation across the execution graph
- **Dynamic shapes**: Handle variable batch sizes and sequence lengths

#### Design a GPU Cluster Scheduler

```
[Job Queue] --> [Scheduler] --> [GPU Cluster]
                    |               |
              [Topology Aware] [NVLink/NVSwitch Topology]
                    |               |
              [Job Placement]  [GPU Health Monitor]
```

- **Topology-aware scheduling**: Place multi-GPU jobs on GPUs with NVLink connectivity
- **Gang scheduling**: All GPUs for a training job start simultaneously
- **Preemption**: High-priority inference workloads preempt training
- **GPU health**: Monitor for ECC errors, thermal throttling, GPU hang
- **Fractional GPUs**: MIG (Multi-Instance GPU) for sharing a single GPU

#### Design NVIDIA DGX Cloud

- **Multi-tenant GPU sharing**: MIG + time-slicing + Kubernetes
- **Storage**: High-throughput distributed filesystem (GPUDirect Storage)
- **Networking**: RDMA / RoCE / InfiniBand for inter-node communication
- **Monitoring**: GPU metrics (SM utilization, memory bandwidth, tensor core usage)
- **Cost optimization**: Spot instances for training, reserved for inference

### Performance Optimization Concepts

| Optimization | Description | Impact |
|-------------|-------------|--------|
| Memory coalescing | Adjacent threads access adjacent memory | 10-100x bandwidth improvement |
| Shared memory tiling | Cache data in shared memory | 5-10x for memory-bound kernels |
| Occupancy tuning | Balance threads/registers/shared mem | 20-50% throughput improvement |
| Warp divergence minimization | Avoid branching within warps | Up to 32x for divergent code |
| Tensor Core utilization | Use TC-friendly shapes (multiples of 16) | 5-10x for matrix ops |
| Kernel fusion | Combine multiple kernels | Eliminate memory round-trips |
| Asynchronous data transfer | Overlap compute and memory transfer | Hide transfer latency |

## Behavioral / Hiring Manager

### NVIDIA Culture

- **Innovation driven** -- "The way we do the impossible is by imagining the unimaginable"
- **Speed** -- Jensen Huang is famous for fast iteration and pivoting
- **Technical depth** -- Engineers are expected to go deep
- **Collaborative** -- Cross-team work between hardware and software

### Common Questions

- "What excites you about GPU computing?"
- "Tell me about a performance optimization you're proud of."
- "Describe a time you had to learn a new domain quickly."
- "How do you approach debugging a performance regression?"
- "What's your view on the future of AI hardware?"

## Preparation Tips

1. **Learn CUDA basics** -- Even if you won't write CUDA daily, understand the execution model, memory hierarchy, and common patterns.
2. **C++ proficiency** -- For systems roles, modern C++ (C++17/20), smart pointers, move semantics, templates.
3. **GPU architecture** -- Know SM, warp, thread block, memory hierarchy. Read the H100 whitepaper.
4. **Performance mindset** -- Think in terms of memory bandwidth, compute utilization, latency hiding.
5. **Read NVIDIA tech blog** -- developer.nvidia.com/blog has excellent technical content.
6. **Know NCCL** -- NVIDIA's multi-GPU communication library. Understand AllReduce, broadcast, etc.
7. **Understand TensorRT** -- Model optimization and inference serving on GPUs.
8. **Practice systems problems** -- Memory allocators, lock-free structures, cache-friendly algorithms.

## Sources

- [NVIDIA Developer Blog](https://developer.nvidia.com/blog)
- [NVIDIA H100 Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Glassdoor - NVIDIA SWE Interview Questions](https://www.glassdoor.com/Interview/NVIDIA-Software-Engineer-Interview-Questions-EI_IE7633.0,6_KO7,24.htm)
- [levels.fyi - NVIDIA Compensation Data](https://www.levels.fyi/companies/nvidia/salaries/software-engineer)
- r/cscareerquestions, r/ExperiencedDevs, Blind (community reports)
