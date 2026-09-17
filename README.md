# ⚡ CUDA High-Performance Computing & Sobel Edge Detection

A structured exploration of GPU-accelerated computing using modern C++ and NVIDIA CUDA, scaling from core hardware execution models to a 2D Sobel convolution filter with global vs. shared memory benchmarking.

---

## 📌 Project Overview

This repository documents the step-by-step development of parallel algorithms on NVIDIA GPUs, focusing on memory hierarchy optimization, thread synchronization, and spatial image filtering:

1. **`src/sobel_cpu.cpp`**: Sequential CPU implementation.
2. **`src/sobel_naive.cu`**: Basic 2D convolution using global memory.
3. **`src/sobel_shared_tiled.cu`**: Optimized 2D convolution featuring shared memory and 1D coalesced memory access.
4. **`src/baenchmarks.cu`**: Comparative benchmark suite (CPU vs. Naive GPU vs. Shared GPU) measuring execution time and effective bandwidth (GB/s).

---

## 📐 Mathematical Model (Sobel Convolution)

The 2D Sobel operator computes spatial gradient vectors for each pixel coordinate $(x, y)$:

$$
G_x = \begin{bmatrix} -1 & 0 & +1 \\\\ -2 & 0 & +2 \\\\ -1 & 0 & +1 \end{bmatrix} * I \quad , \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\\\ 0 & 0 & 0 \\\\ +1 & +2 & +1 \end{bmatrix} * I
$$

Gradient magnitude approximation:

$$
|\nabla I| = \min(|G_x| + |G_y|, 255)
$$

---

## 🛠️ Build & Run

### Prerequisites
- NVIDIA GPU with Compute Capability $\ge$ 6.0
- NVIDIA CUDA Toolkit $\ge$ 11.0
- C++17 compatible host compiler (`g++` or `clang++`)

### Compilation with Make
```bash
# Build all examples
make

## 📊 Benchmark & Performance Audit

Tests conducted on an NVIDIA GPU via Google Colab with an input matrix size of **2048 × 2048** (8-bit grayscale pixels, ~4.19 MB):

| Implementation | Execution Time | Speedup vs. CPU | Effective Bandwidth |
| :--- | :--- | :--- | :--- |
| **CPU (Sequential)** | 15.86 ms | 1.00× (Baseline) | — |
| **GPU Naive (Global Memory)** | **9.32 ms** | **1.70×** | **0.90 GB/s** |
| **GPU Tiled (Shared Memory)** | 110.41 ms | 0.14× | 0.08 GB/s |

### Architectural Analysis & Insights

- **L2 Cache vs. Shared Memory:** The 2D Sobel kernel exhibits low arithmetic intensity (few arithmetic operations per memory transaction). On modern NVIDIA architectures, spatial locality is efficiently captured by hardware L1/L2 caches in the global memory implementation, avoiding manual buffering penalties.
- **Divergence & Boundary Overhead:** The shared memory implementation incurs extra overhead from conditional zero-padding on tile halos, leading to warp divergence and thread serialization at matrix boundaries.
- **Synchronization Penalty:** Explicit barrier synchronization (`__syncthreads()`) adds latency that outweighs the memory access gains for small stencil radii on lightweight computational kernels.

# Run baenchmarks
./sobel_benchmark
