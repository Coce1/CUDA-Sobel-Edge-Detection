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

# Run baenchmarks
./sobel_benchmark
