# ⚡ CUDA High-Performance Computing & Sobel Edge Detection

A structured exploration of GPU-accelerated computing using modern C++ and NVIDIA CUDA, scaling from core hardware execution models to a 2D Sobel convolution filter with global vs. shared memory benchmarking.

---

## 📌 Project Overview

This repository documents the step-by-step development of parallel algorithms on NVIDIA GPUs, focusing on memory hierarchy optimization, thread synchronization, and spatial image filtering:

1. **`01_hello_cuda.cu`**: GPU thread hierarchy (`<<<grid, block>>>`), thread indexing, and hardware execution verification.
2. **`02_vector_multiply.cu`**: Explicit memory management (`cudaMalloc`, `cudaMemcpy`, `cudaFree`) and 1D boundary handling.
3. **`03_sobel_basic.cu`**: 2D discrete convolution operator applied on a synthetic image matrix.
4. **`04_sobel_cpu_vs_gpu.cu`**: Sequential CPU vs. Parallel GPU execution benchmark timed via `cudaEvent_t` ($2048 \times 2048$ resolution).
5. **`05_sobel_shared_memory.cu`**: Advanced L1 Cache / Shared Memory (`__shared__`) tiling with halo exchange benchmarked on 16 Megapixels ($4096 \times 4096$).

---

## 📐 Mathematical Model (Sobel Convolution)

The 2D Sobel operator computes spatial gradient vectors for each pixel coordinate $(x, y)$:

$$
G_x = \begin{bmatrix} -1 & 0 & +1 \\\\ -2 & 0 & +2 \\\\ -1 & 0 & +1 \end{bmatrix} * I \quad , \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\\\ 0 & 0 & 0 \\\\ +1 & +2 & +1 \end{bmatrix} * I
$$

Gradient magnitude approximation:

$$
|\nabla I| = \min\left(|G_x| + |G_y|, \, 255\right)
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

# Run specific modules
./01_hello
./04_benchmark
./05_shared
