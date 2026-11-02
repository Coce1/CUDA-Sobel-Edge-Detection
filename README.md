# ⚡ CUDA High-Performance Computing & Sobel Edge Detection

A structured exploration of GPU-accelerated computing using modern C++ and NVIDIA CUDA, scaling from core hardware execution models to a 2D Sobel convolution filter with global vs. shared memory benchmarking. All implementations are validated against the CPU reference before being timed.

---

## 📌 Project Overview

This repository documents the step-by-step development of parallel algorithms on NVIDIA GPUs, focusing on memory hierarchy optimization, thread synchronization, and spatial image filtering:

1. **`src/sobel_cpu.cpp`**: Sequential CPU implementation (reference).
2. **`src/sobel_naive.cu`**: Basic 2D convolution using global memory (32×8 blocks).
3. **`src/sobel_shared_tiled.cu`**: 2D convolution with a shared-memory tile and halo. Each thread loads its own pixel; only border threads load the halo (sides and corners), with no integer division.
4. **`src/baenchmarks.cu`**: Comparative benchmark suite (CPU vs. Naive GPU vs. Shared GPU) on two image sizes, measuring execution time, speedup, effective bandwidth (GB/s) and correctness.
5. **`include/sobel.cuh`**: CPU reference and GPU launchers (kernels stay private to their `.cu` file).
6. **`include/cuda_utils.cuh`**: `CUDA_CHECK` / `CUDA_CHECK_KERNEL` macros, `DeviceBuffer<T>` (RAII GPU memory), `benchmarkMs` (warm-up + averaged `cudaEvent` timing).

> The file name `baenchmarks.cu` (typo included) is kept to avoid breaking existing references. It can be renamed to `benchmarks.cu` together with the Makefile.

---

## 📐 Mathematical Model (Sobel Convolution)

The 2D Sobel operator computes spatial gradient vectors for each pixel coordinate $(x, y)$:

$$
G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} * I \quad , \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix} * I
$$

Gradient magnitude approximation:

$$
|\nabla I| = \min(|G_x| + |G_y|, 255)
$$

**Border convention (identical for CPU and both GPU kernels):** pixels on the image border are written as `0`. Previously the naive kernel left them unchanged while the tiled kernel used zero-padding, so the outputs were not equivalent.

---

## 🧩 Implementation Notes

- **32×8 thread blocks:** a warp covers 32 contiguous pixels of one row (instead of two half-rows with 16×16 blocks).
- **Tile of $(8+2) \times (32+2)$ bytes** in shared memory with halo and 4 corners (required by the 3×3 kernel).
- **`__syncthreads()` placed before any early `return`:** every thread of the block must reach the barrier (a barrier in a divergent branch is undefined behavior).
- **Explicit border writes** in both kernels, so the output never depends on the previous content of `d_out`.
- **Error handling and memory:** every CUDA call is wrapped in `CUDA_CHECK`, kernel launches are followed by `CUDA_CHECK_KERNEL()`, and GPU buffers are released automatically (RAII).

---

## 🛠️ Build & Run

### Prerequisites
- NVIDIA GPU (default target `sm_75`, Turing / T4). CUDA Toolkit 13 requires Compute Capability $\ge$ 7.5; with CUDA 11/12, older architectures (Pascal `sm_60/61`, Volta `sm_70`) still work.
- NVIDIA CUDA Toolkit $\ge$ 11.0 (tested with 13.0), providing `nvcc`
- C++17 compatible host compiler (`g++` or `clang++`)

### Compilation with Make
```bash
# Build all examples (set ARCH to your GPU: sm_75, sm_80, sm_86, sm_89...)
make ARCH=sm_75

# Run the benchmarks
./sobel_benchmark
```

Equivalent direct command (useful if the Makefile tabs were altered by a copy-paste):
```bash
nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo -Iinclude \
     src/baenchmarks.cu src/sobel_naive.cu src/sobel_shared_tiled.cu src/sobel_cpu.cpp \
     -o sobel_benchmark
./sobel_benchmark
```

### Memory and race checking (optional)
```bash
compute-sanitizer --tool memcheck  ./sobel_benchmark
compute-sanitizer --tool racecheck ./sobel_benchmark
```

---

## 🧪 Benchmark Protocol

- **Input:** 8-bit grayscale images of two sizes: **2048 × 2048** (4 MB) and **8192 × 8192** (64 MB). Content: sinusoidal texture, light noise and a white square. A single white square would make almost all outputs equal to 0 and the exact comparison meaningless.
- **CPU:** average of 3 runs.
- **GPU:** **5 warm-up runs, then 20 measurements**, each measurement chaining 20 kernel launches (a single launch lasts only tens to hundreds of microseconds). Timed with `cudaEvent`.
- **Order independence:** both kernels are warmed up, so the execution order no longer affects the results. In the previous version the shared kernel ran first and absorbed the CUDA context initialization.
- **Validation:** `d_out` is filled with a sentinel (`0xAA`) and the GPU output is compared byte by byte with the CPU reference (`OK` only with 0 mismatches).
- **Effective bandwidth:** $(\text{bytes read} + \text{bytes written}) / \text{time}$ with an ideal model of 2 × image size (each pixel read once and written once).

---

## 📊 Benchmark & Performance Audit

**Hardware:** Google Colab | GPU: Tesla T4 (sm_75, 40 SM, 15 GB) | CUDA Toolkit: 13.0 (V13.0.88) | Driver: 580.82.07

### 2048 × 2048 (4 MB)

| Implementation | Execution Time (ms) | Speedup vs. CPU | Effective Bandwidth | Throughput (Mpix/s) | Mismatches | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CPU (Sequential)** | 14.4755 | 1.0× (Baseline) | — | 290 | (reference) | — |
| **GPU Naive (Global Memory)** | **0.14826** | **97.6×** | **56.6 GB/s** | 28289 | 0 | OK |
| **GPU Tiled (Shared Memory)** | 0.31056 | 46.6× | 27.0 GB/s | 13505 | 0 | OK |

### 8192 × 8192 (64 MB)

| Implementation | Execution Time (ms) | Speedup vs. CPU | Effective Bandwidth | Throughput (Mpix/s) | Mismatches | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CPU (Sequential)** | 257.2687 | 1.0× (Baseline) | — | 261 | (reference) | — |
| **GPU Naive (Global Memory)** | **1.67417** | **153.7×** | **80.2 GB/s** | 40085 | 0 | OK |
| **GPU Tiled (Shared Memory)** | 3.70998 | 69.3× | 36.2 GB/s | 18089 | 0 | OK |

Total program run time: 5.52 s.

### Architectural Analysis & Insights

- **The naive kernel is about 2× faster than the shared-memory kernel, on both sizes** (2.09× at 2048², 2.22× at 8192²). The ratio is stable when the image grows 16× in size, so it is not a small-image or cache-size artifact. For a 3×3 stencil, the L1/L2 caches already capture the spatial reuse between neighboring threads; the explicit tile adds work without removing any global traffic: halo loads by border threads (divergent branches), byte-wide shared-memory stores and loads, and a barrier per block. This explanation is consistent with the measurements but has **not been confirmed with a profiler** (Nsight Compute would show the instruction and stall breakdown).
- **Neither kernel is limited by DRAM bandwidth.** The best result (80.2 GB/s at 8192²) is about a quarter of the 320 GB/s theoretical peak of the T4. Each thread processes a single 1-byte pixel with 8 byte-wide loads, so the cost is likely dominated by the number of instructions and memory transactions per byte rather than by DRAM traffic. This is a hypothesis; the natural test is to process 4 pixels per thread with `uchar4` accesses.
- **Larger image, better throughput.** The naive kernel goes from 28.3 to 40.1 Gpixel/s between 2048² and 8192²: at 0.148 ms, launch and tail effects weigh more on the small image.
- **The GPU speedup over the CPU (97× to 154×)** compares one optimized GPU kernel to a single sequential CPU thread; it is not a multi-core CPU comparison.

### Correction of the previous version
The previously reported times (naive 9.32 ms vs. tiled 110.41 ms, 0.90 and 0.08 GB/s) were measured without warm-up, with the tiled kernel launched first, so they mostly captured CUDA initialization overhead. The conclusions drawn from them (divergence and synchronization penalty as the main cause) were not demonstrated. The corrected measurements still show the tiled kernel as slower, but by a factor of about 2 instead of about 12, and with a different likely cause (the caches already provide the reuse).
