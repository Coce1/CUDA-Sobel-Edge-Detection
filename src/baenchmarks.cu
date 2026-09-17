#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include "../include/sobel.cuh"

int main() {
    int width = 2048, height = 2048;
    int total_pixels = width * height;
    size_t bytes = total_pixels * sizeof(unsigned char);

    // Create a synthetic image with a white square in the middle
    std::vector<unsigned char> h_in(total_pixels, 0);
    for (int y = 500; y < 1500; y++) {
        for (int x = 500; x < 1500; x++) {
            h_in[y * width + x] = 255;
        }
    }

    std::vector<unsigned char> h_out_cpu(total_pixels, 0);
    std::vector<unsigned char> h_out_naive(total_pixels, 0);
    std::vector<unsigned char> h_out_shared(total_pixels, 0);

    // 1. CPU Benchmark
    auto start_cpu = std::chrono::high_resolution_clock::now();
    sobelCPU(h_in, h_out_cpu, width, height);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;


    // 2. GPU Benchmark for Shared Memory version
    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                    (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    sobelFilter<<<numBlocks, threadsPerBlock>>>(d_in, d_out, width, height);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float shared_duration = 0;
    cudaEventElapsedTime(&shared_duration, start_gpu, stop_gpu);
    cudaMemcpy(h_out_shared.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    // 3. GPU Benchmark for Naive version

    cudaEventRecord(start_gpu);
    sobelGPUNaive<<<numBlocks, threadsPerBlock>>>(d_in, d_out, width, height);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float naive_duration = 0;
    cudaEventElapsedTime(&naive_duration, start_gpu, stop_gpu);
    cudaMemcpy(h_out_naive.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    // HPC metric calculation: Effective Bandwidth (GB/s)
    // Formula: (Total bytes read + Total bytes written) / (Time in milliseconds * 1e6)
    float total_GB = (2.0f * bytes) / 1.0e9f;                                             // Reading (1x) + Writing (1x) = 2 * bytes
    float naive_bw = total_GB / (naive_duration / 1000.0f);
    float shared_bw = total_GB / (shared_duration / 1000.0f);

    // Technical report
    std::cout << "=== Performance Audit – Sobel Filter (" << width << "x" << height << ") ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "\n[CPU] Sequential Implementation :" << std::endl;
    std::cout << "- Execution time : " << cpu_duration.count() << " ms" << std::endl;

    std::cout << "\n[GPU] Version 1 : Naive (Global Memory) :" << std::endl;
    std::cout << "- Execution time : " << naive_duration << " ms" << std::endl;
    std::cout << "- Speedup vs CPU    : " << (cpu_duration.count() / naive_duration) << "x" << std::endl;
    std::cout << "- Bandwidth    : " << naive_bw << " Go/s" << std::endl;

    std::cout << "\n[GPU] Version 2 : Optimised (Shared Memory + Coalescing) :" << std::endl;
    std::cout << "- Execution time : " << shared_duration << " ms" << std::endl;
    std::cout << "- Speedup vs CPU    : " << (cpu_duration.count() / shared_duration) << "x" << std::endl;
    std::cout << "- Bandwidth    : " << shared_bw << " Go/s" << std::endl;

    // Memory release
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}