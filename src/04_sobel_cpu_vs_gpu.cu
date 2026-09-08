#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

__global__ void sobelGPU(const int* __restrict__ d_in, int* __restrict__ d_out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int gx = -d_in[(y-1)*width + (x-1)] + d_in[(y-1)*width + (x+1)]
                 -2*d_in[y*width + (x-1)]   + 2*d_in[y*width + (x+1)]
                 -d_in[(y+1)*width + (x-1)] + d_in[(y+1)*width + (x+1)];

        int gy = -d_in[(y-1)*width + (x-1)] - 2*d_in[(y-1)*width + x] - d_in[(y-1)*width + (x+1)]
                 +d_in[(y+1)*width + (x-1)] + 2*d_in[(y+1)*width + x] + d_in[(y+1)*width + (x+1)];

        d_out[y * width + x] = min(abs(gx) + abs(gy), 255);
    }
}

void sobelCPU(const std::vector<int>& in, std::vector<int>& out, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int gx = -in[(y-1)*width + (x-1)] + in[(y-1)*width + (x+1)]
                     -2*in[y*width + (x-1)]   + 2*in[y*width + (x+1)]
                     -in[(y+1)*width + (x-1)] + in[(y+1)*width + (x+1)];

            int gy = -in[(y-1)*width + (x-1)] - 2*in[(y-1)*width + x] - in[(y-1)*width + (x+1)]
                     +in[(y+1)*width + (x-1)] + 2*in[(y+1)*width + x] + in[(y+1)*width + (x+1)];

            out[y * width + x] = std::min(std::abs(gx) + std::abs(gy), 255);
        }
    }
}

int main() {
    int width = 2048, height = 2048;
    int total_pixels = width * height;
    size_t bytes = total_pixels * sizeof(int);

    std::vector<int> h_in(total_pixels, 0);
    for (int y = 500; y < 1500; y++) {
        for (int x = 500; x < 1500; x++) {
            h_in[y * width + x] = 255;
        }
    }
    std::vector<int> h_out_cpu(total_pixels, 0);
    std::vector<int> h_out_gpu(total_pixels, 0);

    // 1. CPU Benchmark
    auto start_cpu = std::chrono::high_resolution_clock::now();
    sobelCPU(h_in, h_out_cpu, width, height);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;

    // 2. GPU Benchmark
    int *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    sobelGPU<<<numBlocks, threadsPerBlock>>>(d_in, d_out, width, height);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float gpu_duration = 0;
    cudaEventElapsedTime(&gpu_duration, start_gpu, stop_gpu);
    cudaMemcpy(h_out_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    std::cout << "--- Benchmark Filtre de Sobel (" << width << "x" << height << " - " << total_pixels << " pixels) ---" << std::endl;
    std::cout << "Temps CPU : " << cpu_duration.count() << " ms" << std::endl;
    std::cout << "Temps GPU (Kernel pur) : " << gpu_duration << " ms" << std::endl;
    std::cout << "Acceleration (Speedup) : " << (cpu_duration.count() / gpu_duration) << "x" << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}