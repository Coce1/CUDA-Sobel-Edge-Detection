#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#define TILE_SIZE 16

__global__ void sobelGlobal(const int* __restrict__ d_in, int* __restrict__ d_out, int width, int height) {
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

__global__ void sobelShared(const int* __restrict__ d_in, int* __restrict__ d_out, int width, int height) {
    __shared__ int s_in[TILE_SIZE + 2][TILE_SIZE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;

    // Chargement coopératif
    if (x < width && y < height) {
        s_in[ty + 1][tx + 1] = d_in[y * width + x];
    } else {
        s_in[ty + 1][tx + 1] = 0;
    }

    // Halo
    if (tx == 0 && x > 0) s_in[ty + 1][0] = d_in[y * width + (x - 1)];
    if (tx == TILE_SIZE - 1 && x < width - 1) s_in[ty + 1][TILE_SIZE + 1] = d_in[y * width + (x + 1)];
    if (ty == 0 && y > 0) s_in[0][tx + 1] = d_in[(y - 1) * width + x];
    if (ty == TILE_SIZE - 1 && y < height - 1) s_in[TILE_SIZE + 1][tx + 1] = d_in[(y + 1) * width + x];

    // Coins
    if (tx == 0 && ty == 0 && x > 0 && y > 0) s_in[0][0] = d_in[(y - 1) * width + (x - 1)];
    if (tx == TILE_SIZE - 1 && ty == 0 && x < width - 1 && y > 0) s_in[0][TILE_SIZE + 1] = d_in[(y - 1) * width + (x + 1)];
    if (tx == 0 && ty == TILE_SIZE - 1 && x > 0 && y < height - 1) s_in[TILE_SIZE + 1][0] = d_in[(y + 1) * width + (x - 1)];
    if (tx == TILE_SIZE - 1 && ty == TILE_SIZE - 1 && x < width - 1 && y < height - 1) s_in[TILE_SIZE + 1][TILE_SIZE + 1] = d_in[(y + 1) * width + (x + 1)];

    __syncthreads();

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int gx = -s_in[ty][tx]     + s_in[ty][tx+2]
                 -2*s_in[ty+1][tx] + 2*s_in[ty+1][tx+2]
                 -s_in[ty+2][tx]   + s_in[ty+2][tx+2];

        int gy = -s_in[ty][tx]     - 2*s_in[ty][tx+1]   - s_in[ty][tx+2]
                 +s_in[ty+2][tx]   + 2*s_in[ty+2][tx+1] + s_in[ty+2][tx+2];

        d_out[y * width + x] = min(abs(gx) + abs(gy), 255);
    }
}

float benchmarkKernel(void (*kernel)(const int*, int*, int, int), const int* d_in, int* d_out, int width, int height) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    kernel<<<blocks, threads>>>(d_in, d_out, width, height); // warmup
    
    cudaEventRecord(start);
    kernel<<<blocks, threads>>>(d_in, d_out, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms;
}

int main() {
    int width = 4096, height = 4096;
    size_t bytes = width * height * sizeof(int);
    
    std::vector<int> h_in(width * height, 0);
    int *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    float timeGlobal = benchmarkKernel(sobelGlobal, d_in, d_out, width, height);
    float timeShared = benchmarkKernel(sobelShared, d_in, d_out, width, height);

    std::cout << "--- Traitement d'une image " << width << "x" << height << " (16 Millions de pixels) ---" << std::endl;
    std::cout << "Temps VRAM (Global Memory) : " << timeGlobal << " ms" << std::endl;
    std::cout << "Temps Cache L1 (Shared Memory) : " << timeShared << " ms" << std::endl;
    std::cout << "Gain marginal materiel : " << (timeGlobal / timeShared) << "x plus rapide" << std::endl;

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}