#ifndef SOBEL_CUH
#define SOBEL_CUH

#include <vector>

// CPU version declaration
void sobelCPU(const std::vector<unsigned char>& in, std::vector<unsigned char>& out, int width, int height);


#ifdef __CUDACC__
// Declaration of the Naive GPU Version (Global Memory)
__global__ void sobelGPUNaive(const unsigned char* __restrict__ d_in, unsigned char* __restrict__ d_out, int width, int height);

// Declaration of the GPU-optimized version (Shared Memory + Coalescing)
__global__ void sobelFilter(const unsigned char* __restrict__ d_in, unsigned char* __restrict__ d_out, int width, int height);
#endif

#endif