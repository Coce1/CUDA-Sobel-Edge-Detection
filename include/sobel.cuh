#ifndef SOBEL_CUH
#define SOBEL_CUH

#include <vector>

// Déclaration de la version CPU
void sobelCPU(const std::vector<unsigned char>& in, std::vector<unsigned char>& out, int width, int height);

// Déclaration de la version GPU Naïve (Mémoire Globale)
__global__ void sobelGPUNaive(const unsigned char* __restrict__ d_in, unsigned char* __restrict__ d_out, int width, int height);

// Déclaration de la version GPU Optimisée (Mémoire Partagée + Coalescence)
__global__ void sobelFilter(const unsigned char* __restrict__ d_in, unsigned char* __restrict__ d_out, int width, int height);

#endif