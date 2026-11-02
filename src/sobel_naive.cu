#include "../include/sobel.cuh"
#include "../include/cuda_utils.cuh"

constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;

__global__ void sobelGPUNaive(const unsigned char* __restrict__ d_in,
                              unsigned char* __restrict__ d_out, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const size_t o = static_cast<size_t>(y) * width + x;

    // Bord explicitement écrit à 0 : la sortie ne dépend plus du contenu précédent de
    // d_out (avant : valeurs résiduelles du kernel précédent sur les bords).
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        d_out[o] = 0;
        return;
    }

    const unsigned char* r0 = d_in + static_cast<size_t>(y - 1) * width + x;
    const unsigned char* r1 = d_in + static_cast<size_t>(y) * width + x;
    const unsigned char* r2 = d_in + static_cast<size_t>(y + 1) * width + x;

    const int gx = -r0[-1] + r0[1] - 2 * r1[-1] + 2 * r1[1] - r2[-1] + r2[1];
    const int gy = -r0[-1] - 2 * r0[0] - r0[1] + r2[-1] + 2 * r2[0] + r2[1];

    d_out[o] = static_cast<unsigned char>(min(abs(gx) + abs(gy), 255));
}

void launchSobelNaive(const unsigned char* d_in, unsigned char* d_out, int width, int height) {
    // Bloc 32x8 : un warp = 32 pixels contigus d'une ligne (au lieu de 2 demi-lignes en 16x16).
    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);
    sobelGPUNaive<<<grid, block>>>(d_in, d_out, width, height);
    CUDA_CHECK_KERNEL();
}
