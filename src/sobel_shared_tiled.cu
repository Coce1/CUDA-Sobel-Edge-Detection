#include <iostream>
#include <vector>
#include <cmath>
#define TILE_DIM 16
#define HALO 1
#define SHARED_DIM (TILE_DIM + 2 * HALO) // 18

__global__ void sobelFilter(const unsigned char* __restrict__ d_in,unsigned char* __restrict__ d_out, int width, int height) {
    __shared__ unsigned char s_tile[SHARED_DIM * SHARED_DIM];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int tid = threadIdx.y * TILE_DIM + threadIdx.x;

    // 1. Chargement coopératif avec gestion du Zero Padding sur les bords extrêmes
    for (int i = tid; i < SHARED_DIM * SHARED_DIM; i += TILE_DIM * TILE_DIM) {
        int local_y = i / SHARED_DIM;
        int local_x = i % SHARED_DIM;
        
        int global_x = blockIdx.x * TILE_DIM + local_x - HALO;
        int global_y = blockIdx.y * TILE_DIM + local_y - HALO;

        if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
            s_tile[local_y * SHARED_DIM + local_x] = d_in[global_y * width + global_x];
        } else {
            s_tile[local_y * SHARED_DIM + local_x] = 0;
        }
    }

    __syncthreads();

    if (x < width  && y < height ) {
        int lx = threadIdx.x + HALO;
        int ly = threadIdx.y + HALO;

        int gx = -s_tile[(ly-1)*SHARED_DIM + (lx-1)] + s_tile[(ly-1)*SHARED_DIM + (lx+1)]
                 -2*s_tile[ly*SHARED_DIM + (lx-1)]   + 2*s_tile[ly*SHARED_DIM + (lx+1)]
                 -s_tile[(ly+1)*SHARED_DIM + (lx-1)] + s_tile[(ly+1)*SHARED_DIM + (lx+1)];

        int gy = -s_tile[(ly-1)*SHARED_DIM + (lx-1)] - 2*s_tile[(ly-1)*SHARED_DIM + lx] - s_tile[(ly-1)*SHARED_DIM + (lx+1)]
                 +s_tile[(ly+1)*SHARED_DIM + (lx-1)] + 2*s_tile[(ly+1)*SHARED_DIM + lx] + s_tile[(ly+1)*SHARED_DIM + (lx+1)];

        d_out[y * width + x] = min(abs(gx) + abs(gy), 255);
    }
}

int main() {
    int width = 16, height = 16;
    size_t bytes = width * height * sizeof(unsigned char);

    std::vector<unsigned char> h_image(width * height, 0);
    for(int y = 4; y < 12; y++) {
        for(int x = 4; x < 12; x++) {
            h_image[y * width + x] = 255;
        }
    }
    std::vector<unsigned char> h_output(width * height, 0);

    unsigned char *d_in, *d_out;
    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, bytes);
    cudaMemcpy(d_in, h_image.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sobelFilter<<<numBlocks, threadsPerBlock>>>(d_in, d_out, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    std::cout << "--- Contours detectes par le filtre de Sobel (GPU) ---" << std::endl;
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            if (h_output[y * width + x] > 100) std::cout << "# ";
            else std::cout << ". ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}