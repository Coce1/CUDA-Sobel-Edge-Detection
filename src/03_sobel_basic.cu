#include <iostream>
#include <vector>
#include <cmath>

__global__ void sobelFilter(int* d_in, int* d_out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;

        int gx = -d_in[(y-1)*width + (x-1)] + d_in[(y-1)*width + (x+1)]
                 -2*d_in[y*width + (x-1)]   + 2*d_in[y*width + (x+1)]
                 -d_in[(y+1)*width + (x-1)] + d_in[(y+1)*width + (x+1)];

        int gy = -d_in[(y-1)*width + (x-1)] - 2*d_in[(y-1)*width + x] - d_in[(y-1)*width + (x+1)]
                 +d_in[(y+1)*width + (x-1)] + 2*d_in[(y+1)*width + x] + d_in[(y+1)*width + (x+1)];

        d_out[idx] = min(abs(gx) + abs(gy), 255);
    }
}

int main() {
    int width = 16, height = 16;
    size_t bytes = width * height * sizeof(int);

    std::vector<int> h_image(width * height, 0);
    for(int y = 4; y < 12; y++) {
        for(int x = 4; x < 12; x++) {
            h_image[y * width + x] = 255;
        }
    }
    std::vector<int> h_output(width * height, 0);

    int *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
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