

__global__ void sobelGPUNaive(const int* __restrict__ d_in, int* __restrict__ d_out, int width, int height) {
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