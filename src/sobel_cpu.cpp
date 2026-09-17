#include <vector>
#include <cmath>
#include <algorithm>
#include "../include/sobel.cuh"

void sobelCPU(const std::vector<unsigned char>& in, std::vector<unsigned char>& out, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int gx = -in[(y - 1) * width + (x - 1)] + in[(y - 1) * width + (x + 1)]
                - 2 * in[y * width + (x - 1)] + 2 * in[y * width + (x + 1)]
                - in[(y + 1) * width + (x - 1)] + in[(y + 1) * width + (x + 1)];

            int gy = -in[(y - 1) * width + (x - 1)] - 2 * in[(y - 1) * width + x] - in[(y - 1) * width + (x + 1)]
                + in[(y + 1) * width + (x - 1)] + 2 * in[(y + 1) * width + x] + in[(y + 1) * width + (x + 1)];

            out[y * width + x] = std::min(std::abs(gx) + std::abs(gy), 255);
        }
    }
}