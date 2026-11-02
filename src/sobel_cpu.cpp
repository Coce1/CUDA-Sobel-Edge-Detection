#include "../include/sobel.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdlib>

void sobelCPU(const std::vector<unsigned char>& in, std::vector<unsigned char>& out,
              int width, int height) {

    // Bord à 0 (convention commune CPU/GPU)
    std::fill(out.begin(), out.end(), static_cast<unsigned char>(0));

    for (int y = 1; y < height - 1; ++y) {
        const unsigned char* r0 = &in[static_cast<std::size_t>(y - 1) * width];
        const unsigned char* r1 = &in[static_cast<std::size_t>(y) * width];
        const unsigned char* r2 = &in[static_cast<std::size_t>(y + 1) * width];

        for (int x = 1; x < width - 1; ++x) {
            const int gx = -r0[x - 1] + r0[x + 1]
                           - 2 * r1[x - 1] + 2 * r1[x + 1]
                           - r2[x - 1] + r2[x + 1];

            const int gy = -r0[x - 1] - 2 * r0[x] - r0[x + 1]
                           + r2[x - 1] + 2 * r2[x] + r2[x + 1];

            out[static_cast<std::size_t>(y) * width + x] =
                static_cast<unsigned char>(std::min(std::abs(gx) + std::abs(gy), 255));
        }
    }
}
