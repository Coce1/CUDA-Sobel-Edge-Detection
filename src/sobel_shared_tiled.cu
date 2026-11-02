#include "../include/sobel.cuh"
#include "../include/cuda_utils.cuh"

constexpr int TILE_X = 32;
constexpr int TILE_Y = 8;
constexpr int HALO = 1;

__device__ __forceinline__ unsigned char loadPixel(const unsigned char* __restrict__ in,
                                                   int width, int height, int gx, int gy) {
    return (gx >= 0 && gx < width && gy >= 0 && gy < height)
               ? in[static_cast<size_t>(gy) * width + gx]
               : static_cast<unsigned char>(0);
}

// Tuile (TILE_Y+2) x (TILE_X+2) octets. Chaque thread charge son pixel central ; seuls
// les threads situés sur le pourtour de la tuile chargent le halo (côtés + 4 coins).
// Cela supprime la boucle de chargement avec i / 18 et i % 18 (divisions entières par
// une constante non puissance de 2) de la version d'origine.
__global__ void sobelFilter(const unsigned char* __restrict__ d_in,
                            unsigned char* __restrict__ d_out, int width, int height) {

    __shared__ unsigned char tile[TILE_Y + 2 * HALO][TILE_X + 2 * HALO];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * TILE_X + tx;
    const int y = blockIdx.y * TILE_Y + ty;

    tile[ty + 1][tx + 1] = loadPixel(d_in, width, height, x, y);

    // Côtés du halo
    if (ty == 0)          tile[0][tx + 1]          = loadPixel(d_in, width, height, x, y - 1);
    if (ty == TILE_Y - 1) tile[TILE_Y + 1][tx + 1] = loadPixel(d_in, width, height, x, y + 1);
    if (tx == 0)          tile[ty + 1][0]          = loadPixel(d_in, width, height, x - 1, y);
    if (tx == TILE_X - 1) tile[ty + 1][TILE_X + 1] = loadPixel(d_in, width, height, x + 1, y);

    // Coins du halo (nécessaires au noyau 3x3)
    if (tx == 0 && ty == 0)                    tile[0][0]                   = loadPixel(d_in, width, height, x - 1, y - 1);
    if (tx == TILE_X - 1 && ty == 0)           tile[0][TILE_X + 1]          = loadPixel(d_in, width, height, x + 1, y - 1);
    if (tx == 0 && ty == TILE_Y - 1)           tile[TILE_Y + 1][0]          = loadPixel(d_in, width, height, x - 1, y + 1);
    if (tx == TILE_X - 1 && ty == TILE_Y - 1)  tile[TILE_Y + 1][TILE_X + 1] = loadPixel(d_in, width, height, x + 1, y + 1);

    // Barrière placée AVANT toute sortie anticipée : tous les threads du bloc doivent
    // l'atteindre (une barrière dans une branche divergente est un comportement indéfini).
    __syncthreads();

    if (x >= width || y >= height) return;

    const size_t o = static_cast<size_t>(y) * width + x;
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) { // même convention que le naïf
        d_out[o] = 0;
        return;
    }

    // tile[ty][tx] = voisin haut-gauche du pixel (x, y) ; tile[ty+1][tx+1] = pixel central.
    const int gx = -tile[ty][tx]     + tile[ty][tx + 2]
                   - 2 * tile[ty + 1][tx] + 2 * tile[ty + 1][tx + 2]
                   - tile[ty + 2][tx] + tile[ty + 2][tx + 2];

    const int gy = -tile[ty][tx] - 2 * tile[ty][tx + 1] - tile[ty][tx + 2]
                   + tile[ty + 2][tx] + 2 * tile[ty + 2][tx + 1] + tile[ty + 2][tx + 2];

    d_out[o] = static_cast<unsigned char>(min(abs(gx) + abs(gy), 255));
}

void launchSobelShared(const unsigned char* d_in, unsigned char* d_out, int width, int height) {
    const dim3 block(TILE_X, TILE_Y);
    const dim3 grid((width + TILE_X - 1) / TILE_X, (height + TILE_Y - 1) / TILE_Y);
    sobelFilter<<<grid, block>>>(d_in, d_out, width, height);
    CUDA_CHECK_KERNEL();
}
