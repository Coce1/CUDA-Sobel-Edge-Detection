#ifndef SOBEL_CUH
#define SOBEL_CUH

#include <vector>

// Convention commune aux 3 implémentations : les pixels du BORD de l'image valent 0
// en sortie. (Avant : le naïf les laissait inchangés, le tuilé les calculait en
// zero-padding -> sorties non équivalentes.) Images 8 bits, row-major.

// Référence CPU.
void sobelCPU(const std::vector<unsigned char>& in, std::vector<unsigned char>& out,
              int width, int height);

// Launchers hôte (les kernels restent privés à leur .cu). Pointeurs d_* en mémoire GPU.
// Chaque launcher vérifie sa configuration de lancement ; l'appelant synchronise.
void launchSobelNaive(const unsigned char* d_in, unsigned char* d_out, int width, int height);
void launchSobelShared(const unsigned char* d_in, unsigned char* d_out, int width, int height);

#endif // SOBEL_CUH
