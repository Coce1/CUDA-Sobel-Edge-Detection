#include <iostream>
#include <stdio.h>

// Le Kernel (exécuté sur le GPU)
__global__ void helloCUDA() {
    // threadIdx.x donne l'identifiant matériel unique du cœur en train de travailler
    printf("Execution sur le coeur GPU (Thread) numero : %d\n", threadIdx.x);
}

int main() {
    std::cout << "--- CPU : Lancement de l'ordre ---" << std::endl;
    
    // Appel du Kernel : <<<Nombre de Blocs, Nombre de Threads par bloc>>>
    helloCUDA<<<1, 5>>>();
    
    // Synchronisation GPU / CPU
    cudaDeviceSynchronize(); 
    
    std::cout << "--- CPU : Fin de l'execution ---" << std::endl;
    return 0;
}