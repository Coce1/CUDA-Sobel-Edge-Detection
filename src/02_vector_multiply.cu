#include <iostream>
#include <vector>

__global__ void multiplyKernel(int* d_array, int size, int a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = d_array[idx] * a;
    }
}

int main() {
    int N = 5;
    int a = 10;
    size_t bytes = N * sizeof(int);

    // 1. Host allocation
    std::vector<int> h_array = {1, 2, 3, 4, 5};
    std::cout << "Avant GPU : ";
    for(int i = 0; i < N; i++) std::cout << h_array[i] << " ";
    std::cout << std::endl;

    // 2. Device allocation
    int* d_array;
    cudaMalloc((void**)&d_array, bytes);

    // 3. Transfert Host -> Device
    cudaMemcpy(d_array, h_array.data(), bytes, cudaMemcpyHostToDevice);

    // 4. Exécution du Kernel
    multiplyKernel<<<1, N>>>(d_array, N);
    cudaDeviceSynchronize();

    // 5. Transfert Device -> Host
    cudaMemcpy(h_array.data(), d_array, bytes, cudaMemcpyDeviceToHost);
    // ou cudaMemcpy(&h_array[0], d_array, bytes, cudaMemcpyDeviceToHost)
    cudaFree(d_array);

    std::cout << "Apres GPU : ";
    for(int i = 0; i < N; i++) std::cout << h_array[i] << " ";
    std::cout << std::endl;

    return 0;
}