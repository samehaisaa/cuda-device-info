#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// CUDA kernel for vector addition
__global__ void VecAdd(float* A, float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Host function to initialize vectors with random values
void initializeVector(float* vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = static_cast<float>(rand()) / RAND_MAX; // Random float between 0 and 1
    }
}

int main() {
    // Set the size of the vectors
    int N = 1024; // Number of elements in each vector
    size_t size = N * sizeof(float); // Size in bytes

    // Allocate memory on the host
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize input vectors
    initializeVector(h_A, N);
    initializeVector(h_B, N);

    // Allocate memory on the device
    float* d_A; 
    float* d_B; 
    float* d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    bool success = true;
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-8) { // Allow small numerical errors
            success = false;
            std::cerr << "Error at index " << i << ": expected " << expected << ", got " << h_C[i] << std::endl;
        }
    }

    if (success) {
        std::cout << "All results are correct!" << std::endl;
    } else {
        std::cerr << "Some results are incorrect!" << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
