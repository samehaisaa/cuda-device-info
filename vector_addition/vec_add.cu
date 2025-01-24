#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// Include the utility functions
#include "../utilities/utils.cu"

__global__ void VecAdd(float* A, float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "Error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int N = 1024; 
    size_t size = N * sizeof(float); 

    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    initializeVector(h_A, N);
    initializeVector(h_B, N);

    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc((void**)&d_A, size), "Failed to allocate device memory for A");
    checkCudaError(cudaMalloc((void**)&d_B, size), "Failed to allocate device memory for B");
    checkCudaError(cudaMalloc((void**)&d_C, size), "Failed to allocate device memory for C");

    checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "Failed to copy A to device");
    checkCudaError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "Failed to copy B to device");

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed");

    checkCudaError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "Failed to copy C to host");

    for (int i = 0; i < N; i++) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            std::cerr << "Result verification failed at element " << i << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "Test PASSED" << std::endl;

    free(h_A);
    free(h_B);
    free(h_C);
    checkCudaError(cudaFree(d_A), "Failed to free device memory for A");
    checkCudaError(cudaFree(d_B), "Failed to free device memory for B");
    checkCudaError(cudaFree(d_C), "Failed to free device memory for C");

    return 0;
}