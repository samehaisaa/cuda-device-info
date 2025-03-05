#include <iostream>
#include <cuda_runtime.h>

__global__ void VecAddKernel(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void vector_add(float* A, float* B, float* C, int N) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void initializeVector(float* vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void verify_vector_addition(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        if (fabs(C[i] - (A[i] + B[i])) > 1e-5) {
            std::cerr << "Verification failed at index " << i << std::endl;
            return;
        }
    }
    std::cout << "Vector addition verification passed!" << std::endl;
}