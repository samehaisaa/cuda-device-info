#include <cuda_runtime.h>
#include <iostream>
#include "kernel.h"
#include "../matrix_ops/matrix_ops.h"
#include "../utilities/matrix_utils.h"
#include "../vector_addition/vec_add.h"

int main() {
    std::cout << "Starting CUDA program..." << std::endl;
    
    print_gpu_info();
    
    kernel_function();

    std::cout << "\n=== Matrix Multiplication Demo ===" << std::endl;
    const int M = 64;  // rows of A
    const int N = 32;  // cols of A, rows of B
    const int K = 16;  // cols of B

    float* A = new float[M * N];
    float* B = new float[N * K];
    float* C = new float[M * K];

    initialize_matrix(A, M, N);
    initialize_matrix(B, N, K);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matrix_multiply(A, B, C, M, N, K);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "Matrix multiplication time: " << milliseconds << " ms" << std::endl;
    verify_matrix_multiplication(A, B, C, M, N, K);

    std::cout << "\n=== Vector Addition Demo ===" << std::endl;
    const int vectorSize = 1024;
    float* vec1 = new float[vectorSize];
    float* vec2 = new float[vectorSize];
    float* result = new float[vectorSize];

    initializeVector(vec1, vectorSize);
    initializeVector(vec2, vectorSize);

    cudaEventRecord(start);
    vector_add(vec1, vec2, result, vectorSize);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "Vector addition time: " << milliseconds << " ms" << std::endl;
    verify_vector_addition(vec1, vec2, result, vectorSize);

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] vec1;
    delete[] vec2;
    delete[] result;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}