#include <iostream>

#include "cpl/common.h"
#include "cpl/matrix.h"
#include "cpl/utils.h"
#include "cpl/vector.h"
#include "cpl/logging.h"

int main() {
    cpl::logging::set_log_level(cpl::logging::LogLevel::INFO);
    cpl::logging::log_info("Starting CUDA program...");
    
    print_gpu_info();

    cpl::logging::log_info("\n=== Matrix Multiplication Demo ===");
    const int M = 64;
    const int N = 32;
    const int K = 16;

    cpl::Matrix<float> A(M, N);
    cpl::Matrix<float> B(N, K);
    cpl::Matrix<float> C(M, K);

    initialize_matrix(A);
    initialize_matrix(B);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    matrix_multiply(A, B, C);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    cpl::logging::log_info("Matrix multiplication time: ", milliseconds, " ms");
    verify_matrix_multiplication(A, B, C);

    cpl::logging::log_info("\n=== Vector Addition Demo ===");
    const int vectorSize = 1024;
    cpl::Vector<float> vec1(vectorSize);
    cpl::Vector<float> vec2(vectorSize);
    cpl::Vector<float> result(vectorSize);

    initializeVector(vec1);
    initializeVector(vec2);

    CUDA_CHECK(cudaEventRecord(start));
    vector_add(vec1, vec2, result);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    cpl::logging::log_info("Vector addition time: ", milliseconds, " ms");
    verify_vector_addition(vec1, vec2, result);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}