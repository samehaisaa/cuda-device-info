#include <iostream>
#include <cstdlib>
#include <cmath>
#include "matrix_utils.h"

void initialize_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void print_matrix(float* matrix, int rows, int cols, const char* name) {
    std::cout << "Matrix " << name << " (" << rows << "x" << cols << "):" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void verify_matrix_multiplication(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * K + j];
            }
            if (fabs(C[i * K + j] - sum) > 1e-5) {
                std::cerr << "Verification failed at (" << i << "," << j << ")" << std::endl;
                return;
            }
        }
    }
    std::cout << "Matrix multiplication verification passed!" << std::endl;
}