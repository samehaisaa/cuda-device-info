#include <iostream>
#include <cstdlib>
#include <cmath>
#include "cpl/cpl.h"

void initialize_matrix(cpl::Matrix<float>& matrix) {
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            matrix.at(i, j) = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

void print_matrix(const cpl::Matrix<float>& matrix, const char* name) {
    std::cout << "Matrix " << name << " (" << matrix.rows() << "x" << matrix.cols() << "):" << std::endl;
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            std::cout << matrix.at(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

void verify_matrix_multiplication(const cpl::Matrix<float>& A, const cpl::Matrix<float>& B, const cpl::Matrix<float>& C) {
    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < B.cols(); j++) {
            float sum = 0.0f;
            for (int k = 0; k < A.cols(); k++) {
                sum += A.at(i, k) * B.at(k, j);
            }
            if (fabs(C.at(i, j) - sum) > 1e-5) {
                std::cerr << "Verification failed at (" << i << "," << j << ")" << std::endl;
                return;
            }
        }
    }
    std::cout << "Matrix multiplication verification passed!" << std::endl;
}