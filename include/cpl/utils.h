#pragma once

#include <cuda_runtime.h>

// Forward declare the class template to break include cycles
namespace cpl {
template <typename T>
class Matrix;
}

void print_gpu_info();
void initialize_matrix(cpl::Matrix<float>& matrix);
void verify_matrix_multiplication(const cpl::Matrix<float>& A, const cpl::Matrix<float>& B, const cpl::Matrix<float>& C);
void transpose(const cpl::Matrix<float>& input, cpl::Matrix<float>& output);