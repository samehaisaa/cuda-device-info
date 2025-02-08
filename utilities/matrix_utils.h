#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

void initialize_matrix(float* matrix, int rows, int cols);
void print_matrix(float* matrix, int rows, int cols, const char* name);
void verify_matrix_multiplication(float* A, float* B, float* C, int M, int N, int K);

#endif