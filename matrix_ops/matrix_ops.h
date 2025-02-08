#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

void matrix_multiply(float* A, float* B, float* C, int M, int N, int K);
void print_matrix(float* matrix, int rows, int cols, const char* name);

#endif