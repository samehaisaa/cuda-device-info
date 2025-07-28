#include <iostream>
#include "cpl/cpl.h"

#define TILE_WIDTH 16

__global__ void TiledMatrixMulKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Cvalue = 0.0f;

    for (int t = 0; t < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < M && t * TILE_WIDTH + tx < N) {
            sA[ty][tx] = A[row * N + t * TILE_WIDTH + tx];
        } else {
            sA[ty][tx] = 0.0f;
        }

        if (t * TILE_WIDTH + ty < N && col < K) {
            sB[ty][tx] = B[(t * TILE_WIDTH + ty) * K + col];
        } else {
            sB[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            Cvalue += sA[ty][i] * sB[i][tx];
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = Cvalue;
    }
}

void matrix_multiply(const cpl::Matrix<float>& A, const cpl::Matrix<float>& B, cpl::Matrix<float>& C) {
    int M = A.rows();
    int N = A.cols();
    int K = B.cols();

    if (N != B.rows() || M != C.rows() || K != C.cols()) {
        cpl::logging::log_error("Matrix dimensions are not valid for multiplication.");
        return;
    }

    cpl::DeviceMemory<float> d_A(A.size());
    cpl::DeviceMemory<float> d_B(B.size());
    cpl::DeviceMemory<float> d_C(C.size());

    A.to_device(d_A);
    B.to_device(d_B);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((K + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    TiledMatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A.get(), d_B.get(), d_C.get(), M, N, K);

    C.from_device(d_C);
}

__global__ void TransposeKernel(const float* input, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int r = idx / cols;
        int c = idx % cols;
        output[c * rows + r] = input[idx];
    }
}

void transpose(const cpl::Matrix<float>& input, cpl::Matrix<float>& output) {
    int rows = input.rows();
    int cols = input.cols();

    if (output.rows() != cols || output.cols() != rows) {
        cpl::logging::log_error("Output matrix dimensions must be swapped for transpose.");
        return;
    }

    cpl::DeviceMemory<float> d_input(input.size());
    cpl::DeviceMemory<float> d_output(output.size());

    input.to_device(d_input);

    int threadsPerBlock = 256;
    int blocksPerGrid = (rows * cols + threadsPerBlock - 1) / threadsPerBlock;
    TransposeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input.get(), d_output.get(), rows, cols);

    output.from_device(d_output);
}

__global__ void ElementWiseAddKernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void element_wise_add(const cpl::Matrix<float>& A, const cpl::Matrix<float>& B, cpl::Matrix<float>& C) {
    if (A.rows() != B.rows() || A.cols() != B.cols() || A.rows() != C.rows() || A.cols() != C.cols()) {
        cpl::logging::log_error("Matrices must have the same dimensions for element-wise addition.");
        return;
    }

    cpl::DeviceMemory<float> d_A(A.size());
    cpl::DeviceMemory<float> d_B(B.size());
    cpl::DeviceMemory<float> d_C(C.size());

    A.to_device(d_A);
    B.to_device(d_B);

    int threadsPerBlock = 256;
    int blocksPerGrid = (A.size() + threadsPerBlock - 1) / threadsPerBlock;
    ElementWiseAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A.get(), d_B.get(), d_C.get(), A.size());

    C.from_device(d_C);
}

__global__ void MatrixVecMulKernel(const float* A, const float* x, float* y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * x[i];
        }
        y[row] = sum;
    }
}

void matrix_vector_multiply(const cpl::Matrix<float>& A, const cpl::Vector<float>& x, cpl::Vector<float>& y) {
    int M = A.rows();
    int N = A.cols();

    if (N != x.size() || M != y.size()) {
        cpl::logging::log_error("Matrix and vector dimensions are not valid for multiplication.");
        return;
    }

    cpl::DeviceMemory<float> d_A(A.size());
    cpl::DeviceMemory<float> d_x(x.size());
    cpl::DeviceMemory<float> d_y(y.size());

    A.to_device(d_A);
    x.to_device(d_x);

    int threadsPerBlock = 256;
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;
    MatrixVecMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A.get(), d_x.get(), d_y.get(), M, N);

    y.from_device(d_y);
}

__global__ void ElementWiseSubKernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] - B[i];
    }
}

void element_wise_subtract(const cpl::Matrix<float>& A, const cpl::Matrix<float>& B, cpl::Matrix<float>& C) {
    if (A.rows() != B.rows() || A.cols() != B.cols() || A.rows() != C.rows() || A.cols() != C.cols()) {
        cpl::logging::log_error("Matrices must have the same dimensions for element-wise subtraction.");
        return;
    }

    cpl::DeviceMemory<float> d_A(A.size());
    cpl::DeviceMemory<float> d_B(B.size());
    cpl::DeviceMemory<float> d_C(C.size());

    A.to_device(d_A);
    B.to_device(d_B);

    int threadsPerBlock = 256;
    int blocksPerGrid = (A.size() + threadsPerBlock - 1) / threadsPerBlock;
    ElementWiseSubKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A.get(), d_B.get(), d_C.get(), A.size());

    C.from_device(d_C);
}