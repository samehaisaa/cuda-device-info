#include <iostream>
#include <cuda_runtime.h>
#include "cpl/cpl.h"

__global__ void VecAddKernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void vector_add(const cpl::Vector<float>& A, const cpl::Vector<float>& B, cpl::Vector<float>& C) {
    if (A.size() != B.size() || A.size() != C.size()) {
        cpl::logging::log_error("Vectors must have the same size for addition.");
        return;
    }
    
    cpl::DeviceMemory<float> d_A(A.size());
    cpl::DeviceMemory<float> d_B(B.size());
    cpl::DeviceMemory<float> d_C(C.size());

    A.to_device(d_A);
    B.to_device(d_B);

    int threadsPerBlock = 256;
    int blocksPerGrid = (A.size() + threadsPerBlock - 1) / threadsPerBlock;
    VecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A.get(), d_B.get(), d_C.get(), A.size());

    C.from_device(d_C);
}

void initializeVector(cpl::Vector<float>& vec) {
    for (int i = 0; i < vec.size(); i++) {
        vec.at(i) = static_cast<float>(rand()) / RAND_MAX;
    }
}

void verify_vector_addition(const cpl::Vector<float>& A, const cpl::Vector<float>& B, const cpl::Vector<float>& C) {
    for (int i = 0; i < A.size(); i++) {
        if (fabs(C.at(i) - (A.at(i) + B.at(i))) > 1e-5) {
            std::cerr << "Verification failed at index " << i << std::endl;
            return;
        }
    }
    std::cout << "Vector addition verification passed!" << std::endl;
}

__global__ void VecSubKernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] - B[i];
    }
}

void vector_subtract(const cpl::Vector<float>& A, const cpl::Vector<float>& B, cpl::Vector<float>& C) {
    if (A.size() != B.size() || A.size() != C.size()) {
        cpl::logging::log_error("Vectors must have the same size for subtraction.");
        return;
    }

    cpl::DeviceMemory<float> d_A(A.size());
    cpl::DeviceMemory<float> d_B(B.size());
    cpl::DeviceMemory<float> d_C(C.size());

    A.to_device(d_A);
    B.to_device(d_B);

    int threadsPerBlock = 256;
    int blocksPerGrid = (A.size() + threadsPerBlock - 1) / threadsPerBlock;
    VecSubKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A.get(), d_B.get(), d_C.get(), A.size());

    C.from_device(d_C);
}

__global__ void DotProductKernel(const float* A, const float* B, float* C, int N) {
    __shared__ float cache[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (i < N) {
        temp += A[i] * B[i];
        i += gridDim.x * blockDim.x;
    }
    cache[cacheIndex] = temp;
    __syncthreads();

    // Reduction in shared memory
    int i_ = blockDim.x / 2;
    while (i_ != 0) {
        if (cacheIndex < i_) {
            cache[cacheIndex] += cache[cacheIndex + i_];
        }
        __syncthreads();
        i_ /= 2;
    }

    if (cacheIndex == 0) {
        C[blockIdx.x] = cache[0];
    }
}

float dot_product(const cpl::Vector<float>& A, const cpl::Vector<float>& B) {
    if (A.size() != B.size()) {
        cpl::logging::log_error("Vectors must have the same size for dot product.");
        return 0.0f;
    }

    int N = A.size();
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    cpl::DeviceMemory<float> d_A(N);
    cpl::DeviceMemory<float> d_B(N);
    cpl::DeviceMemory<float> d_C(blocksPerGrid);
    
    A.to_device(d_A);
    B.to_device(d_B);

    DotProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A.get(), d_B.get(), d_C.get(), N);

    // Sum the partial results from each block
    std::vector<float> h_C(blocksPerGrid);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C.get(), d_C.bytes(), cudaMemcpyDeviceToHost));

    float result = 0.0f;
    for (int i = 0; i < blocksPerGrid; ++i) {
        result += h_C[i];
    }

    return result;
}