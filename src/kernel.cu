#include <iostream>
#include <cuda_runtime.h>
#include "kernel.h"

__global__ void kernel() {
    printf("Hello from the CUDA kernel!\n");
}

void kernel_function() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}

void print_gpu_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "GPU " << i << ": " << prop.name << std::endl;
        std::cout << "  Total Global Memory: " << prop.totalGlobalMem << std::endl;
        std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock << std::endl;
        std::cout << "  Registers per Block: " << prop.regsPerBlock << std::endl;
        std::cout << "  Warp Size: " << prop.warpSize << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Total Constant Memory: " << prop.totalConstMem << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    }
}