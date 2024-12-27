#include <opencv2/opencv.hpp>
#include <cudnn.h>

#include <iostream>
#include "kernel.h"
#include <cuda_runtime.h>
void print_gpu_info() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cout << "No CUDA devices found!" << std::endl;
        return;
    }

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, i);

        std::cout << "Device " << i << ": " << device_prop.name << std::endl;
        std::cout << "  Compute capability: " << device_prop.major << "." << device_prop.minor << std::endl;
        std::cout << "  Total global memory: " << device_prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared memory per block: " << device_prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Number of multiprocessors: " << device_prop.multiProcessorCount << std::endl;
        std::cout << "  Max threads per block: " << device_prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max block dimensions: (" 
                  << device_prop.maxThreadsDim[0] << ", "
                  << device_prop.maxThreadsDim[1] << ", "
                  << device_prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max grid dimensions: (" 
                  << device_prop.maxGridSize[0] << ", "
                  << device_prop.maxGridSize[1] << ", "
                  << device_prop.maxGridSize[2] << ")" << std::endl;
    }
}

int main() {
    std::cout << "Starting CUDA program..." << std::endl;
    
    // Print GPU information
    print_gpu_info();
    
    kernel_function();  // Call the kernel function

    return 0;
}
