#include <iostream>
#include <cuda_runtime.h>
#include "cpl/cpl.h"

void print_gpu_info() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        cpl::logging::log_error("No CUDA devices found!");
        return;
    }

    cpl::logging::log_info("Detected ", deviceCount, " CUDA device(s).");

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        cpl::logging::log_info("--- GPU ", i, ": ", prop.name, " ---");
        cpl::logging::log_info("  Compute Capability: ", prop.major, ".", prop.minor);
        cpl::logging::log_info("  Total Global Memory: ", prop.totalGlobalMem / (1024 * 1024), " MB");
        cpl::logging::log_info("  Shared Memory per Block: ", prop.sharedMemPerBlock / 1024, " KB");
        cpl::logging::log_info("  Registers per Block: ", prop.regsPerBlock);
        cpl::logging::log_info("  Warp Size: ", prop.warpSize);
        cpl::logging::log_info("  Max Threads per Block: ", prop.maxThreadsPerBlock);
        cpl::logging::log_info("  Total Constant Memory: ", prop.totalConstMem / 1024, " KB");
        cpl::logging::log_info("-------------------------");
    }
}