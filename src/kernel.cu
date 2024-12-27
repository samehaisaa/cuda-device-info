#include <iostream>
#include "kernel.h"

__global__ void simple_kernel() {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;

    printf("Hello from CUDA kernel! Thread %d out of %d threads.\n", thread_id, num_threads);
}

void kernel_function() {
    int threads_per_block = 256; // Number of threads per block
    int num_blocks = 1;          // Number of blocks

    simple_kernel<<<num_blocks, threads_per_block>>>();
    cudaDeviceSynchronize();
}
