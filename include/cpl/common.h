#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

// Use CUDA_CHECK macro for all CUDA calls
#define CUDA_CHECK(call) do { \
  cudaError_t error = call; \
  if (error != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d - %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(error)); \
    exit(1); \
  } \
} while(0)
