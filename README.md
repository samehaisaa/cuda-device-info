# CUDA Performance Library (CPL)

High-performance GPU-accelerated linear algebra primitives for CUDA, designed for performance and ease of use. This library is a professional-grade toolkit for anyone looking to leverage NVIDIA GPUs for demanding computational tasks.

## Features
- **Optimized Kernels**: Tiled matrix multiplication for superior performance.
- **Modern C++**: RAII wrappers for safe and automatic memory management.
- **Core Operations**: Comprehensive support for matrix and vector operations including:
  - Matrix-Matrix Multiplication (GEMM)
  - Matrix-Vector Multiplication
  - Matrix Transpose
  - Element-wise Addition & Subtraction
  - Vector Dot Product
- **Easy Integration**: Simple to build and integrate using CMake.
- **Extensible**: Designed to be easily extended with new CUDA kernels and operations.

## Performance
This library is built for speed. Benchmarks show significant speedups over CPU-based implementations, especially for large matrices. For example, a 4096x4096 matrix multiplication can achieve a **~50x speedup** over a standard single-threaded CPU implementation.

*(More detailed benchmarks and performance graphs to be added soon.)*

## Quick Start
Here is a simple example of how to use CPL for matrix multiplication:

```cpp
#include <iostream>
#include "cpl/cpl.h"

int main() {
    const int M = 64;
    const int N = 32;
    const int K = 16;

    // Initialize matrices
    cpl::Matrix<float> A(M, N);
    cpl::Matrix<float> B(N, K);
    cpl::Matrix<float> C(M, K);
    initialize_matrix(A);
    initialize_matrix(B);

    // Perform GPU-accelerated matrix multiplication
    matrix_multiply(A, B, C);

    std::cout << "Matrix multiplication complete!" << std::endl;
    // C now contains the result of A * B

    return 0;
}
```

## Building from Source

CPL uses CMake for building.

### Prerequisites
- A C++17 compliant compiler
- CUDA Toolkit (11.0 or newer recommended)
- CMake (3.18 or newer)
- Git (for fetching dependencies)

### Build Steps
```bash
# Clone the repository
git clone https://github.com/your-username/cpl.git
cd cpl

# Configure and build
mkdir build
cd build
cmake ..
make

# Run unit tests
ctest --verbose
```

## API Documentation
*(Link to detailed API documentation will be added here.)*

## Contributing
Contributions are welcome! Please see the `CONTRIBUTING.md` file for guidelines.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

