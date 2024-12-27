# My CUDA Project

This project demonstrates a simple CUDA program that prints GPU information and executes a kernel function.

## Project Structure
# cuda-device-info
## Files

- `src/main.cpp`: Contains the main function that initializes the CUDA program, prints GPU information, and calls the kernel function.
- `src/kernel.cu`: Contains the CUDA kernel function.
- `include/kernel.h`: Header file for the kernel function.
- `CMakeLists.txt`: CMake configuration file for building the project.
- `Makefile`: Makefile for building the project.

## Building the Project

To build the project, use the following commands:

```sh
mkdir build
cd build
cmake ..
make

