cmake_minimum_required(VERSION 3.10)
project(MyCudaProject)

# Enable CUDA support
find_package(CUDA REQUIRED)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS} include)

# Add executable
add_executable(my_cuda_program src/main.cpp src/kernel.cu)

# Link CUDA libraries
target_link_libraries(my_cuda_program ${CUDA_LIBRARIES})

# Enable CUDA flags if necessary
set_target_properties(my_cuda_program PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
