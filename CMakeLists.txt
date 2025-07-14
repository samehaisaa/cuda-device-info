cmake_minimum_required(VERSION 3.18)
project(CPL LANGUAGES CXX CUDA)

message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "Configuring CUDA Performance Library (CPL)")

# --- Standard settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Find CUDA and other NVIDIA libraries ---
find_package(CUDA REQUIRED)
find_package(CUDAToolkit) # For cuBLAS

# --- Library: CPL ---
# Gather source files
file(GLOB_RECURSE CPL_SOURCES
    "src/matrix/*.cu"
    "src/vector/*.cu"
    "src/utilities/*.cu"
)

# Create the static library
add_library(cpl STATIC ${CPL_SOURCES})
target_link_libraries(cpl PUBLIC CUDA::toolkit)

# Set CUDA specific properties
set_target_properties(cpl PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Define public include directory
target_include_directories(cpl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# --- Examples ---
message(STATUS "Configuring examples...")
file(GLOB EXAMPLE_SOURCES "examples/*.cpp")
foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
    target_link_libraries(${EXAMPLE_NAME} PRIVATE cpl)
    message(STATUS "Added example: ${EXAMPLE_NAME}")
endforeach()

# --- Benchmarks ---
message(STATUS "Configuring benchmarks...")
file(GLOB BENCHMARK_SOURCES "benchmarks/*.cu")
foreach(BENCHMARK_SOURCE ${BENCHMARK_SOURCES})
    get_filename_component(BENCHMARK_NAME ${BENCHMARK_SOURCE} NAME_WE)
    add_executable(${BENCHMARK_NAME} ${BENCHMARK_SOURCE})
    target_link_libraries(${BENCHMARK_NAME} PRIVATE cpl)
    message(STATUS "Added benchmark: ${BENCHMARK_NAME}")
endforeach()

# Link cuBLAS to the specific benchmark
if(TARGET compare_with_cublas)
    target_link_libraries(compare_with_cublas PRIVATE ${CUDAToolkit_CUBLAS_LIBRARIES})
    message(STATUS "Linking cuBLAS to compare_with_cublas benchmark")
endif()

# --- Testing ---
enable_testing()

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG    release-1.12.1
)
FetchContent_MakeAvailable(googletest)

file(GLOB TEST_SOURCES "tests/unit/*.cpp" "tests/integration/*.cpp")

add_executable(run_tests ${TEST_SOURCES})
target_link_libraries(run_tests PRIVATE gtest_main cpl)

add_test(NAME unit_tests COMMAND run_tests)

# --- Installation ---
# Configuration for installation will be added here later
# install(TARGETS cpl
#     EXPORT CPLTargets
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib
#     RUNTIME DESTINATION bin
# )
# install(DIRECTORY include/ DESTINATION include)
