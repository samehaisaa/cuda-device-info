# Compilers
CXX = g++
CUDA_COMPILER = nvcc

# Compiler flags
CXX_FLAGS = -O3 -std=c++14
CUDA_FLAGS = -arch=sm_75

# Directories
SRC_DIR = src
OBJ_DIR = build
INCLUDE_DIR = include
MATRIX_OPS_DIR = matrix_ops
UTILITIES_DIR = utilities
VECTOR_ADD_DIR = vector_addition
CUDA_INCLUDE_DIR = /usr/local/cuda/include

# Source files
SOURCES = $(SRC_DIR)/main.cpp \
          $(SRC_DIR)/kernel.cu \
          $(MATRIX_OPS_DIR)/matrix_ops.cu \
          $(UTILITIES_DIR)/matrix_utils.cu \
          $(VECTOR_ADD_DIR)/vec_add.cu

# Object files
OBJECTS = $(OBJ_DIR)/main.o \
          $(OBJ_DIR)/kernel.o \
          $(OBJ_DIR)/matrix_ops.o \
          $(OBJ_DIR)/matrix_utils.o \
          $(OBJ_DIR)/vec_add.o

# Executable name
EXEC = my_cuda_program

# Include paths
INCLUDES = -I$(INCLUDE_DIR) \
           -I$(MATRIX_OPS_DIR) \
           -I$(UTILITIES_DIR) \
           -I$(VECTOR_ADD_DIR) \
           -I$(CUDA_INCLUDE_DIR)

# Main target
$(EXEC): $(OBJECTS)
    $(CUDA_COMPILER) $(CUDA_FLAGS) $(OBJECTS) -o $(EXEC)

# Rule for C++ source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
    @mkdir -p $(OBJ_DIR)
    $(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Rule for CUDA source files in src directory
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
    @mkdir -p $(OBJ_DIR)
    $(CUDA_COMPILER) $(CUDA_FLAGS) $(INCLUDES) -c $< -o $@

# Rule for CUDA source files in matrix_ops directory
$(OBJ_DIR)/matrix_ops.o: $(MATRIX_OPS_DIR)/matrix_ops.cu
    @mkdir -p $(OBJ_DIR)
    $(CUDA_COMPILER) $(CUDA_FLAGS) $(INCLUDES) -c $< -o $@

# Rule for CUDA source files in utilities directory
$(OBJ_DIR)/matrix_utils.o: $(UTILITIES_DIR)/matrix_utils.cu
    @mkdir -p $(OBJ_DIR)
    $(CUDA_COMPILER) $(CUDA_FLAGS) $(INCLUDES) -c $< -o $@

# Rule for CUDA source files in vector_addition directory
$(OBJ_DIR)/vec_add.o: $(VECTOR_ADD_DIR)/vec_add.cu
    @mkdir -p $(OBJ_DIR)
    $(CUDA_COMPILER) $(CUDA_FLAGS) $(INCLUDES) -c $< -o $@

# Clean target
clean:
    rm -rf $(OBJ_DIR) $(EXEC)

# Make build directory
$(shell mkdir -p $(OBJ_DIR))

.PHONY: clean