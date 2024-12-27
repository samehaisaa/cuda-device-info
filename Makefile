# Compiler and flags
CXX = g++
CUDA_COMPILER = nvcc
CXX_FLAGS = -O3 -std=c++14
CUDA_FLAGS = -arch=sm_75

# Directories
SRC_DIR = src
OBJ_DIR = build
INCLUDE_DIR = include
CUDA_INCLUDE_DIR = /usr/local/cuda/include

# Files
SOURCES = $(SRC_DIR)/main.cpp $(SRC_DIR)/kernel.cu
OBJECTS = $(OBJ_DIR)/main.o $(OBJ_DIR)/kernel.o
EXEC = my_cuda_program

# Targets
$(EXEC): $(OBJECTS)
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(OBJECTS) -o $(EXEC)

# Compile C++ files using g++
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE_DIR) -I$(CUDA_INCLUDE_DIR) -c $< -o $@

# Compile CUDA files using nvcc
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(CUDA_COMPILER) $(CUDA_FLAGS) -I$(INCLUDE_DIR) -I$(CUDA_INCLUDE_DIR) -c $< -o $@

clean:
	rm -f $(OBJ_DIR)/*.o $(EXEC)
