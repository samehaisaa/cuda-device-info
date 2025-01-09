CXX = g++
CUDA_COMPILER = nvcc
CXX_FLAGS = -O3 -std=c++14
CUDA_FLAGS = -arch=sm_75

SRC_DIR = src
OBJ_DIR = build
INCLUDE_DIR = include
CUDA_INCLUDE_DIR = /usr/local/cuda/include
TARG_DIR = targ
SOURCES = $(SRC_DIR)/main.cpp $(SRC_DIR)/kernel.cu
OBJECTS = $(OBJ_DIR)/main.o $(OBJ_DIR)/kernel.o
EXEC = my_cuda_program

$(EXEC): $(OBJECTS)
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(OBJECTS) -o $(EXEC)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE_DIR) -I$(CUDA_INCLUDE_DIR) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(CUDA_COMPILER) $(CUDA_FLAGS) -I$(INCLUDE_DIR) -I$(CUDA_INCLUDE_DIR) -c $< -o $@

clean:
	rm -f $(OBJ_DIR)/*.o $(EXEC)
