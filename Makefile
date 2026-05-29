NVCC     = nvcc
CXX      = g++
CXXFLAGS = -O2 -std=c++20 -Iinc
NVCCFLAGS = -O2 -std=c++20 -Iinc

SRC_DIR = src
INC_DIR = inc

CUDA_SRC  = $(SRC_DIR)/edge.cu
CUDA_BIN  = edge_cuda

cuda: $(CUDA_SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $(CUDA_BIN)
	./$(CUDA_BIN)

CPU_SRC  = $(SRC_DIR)/edge.cpp
CPU_BIN  = edge_cpu

cpu: $(CPU_SRC)
	$(CXX) $(CXXFLAGS) $< -o $(CPU_BIN)
	./$(CPU_BIN)

clean:
	rm -f $(CUDA_BIN) $(CPU_BIN)

.PHONY: cuda cpu clean
