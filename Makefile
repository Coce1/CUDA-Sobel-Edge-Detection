# Compiler
NVCC = nvcc

# Compilation options (Maximum optimization)
NVCC_FLAGS = -O3

# Directory tree folders
SRC_DIR = src
INC_DIR = include

# List of source files
CU_SOURCES = $(SRC_DIR)/baenchmarks.cu $(SRC_DIR)/sobel_naive.cu $(SRC_DIR)/sobel_shared_tiled.cu
CPP_SOURCES = $(SRC_DIR)/sobel_cpu.cpp

# Name of the final executable
TARGET = sobel_benchmark

# Default rule
all: $(TARGET)

# Compilation
$(TARGET):
	$(NVCC) $(NVCC_FLAGS) -I$(INC_DIR) $(CU_SOURCES) $(CPP_SOURCES) -o $(TARGET)

# Cleaning
clean:
	rm -f $(TARGET)