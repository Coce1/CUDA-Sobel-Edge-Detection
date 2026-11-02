# Les commandes commencent par '>' au lieu d'une tabulation (robuste au copier-coller)
.RECIPEPREFIX = >

NVCC ?= nvcc

# Architecture GPU : sm_75 = T4. Surcharger : make ARCH=sm_86
ARCH ?= sm_75

NVCC_FLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo -I./include

SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj

TARGET = sobel_benchmark

CU_SOURCES  = $(SRC_DIR)/baenchmarks.cu $(SRC_DIR)/sobel_naive.cu $(SRC_DIR)/sobel_shared_tiled.cu
CPP_SOURCES = $(SRC_DIR)/sobel_cpu.cpp

OBJS = $(patsubst $(SRC_DIR)/%.cu,  $(OBJ_DIR)/%.o, $(CU_SOURCES)) \
       $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SOURCES))

# Tout header modifié => recompilation de tous les objets
HEADERS = $(wildcard $(INC_DIR)/*.cuh)

all: $(TARGET)

$(TARGET): $(OBJS)
> $(NVCC) $(NVCC_FLAGS) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS) | $(OBJ_DIR)
> $(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(OBJ_DIR)
> $(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ_DIR):
> mkdir -p $@

clean:
> rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean
