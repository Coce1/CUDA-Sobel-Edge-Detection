# Compilateur
NVCC = nvcc

# Options de compilation (Optimisation maximale)
NVCC_FLAGS = -O3

# Dossiers de l'arborescence
SRC_DIR = src
INC_DIR = include

# Liste des fichiers sources
CU_SOURCES = $(SRC_DIR)/benchmark.cu $(SRC_DIR)/sobel_naive.cu $(SRC_DIR)/sobel_shared.cu
CPP_SOURCES = $(SRC_DIR)/sobel_cpu.cpp

# Nom de l'exécutable final
TARGET = sobel_benchmark

# Règle par défaut
all: $(TARGET)

# Compilation
$(TARGET):
	$(NVCC) $(NVCC_FLAGS) -I$(INC_DIR) $(CU_SOURCES) $(CPP_SOURCES) -o $(TARGET)

# Nettoyage
clean:
	rm -f $(TARGET)