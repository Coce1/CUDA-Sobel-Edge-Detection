NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17

all: 01_hello 02_multiply 03_sobel_basic 04_benchmark 05_shared

01_hello: src/01_hello_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

02_multiply: src/02_vector_multiply.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

03_sobel_basic: src/03_sobel_basic.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

04_benchmark: src/04_sobel_cpu_vs_gpu.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

05_shared: src/05_sobel_shared_memory.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f 01_hello 02_multiply 03_sobel_basic 04_benchmark 05_shared