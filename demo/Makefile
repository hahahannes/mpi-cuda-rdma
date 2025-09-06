MPICOMP = mpicc
CUCOMP  = nvcc
CUFLAGS = -arch=sm_90

INCLUDES  = -I$(OMPI_DIR)/include
LIBRARIES = -L$(OMPI_DIR)/lib -lmpi_ibm

compile-cpu: cpu.c
	$(MPICOMP) -c cpu.c -o cpu

compile-gpu: gpu.c
	$(CUCOMP) $(CUFLAGS) $(LIBRARIES) gpu.o -o gpu
	$(CUCOMP) $(CUFLAGS) $(INCLUDES) -c gpu.cu

compile-cuda-aware: cuda-aware.c
	$(CUCOMP) $(CUFLAGS) $(LIBRARIES) cuda-aware.o -o cuda-aware
	$(CUCOMP) $(CUFLAGS) $(INCLUDES) -c cuda-aware.cu

run-cpu:
	mpirun -n 2 ./cpu

run-gpu:
	mpirun -n 2 ./gpu

run-cuda-aware:
	mpirun -n 2 ./cuda-aware