CUCOMP  = nvcc
CUFLAGS = -arch=sm_70
OMPI_DIR = /usr/lib64/openmpi

INCLUDES  = -I/usr/include/openmpi-x86_64
LIBRARIES = -L/usr/lib64/openmpi/lib

compile:
	$(CUCOMP) $(CUFLAGS) $(LIBRARIES) $(INCLUDES) staged.cu -o benchmark_cuda_staged -lmpi

.PHONY: clean

clean:
	rm -f pp_cuda_staged *.o

mpirun:
	mpirun -np 2 benchmark_cuda_staged 50