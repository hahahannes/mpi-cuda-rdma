CUCOMP  = nvcc
CUFLAGS = -arch=sm_70
OMPI_DIR = /usr/lib64/openmpi

INCLUDES  = -I/usr/include/openmpi-x86_64
LIBRARIES = -L/usr/lib64/openmpi/lib

compile:
	$(CUCOMP) $(CUFLAGS) $(LIBRARIES) $(INCLUDES) gdr.cu -o benchmark_cuda_staged -lmpi

.PHONY: clean

clean:
	rm -f pp_cuda_aware *.o

mpirun_with_gdr:
    # Autodetect GPUDirect RDMA
	mpirun -np 2 benchmark_cuda_aware 50

mpirun_without_gdr:
    # cuda_copy Copies via host memory (GPU → CPU → GPU)
	mpirun -np 2 -x UCX_IB_GPU_DIRECT_RDMA=no -x UCX_LOG_LEVEL=info -x UCX_TLS=rc,sm,cuda_copy benchmark_cuda_aware 50