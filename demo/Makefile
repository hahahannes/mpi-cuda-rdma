MPICOMP = mpicc

CUCOMP  = nvcc
CUFLAGS = -arch=sm_70
OMPI_DIR = /usr/lib64/openmpi

INCLUDES  = -I/usr/include/openmpi-x86_64
LIBRARIES = -L/usr/lib64/openmpi/lib

compile-cpu: cpu.c
	$(MPICOMP) -c cpu.c -o cpu

compile-gpu: gpu.c
	$(CUCOMP) $(CUFLAGS) $(LIBRARIES) $(INCLUDES) gpu.cu -o gpu -lmpi

compile-cuda-aware: cuda-aware.c
	$(CUCOMP) $(CUFLAGS) $(LIBRARIES) $(INCLUDES) cuda-aware.cu -o cuda-aware -lmpi

run-cpu:
	mpirun -n 2 ./cpu

run-gpu:
	mpirun -n 2 ./gpu

run-cuda-aware:
	mpirun -n 2 ./cuda-aware

run-cuda-aware-disable-rdma:
# TODO test
# TODO TLS=tcp,sm instead
	mpirun -np 2 -x UCX_IB_GPU_DIRECT_RDMA=no -x UCX_LOG_LEVEL=info -x UCX_TLS=rc,sm,cuda_copy ./cuda-aware