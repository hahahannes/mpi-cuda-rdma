MPICOMP = mpicc

CUCOMP  = nvcc
CUFLAGS = -arch=sm_70
OMPI_DIR = /usr/lib64/openmpi

INCLUDES  = -I/usr/include/openmpi-x86_64

# For non cuda aware mpi
#LIBRARIES = -L/usr/lib64/openmpi/lib 

# For cuda aware MPI
LIBRARIES = -L/usr/local/mpi/lib

compile-cpu: cpu.c
	$(MPICOMP) cpu.c -o cpu
	chmod +x ./cpu

compile-gpu: gpu.cu
	$(CUCOMP) $(CUFLAGS) $(LIBRARIES) $(INCLUDES) gpu.cu -o gpu -lmpi
	chmod +x ./gpu

compile-cuda-aware: cuda-aware.cu
	$(CUCOMP) $(CUFLAGS) $(LIBRARIES) $(INCLUDES) cuda-aware.cu -o cuda-aware -lmpi
	chmod +x ./cuda-aware

run-cpu:
	mpirun -n 2 ~/mpi-demo/mpi-cuda-rdma/demo/cpu

run-gpu:
	mpirun -n 2 ~/mpi-demo/mpi-cuda-rdma/demo/gpu

run-cuda-aware:
	mpirun -n 2 ~/mpi-demo/mpi-cuda-rdma/demo/cuda-aware

run-cuda-aware-disable-rdma:
# TODO test on sxm with cudaware mpi
# TODO TLS=tcp,sm instead
	mpirun -np 2 -x UCX_IB_GPU_DIRECT_RDMA=no -x UCX_LOG_LEVEL=info -x UCX_TLS=rc,sm,cuda_copy ./cuda-aware

run-cuda-aware-enabled-rdma:
# TODO test
	mpirun -np 2 -x UCX_IB_GPU_DIRECT_RDMA=yes -x UCX_LOG_LEVEL=info -x UCX_TLS=rc,sm,cuda ./cuda-aware