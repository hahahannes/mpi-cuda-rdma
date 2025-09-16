#! /bin/bash

# INTRA NODE
# With IPC
mpirun --hostfile=/shared/intra-hostfile --allow-run-as-root -np 2 -mca pml ucx -bind-to-core \
-x UCX_TLS=rc,cuda_copy,cuda_ipc \
/eos/home-h/hannesja/osi-benchmark/osu-micro-benchmarks-7.5.1/c/mpi/pt2pt/standard/osu_bw -d cuda D D \
| tee /shared/ngt-benchmarks/osu/intra-ipc

# Without IPC
mpirun --hostfile=/shared/intra-hostfile --allow-run-as-root -np 2 -mca pml ucx -bind-to-core \
-x UCX_TLS=rc,cuda_copy \
/eos/home-h/hannesja/osi-benchmark/osu-micro-benchmarks-7.5.1/c/mpi/pt2pt/standard/osu_bw -d cuda D D \
| tee /shared/ngt-benchmarks/osu/intra-no-ipc

# INTER NODE
# IB - RDMA
mpirun --hostfile=/shared/inter-hostfile --allow-run-as-root -np 2 -npernode 1 -mca pml ucx -bind-to-core \
-x UCX_IB_GPU_DIRECT_RDMA=yes \
-x UCX_TLS=rc,cuda \
/eos/home-h/hannesja/osi-benchmark/osu-micro-benchmarks-7.5.1/c/mpi/pt2pt/standard/osu_bw -d cuda D D \
| tee /shared/ngt-benchmarks/osu/inter-ib-rdma

# IB - No RDMA 
mpirun --hostfile=/shared/inter-hostfile --allow-run-as-root -np 2 -npernode 1 -mca pml ucx -bind-to-core \
-x UCX_IB_GPU_DIRECT_RDMA=no \
-x UCX_TLS=rc,cuda \
/eos/home-h/hannesja/osi-benchmark/osu-micro-benchmarks-7.5.1/c/mpi/pt2pt/standard/osu_bw -d cuda D D \
| tee /shared/ngt-benchmarks/osu/inter-ib-no-rdma

# Ethernet - No RDMA 
mpirun --hostfile=/shared/inter-hostfile --allow-run-as-root -np 2 -npernode 1 -mca pml ucx -bind-to-core \
-x UCX_IB_GPU_DIRECT_RDMA=no \
-x UCX_TLS=tcp,cuda \
/eos/home-h/hannesja/osi-benchmark/osu-micro-benchmarks-7.5.1/c/mpi/pt2pt/standard/osu_bw -d cuda D D \
| tee /shared/ngt-benchmarks/osu/inter-eth-no-rdma

# Non Cuda Aware - Ethernet 
mpirun --hostfile /shared/non-cuda-inter-hostfile --allow-run-as-root -np 2 -npernode 1 -bind-to-core \
/eos/home-h/hannesja/osi-benchmark/no-cuda-osu-micro-benchmarks-7.5.1/c/mpi/pt2pt/standard/osu_bw -d cuda D D \
| tee /shared/ngt-benchmarks/osu/non-cuda-aware-gpu-eth

# HOST
## IB
mpirun --hostfile=/shared/inter-hostfile --allow-run-as-root -np 2 -npernode 1 -mca pml ucx -bind-to-core \
-x UCX_TLS=rc \
/eos/home-h/hannesja/osi-benchmark/osu-micro-benchmarks-7.5.1/c/mpi/pt2pt/standard/osu_bw H H \
| tee /shared/ngt-benchmarks/osu/host-ib

## Ethernet 
mpirun --hostfile=/shared/inter-hostfile --allow-run-as-root -np 2 -npernode 1 -mca pml ucx -bind-to-core \
-x UCX_TLS=tcp,cuda \
/eos/home-h/hannesja/osi-benchmark/no-cuda-osu-micro-benchmarks-7.5.1/c/mpi/pt2pt/standard/osu_bw H H \
| tee /shared/ngt-benchmarks/osu/host-eth

## Ethernet - Non Cuda MPI (more like UCX vs other PML from MPI4)
# Useless ?
# MPI4 --hostfile /shared/non-cuda-inter-hostfile
mpirun --hostfile=/shared/non-cuda-inter-hostfile --allow-run-as-root -np 2 -npernode 1 -bind-to-core \
/eos/home-h/hannesja/osi-benchmark/no-cuda-osu-micro-benchmarks-7.5.1/c/mpi/pt2pt/standard/osu_bw H H \
| tee /shared/ngt-benchmarks/osu/no-cuda-aware-host-eth

