
- 2 nodes with 2 GPUS
- SXM 
- QUDA does P2P by itself https://github.com/lattice/quda/wiki/Multi-GPU-Support/#peer-to-peer-communication
- 48 dim

## Default 
QUDA_ENABLE_P2P=0 QUDA_ENABLE_GDR=1 TUNE_RESULT_DIR=/shared/ngt-benchmarks/lqft/ib-rdma-ipc ./run.sh

## Infiniband - RDMA - IPC
QUDA_ENABLE_P2P=0 UCX_IB_GPU_DIRECT_RDMA=yes UCX_TLS=rc,sm,cuda_ipc,cuda_copy QUDA_ENABLE_GDR=1 TUNE_RESULT_DIR=/shared/ngt-benchmarks/lqft/ib-rdma-ipc ./run.sh


Before Tuning
QUDA_ENABLE_P2P=0
invertQUDA = 38.238
18890.1 Gflops

After Tuning
QUDA_ENABLE_P2P=0
invertQUDA = 3.319
20984.3 Gflops


## Infiniband - RDMA - No IPC
QUDA_ENABLE_P2P=0 UCX_IB_GPU_DIRECT_RDMA=yes UCX_TLS=rc,sm,cuda_copy QUDA_ENABLE_GDR=1 TUNE_RESULT_DIR=/shared/ngt-benchmarks/lqft/ib-rdma-no-ipc ./run.sh

QUDA_ENABLE_P2P=0
invertQUDA = 
 Gflops

## Infiniband - No RDMA - IPC
QUDA_ENABLE_P2P=0 UCX_IB_GPU_DIRECT_RDMA=no UCX_TLS=rc,sm,cuda_ipc,cuda_copy QUDA_ENABLE_GDR=1 TUNE_RESULT_DIR=/shared/ngt-benchmarks/lqft/ib-no-rdma-ipc ./run.sh

Before Tuning
invertQUDA = 42.047
18894.1 Gflops

After Tuning
3.275
22302.6 Gflops

## Infiniband - No RDMA - No IPC
QUDA_ENABLE_P2P=0 UCX_IB_GPU_DIRECT_RDMA=no UCX_TLS=rc,sm,cuda_copy QUDA_ENABLE_GDR=1 TUNE_RESULT_DIR=/shared/ngt-benchmarks/lqft/ib-no-rdma-no-ipc ./run.sh

Before Tuning
invertQUDA = 
Gflops

After Tuning


## Ethernet - IPC
QUDA_ENABLE_P2P=0 UCX_TLS=tcp,sm,cuda_ipc,cuda_copy QUDA_ENABLE_GDR=1 TUNE_RESULT_DIR=/shared/ngt-benchmarks/lqft/eth-no-rdma-ipc ./run.sh

Before tuning:
45.278
10634.1

After tuning:
6.073
11792.8

## Ethernet - No IPC
UCX_IB_GPU_DIRECT_RDMA=no QUDA_ENABLE_P2P=0 UCX_TLS=tcp,sm,cuda_copy QUDA_ENABLE_GDR=1 TUNE_RESULT_DIR=/shared/ngt-benchmarks/lqft/eth-no-rdma-no-ipc ./run.sh

Before tuning:


After tuning:
