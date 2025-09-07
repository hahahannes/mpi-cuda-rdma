# MPI
Supported by the Platform via `MpiJobs` for interactive sessions and job submission.

## Non-CUDA-aware MPI
Developer has to copy data back and forth between GPU and host memory before using the MPI API.

```
# on H100 NVL node
make compile-cpu
make run-cpu
# Output: 
```

```
# on H100 NVL node
make compile-gpu
make run-gpu
# Output
```

## CUDA-aware MPI
CUDA-aware MPI allows MPI to directly send/receive GPU memory buffers (i.e., memory allocated with cudaMalloc) without requiring manual staging through host (CPU) memory.

### With RDMA
If RDMA is available (e.g., using InfiniBand with GPUDirect RDMA), the GPU memory can be directly accessed by the NIC (Network Interface Card).
Data stays on the GPU and is transferred directly over the network, avoiding the host CPU entirely.

```
# On H100 SXM node
make compile-cuda-aware
make run-cuda-aware
Output:
```

### Without RDMA
Without RDMA, CUDA-aware MPI will fall back to staging Through Host Memory

- GPU → Host: Data is first copied from GPU memory to host (CPU) memory.
- MPI Transfer: The host memory is used for MPI communication (standard network transfer).
- Host → GPU: On the receiving end, data is copied back from host memory to the GPU memory.

```
make compile-cuda-aware
make run-cuda-aware-disable-rdma
Output:
```

# UCX
UCX is an open-source optimized communication library which supports multiple networks, including RoCE, InfiniBand, uGNI, TCP, shared memory, and others. UCX mixes-and-matches transports and protocols which are available on the system to provide optimal performance. It also has built-in support for GPU transports (with CUDA and ROCm providers) which lets RDMA-capable transports access the GPU memory directly.

## Transport
Which transports does UCX use?
By default, UCX tries to use all available transports, and select best ones according to their performance capabilities and scale (passed as estimated number of endpoints to ucp_init() API).
For example:
- On machines with Ethernet devices only, shared memory will be used for intra-node communication and TCP sockets for inter-node communication.
- On machines with RDMA devices, RC transport will be used for small scale, and DC transport (available with Connect-IB devices and above) will be used for large scale. If DC is not available, UD will be used for large scale.
- If GPUs are present on the machine, GPU transports will be enabled for detecting memory pointer type and copying to/from GPU memory.

## RDMA
If Open MPI includes UCX support, then UCX is enabled and selected by default for InfiniBand and RoCE network devices; typically, no additional parameters are required. In this case, the network port with the highest bandwidth on the system will be used for inter-node communication, and shared memory will be used for intra-node communication. 

### Select network devices
To select a specific network device to use (for example, mlx5_0 device port 1):

```bash
mpirun -np 2 -x UCX_NET_DEVICES=mlx5_0:1 ./app
```

UCX is integrated into Open MPI as a pml layer:
```
-mca pml ucx 
```

### Disabling RDMA

```
(-mca pml ucx?) -x UCX_TLS=tcp,sm
```

FROM nvidia slides:
To run without any GPUDirect flavor set UCX_TLS to only include cuda_copy, e.g. UCX_TLS=rc,sm,cuda_copy and
UCX_IB_GPU_DIRECT_RDMA=no (rc transport uses GPUDirect RDMA otherwise).


TODO
run lqft with and without rdma on 2 nodes 


### Explicitly Enabling RDMA
```
mpirun -np 2 -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc,cuda -x UCX_IB_GPU_DIRECT_RDMA=yes ./app
```