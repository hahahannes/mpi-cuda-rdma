export HOST=ngt-003-h100-sxm--4ug7wvdv7c43-node-1

TODO
-s 100000000 = 10 MB 

# HOST memory
## Inter Node

### Default
```
ucx_perftest $HOST -t ucp_am_bw -m host -s 100
60 MB/s -> Ethernet
```

```
ucx_perftest $HOST -t ucp_am_bw -m host -s 100000
60600 MB/s = 60 GB/s -> Infiniband
```

### Infiniband
```
UCX_TLS=rc,sm ucx_perftest $HOST -t ucp_am_bw -m host -s 100
400 MB/s
```

```
UCX_TLS=rc,sm ucx_perftest $HOST -t ucp_am_bw -m host -s 100000
60600 MB/s = 60 GB/s
```

### Ethernet
```
UCX_TLS=tcp,sm ucx_perftest $HOST -t ucp_am_bw -m host -s 100
60 MB/s
```

```
UCX_TLS=tcp,sm ucx_perftest $HOST -t ucp_am_bw -m host -s 100000
4000 MB/s = 4 GB/s
```


# CUDA memory

## Intra Node
executed on same node/pod

### Default 
```
CUDA_VISIBLE_DEVICES=0 ucx_perftest -t tag_bw -m cuda -s 10000000 -n 10
CUDA_VISIBLE_DEVICES=1 ucx_perftest 127.0.0.1 -t tag_bw -m cuda -s 100000000 -n 10 
120000 MB/s = 120 GB/s
```

### CUDA IPC
```
CUDA_VISIBLE_DEVICES=0 UCX_TLS=cuda_ipc,cuda_copy,tcp ucx_perftest -t tag_bw -m cuda -s 10000000 -n 10
CUDA_VISIBLE_DEVICES=1 UCX_TLS=cuda_ipc,cuda_copy,tcp ucx_perftest 127.0.0.1 -t tag_bw -m cuda -s 100000000 -n 10 
120000 MB/s = 120 GB/s
```

```
mpirun --hostfile=/shared/hostfile --allow-run-as-root -np 2 -mca pml ucx -bind-to-core -x UCX_RNDV_SCHEME=get_zcopy -x UCX_TLS=rc,cuda_copy,cuda_ipc /eos/home-h/hannesja/osi-benchmark/osu-micro-benchmarks-7.5.1/c/mpi/pt2pt/standard/osu_bw -d cuda D D
120 GB/s
```

### CUDA Copy
```
CUDA_VISIBLE_DEVICES=0 UCX_TLS=cuda_copy,tcp ucx_perftest -t tag_bw -m cuda -s 10000000 -n 10
CUDA_VISIBLE_DEVICES=1 UCX_TLS=cuda_copy,tcp ucx_perftest 127.0.0.1 -t tag_bw -m cuda -s 100000000 -n 10 
1300 MB/s = 1 GB/s
```

```
CUDA_VISIBLE_DEVICES=0 UCX_TLS=cuda_copy,rc ucx_perftest -t tag_bw -m cuda -s 10000000 -n 10
CUDA_VISIBLE_DEVICES=1 UCX_TLS=cuda_copy,rc ucx_perftest 127.0.0.1 -t tag_bw -m cuda -s 100000000 -n 10 
50 GB/s
```

```
mpirun --hostfile=/shared/hostfile --allow-run-as-root -np 2 -mca pml ucx -bind-to-core -x UCX_RNDV_SCHEME=get_zcopy -x UCX_TLS=rc,cuda_copy /eos/home-h/hannesja/osi-benchmark/osu-micro-benchmarks-7.5.1/c/mpi/pt2pt/standard/osu_bw -d cuda D D
50 GB/s
```

## Inter Node
### Default
```
ucx_perftest $HOST -t ucp_am_bw -m cuda -s 15
45 MB/s
```

```
ucx_perftest $HOST -t ucp_am_bw -m cuda -s 100000 -n 10 
4237 MB/s = 4 GB/s -> Infiniband and RDMA
```

### Infiniband
One GPU per node 
```
UCX_TLS=rc,cuda_copy ucx_perftest $HOST -t ucp_am_bw -m cuda -s 100000 -n 10 
3700 MB/s = 4 GB/s
```

```
UCX_TLS=rc,cuda_copy ucx_perftest $HOST -t ucp_am_bw -m cuda -s 1000000 -n 10 
22000 MB/s = 20 GB/s
```

```
UCX_TLS=rc,cuda_copy ucx_perftest $HOST -t ucp_am_bw -m cuda -s 100000000 -n 10 
45933.72 MB/s = 45 GB/s 
```



#### No RDMA
```
UCX_IB_GPU_DIRECT_RDMA=no UCX_TLS=rc,cuda_copy ucx_perftest $HOST -t ucp_am_bw -m cuda -s 100000 -n 10 
134 MB/s 
```

```
UCX_IB_GPU_DIRECT_RDMA=no UCX_TLS=rc,cuda_copy ucx_perftest $HOST -t ucp_am_bw -m cuda -s 1000000 -n 10 
240 MB/s 
```

```
UCX_IB_GPU_DIRECT_RDMA=no UCX_TLS=rc,cuda_copy ucx_perftest $HOST -t ucp_am_bw -m cuda -s 100000000 -n 10 
542.72 MB/s 
```

```
mpirun --hostfile=/shared/hostfile --allow-run-as-root -np 2 -npernode 1 -mca pml ucx -bind-to-core \
-x CUDA_VISIBLE_DEVICES=0 -x UCX_NET_DEVICES=mlx5_0:1 \
-x UCX_IB_GPU_DIRECT_RDMA=no \
/eos/home-h/hannesja/osi-benchmark/osu-micro-benchmarks-7.5.1/c/mpi/pt2pt/standard/osu_bw -d cuda D D
520 MB/s
```

#### With RDMA
```
UCX_IB_GPU_DIRECT_RDMA=yes UCX_TLS=rc,cuda_copy ucx_perftest $HOST -t ucp_am_bw -m cuda -s 100000 -n 10 
4400 MB/s = 4 GB/s 
```

```
UCX_IB_GPU_DIRECT_RDMA=yes UCX_TLS=rc,cuda_copy ucx_perftest $HOST -t ucp_am_bw -m cuda -s 100000000 -n 10 
45768.16 MB/s = 45 GB/s 
```

```
mpirun --hostfile=/shared/hostfile --allow-run-as-root -np 2 -npernode 1 -mca pml ucx -bind-to-core \
-x CUDA_VISIBLE_DEVICES=0 -x UCX_NET_DEVICES=mlx5_0:1 \
-x UCX_IB_GPU_DIRECT_RDMA=yes \
/eos/home-h/hannesja/osi-benchmark/osu-micro-benchmarks-7.5.1/c/mpi/pt2pt/standard/osu_bw -d cuda D D
50 GB/s
```



### Ethernet
```
UCX_TLS=tcp,cuda_copy ucx_perftest $HOST -t ucp_am_bw -m cuda -s 15
1 MB/s
```

```
UCX_TLS=tcp,cuda_copy ucx_perftest $HOST -t ucp_am_bw -m cuda -s 100000 -n 10 
600 MB/s
```

```
UCX_TLS=tcp,cuda_copy ucx_perftest $HOST -t ucp_am_bw -m cuda -s 100000000 -n 10 
1404.89 MB/s
```

```
mpirun --hostfile=/shared/hostfile --allow-run-as-root -np 2 -npernode 1 -mca pml ucx -bind-to-core \
-x CUDA_VISIBLE_DEVICES=0  \
-x UCX_TLS=tcp,cuda \
/eos/home-h/hannesja/osi-benchmark/osu-micro-benchmarks-7.5.1/c/mpi/pt2pt/standard/osu_bw -d cuda D D
500 MB/s
```



#### No RDMA
```
UCX_IB_GPU_DIRECT_RDMA=no UCX_TLS=tcp,cuda_copy ucx_perftest $HOST -t ucp_am_bw -m cuda -s 100000 -n 10 
761 MB/s 
```

```
UCX_IB_GPU_DIRECT_RDMA=no UCX_TLS=tcp,cuda_copy ucx_perftest $HOST -t ucp_am_bw -m cuda -s 100000000 -n 10 
966.07 MB/s
```










# Receiving RDMA device memory
can not be used as sending memory type 

## Sending from host
```
 ucx_perftest $HOST -t ucp_am_bw -m host,rdma -s 15
58 MB/s

 ucx_perftest $HOST -t ucp_am_bw -m host,rdma -s 100000
58 MB/s
```

CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=mlx5_0:1 UCX_TLS=rc,cuda_copy ucx_perftest -t tag_bw -m cuda -s 10000000 -n 10 
CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=mlx5_1:1 UCX_TLS=rc,cuda_copy ucx_perftest $HOST -t tag_bw -m cuda -s 100000000 -n 10 

