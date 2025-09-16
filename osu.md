https://mvapich.cse.ohio-state.edu/benchmarks/

```bash
./configure CC=mpicc CXX=mpicxx -prefix=/eos/home-h/hannesja/osi-benchmark/osu-micro-benchmarks-7.5.1 \
--enable-cuda --with-cuda=/usr/local/cuda-12.8
make
make install
```


