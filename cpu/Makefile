MPICOMP = mpicc

compile: cpu.c
	$(MPICOMP) cpu.c -o benchmark

.PHONY: clean

clean:
	rm -f cpu *.o

mpirun:
	mpirun -np 2 benchmark 50