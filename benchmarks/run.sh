TASKS=2

mpirun -np $TASKS cpu
mpirun -np $TASKS cuda-staged
mpirun -np $TASKS cuda-aware-gdr
mpirun -np $TASKS cuda-aware-no-gdr
