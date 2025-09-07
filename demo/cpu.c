#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Status stat;
	int tag1 = 10;
	int tag2 = 20;

	if(size != 2){
		if(rank == 0){
			printf("This program requires exactly 2 MPI ranks, but you are attempting to use %d! Exiting...\n", size);
		}
		MPI_Finalize();
		exit(0);
	}

	size_t N = 6553600;  // 50 MB worth of doubles
	double *A = (double*)malloc(N*sizeof(double));


    if (A == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

	printf("Initialize random data");
    for (size_t i = 0; i < N; i++) {
        A[i] = 0.0;
    }

	double start_time, stop_time, elapsed_time;
	start_time = MPI_Wtime();
	printf("Start sending and receiving data");

	if(rank == 0){
		MPI_Send(A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
		MPI_Recv(A, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
	}
	else if(rank == 1){
		MPI_Recv(A, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
		MPI_Send(A, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
	}

	stop_time = MPI_Wtime();
	elapsed_time = stop_time - start_time;
	printf("%.9f\n", elapsed_time);

	free(A);

	MPI_Finalize();

	return 0;
}
