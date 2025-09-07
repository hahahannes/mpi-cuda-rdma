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

	if(size != 2){
		if(rank == 0){
			printf("This program requires exactly 2 MPI ranks, but you are attempting to use %d! Exiting...\n", size);
		}
		MPI_Finalize();
		exit(0);
	}

	long int N = 2;
	double *A = (double*)malloc(N*sizeof(double));

	double start_time, stop_time, elapsed_time;
	start_time = MPI_Wtime();
	
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
	fprintf("%li,%.9f,%.9f\n", elapsed_time);

	free(A);

	MPI_Finalize();

	return 0;
}
