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

	double *DATA = (double*)malloc(N*sizeof(double));
	double *DATA_DEVICE;
	cudaMalloc(&DATA_DEVICE, N*sizeof(double));
	cudaMemcpy(DATA_DEVICE, DATA, N*sizeof(double), cudaMemcpyHostToDevice);
	

	if(rank == 0){
		cudaMemcpy(DATA, DATA_DEVICE, N*sizeof(double), cudaMemcpyDeviceToHost);
		MPI_Send(DATA, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
		MPI_Recv(DATA, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
		cudaMemcpy(DATA_DEVICE, DATA, N*sizeof(double), cudaMemcpyHostToDevice);
	}
	else if(rank == 1){
		MPI_Recv(DATA, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
		cudaMemcpy(DATA_DEVICE, DATA, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(DATA, DATA_DEVICE, N*sizeof(double), cudaMemcpyDeviceToHost);
		MPI_Send(DATA, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
	}

	MPI_Finalize();

	return 0;
}
