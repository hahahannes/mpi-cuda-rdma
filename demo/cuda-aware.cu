#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
do{                                                                                       \
	cudaError_t cuErr = call;                                                             \
	if(cudaSuccess != cuErr){                                                             \
		printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
		exit(0);                                                                            \
	}                                                                                     \
}while(0)

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
	double *DATA = (double*)malloc(N*sizeof(double));

	if (DATA == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    for (size_t i = 0; i < N; i++) {
        DATA[i] = 0.0;
    }

	double *DATA_DEVICE;
	cudaErrorCheck(cudaMalloc(&DATA_DEVICE, N*sizeof(double)));
	cudaErrorCheck(cudaMemcpy(DATA_DEVICE, DATA, N*sizeof(double), cudaMemcpyHostToDevice));
	

	double start_time, stop_time, elapsed_time;
	start_time = MPI_Wtime();

	if(rank == 0){
		MPI_Send(DATA_DEVICE, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
		MPI_Recv(DATA_DEVICE, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
	}
	else if(rank == 1){
		MPI_Recv(DATA_DEVICE, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
		MPI_Send(DATA_DEVICE, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
	}

	stop_time = MPI_Wtime();
	elapsed_time = stop_time - start_time;
	printf("%.9f\n", elapsed_time);

	cudaErrorCheck(cudaFree(DATA_DEVICE));
	free(DATA);
	
	MPI_Finalize();

	return 0;
}
