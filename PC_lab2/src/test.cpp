#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Generate data with different length for each process
    int* sendbuf = NULL;
    int* recvbuf = NULL;
    int* displs = NULL;
    int* recvcounts = NULL;
    int sendcount = rank + 1;
    int sum_recvcounts = 0;

    sendbuf = (int*)malloc(sendcount * sizeof(int));
    for (int i = 0; i < sendcount; i++) {
        sendbuf[i] = rank;
    }

    recvcounts = (int*)malloc(size * sizeof(int));
    displs = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        recvcounts[i] = i + 1;
        displs[i] = sum_recvcounts;
        sum_recvcounts += recvcounts[i];
    }

    recvbuf = (int*)malloc(sum_recvcounts * sizeof(int));

    // Call Allgatherv to gather data
    MPI_Allgatherv(sendbuf, sendcount, MPI_INT, recvbuf, recvcounts, displs, MPI_INT, MPI_COMM_WORLD);

    // Print result
    for (int i = 0; i < sum_recvcounts; i++) {
        printf("Process %d: recvbuf[%d] = %d\n", rank, i, recvbuf[i]);
    }

    free(sendbuf);
    free(recvbuf);
    free(recvcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}
