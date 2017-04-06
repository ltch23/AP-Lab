#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int main() {
    int size, rank;
    
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(time(NULL));
    
    int *globaldata=NULL;
    int localdata;

    if (rank == 0) {
        globaldata = malloc(size * sizeof(int) );
        for (int i=0; i<size; i++)
            globaldata[i] = 0;

        printf("Processor %d has data: ", rank);
        for (int i=0; i<size; i++)
            printf("%d ", globaldata[i]);
        printf("\n");
    }
    
    //reparte elemento globaldata a localdata
    MPI_Scatter(globaldata, 1, MPI_INT, &localdata, 1, MPI_INT, 0, MPI_COMM_WORLD);
    srand(time(NULL) + rank);
    //llena numeros aleatorios
    localdata=rand()%100;
    printf("Processor %d has data %d\n", rank, localdata);
    //envia localdata a globaldata
    MPI_Gather(&localdata, 1, MPI_INT, globaldata, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // devuele a proceso 0 nuevo matrix
    if (rank == 0) {
        printf("Processor %d has data: ", rank);
        for (int i=0; i<size; i++)
            printf("%d ", globaldata[i]);
        printf("\n");
    }
    
    //libera
    if (rank == 0)
        free(globaldata);

    MPI_Finalize();
    return 0;
}
