#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>

float *create_rand_nums(int num_elements) {
  float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
  assert(rand_nums != NULL);
  int i;
  for (i = 0; i < num_elements; i++) {
    rand_nums[i] = (rand() / (float)RAND_MAX);
  }
  return rand_nums;
}
void my_reduce(void* data, void* recibe, int count, MPI_Datatype datatype,MPI_Op op, int root,
              MPI_Comm communicator) {
  int world_rank;
  MPI_Comm_rank(communicator, &world_rank);
  int world_size;
  MPI_Comm_size(communicator, &world_size);

  if (world_rank != root) {
         MPI_Send(data, count, datatype, root, 0, communicator);
    
  } else {
    
    
 int i;
 for (i = 0; i < world_size; i++) {
      if (i != world_rank) {    
	MPI_Recv(data, count, datatype, i, 0, communicator, MPI_STATUS_IGNORE);
          }
    }
}
}

int main(int argc, char** argv) {

  int num_elements_per_proc = 1000000;
  MPI_Init(NULL, NULL);

  double total_my_bcast_time = 0.0;
  double total_mpi_bcast_time = 0.0;

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  srand(time(NULL)*world_rank);
  float *rand_nums = NULL;

  rand_nums = create_rand_nums(num_elements_per_proc);

  float local_sum = 0;
  int i;
  for (i = 0; i < num_elements_per_proc; i++) {
    local_sum += rand_nums[i];
  }

  printf("Suma local para el proceso %d es  %f, promedio = %f\n",
         world_rank, local_sum, local_sum / num_elements_per_proc);

  float global_sum;
   MPI_Barrier(MPI_COMM_WORLD);
   total_my_bcast_time -= MPI_Wtime(); 

 my_reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0,
             MPI_COMM_WORLD);

MPI_Barrier(MPI_COMM_WORLD);
    total_my_bcast_time += MPI_Wtime();

 MPI_Barrier(MPI_COMM_WORLD);
    total_mpi_bcast_time -= MPI_Wtime();

  MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0,
             MPI_COMM_WORLD);
     MPI_Barrier(MPI_COMM_WORLD);
    total_mpi_bcast_time += MPI_Wtime();
 

  if (world_rank == 0) {
    printf("Suma total = %f, promedio = %f\n", global_sum,
           global_sum / (world_size * num_elements_per_proc));
  }
  printf("my_reduce tiempo promedio = %lf\n", total_my_bcast_time / num_elements_per_proc );
  printf("MPI_Reduce tiempo promedio= %lf\n", total_mpi_bcast_time / num_elements_per_proc );

  free(rand_nums);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
 
