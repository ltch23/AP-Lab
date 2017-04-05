#include <mpi.h>
#include <stdio.h>

int main() {
  const int PING_PONG_LIMIT = 10;

  MPI_Init(NULL, NULL);

  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);//identificador de procesos
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);//procesos 


  int count = 0;
  int partner_rank=(my_rank+1)% 2;
  
  while ((count< PING_PONG_LIMIT)) {
    if (my_rank == count% 2) {
      count++;// Incrementar el contador antes de enviar 
      MPI_Send(&count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);
      printf("ENVIAR\tRank actual:%d  contador :%d 	destino:%d\n", my_rank, count, partner_rank);
      }    
    else {
      MPI_Recv(&count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("RECIVIR\tRank acual:%d contador :%d 	origen :%d\n",
             my_rank, count, partner_rank);
        }
    }
    MPI_Finalize();
     
return 0;
}
