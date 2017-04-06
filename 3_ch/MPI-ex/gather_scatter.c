#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int my_rank, comm_sz;
MPI_Comm comm;

void Get_dims(int argc, char* argv[],int* n_p, int* local_n_p);
void Allocate_arrays(double** local_x_pp, int n, int local_n);
void Read_vector(char prompt[], double local_vec[], int n, int local_n);
void Generate_vector(double local_x[], int local_n);
void Print_vector(char title[], double local_vec[], int n,
      int local_n);
/*-------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
   double* local_x;
   int  n, local_n;

   MPI_Init(&argc, &argv);
   comm = MPI_COMM_WORLD;
   MPI_Comm_size(comm, &comm_sz);
   MPI_Comm_rank(comm, &my_rank);

   Get_dims(argc, argv, &n, &local_n);
   Allocate_arrays(&local_x, n, local_n);

   srandom(my_rank);
   Read_vector("x", local_x, n, local_n);
   //Generate_vector(local_x, local_n);
   Print_vector("x", local_x, n, local_n);
   free(local_x);
   MPI_Finalize();
   return 0;
}  

/*-------------------------------------------------------------------*/
void Get_dims(
      int       argc       /* in  */,
      char*     argv[]     /* in  */,
      int*      n_p        /* out */,
      int*      local_n_p  /* out */) {

   if (my_rank == 0) {
      if (argc != 2) 
         *n_p = 0;
      else 
         *n_p = strtol(argv[1], NULL, 10);
   }
   MPI_Bcast(n_p, 1, MPI_INT, 0, comm);
   *local_n_p = *n_p/comm_sz;
}  /* Get_dims */

/*-------------------------------------------------------------------*/
void Allocate_arrays(
      double**  local_x_pp  /* out */, 
      int       n           /* in  */,   
      int       local_n     /* in  */) {

   *local_x_pp = malloc(local_n*sizeof(double));

}  

/*-------------------------------------------------------------------*/
void Read_vector(
      char      prompt[]     /* in  */, 
      double    local_vec[]  /* out */, 
      int       n            /* in  */,
      int       local_n      /* in  */) {
   double* vec = NULL;
    int i;
   if (my_rank == 0) {
      vec = malloc(n*sizeof(double));
      printf("Enter the vector %s\n", prompt);
      for (i = 0; i < n; i++)
         scanf("%lf", &vec[i]);
      MPI_Scatter(vec, local_n, MPI_DOUBLE,local_vec, local_n, MPI_DOUBLE, 0, comm);
    printf("Processor %d has data: ", my_rank);  
    for (int i=0; i<n/comm_sz; i++)
            printf("%f ", local_vec[i]);
        printf("\n");
    free(vec);
   } else {
      MPI_Scatter(vec, local_n, MPI_DOUBLE,local_vec, local_n, MPI_DOUBLE, 0, comm);
        printf("Processor %d has data: ", my_rank);
      for (int i=0; i<n/comm_sz; i++)
            printf("%f ", local_vec[i]);
        printf("\n");
   }
}  

/*-------------------------------------------------------------------*/
void Generate_vector(double local_x[], int local_n) {
   int i;

#  ifndef DEBUG
   for (i = 0; i < local_n; i++)
      local_x[i] = ((double) random())/((double) RAND_MAX);
#  else
   for (i = 0; i < local_n; i++)
      local_x[i] = my_rank + 1;

#  endif
}  

/*-------------------------------------------------------------------*/
void Print_vector(
      char      title[]     /* in */, 
      double    local_vec[] /* in */, 
      int       n           /* in */,
      int       local_n     /* in */) {
   double* vec = NULL;
   int i;

   if (my_rank == 0) {
      vec = malloc(n*sizeof(double));
      MPI_Gather(local_vec, local_n, MPI_DOUBLE,
            vec, local_n, MPI_DOUBLE, 0, comm);
      printf("\nThe vector %s\n", title);
      for (i = 0; i < n; i++)
         printf("%f ", vec[i]);
      printf("\n");
      free(vec);
   }  else {
      MPI_Gather(local_vec, local_n, MPI_DOUBLE,
            vec, local_n, MPI_DOUBLE, 0, comm);
   }
}  
