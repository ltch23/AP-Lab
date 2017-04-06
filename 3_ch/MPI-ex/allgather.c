
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int my_rank, comm_sz;
MPI_Comm comm;

void Get_dims(int argc, char* argv[], int* m_p, int* local_m_p, int* n_p, int* local_n_p);
void Allocate_arrays(double** local_A_pp, double** local_x_pp, double** local_y_pp, int local_m, int n, int local_n);
void Read_matrix(char prompt[], double local_A[], int m, int local_m,int n, int local_n);
void Generate_matrix(double local_A[], int local_m, int n);
void Print_matrix(char title[], double local_A[], int m, int local_m,int n);
void Mat_vect_mult(double local_A[], double local_x[], double local_y[], double x[], 
                   int m, int local_m, int n, int local_n);

/*-------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
   double* local_A;
   double* local_x;
   double* local_y;
   double* x;
   int m, local_m, n, local_n;
   double start, finish, loc_elapsed, elapsed;

   MPI_Init(&argc, &argv);
   comm = MPI_COMM_WORLD;
   MPI_Comm_size(comm, &comm_sz);
   MPI_Comm_rank(comm, &my_rank);

   Get_dims(argc, argv, &m, &local_m, &n, &local_n);
   Allocate_arrays(&local_A, &local_x, &local_y, local_m, n, local_n);

   Read_matrix("A", local_A, m, local_m, n, local_n);
   Read_matrix("X", local_x, m, local_m, n, local_n);
   
   srandom(my_rank);
   //Generate_matrix(local_A, local_m, n);
   //Generate_matrix(local_x, local_m, n);
   
   Print_matrix("A", local_A, m, local_m, n);
   Print_matrix("X", local_x, m, local_m, n);
   
   
   x = malloc(n*sizeof(double));
  
   Mat_vect_mult(local_A, local_x, local_y, x, m, local_m, n, local_n);
   Print_matrix("Y", local_y, m, local_m, n);
   
   free(local_A);
   free(local_x);
   free(local_y);
   free(x);
   MPI_Finalize();
   return 0;
}  

/*-------------------------------------------------------------------*/
void Get_dims(
      int       argc       /* in  */,
      char*     argv[]     /* in  */,
      int*      m_p        /* out */, 
      int*      local_m_p  /* out */,
      int*      n_p        /* out */,
      int*      local_n_p  /* out */) {
    if (my_rank == 0) {
      if (argc != 2) 
         *m_p = *n_p = 0;
      else 
         *m_p = *n_p = strtol(argv[1], NULL, 10);
   }
   MPI_Bcast(m_p, 1, MPI_INT, 0, comm);
   MPI_Bcast(n_p, 1, MPI_INT, 0, comm);
   *local_m_p = *m_p/comm_sz;
   *local_n_p = *n_p/comm_sz;
}  

/*-------------------------------------------------------------------*/
void Allocate_arrays(
      double**  local_A_pp  /* out */, 
      double**  local_x_pp  /* out */, 
      double**  local_y_pp  /* out */, 
      int       local_m     /* in  */, 
      int       n           /* in  */,   
      int       local_n     /* in  */) {

   *local_A_pp = malloc(local_m*n*sizeof(double));
   *local_x_pp = malloc(local_n*sizeof(double));
   *local_y_pp = malloc(local_m*sizeof(double));

   
}  

/*-------------------------------------------------------------------*/
void Read_matrix(
      char      prompt[]   /* in  */, 
      double    local_A[]  /* out */, 
      int       m          /* in  */, 
      int       local_m    /* in  */, 
      int       n          /* in  */,
      int       local_n    /* in  */) {
   double* A = NULL;
   int i, j;

   if (my_rank == 0) {
      A = malloc(m*n*sizeof(double));
      printf("Enter the matrix %s\n", prompt);
      for (i = 0; i < m; i++)
         for (j = 0; j < n; j++)
            scanf("%lf", &A[i*n+j]);
      MPI_Scatter(A, local_m*n, MPI_DOUBLE, 
            local_A, local_m*n, MPI_DOUBLE, 0, comm);
      free(A);
   } else {
      MPI_Scatter(A, local_m*n, MPI_DOUBLE, 
            local_A, local_m*n, MPI_DOUBLE, 0, comm);
   }
} 


/*-------------------------------------------------------------------*/
void Generate_matrix(double local_A[], int local_m, int n) {
   int i, j;

#  ifndef DEBUG
   for (i = 0; i < local_m; i++)
      for (j = 0; j < n; j++) 
         local_A[i*n + j] = ((double) random())/((double) RAND_MAX);
#  else
   for (i = 0; i < local_m; i++)
      for (j = 0; j < n; j++) 
         local_A[i*n + j] = my_rank + i;
#  endif
}

/*-------------------------------------------------------------------*/
void Print_matrix(
      char      title[]    /* in */,
      double    local_A[]  /* in */, 
      int       m          /* in */, 
      int       local_m    /* in */, 
      int       n          /* in */) {
   double* A = NULL;
   int i, j;

   if (my_rank == 0) {
      A = malloc(m*n*sizeof(double));
      MPI_Gather(local_A, local_m*n, MPI_DOUBLE,
            A, local_m*n, MPI_DOUBLE, 0, comm);
      printf("\nThe matrix %s\n", title);
      for (i = 0; i < m; i++) {
         for (j = 0; j < n; j++)
            printf("%f ", A[i*n+j]);
         printf("\n");
      }
      printf("\n");
      free(A);
   } else {
      MPI_Gather(local_A, local_m*n, MPI_DOUBLE,
            A, local_m*n, MPI_DOUBLE, 0, comm);
   }
} 

/*-------------------------------------------------------------------*/
void Mat_vect_mult(
      double    local_A[]  /* in  */, 
      double    local_x[]  /* in  */, 
      double    local_y[]  /* out */,
      double    x[]        /* scratch */,
      int       m          /* in  */,
      int       local_m    /* in  */, 
      int       n          /* in  */,
      int       local_n    /* in  */) {
   int local_i, j;

   MPI_Allgather(local_x, local_n, MPI_DOUBLE,
         x, local_n, MPI_DOUBLE, comm);

   for (local_i = 0; local_i < local_m; local_i++) {
      local_y[local_i] = 0.0;
      for (j = 0; j < n; j++)
         local_y[local_i] += local_A[local_i*n+j]*x[j];
   }
}  
