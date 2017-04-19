#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>



int     thread_count;
int     m, n;
double* A;
double* x;
double* y;

 
void GET_TIME(double now) {                        
    struct timeval t;                           
    gettimeofday(&t, NULL);                     
    now = t.tv_sec + t.tv_usec/1000000.0;       
  }


void Gen_vector(double x[], int n) {
   int i;
   
   for (i = 0; i < n; i++)
      x[i] = random()/((double) RAND_MAX);
}  
void Gen_matrix(double A[], int m, int n) {
   int i, j;
   for (i = 0; i < m; i++)
      for (j = 0; j < n; j++)
         A[i*n+j] = random()/((double) RAND_MAX);
} 


void *Pth_mat_vect(void* rank) {
   long my_rank = (long) rank;
   int i;
   int j; 
   int local_m = m/thread_count; 
   int my_first_row = my_rank*local_m;
   int my_last_row = (my_rank+1)*local_m - 1;

   for (i = my_first_row; i <= my_last_row; i++) {
      y[i] = 0.0;
      for (j = 0; j < n; j++)
          y[i] += A[i*n+j]*x[j];
   }

   return NULL;
}

void Print_matrix( char* title, double A[], int m, int n) {
   int   i, j;

   printf("%s\n", title);
   for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++)
         printf("%6.3f ", A[i*n + j]);
      printf("\n");
   }
}

void Print_vector(char* title, double y[], double m) {
   int   i;

   printf("%s\n", title);
   for (i = 0; i < m; i++)
      printf("%6.3f ", y[i]);
   printf("\n");
}


int main(int argc, char* argv[]) {
   long       thread;
   pthread_t* thread_handles;
   double start, finish;

   thread_count = strtol(argv[1], NULL, 10);
   m = strtol(argv[2], NULL, 10);
   n = strtol(argv[3], NULL, 10);


   thread_handles = malloc(thread_count*sizeof(pthread_t));
   A = malloc(m*n*sizeof(double));
   x = malloc(n*sizeof(double));
   y = malloc(m*sizeof(double));
   
   srandom(1);
   Gen_matrix(A, m, n);
   Print_matrix("matrix: ", A, m, n); 

   Gen_vector(x, n);
   Print_vector("vector: ", x, n); 

   GET_TIME(start);
   
   for (thread = 0; thread < thread_count; thread++)
      pthread_create(&thread_handles[thread], NULL,
         Pth_mat_vect, (void*) thread);

   for (thread = 0; thread < thread_count; thread++)
      pthread_join(thread_handles[thread], NULL);
   GET_TIME(finish);

   Print_vector("multiplicacion: ", y, m); 
   printf("tiempo: %e", finish - start);

   free(A);
   free(x);
   free(y);
   free(thread_handles);

   return 0;
}
