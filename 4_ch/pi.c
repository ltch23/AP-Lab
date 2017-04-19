/*
               arctan(x) = Sum_n=0 ^infty (-1)^n x^(2n+1)/(2n+1), |x| <= 
            Since arctan(1) = pi/4, we can compute
 
               pi = 4*[1 - 1/3 + 1/5 - 1/7 + 1/9 - . . . ]
 
  Run:      pth_pi <numero de threads> <n>
            n es el numero de condiciones del uso de la serie de Maclaurin
            n  sera dividido por numero de threads
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

long thread_count;
long long n;
double sum;

/*Añadir en las condiciones calculados por el hilo que ejecuta este*/
void* Thread_sum(void* rank) {
   long my_rank = (long) rank;
   double factor;
   long long i;
   long long my_n = n/thread_count;
   long long my_first_i = my_n*my_rank;
   long long my_last_i = my_first_i + my_n;

   if (my_first_i % 2 == 0)
      factor = 1.0;
   else
      factor = -1.0;

   for (i = my_first_i; i < my_last_i; i++, factor = -factor) {
      sum += factor/(2*i+1);  
   }

   return NULL;
}

/* solo 1 thread*/
/*Estimar pi usanfo 1 thread (serial)*/
 double Serial_pi(long long n) {
   double sum = 0.0;
   long long i;
   double factor = 1.0;

   for (i = 0; i < n; i++, factor = -factor) {
      sum += factor/(2*i+1);
   }
   return 4.0*sum;

}

int main(int argc, char* argv[]) {
   long       thread;  /* long en caso de sistema de 64-bit */
   pthread_t* thread_handles;

   /* Obtener numero de threads  y condiciones */
   thread_count = strtol(argv[1], NULL, 10);  
   n = strtoll(argv[2], NULL, 10);
   
   thread_handles = malloc (thread_count*sizeof(pthread_t)); 

   sum = 0.0;
   for (thread = 0; thread < thread_count; thread++)  
      pthread_create(&thread_handles[thread], NULL,
          Thread_sum, (void*)thread);  

   for (thread = 0; thread < thread_count; thread++) 
      pthread_join(thread_handles[thread], NULL); 
   sum = 4.0*sum;

   printf("Con n = %lld condiciones,\n", n);
   printf("estimado de pi= %.15f\n", sum);
   sum = Serial_pi(n);
   printf("simple thread estimado   = %.15f\n", sum);
   printf("pi = %.15f\n", 4.0*atan(1.0));
   
   free(thread_handles);
   return 0;
}  

