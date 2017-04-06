#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void Generate_list(int a[], int n);
void Print_list(int a[], int n, char* title);
void Read_list(int a[], int n);
void Odd_even_sort(int a[], int n);
void Swap(int* x_p, int* y_p);
void Get_args(int argc, char* argv[], int* n_p);

/*-----------------------------------------------------------------*/
int main(int argc, char* argv[]) {
   int  n;
   Get_args(argc, argv, &n);
   int* a= (int*) malloc(n*sizeof(int));
   
   Generate_list(a, n);
   Print_list(a, n, "Before sort");
   Odd_even_sort(a, n);

   Print_list(a, n, "After sort");
   free(a);
   return 0;
}  


void Generate_list(int a[], int n) {
   int i;

   srandom(time(NULL));
   for (i = 0; i < n; i++)
      a[i] = random() % 100;
} 

void Print_list(int a[], int n, char* title) {
   int i;

   printf("%s:\n", title);
   for (i = 0; i < n; i++)
      printf("%d ", a[i]);
   printf("\n\n");
}  

void Odd_even_sort(int a[], int n) {
   int phase,left,right,i;
   for (phase = 0; phase < n; phase++) {
   if (phase % 2 == 0) {  /* Fase par: los subíndices impares parecen a la izquierda */
      for (i = 1; i < n; i += 2) {
         left = i-1;
         if (a[left] > a[i]) Swap(&a[left],&a[i]);
      }
   } else {  /* Fase impar: los subíndices impares parecen correctos */
      for (i = 1; i < n-1; i += 2) {
         right = i+1;
         if (a[i] > a[right]) Swap(&a[i], &a[right]);
      }
   }
}  
}

void Swap(int* x_p, int* y_p) {
   int temp = *x_p;
   *x_p = *y_p;
   *y_p = temp;
}


void Get_args(int argc, char* argv[], int* n_p) {
   *n_p = atoi(argv[1]);
} 

