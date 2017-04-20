#include <stdio.h>
#include <stdlib.h>


#ifndef _MY_RAND_H_
#define _MY_RAND_H_

unsigned my_rand(unsigned* a_p);
double my_drand(unsigned* a_p);

#endif

/* Archivo: my_rand.c
  *
implementar un generador de números aleatorios congruenciales lineales
  *
  * My_rand: genera un int sin signo aleatorio en el rango 0 - MR_MODULUS
  * My_drand: genera un double aleatorio en el rango 0 - 1
  *
  * Notas:
  * 1. El generador se toma del artículo de Wikipedia "Congruencial linealGenerador
  * 2. Esto es * no * un generador de números aleatorios muy bueno. Sin embargo, a diferencia de
  * La función de la biblioteca C random (), es * threadsafe: el "estado" de
  * El generador se devuelve en el argumento seed_p.
  * 3. La función principal es sólo un controlador simple.
*/


#define MR_MULTIPLIER 279470273 
#define MR_INCREMENT 0
#define MR_MODULUS 4294967291U
#define MR_DIVISOR ((double) 4294967291U)


#ifdef _MAIN_
int main(void) {
   int n, i; 
   unsigned seed = 1, x;
   double y;

   printf("Cuantos numeros randoms?\n");
   scanf("%d", &n);

   x = my_rand(&seed);
   for (i = 0; i < n; i++) {
      x = my_rand(&x);
      printf("%u\n", x);
   }
   for (i = 0; i < n; i++) {
      y = my_drand(&x);
      printf("%e\n", y);
   }
   return 0;
}
#endif
/* Función: my_rand
  * In / out arg: seed_p
  * Valor de retorno: Un nuevo pseudo-random unsigned int en el rango
  * 0 - MR_MODULUS
  *
  * Notas:
  * 1. Esta es una versión ligeramente modificada del generador en el
  * Artículo de Wikipedia "Generador lineal congruencial"
  * 2. El argumento seed_p almacena el "estado" para la siguiente llamada a
  * la función.
  */

unsigned my_rand(unsigned* seed_p) {
   long long z = *seed_p;
   z *= MR_MULTIPLIER; 
// z += MR_INCREMENT;
   z %= MR_MODULUS;
   *seed_p = z;
   return *seed_p;
}

/* Función: my_drand
  * In / out arg: seed_p
  * Valor de retorno: Un nuevo doble pseudoaleatorio en el rango 0 - 1
  */
double my_drand(unsigned* seed_p) {
   unsigned x = my_rand(seed_p);
   double y = x/MR_DIVISOR;
   return y;
}
