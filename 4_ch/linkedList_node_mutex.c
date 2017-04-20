/* linkedList_List_mutex.c
 
 Implementar una lista enlazada ordenada de varios hilos de
  Ints con ops insertar, imprimir, miembro, eliminar, lista libre.
  Esta versión utiliza un mutex simple
 
  Compile: gcc linkedList_node_mutex.c -lpthread
  Uso: linkedList_node_mutex.c <thread_count>
  Entrada: número total de teclas insertadas por el hilo principal
  Número total de operaciones realizadas
  Porcentaje de operaciones que son búsquedas e inserciones (operaciones restantes
  Son borrados).
  Salida: Tiempo transcurrido para realizar las operaciones
 
  Utilice un mutex para controlar el acceso a la lista
 La función aleatoria no es threadsafe. Así que este programa
  Utilizar un generador lineal simple 
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "my_rand.h"
#include "sys/time.h"

/* Las entradas aleatorias son inferiores a MAX_KEY*/
const int MAX_KEY = 100000000;

void GET_TIME(double now) {                        
    struct timeval t;                           
    gettimeofday(&t, NULL);                     
    now = t.tv_sec + t.tv_usec/1000000.0;       
}
/* Struct para lista de nodos */
struct list_node_s {
   int    data;
   struct list_node_s* next;
};

/* variables compartidas*/
struct      list_node_s* head = NULL;  
int         thread_count;
int         total_ops;
double      insert_percent;
double      search_percent;
double      delete_percent;
pthread_mutex_t mutex;
pthread_mutex_t count_mutex;
int         member_total=0, insert_total=0, delete_total=0;


int  Is_empty(void) {
   if (head == NULL)
      return 1;
   else
      return 0;
}


/* Inserte el valor en la ubicación numérica correcta en la lista */
/* Si el valor no está en la lista, devuelve 1, de lo contrario devuelve 0 */
int Insert(int value) {
   struct list_node_s* curr = head;
   struct list_node_s* pred = NULL;
   struct list_node_s* temp;
   int rv = 1;
   
   while (curr != NULL && curr->data < value) {
      pred = curr;
      curr = curr->next;
   }

   if (curr == NULL || curr->data > value) {
      temp = malloc(sizeof(struct list_node_s));
      temp->data = value;
      temp->next = curr;
      if (pred == NULL)
         head = temp;
      else
         pred->next = temp;
   } else { /* valor en lista */
      rv = 0;
   }

   return rv;
}  

void Print(void) {
   struct list_node_s* temp;

   printf("list = ");

   temp = head;
   while (temp != (struct list_node_s*) NULL) {
      printf("%d ", temp->data);
      temp = temp->next;
   }
   printf("\n");
}  

int  Member(int value) {
   struct list_node_s* temp;

   temp = head;
   while (temp != NULL && temp->data < value)
      temp = temp->next;

   if (temp == NULL || temp->data > value) {
      return 0;
   } else {
      return 1;
   }
}

/* Elimina el valor de la lista */
/* Si el valor está en la lista, devuelve 1, else return 0 */
int Delete(int value) {
   struct list_node_s* curr = head;
   struct list_node_s* pred = NULL;
   int rv = 1;

   /* Buscar valor*/
   while (curr != NULL && curr->data < value) {
      pred = curr;
      curr = curr->next;
   }
   
   if (curr != NULL && curr->data == value) {
      if (pred == NULL) { /* prmer elemento en lista */
         head = curr->next;
#        ifdef DEBUG
         printf("Freeing %d\n", value);
#        endif
         free(curr);
      } else { 
         pred->next = curr->next;
#        ifdef DEBUG
         printf("Freeing %d\n", value);
#        endif
         free(curr);
      }
   } else { /* No en lista */
      rv = 0;
   }

   return rv;
}  

void Free_list(void) {
   struct list_node_s* current;
   struct list_node_s* following;

   if (Is_empty()) return;
   current = head; 
   following = current->next;
   while (following != NULL) {
      printf("Freeing %d\n", current->data);
      free(current);
      current = following;
      following = current->next;
   }
   printf("Freeing %d\n", current->data);
   free(current);
}  


void* Thread_work(void* rank) {
   long my_rank = (long) rank;
   int i, val;
   double which_op;
   unsigned seed = my_rank + 1;
   int my_member=0, my_insert=0, my_delete=0;
   int ops_per_thread = total_ops/thread_count;

   for (i = 0; i < ops_per_thread; i++) {
      which_op = my_drand(&seed);
      val = my_rand(&seed) % MAX_KEY;
      if (which_op < search_percent) {
         pthread_mutex_lock(&mutex);
         Member(val);
         pthread_mutex_unlock(&mutex);
         my_member++;
      } else if (which_op < search_percent + insert_percent) {
         pthread_mutex_lock(&mutex);
         Insert(val);
         pthread_mutex_unlock(&mutex);
         my_insert++;
      } else { /* delete */
         pthread_mutex_lock(&mutex);
         Delete(val);
         pthread_mutex_unlock(&mutex);
         my_delete++;
      }
   }  
   pthread_mutex_lock(&count_mutex);
   member_total += my_member;
   insert_total += my_insert;
   delete_total += my_delete;
   pthread_mutex_unlock(&count_mutex);

   return NULL;
}  


int main(int argc, char* argv[]) {
   long i; 
   int key, success, attempts;
   pthread_t* thread_handles;
   int inserts_in_main;
   unsigned seed = 1;
   double start, finish;

   if (argc != 2) Usage(argv[0]);
   thread_count = strtol(argv[1],NULL,10);

    printf("¿Cuántas claves se deben insertar en el hilo principal?\n");
   scanf("%d", &inserts_in_main);
   printf("¿Cuántas operaciones totales deben realizar los hilos?\n");
   scanf("%d", &total_ops);
   printf("Porcentaje de operaciones que deberían ser búsquedas? (Entre 0 y 1)\n");
   scanf("%lf", &search_percent);
   printf("Porcentaje de operaciones que deberían ser búsquedas? (Entre 0 y 1))\n");
   scanf("%lf", &insert_percent);
   delete_percent = 1.0 - (search_percent + insert_percent);
   
   /* Intentar insertar las llaves de inserts_in_main, pero dar para arriba después de */
   i = attempts = 0;
   while ( i < inserts_in_main && attempts < 2*inserts_in_main ) {
      key = my_rand(&seed) % MAX_KEY;
      success = Insert(key);
      attempts++;
      if (success) i++;
   }
   printf("Insertar %ld claves en lista vacias\n", i);

   printf("antes threads, list = \n");
   Print();
   printf("\n");

   thread_handles = malloc(thread_count*sizeof(pthread_t));
   pthread_mutex_init(&mutex, NULL);
   pthread_mutex_init(&count_mutex, NULL);

   GET_TIME(start);
   for (i = 0; i < thread_count; i++)
      pthread_create(&thread_handles[i], NULL, Thread_work, (void*) i);

   for (i = 0; i < thread_count; i++)
      pthread_join(thread_handles[i], NULL);
   GET_TIME(finish);

   printf("tiempo = %e \n", finish - start);
   printf("Total ops = %d\n", total_ops);
   printf("ops member = %d\n", member_total);
   printf("ops insert = %d\n", insert_total);
   printf("ops delete = %d\n", delete_total);
   
   printf("After threads terminate, list = \n");
    Print();
   printf("\n");

   Free_list();
   pthread_mutex_destroy(&mutex);
   pthread_mutex_destroy(&count_mutex);
   free(thread_handles);

   return 0;
}  
