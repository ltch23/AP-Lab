/* 
 * Implementar una lista enlazada ordenada de varios hilos de Ints con ops insert, print, member, delete, free list.
   Esta versión usa un mutex por nodo de lista
   Compilar: gcc linked_list_mutex.c  -lpthread
   Uso: ./linked_list_mutex.c <thread_count>
   número total de teclas insertadas por el hilo principal
   Número total de operaciones realizadas
   Porcentaje de operaciones que son búsquedas e inserciones
   (operaciones restantes Son borrados.
*/


#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "my_rand.h"
#include <sys/time.h>


/* Las entradas aleatorias son inferiores a MAX_KEY*/
const int MAX_KEY = 100000000;

/* Valores devueltos de Advance_ptrs*/
const int IN_LIST = 1;
const int EMPTY_LIST = -1;
const int END_OF_LIST = 0;

void GET_TIME(double now) {                        
    struct timeval t;                           
    gettimeofday(&t, NULL);                     
    now = t.tv_sec + t.tv_usec/1000000.0;       
}


/* Struct para lista de nodos */
struct list_node_s {
   int    data;
   pthread_mutex_t mutex;
   struct list_node_s* next;
};

/* variables compartidas*/
struct list_node_s* head = NULL;  
pthread_mutex_t head_mutex;
int         thread_count;
int         total_ops;
double      insert_percent;
double      search_percent;
double      delete_percent;
pthread_mutex_t count_mutex;
int         member_total=0, insert_total=0, delete_total=0;


int  Is_empty(void) {
   if (head == NULL)
      return 1;
   else
      return 0;
}


/*Inicialice los punteros pred y curr antes de iniciar la búsqueda 
 * realizada mediante Insertar o Borrar*/
void Init_ptrs(struct list_node_s** curr_pp, struct list_node_s** pred_pp) {
   *pred_pp = NULL;
   pthread_mutex_lock(&head_mutex);
   *curr_pp = head;
   if (head != NULL)
      pthread_mutex_lock(&(head->mutex));
} 

/*Avanzar el par de punters pred y curr durante*/
int Advance_ptrs(struct list_node_s** curr_pp, struct list_node_s** pred_pp) {
   int rv = IN_LIST;
   struct list_node_s* curr_p = *curr_pp;
   struct list_node_s* pred_p = *pred_pp;

   if (curr_p == NULL)
      if (pred_p == NULL)
         return EMPTY_LIST;
      else 
         return END_OF_LIST;
   else { // *curr_pp != NULL
      if (curr_p->next != NULL)
         pthread_mutex_lock(&(curr_p->next->mutex));
      else
         rv = END_OF_LIST;
      if (pred_p != NULL)
         pthread_mutex_unlock(&(pred_p->mutex));
      else
         pthread_mutex_unlock(&head_mutex);
      *pred_pp = curr_p;
      *curr_pp = curr_p->next;
      return rv;
   }
}
   
/* Inserte el valor en la ubicación numérica correcta en la lista */
/* Si el valor no está en la lista, devuelve 1, de lo contrario devuelve 0 */
int Insert(int value) {
   struct list_node_s* curr;
   struct list_node_s* pred;
   struct list_node_s* temp;
   int rv = 1;

   Init_ptrs(&curr, &pred);
   
   while (curr != NULL && curr->data < value) {
      Advance_ptrs(&curr, &pred);
   }

   if (curr == NULL || curr->data > value) {
      temp = malloc(sizeof(struct list_node_s));
      pthread_mutex_init(&(temp->mutex), NULL);
      temp->data = value;
      temp->next = curr;
      if (curr != NULL) 
         pthread_mutex_unlock(&(curr->mutex));
      if (pred == NULL) {
         // Inserting in head of list
         head = temp;
         pthread_mutex_unlock(&head_mutex);
      } else {
         pred->next = temp;
         pthread_mutex_unlock(&(pred->mutex));
      }
   } else { /* value in list */
      if (curr != NULL) 
         pthread_mutex_unlock(&(curr->mutex));
      if (pred != NULL)
         pthread_mutex_unlock(&(pred->mutex));
      else
         pthread_mutex_unlock(&head_mutex);
      rv = 0;
   }

   return rv;
}  

/* No usa bloqueos: no se puede ejecutar con los otros subprocesos */
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

   pthread_mutex_lock(&head_mutex);
   temp = head;
   while (temp != NULL && temp->data < value) {
      if (temp->next != NULL) 
         pthread_mutex_lock(&(temp->next->mutex));
      if (temp == head)
         pthread_mutex_unlock(&head_mutex);
      pthread_mutex_unlock(&(temp->mutex));
      temp = temp->next;
   }

   if (temp == NULL || temp->data > value) {
      if (temp == head)
         pthread_mutex_unlock(&head_mutex);
      if (temp != NULL) 
         pthread_mutex_unlock(&(temp->mutex));
      return 0;
   } else {
      if (temp == head)
         pthread_mutex_unlock(&head_mutex);
      pthread_mutex_unlock(&(temp->mutex));
      return 1;
   }
}

/* Elimina el valor de la lista */
/* Si el valor está en la lista, devuelve 1, else return 0 */
int Delete(int value) {
   struct list_node_s* curr;
   struct list_node_s* pred;
   int rv = 1;

   Init_ptrs(&curr, &pred);

   /* busar valor*/
   while (curr != NULL && curr->data < value) {
      Advance_ptrs(&curr, &pred);
   }
   
   if (curr != NULL && curr->data == value) {
      if (pred == NULL) { /* primer elemento de la lista*/
         head = curr->next;
         pthread_mutex_unlock(&head_mutex);
         pthread_mutex_unlock(&(curr->mutex));
         pthread_mutex_destroy(&(curr->mutex));
         free(curr);
      } else { 
         pred->next = curr->next;
         pthread_mutex_unlock(&(pred->mutex));
         pthread_mutex_unlock(&(curr->mutex));
         pthread_mutex_destroy(&(curr->mutex));
         free(curr);
      }
   } else { /* No en lista */
      if (pred != NULL)
         pthread_mutex_unlock(&(pred->mutex));
      if (curr != NULL)
         pthread_mutex_unlock(&(curr->mutex));
      if (curr == head)
         pthread_mutex_unlock(&head_mutex);
      rv = 0;
   }

   return rv;
}  
/* No usa cerraduras. Sólo se puede ejecutar cuando no hay otros subprocesos
    Acceder a la lista*/
void Free_list(void) {
   struct list_node_s* current;
   struct list_node_s* following;

   if (Is_empty()) return;
   current = head; 
   following = current->next;
   while (following != NULL) {
      free(current);
      current = following;
      following = current->next;
   }
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
         Member(val);
         my_member++;
      } else if (which_op < search_percent + insert_percent) {
         Insert(val);
         my_insert++;
      } else { /* delete */
         Delete(val);
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

   thread_count = strtol(argv[1], NULL, 10);
    
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
    /* 2 * inserts_in_main intentos. */
   i = attempts = 0;
   pthread_mutex_init(&head_mutex, NULL);
   while ( i < inserts_in_main && attempts < 2*inserts_in_main ) {
      key = my_rand(&seed) % MAX_KEY;
      success = Insert(key);
      attempts++;
      if (success) i++;
   }
   printf("Insertar %ld claves en la lista vacía\n", i);

   printf("Antes de threads, list = \n");
   Print();
   printf("\n");

   thread_handles = malloc(thread_count*sizeof(pthread_t));
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
   pthread_mutex_destroy(&head_mutex);
   pthread_mutex_destroy(&count_mutex);
   free(thread_handles);

   return 0;
}
