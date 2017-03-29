/*Ping pong con MPI*/ 
#include  <iostream>
#include <string.h>
#include <mpi.h>

using namespace std;
const int MAX_STRING=100;

int main(){
    char greeting[MAX_STRING]; 
    int comm_sz; /*numeros de procesos*/
    int my_rank; /*rank del proceso*/
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if(my_rank!=0){
    cout<<my_rank<<"\t"<<comm_sz<<endl;
    MPI_Send(greeting,strlen(greeting)+1,MPI_CHAR,0,0,MPI_COMM_WORLD);
    }
    else{
    cout<<my_rank<<"\t"<<comm_sz<<endl;
        for (int q=1;q<comm_sz;q++){
        MPI_Recv(greeting,MAX_STRING,MPI_CHAR,q,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        cout<<greeting<<endl;
        }
    }
    MPI_Finalize();
    return 0;
        
    }

